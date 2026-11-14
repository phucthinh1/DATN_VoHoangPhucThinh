import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# KHỐI ĐỊNH TUYẾN ĐƯỜNG DẪN LIÊN THƯ MỤC (CROSS-DIRECTORY PATH RESOLVER)
# ==============================================================================
# Đảm bảo dù chạy script từ bất kỳ thư mục nào trên Windows/Linux/Colab,
# tiến trình chính và các tiến trình con (worker spawn) đều import thông suốt.
CURRENT_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = CURRENT_DIR.parent
CODE_DIR = SYSTEM_DIR.parent
PREPROC_DIR = CODE_DIR / "pre_processing" / "pre_processing_all_and_branchs"
# Suy ra từ vị trí file thay vì gán cứng ổ D:\ để chạy được khi clone sang Colab/Linux
DEFAULT_DATASET_ROOT = CODE_DIR.parent / "dataset" / "Defactify_dataset"

for p in (str(CURRENT_DIR), str(PREPROC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import 5 module nền tảng đã nghiệm thu ở các bước trước
from split_dataset import get_split_dataframes
from balance_sample import BalancedGroupSampler
from pre_processing_for_all import CommonProcess
from pre_processing_EfficientNet import EfficientNetPreprocessor
from pre_processing_CLIP import CLIPPreprocessor
from pre_processing_FFT import FFTPreprocessor


# ==============================================================================
# CLASS ĐIỀU PHỐI DỮ LIỆU ĐA NHÁNH CHUẨN SOTA: DefactifyDataset
# ==============================================================================
class DefactifyDataset(Dataset):
    """
    BƯỚC 5A: Trạm điều phối dữ liệu trung tâm (Central Multi-Branch Dataset Orchestrator).

    Vai trò cốt lõi:
    1. Tiếp nhận danh sách mẫu từ split_dataset.py hoặc DataFrame đã lọc.
    2. Zero-DataFrame-Overhead: Rút trích DataFrame thành các mảng NumPy / Python List 1 chiều
       ngay trong __init__ để truy xuất O(1), không gọi df.iloc[idx] trong vòng lặp đa luồng.
    3. Điều phối tiền xử lý liên hoàn theo cấu hình `active_branches`:
       - Tầng 1: Gọi CommonProcess đọc ảnh đĩa duy nhất 1 lần, lật ngang đồng bộ, sinh cặp view.
       - Tầng 2: Rẽ nhánh linh hoạt (chỉ kích hoạt các nhánh được yêu cầu, ngắt hoàn toàn
         các nhánh không dùng để tiết kiệm tài nguyên CPU/RAM).
    4. Cơ chế đồng bộ kép set_epoch(): Đồng bộ hạt giống lật ảnh qua shared-memory của CommonProcess
       và xoay vòng tập AI của BalancedGroupSampler (thông qua attach_sampler).
    5. Đóng gói Collate-Safe: Loại bỏ hoàn toàn đối tượng PIL.Image, chuẩn hóa metadata thành
       Tensor phẳng tương thích 100% với default_collate của PyTorch DataLoader.
    """

    VALID_MODES = ("train", "eval", "extract")
    VALID_BRANCHES = ("spatial", "semantic", "spectral")

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        mode: str = "train",
        active_branches: Tuple[str, ...] = ("spatial", "semantic", "spectral"),
        force_flip: Optional[bool] = None,
        crop_size: int = 224,
        flip_prob: float = 0.5,
        jpeg_prob: float = 0.0,
        jpeg_quality_range: Tuple[int, int] = (70, 95),
        fixed_jpeg_quality: Optional[int] = None,
        spatial_view_mode: str = "native",
        fft_mode: str = "raw",
        fft_use_hann: bool = False,
        fft_remove_dc: bool = True,
        seed: int = 42,
    ):
        """
        Khởi tạo DefactifyDataset.

        Args:
            df: pandas DataFrame chứa danh sách mẫu (cần có các cột filepath, label_a, label_b, group_id, model_name).
            dataset_root: Thư mục gốc chứa ảnh thực tế trên ổ đĩa.
            mode: 'train', 'val'/'validation', 'test' hoặc 'eval', 'extract'. Tự động chuẩn hóa về VALID_MODES.
            active_branches: Bộ các nhánh hoạt động, ví dụ ('spatial', 'semantic', 'spectral') hoặc chỉ ('spectral',).
            force_flip: Chỉ định cưỡng bức lật ảnh (True/False) khi mode='extract', mặc định None (tự động theo seed).
            crop_size: Kích thước cạnh vuông cho view_native (mặc định 224).
            flip_prob: Xác suất lật ngang ngẫu nhiên khi mode='train' (mặc định 0.5).
            jpeg_prob: Xác suất nén JPEG ngẫu nhiên khi mode='train' (mặc định 0.0).
            jpeg_quality_range: Dải chất lượng nén JPEG (min, max), mặc định (70, 95).
            fixed_jpeg_quality: Mức nén JPEG cố định phục vụ thực nghiệm độ bền (Robustness test) ở Chương 4.
            spatial_view_mode: 'native' (mặc định) hoặc 'global' cho nhánh EfficientNet.
            fft_mode: 'raw' (mặc định Baseline) hoặc 'gaussian_residual' cho nhánh FFT.
            fft_use_hann: True để bật cửa sổ Hann, mặc định False (Baseline).
            fft_remove_dc: True để triệt tiêu thành phần DC, mặc định True (Baseline tối ưu SOTA).
            seed: Hạt giống ngẫu nhiên khởi tạo cho bộ tiền xử lý chung.
        """
        super().__init__()

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df phải là pandas.DataFrame, nhận được: {type(df)}")
        if len(df) == 0:
            raise ValueError("DataFrame truyền vào DefactifyDataset không được rỗng!")

        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục dataset_root: {self.dataset_root}")

        # Chuẩn hóa tham số mode thân thiện: 'val', 'validation', 'test' -> 'eval'
        raw_mode_str = str(mode).strip().lower()
        if raw_mode_str in ("val", "validation", "test", "eval"):
            self.mode = "eval"
        elif raw_mode_str in ("train", "extract"):
            self.mode = raw_mode_str
        else:
            raise ValueError(
                f"mode không hợp lệ: '{mode}'. Chỉ chấp nhận 'train', 'eval' ('val'/'test'), hoặc 'extract'."
            )

        # Kiểm tra và khóa cứng active_branches
        if isinstance(active_branches, str):
            active_branches = (active_branches,)
        for b in active_branches:
            if b not in self.VALID_BRANCHES:
                raise ValueError(f"Nhánh không hợp lệ: '{b}'. Chỉ chấp nhận trong {self.VALID_BRANCHES}")
        if len(active_branches) == 0:
            raise ValueError("active_branches không được rỗng! Phải bật ít nhất 1 nhánh.")
        self.active_branches: Tuple[str, ...] = tuple(active_branches)

        # Chặn sớm ở __init__ thay vì để CommonProcess văng lỗi giữa chừng trong worker
        if force_flip is not None and self.mode != "extract":
            raise ValueError(f"force_flip chỉ dùng với mode='extract' (hiện tại mode='{self.mode}').")
        self.force_flip = force_flip
        self.crop_size = crop_size
        self.sampler: Optional[Any] = None

        # ----------------------------------------------------------------------
        # Zero-DataFrame-Overhead: Trích xuất DataFrame thành mảng phẳng 1 chiều
        # ----------------------------------------------------------------------
        required_cols = {"filepath", "label_a", "label_b", "group_id", "model_name"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(f"DataFrame thiếu các cột bắt buộc: {missing_cols}")

        # metadata.csv lưu dấu '\' của Windows; trên Linux '\' không phải dấu phân cách thư mục
        # nên phải đổi sang '/', dạng mà pathlib hiểu đúng trên cả Windows lẫn Linux
        self.filepaths: List[str] = [str(p).replace("\\", "/") for p in df["filepath"].tolist()]
        self.labels_a: np.ndarray = df["label_a"].to_numpy(dtype=np.float32)
        self.labels_b: np.ndarray = df["label_b"].to_numpy(dtype=np.int64)
        self.group_ids: np.ndarray = df["group_id"].to_numpy(dtype=np.int64)
        self.model_names: List[str] = [str(m) for m in df["model_name"].tolist()]
        self._num_samples: int = len(self.filepaths)

        # ----------------------------------------------------------------------
        # Khởi tạo các Bộ tiền xử lý chuyên trách (chỉ tạo nhánh được yêu cầu)
        # ----------------------------------------------------------------------
        self.common_proc = CommonProcess(
            mode=self.mode,
            crop_size=self.crop_size,
            flip_prob=flip_prob if self.mode == "train" else 0.0,
            jpeg_prob=jpeg_prob if self.mode == "train" else 0.0,
            jpeg_quality_range=jpeg_quality_range,
            fixed_jpeg_quality=fixed_jpeg_quality,
            seed=seed,
        )

        self.spatial_proc: Optional[EfficientNetPreprocessor] = None
        if "spatial" in self.active_branches:
            self.spatial_proc = EfficientNetPreprocessor(
                view_mode=spatial_view_mode,
                normalize=True,
            )

        self.semantic_proc: Optional[CLIPPreprocessor] = None
        if "semantic" in self.active_branches:
            self.semantic_proc = CLIPPreprocessor(
                image_size=self.crop_size,
                normalize=True,
            )

        self.spectral_proc: Optional[FFTPreprocessor] = None
        if "spectral" in self.active_branches:
            self.spectral_proc = FFTPreprocessor(
                mode=fft_mode,
                use_hann=fft_use_hann,
                remove_dc=fft_remove_dc,
                normalize=True,
                image_size=self.crop_size,
            )

    def attach_sampler(self, sampler: Any) -> "DefactifyDataset":
        """
        Gắn liên kết tham chiếu tới Sampler (ví dụ BalancedGroupSampler) để đồng bộ set_epoch().
        """
        self.sampler = sampler
        return self

    def set_epoch(self, epoch: int) -> None:
        """
        Đồng bộ chu kỳ Epoch:
        1. Cập nhật hạt giống lật ảnh ngẫu nhiên qua Shared-Memory của CommonProcess.
        2. Tự động chuyển epoch sang Sampler được gắn kèm để xoay vòng tập AI nếu có.
        """
        self.common_proc.set_epoch(epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Trích xuất 1 mẫu dữ liệu đã qua tiền xử lý, đóng gói an toàn với PyTorch DataLoader.
        """
        # range()[idx] nhận int/np.int64/Tensor 0-d, quy đổi chỉ số âm, nhưng vẫn báo IndexError khi vượt biên
        # và TypeError với số thực (int() sẽ âm thầm cắt 2.9 -> 2, còn -n-1 -> mẫu cuối)
        idx = range(self._num_samples)[idx]

        rel_path = self.filepaths[idx]
        full_path = self.dataset_root / rel_path

        # Tầng 1: Tiền xử lý chung (Đọc đĩa 1 lần -> EXIF -> Lật đồng bộ -> 2 view PIL)
        views = self.common_proc(
            full_path,
            force_flip=self.force_flip,
            sample_id=rel_path,
        )

        # Trích xuất metadata sạch bằng hàm tiện ích tĩnh đã chuẩn hóa ở Bước 3
        meta = CommonProcess.extract_batch_meta(views)
        meta["kept_area_ratio"] = torch.tensor(
            CLIPPreprocessor.compute_kept_area_ratio(views["orig_size"], image_size=self.crop_size),
            dtype=torch.float32,
        )

        # Chuẩn hóa nhãn phân loại:
        # - 'label'  : int64 (0 hoặc 1) cho tiêu chuẩn CrossEntropy / Metrics
        # - 'label_a': float32 (0.0 hoặc 1.0) cho BCEWithLogitsLoss
        # - 'label_b': int64 (0..5) định danh cụ thể Generator
        label_a_val = float(self.labels_a[idx])
        item: Dict[str, Any] = {
            "label": torch.tensor(int(round(label_a_val)), dtype=torch.int64),
            "label_a": torch.tensor(label_a_val, dtype=torch.float32),
            "label_b": torch.tensor(int(self.labels_b[idx]), dtype=torch.int64),
            "group_id": torch.tensor(int(self.group_ids[idx]), dtype=torch.int64),
            "sample_idx": idx,
            "model_name": self.model_names[idx],
            "filepath": rel_path,
            "meta": meta,
        }

        # Tầng 2: Rẽ nhánh tiền xử lý (chỉ chạy những nhánh được kích hoạt)
        if self.spatial_proc is not None:
            item["spatial"] = self.spatial_proc(views)
        if self.semantic_proc is not None:
            item["semantic"] = self.semantic_proc(views)
        if self.spectral_proc is not None:
            item["spectral"] = self.spectral_proc(views)

        return item


# ==============================================================================
# HÀM TIỆN ÍCH TRỌN GÓI: build_dataloaders
# ==============================================================================
def _loader_kwargs(num_workers: int, pin_memory: Optional[bool], persistent_workers: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
    }
    if num_workers > 0:
        # Epoch được đồng bộ qua shared memory của CommonProcess nên worker sống lâu vẫn nhận đúng epoch mới,
        # đồng thời tránh phải spawn lại worker (rất chậm trên Windows) ở mỗi epoch
        kwargs["persistent_workers"] = persistent_workers
    return kwargs


def build_dataloaders(
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    metadata_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    active_branches: Tuple[str, ...] = ("spatial", "semantic", "spectral"),
    spatial_view_mode: str = "native",
    flip_prob: float = 0.5,
    jpeg_prob: float = 0.0,
    jpeg_quality_range: Tuple[int, int] = (70, 95),
    fixed_jpeg_quality: Optional[int] = None,
    fft_mode: str = "raw",
    fft_use_hann: bool = False,
    fft_remove_dc: bool = True,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    persistent_workers: bool = True,
    eval_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, BalancedGroupSampler]:
    """
    Khởi tạo trọn gói toàn bộ pipeline dữ liệu (Train, Val, Test DataLoaders và BalancedGroupSampler).

    Quy tắc nghiệp vụ cốt lõi:
    - Train Loader: Dùng BalancedGroupSampler (đảm bảo 16 Real : 16 AI mỗi batch, xoay vòng 100% AI qua 3 epochs).
      Bắt buộc đặt shuffle=False (do Sampler đã xáo trộn). `drop_last` chỉ áp dụng cho Train.
    - Val Loader: Đánh giá trên tập Known (5,836 ảnh), phân phối tự nhiên, shuffle=False, LUÔN drop_last=False.
    - Test Loader: Đánh giá toàn diện trên cả 6 nguồn Known + Unseen (43,800 ảnh), shuffle=False, LUÔN drop_last=False
      (nếu bỏ batch cuối thì metric sai lệch vì thiếu ảnh của nguồn cuối cùng trong metadata).
    - Tái lập 100%: `generator` của Train Loader được gieo bằng `seed`, nên worker_info.seed (đầu vào của RNG lật/cắt
      trong CommonProcess) giống hệt nhau giữa các lần chạy cùng seed.
    - pin_memory: Tự động nhận diện GPU (True nếu có CUDA, False nếu CPU) để không bắn cảnh báo vàng.
    - Tối ưu RAM Windows: persistent_workers=True chỉ áp dụng cho Train Loader để tăng tốc lặp epoch;
      Val và Test Loaders dùng persistent_workers=False để tự giải phóng tiến trình worker khi xong lượt đánh giá.
    - Bắt buộc gọi `train_loader.dataset.set_epoch(epoch)` ở đầu mỗi epoch, trước vòng lặp batch.

    Returns:
        (train_loader, val_loader, test_loader, train_sampler)
    """
    root_path = Path(dataset_root)
    meta_p = Path(metadata_path) if metadata_path else root_path / "metadata.csv"

    # Bước 1: Rút trích bảng dữ liệu phân chia chuẩn trên RAM (không ghi đè file đĩa)
    train_df, val_df, test_df = get_split_dataframes(metadata_path=str(meta_p))

    train_loader_kwargs = _loader_kwargs(num_workers, pin_memory, persistent_workers)
    eval_loader_kwargs = _loader_kwargs(num_workers, pin_memory, persistent_workers=False)
    eval_bs = eval_batch_size or batch_size

    # Bước 2: Khởi tạo Sampler cân bằng nhóm theo từng cặp Real - AI cho tập Train
    train_sampler = BalancedGroupSampler(train_df, seed=seed)

    # Bước 3: Khởi tạo 3 Dataset chuyên biệt
    train_dataset = DefactifyDataset(
        df=train_df,
        dataset_root=root_path,
        mode="train",
        active_branches=active_branches,
        spatial_view_mode=spatial_view_mode,
        flip_prob=flip_prob,
        jpeg_prob=jpeg_prob,
        jpeg_quality_range=jpeg_quality_range,
        fixed_jpeg_quality=None,  # Train không dùng fixed JPEG
        fft_mode=fft_mode,
        fft_use_hann=fft_use_hann,
        fft_remove_dc=fft_remove_dc,
        seed=seed,
    )
    # Tự động gắn liên kết để đồng bộ kép khi gọi train_dataset.set_epoch(epoch)
    train_dataset.attach_sampler(train_sampler)

    val_dataset = DefactifyDataset(
        df=val_df,
        dataset_root=root_path,
        mode="eval",
        active_branches=active_branches,
        spatial_view_mode=spatial_view_mode,
        flip_prob=0.0,
        jpeg_prob=0.0,
        jpeg_quality_range=jpeg_quality_range,
        fixed_jpeg_quality=None,  # Val tập Known nguyên bản
        fft_mode=fft_mode,
        fft_use_hann=fft_use_hann,
        fft_remove_dc=fft_remove_dc,
        seed=seed,
    )

    test_dataset = DefactifyDataset(
        df=test_df,
        dataset_root=root_path,
        mode="eval",
        active_branches=active_branches,
        spatial_view_mode=spatial_view_mode,
        flip_prob=0.0,
        jpeg_prob=0.0,
        jpeg_quality_range=jpeg_quality_range,
        fixed_jpeg_quality=fixed_jpeg_quality,  # Nhận fixed_jpeg_quality phục vụ Robustness Test
        fft_mode=fft_mode,
        fft_use_hann=fft_use_hann,
        fft_remove_dc=fft_remove_dc,
        seed=seed,
    )

    # Bước 4: Đóng gói thành PyTorch DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
        **train_loader_kwargs,
    )

    val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, drop_last=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False, drop_last=False, **eval_loader_kwargs)

    return train_loader, val_loader, test_loader, train_sampler


def build_extract_loader(
    split: str,
    force_flip: bool = False,
    active_branches: Tuple[str, ...] = ("semantic",),
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    metadata_path: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """
    Loader tất định (mode='extract') phục vụ trích xuất & lưu đệm đặc trưng của nhánh đóng băng (CLIP) cho Fusion MLP.

    - Duyệt tuần tự TOÀN BỘ ảnh của split (Train = 27,812 ảnh, không qua Sampler), CenterCrop, không ngẫu nhiên.
    - Gọi 2 lần với force_flip=False/True để có đặc trưng cả ảnh gốc lẫn ảnh lật, giữ nguyên augmentation lật ngang
      khi huấn luyện Fusion mà không phải chạy lại ViT-B/16 mỗi epoch.
    - Mỗi batch có 'sample_idx' (chỉ số dòng trong DataFrame của split) và 'filepath' để làm khóa tra cứu đặc trưng.
    """
    split_key = str(split).strip().lower()
    root_path = Path(dataset_root)
    meta_p = Path(metadata_path) if metadata_path else root_path / "metadata.csv"
    train_df, val_df, test_df = get_split_dataframes(metadata_path=str(meta_p))
    split_map = {"train": train_df, "val": val_df, "validation": val_df, "test": test_df}
    if split_key not in split_map:
        raise ValueError(f"split không hợp lệ: '{split}'. Chỉ chấp nhận 'train', 'val', 'test'.")

    dataset = DefactifyDataset(
        df=split_map[split_key],
        dataset_root=root_path,
        mode="extract",
        active_branches=active_branches,
        force_flip=force_flip,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(num_workers, pin_memory, persistent_workers=False),
    )


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP NGHIỆM THU TOÀN DIỆN BƯỚC 5A (dataset.py)
# ==============================================================================
def run_unit_tests():
    """
    Kiểm chứng nghiệm thu độc lập 6 tiêu chuẩn hệ thống & toán học cho `dataset.py`:
    1. Import liên thư mục thành công và khởi tạo DefactifyDataset với DataFrame mẫu.
    2. Kiểm tra đơn mẫu dataset[0]: Đủ 3 nhánh, đúng Shape, Tensor liên tục, không sót PIL/None.
    3. Kiểm tra cơ chế ngắt mạch `active_branches` (chỉ chạy riêng 'spatial', 'semantic', 'spectral').
    4. Kiểm tra chuẩn hóa mode ('val'/'test' -> 'eval'), force_flip và đóng gói metadata an toàn.
    5. Kiểm tra cơ chế đồng bộ kép set_epoch() giữa Dataset và BalancedGroupSampler.
    6. Kiểm tra tích hợp build_dataloaders() thực tế trên tập Defactify thật với DataLoader đa luồng
       (num_workers=2): Xác nhận 16 Real : 16 AI mỗi batch và kiểm tra tính toán vẹn toàn khi drop_last=False.
    7. Đường dẫn đa nền tảng (không còn dấu '\' gây hỏng trên Linux/Colab).
    8. Tái lập lật/cắt ngẫu nhiên giữa 2 lần chạy cùng seed với num_workers=2.
    9. drop_last chỉ áp dụng cho Train, Val/Test luôn đủ mẫu.
    10. build_extract_loader: tuần tự, tất định, cặp gốc/lật đối xứng gương; chặn force_flip sai mode.
    """
    print("=" * 85)
    print("BẮT ĐẦU KIỂM THỬ NGHIỆM THU BƯỚC 5A: dataset.py (Unified Multi-Branch Dataset)")
    print("=" * 85)

    dataset_root = DEFAULT_DATASET_ROOT
    meta_path = dataset_root / "metadata.csv"

    if not meta_path.exists():
        print(f"  [SKIP] Không tìm thấy {meta_path}. Bỏ qua kiểm thử dữ liệu đĩa.")
        return

    train_df, val_df, test_df = get_split_dataframes(metadata_path=str(meta_path))

    # -------------------------------------------------------------------------
    # TEST 1: Kiểm tra khởi tạo DefactifyDataset & Zero-DataFrame-Overhead
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 1] Khởi tạo DefactifyDataset & kiểm tra trích xuất mảng phẳng O(1):")
    ds_all = DefactifyDataset(
        df=train_df,
        dataset_root=dataset_root,
        mode="train",
        active_branches=("spatial", "semantic", "spectral"),
        seed=42,
    )
    assert len(ds_all) == 27812, f"Độ dài dataset sai: {len(ds_all)}, kỳ vọng 27,812!"
    assert len(ds_all.filepaths) == 27812
    assert isinstance(ds_all.labels_a, np.ndarray) and ds_all.labels_a.dtype == np.float32
    assert isinstance(ds_all.labels_b, np.ndarray) and ds_all.labels_b.dtype == np.int64
    assert isinstance(ds_all.group_ids, np.ndarray) and ds_all.group_ids.dtype == np.int64
    print(f"  + Số lượng mẫu: {len(ds_all):,} | Mảng phẳng filepaths, labels_a, labels_b, group_ids => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 2: Kiểm tra đơn mẫu dataset[0] với đủ 3 nhánh (Shape, Dtype, Collate-Safe)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 2] Kiểm tra trích xuất mẫu đơn dataset[0] với đủ 3 nhánh:")
    sample0 = ds_all[0]
    expected_keys = {
        "spatial", "semantic", "spectral",
        "label", "label_a", "label_b", "group_id",
        "sample_idx", "model_name", "filepath", "meta"
    }
    assert expected_keys.issubset(set(sample0.keys())), f"Thiếu key trong output: {set(sample0.keys())}"

    assert sample0["spatial"].shape == (3, 224, 224) and sample0["spatial"].dtype == torch.float32
    assert sample0["semantic"].shape == (3, 224, 224) and sample0["semantic"].dtype == torch.float32
    assert sample0["spectral"].shape == (1, 224, 224) and sample0["spectral"].dtype == torch.float32
    assert sample0["label"].dtype == torch.int64 and sample0["label"].ndim == 0
    assert sample0["label_a"].dtype == torch.float32 and sample0["label_a"].ndim == 0
    assert sample0["label_b"].dtype == torch.int64 and sample0["label_b"].ndim == 0
    assert sample0["group_id"].dtype == torch.int64 and sample0["group_id"].ndim == 0
    assert isinstance(sample0["sample_idx"], int)
    assert isinstance(sample0["model_name"], str)
    assert isinstance(sample0["filepath"], str)
    assert isinstance(sample0["meta"], dict)

    # Đảm bảo không còn đối tượng PIL.Image hay None nào sót lại
    for k, v in sample0.items():
        if k != "meta":
            assert not hasattr(v, "size") or isinstance(v, torch.Tensor), f"Phát hiện kiểu dữ liệu lạ tại key '{k}': {type(v)}"
    print(
        f"  + Spatial: {tuple(sample0['spatial'].shape)} | Semantic: {tuple(sample0['semantic'].shape)} | "
        f"Spectral: {tuple(sample0['spectral'].shape)} | Nhãn: {sample0['label'].item()} (Collate-Safe) => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 3: Kiểm tra cơ chế ngắt mạch active_branches (Tiết kiệm CPU)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 3] Kiểm tra công tắc ngắt mạch active_branches (Đơn nhánh):")
    # 3A: Chỉ chạy nhánh Spectral (FFT)
    ds_spectral_only = DefactifyDataset(
        df=train_df.iloc[:20],
        dataset_root=dataset_root,
        mode="eval",
        active_branches=("spectral",),
    )
    s_fft = ds_spectral_only[0]
    assert "spectral" in s_fft and "spatial" not in s_fft and "semantic" not in s_fft
    assert ds_spectral_only.spatial_proc is None and ds_spectral_only.semantic_proc is None

    # 3B: Chỉ chạy nhánh Spatial (EfficientNet)
    ds_spatial_only = DefactifyDataset(
        df=train_df.iloc[:20],
        dataset_root=dataset_root,
        mode="eval",
        active_branches=("spatial",),
    )
    s_eff = ds_spatial_only[0]
    assert "spatial" in s_eff and "spectral" not in s_eff and "semantic" not in s_eff
    assert ds_spatial_only.spectral_proc is None and ds_spatial_only.semantic_proc is None
    print("  + Bật riêng 'spectral': Chỉ chạy FFT | Bật riêng 'spatial': Chỉ chạy EfficientNet => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 4: Kiểm tra chuẩn hóa mode ('val'/'test' -> 'eval') & force_flip
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 4] Kiểm tra ánh xạ chế độ mode và cưỡng bức lật ảnh (force_flip):")
    for test_m in ("val", "validation", "test", "EVAL"):
        ds_m = DefactifyDataset(df=val_df.iloc[:5], dataset_root=dataset_root, mode=test_m)
        assert ds_m.mode == "eval", f"Ánh xạ mode '{test_m}' sai: {ds_m.mode}"

    # Kiểm tra force_flip đối xứng
    ds_extract_noflip = DefactifyDataset(
        df=val_df.iloc[:5],
        dataset_root=dataset_root,
        mode="extract",
        active_branches=("spatial",),
        force_flip=False,
    )
    ds_extract_flip = DefactifyDataset(
        df=val_df.iloc[:5],
        dataset_root=dataset_root,
        mode="extract",
        active_branches=("spatial",),
        force_flip=True,
    )
    t_noflip = ds_extract_noflip[0]["spatial"]
    t_flip = ds_extract_flip[0]["spatial"]
    assert torch.allclose(t_flip, t_noflip.flip(-1), atol=1e-5), "force_flip=True không đối xứng gương với force_flip=False!"
    # Chỉ số kiểu Tensor/âm hợp lệ thì quy đổi đúng; vượt biên hoặc số thực phải báo lỗi, không âm thầm trả mẫu khác
    ds5 = ds_extract_noflip
    assert ds5[torch.tensor(1)]["sample_idx"] == 1 and ds5[np.int64(1)]["sample_idx"] == 1
    assert ds5[-1]["sample_idx"] == 4
    for bad_idx, err in ((5, IndexError), (-6, IndexError), (2.9, TypeError)):
        try:
            ds5[bad_idx]
            raise AssertionError(f"ds[{bad_idx}] lẽ ra phải báo {err.__name__}!")
        except err:
            pass
    print("  + Ánh xạ 'val'/'test' -> 'eval' an toàn | force_flip đối xứng gương chuẩn xác | idx Tensor/âm/vượt biên => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 5: Kiểm tra cơ chế đồng bộ kép set_epoch() giữa Dataset và Sampler
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 5] Kiểm tra đồng bộ kép set_epoch() (CommonProcess + BalancedGroupSampler):")
    sampler_mock = BalancedGroupSampler(train_df, seed=42)
    ds_all.attach_sampler(sampler_mock)

    assert ds_all.common_proc._shared_epoch[0].item() == 0
    assert sampler_mock.epoch == 0

    ds_all.set_epoch(1)
    assert ds_all.common_proc._shared_epoch[0].item() == 1, "CommonProcess chưa cập nhật epoch 1!"
    assert sampler_mock.epoch == 1, "BalancedGroupSampler chưa cập nhật epoch 1!"

    ds_all.set_epoch(2)
    assert ds_all.common_proc._shared_epoch[0].item() == 2
    assert sampler_mock.epoch == 2
    print("  + Gọi dataset.set_epoch(epoch) tự động đồng bộ cả CommonProcess và BalancedGroupSampler => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 6: Tích hợp build_dataloaders() thực tế trên tập Defactify (num_workers=2)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 6] Tích hợp build_dataloaders() đa luồng (num_workers=2, batch_size=32):")
    train_loader, val_loader, test_loader, t_sampler = build_dataloaders(
        dataset_root=dataset_root,
        metadata_path=meta_path,
        batch_size=32,
        num_workers=2,
        active_branches=("spatial", "semantic", "spectral"),
        drop_last=False,
    )

    assert len(train_loader.dataset) == 27812
    assert len(val_loader.dataset) == 5836
    assert len(test_loader.dataset) == 43800
    assert len(t_sampler) == 13906
    # 13,906 // 32 = 434 batches đủ + 1 batch 18 ảnh = 435 batches
    assert len(train_loader) == 435, f"Số lượng batch sai: {len(train_loader)}, kỳ vọng 435!"

    # Lấy thử 2 batch thực tế qua DataLoader
    batches_checked = 0
    for batch in train_loader:
        b_spatial = batch["spatial"]
        b_semantic = batch["semantic"]
        b_spectral = batch["spectral"]
        b_labels = batch["label"]
        b_meta = batch["meta"]

        assert b_spatial.shape == (32, 3, 224, 224) and b_spatial.dtype == torch.float32
        assert b_semantic.shape == (32, 3, 224, 224) and b_semantic.dtype == torch.float32
        assert b_spectral.shape == (32, 1, 224, 224) and b_spectral.dtype == torch.float32
        assert torch.isfinite(b_spatial).all() and torch.isfinite(b_semantic).all() and torch.isfinite(b_spectral).all()

        real_count = int((b_labels == 0).sum().item())
        ai_count = int((b_labels == 1).sum().item())
        assert real_count == 16 and ai_count == 16, f"Batch không cân bằng: {real_count} Real : {ai_count} AI"

        # Kiểm tra metadata gom lô không bị lỗi tuple
        assert b_meta["orig_size"].shape == (32, 2)
        assert b_meta["padding"].shape == (32, 4)
        assert b_meta["native_crop_box"].shape == (32, 4)
        assert b_meta["kept_area_ratio"].shape == (32,)

        batches_checked += 1
        print(
            f"  + Batch {batches_checked}: Spatial={tuple(b_spatial.shape)} | "
            f"Semantic={tuple(b_semantic.shape)} | Spectral={tuple(b_spectral.shape)} | "
            f"Nhãn={real_count} Real : {ai_count} AI | Meta Collate OK => [PASS]"
        )
        if batches_checked >= 2:
            break

    # -------------------------------------------------------------------------
    # TEST 7: Đường dẫn đa nền tảng (Windows '\' -> '/')
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 7] Đường dẫn đa nền tảng (metadata lưu '\\' của Windows):")
    assert all("\\" not in p for p in ds_all.filepaths), "Còn dấu '\\' trong filepaths -> hỏng trên Linux/Colab!"
    from pathlib import PurePosixPath
    posix_parts = PurePosixPath(ds_all.filepaths[0]).parts
    assert posix_parts[0] == "train" and len(posix_parts) == 3, f"Linux sẽ tách sai đường dẫn: {posix_parts}"
    print(f"  + '{ds_all.filepaths[0]}' -> Linux tách thành {posix_parts} => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 8: Tái lập augmentation giữa 2 lần chạy cùng seed (num_workers=2)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 8] Tái lập lật/cắt ngẫu nhiên giữa 2 lần build_dataloaders(seed=42):")

    def _first_batch_aug(epoch: int) -> Tuple[List[int], List[int]]:
        tl, _, _, _ = build_dataloaders(
            dataset_root=dataset_root, batch_size=32, num_workers=2,
            active_branches=("spectral",), seed=42, persistent_workers=False,
        )
        tl.dataset.set_epoch(epoch)
        b = next(iter(tl))
        return b["meta"]["is_flipped"].int().tolist(), b["meta"]["native_crop_box"][:, 0].tolist()

    run_a, run_b = _first_batch_aug(0), _first_batch_aug(0)
    assert run_a == run_b, "Cùng seed nhưng lật/cắt khác nhau giữa 2 lần chạy -> không tái lập được thực nghiệm!"
    assert _first_batch_aug(1) != run_a, "Epoch 1 lặp lại y hệt augmentation của epoch 0!"
    print("  + Cùng seed -> lật & tọa độ cắt trùng khớp 100% | Đổi epoch -> augmentation mới => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 9: Val/Test không bao giờ rơi mẫu, kể cả khi drop_last=True cho Train
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 9] drop_last=True chỉ tác động lên Train, Val/Test giữ đủ mẫu:")
    tl, vl, te, _ = build_dataloaders(
        dataset_root=dataset_root, batch_size=32, num_workers=0, active_branches=("spectral",), drop_last=True,
    )
    assert len(tl) == 13906 // 32 and not vl.drop_last and not te.drop_last
    assert len(te) == -(-43800 // 32), f"Test loader rơi mẫu: {len(te)} batches"
    print(f"  + Train {len(tl)} batch (bỏ 18 ảnh) | Val {len(vl)} batch | Test {len(te)} batch đủ 43,800 ảnh => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 10: build_extract_loader cho trích xuất đặc trưng CLIP đóng băng
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 10] build_extract_loader (mode='extract', tuần tự, tất định, gốc + lật):")
    ex_plain = build_extract_loader("val", force_flip=False, active_branches=("spatial",), batch_size=8, num_workers=0)
    ex_flip = build_extract_loader("val", force_flip=True, active_branches=("spatial",), batch_size=8, num_workers=0)
    assert len(ex_plain.dataset) == 5836 and ex_plain.dataset.mode == "extract"
    b_plain, b_flip = next(iter(ex_plain)), next(iter(ex_flip))
    assert b_plain["sample_idx"].tolist() == list(range(8)), "Extract loader phải duyệt tuần tự theo thứ tự DataFrame!"
    assert torch.allclose(b_flip["spatial"], b_plain["spatial"].flip(-1), atol=1e-5)
    try:
        DefactifyDataset(df=val_df.iloc[:5], dataset_root=dataset_root, mode="train", force_flip=True)
        raise AssertionError("force_flip với mode='train' phải bị chặn ngay ở __init__!")
    except ValueError:
        pass
    print("  + Duyệt tuần tự 5,836 ảnh | bản lật đối xứng gương bản gốc | chặn force_flip sai mode => [PASS]")

    print("\n" + "=" * 85)
    print("TẤT CẢ 10 TIÊU CHUẨN NGHIỆM THU BƯỚC 5A (dataset.py) ĐÃ VƯỢT QUA 100%!")
    print("=" * 85)


if __name__ == "__main__":
    run_unit_tests()
