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
        dataset_root: Union[str, Path] = Path(r"D:\nam_4\DATN\dataset\Defactify_dataset"),
        mode: str = "train",
        active_branches: Tuple[str, ...] = ("spatial", "semantic", "spectral"),
        force_flip: Optional[bool] = None,
        crop_size: int = 224,
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

        self.filepaths: List[str] = [str(p).replace("/", "\\") for p in df["filepath"].tolist()]
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
def build_dataloaders(
    dataset_root: Union[str, Path] = Path(r"D:\nam_4\DATN\dataset\Defactify_dataset"),
    metadata_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    active_branches: Tuple[str, ...] = ("spatial", "semantic", "spectral"),
    fft_mode: str = "raw",
    fft_use_hann: bool = False,
    fft_remove_dc: bool = True,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, BalancedGroupSampler]:
    """
    Khởi tạo trọn gói toàn bộ pipeline dữ liệu (Train, Val, Test DataLoaders và BalancedGroupSampler).

    Quy tắc nghiệp vụ cốt lõi:
    - Train Loader: Dùng BalancedGroupSampler (đảm bảo 16 Real : 16 AI mỗi batch, xoay vòng 100% AI qua 3 epochs).
      Bắt buộc đặt shuffle=False (do Sampler đã xáo trộn) và drop_last=False (để không làm rơi 18 ảnh cuối).
    - Val Loader: Đánh giá trên tập Known (5,836 ảnh), phân phối tự nhiên, shuffle=False, drop_last=False.
    - Test Loader: Đánh giá toàn diện trên cả 6 nguồn Known + Unseen (43,800 ảnh), shuffle=False, drop_last=False.
    - pin_memory: Tự động nhận diện GPU (True nếu có CUDA, False nếu CPU) để không bắn cảnh báo vàng.

    Returns:
        (train_loader, val_loader, test_loader, train_sampler)
    """
    root_path = Path(dataset_root)
    meta_p = Path(metadata_path) if metadata_path else root_path / "metadata.csv"

    # Bước 1: Rút trích bảng dữ liệu phân chia chuẩn trên RAM (không ghi đè file đĩa)
    train_df, val_df, test_df = get_split_dataframes(metadata_path=str(meta_p))

    # Tự động nhận diện pin_memory an toàn
    auto_pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)

    # Bước 2: Khởi tạo Sampler cân bằng nhóm theo từng cặp Real - AI cho tập Train
    train_sampler = BalancedGroupSampler(train_df, seed=seed)

    # Bước 3: Khởi tạo 3 Dataset chuyên biệt
    train_dataset = DefactifyDataset(
        df=train_df,
        dataset_root=root_path,
        mode="train",
        active_branches=active_branches,
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
        num_workers=num_workers,
        pin_memory=auto_pin,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=auto_pin,
        drop_last=drop_last,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=auto_pin,
        drop_last=drop_last,
    )

    return train_loader, val_loader, test_loader, train_sampler


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
    """
    print("=" * 85)
    print("BẮT ĐẦU KIỂM THỬ NGHIỆM THU BƯỚC 5A: dataset.py (Unified Multi-Branch Dataset)")
    print("=" * 85)

    dataset_root = Path(r"D:\nam_4\DATN\dataset\Defactify_dataset")
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
    print("  + Ánh xạ 'val'/'test' -> 'eval' an toàn | force_flip đối xứng gương chuẩn xác => [PASS]")

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

    print("\n" + "=" * 85)
    print("TẤT CẢ 6 TIÊU CHUẨN NGHIỆM THU BƯỚC 5A (dataset.py) ĐÃ VƯỢT QUA 100%!")
    print("=" * 85)


if __name__ == "__main__":
    run_unit_tests()
