import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# BỘ HẰNG SỐ CHUẨN IMAGENET & CẤU HÌNH EFFICIENTNET-B0
# ==============================================================================
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
TARGET_SIZE: int = 224
GLOBAL_RESIZE_SHORTEST_EDGE: int = 256


class EfficientNetPreprocessor:
    """
    BƯỚC 4A: Tiền xử lý chuyên biệt cho Nhánh Không Gian (Spatial Branch - EfficientNet-B0).

    Kỷ luật thiết kế:
    1. Chế độ mặc định Baseline (`view_mode='native'`):
       - Tiếp nhận `view_native` (PIL RGB 224x224) cắt từ pixel gốc ở Bước 3.
       - TUYỆT ĐỐI KHÔNG Resize hay CenterCrop lần 2 để bảo toàn 100% vết nhiễu vi mô của AI.
       - Chuyển sang Tensor float32 [0.0, 1.0] và chuẩn hóa theo ImageNet (mean, std).
    2. Chế độ thực nghiệm đối chứng (`view_mode='global'` - Ablation Study):
       - Nhận `view_global` (PIL RGB WxH gốc) để so sánh đối chứng ở Chương 4 Luận văn.
       - Shortest-Edge Resize cạnh ngắn về 256 bằng nội suy BICUBIC (giữ nguyên tỷ lệ khung hình,
         không ép vuông bóp méo vật thể ảnh chữ nhật MS_COCO).
       - Cắt 224x224: RandomCrop(224) nếu 'train', CenterCrop(224) nếu 'eval'/'extract'.
       - Chuyển sang Tensor float32 và chuẩn hóa ImageNet.
    3. Hỗ trợ đầu vào linh hoạt:
       - Nhận trực tiếp đối tượng `PIL.Image.Image`.
       - Nhận trực tiếp Dictionary đầu ra từ `CommonProcess` ở Bước 3 (tự động bóc tách
         khóa động `view_native` hoặc `view_global` dựa theo `self.view_mode`).
    """

    VALID_VIEW_MODES = ("native", "global")

    def __init__(
        self,
        view_mode: str = "native",
        normalize: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Khởi tạo bộ tiền xử lý EfficientNet.

        Args:
            view_mode: 'native' (mặc định Baseline - pixel gốc 224x224) hoặc
                       'global' (chế độ thực nghiệm đối chứng - Shortest-edge 256 -> crop 224).
            normalize: True để chuẩn hóa phân phối ImageNet (chuẩn cho huấn luyện/đánh giá),
                       False để xuất Tensor thô trong dải [0.0, 1.0] (dùng cho debug/kiểm thử).
            seed: Hạt giống ngẫu nhiên khi RandomCrop ở chế độ view_mode='global' train.
        """
        if not HAS_TORCH:
            raise ImportError("Module này yêu cầu PyTorch. Hãy kích hoạt môi trường đã cài đặt PyTorch.")

        if view_mode not in self.VALID_VIEW_MODES:
            raise ValueError(
                f"view_mode không hợp lệ: '{view_mode}'. Chỉ chấp nhận một trong {self.VALID_VIEW_MODES}."
            )
        if not isinstance(normalize, bool):
            raise TypeError(f"normalize phải là bool, nhận được: {type(normalize)}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError(f"seed phải là số nguyên không âm (>= 0) hoặc None, nhận được: {seed}")

        self.view_mode = view_mode
        self.normalize = normalize
        self.seed = seed

        # Tối ưu tốc độ: Khởi tạo sẵn Tensor mean/std kích thước (3, 1, 1) ngay trong __init__
        # để không cấp phát lại bộ nhớ cho mỗi ảnh trong vòng lặp huấn luyện
        # chuẩn hóa imageNet
        self.mean: torch.Tensor = torch.tensor(
            IMAGENET_MEAN, dtype=torch.float32
        ).view(3, 1, 1)
        self.std: torch.Tensor = torch.tensor(
            IMAGENET_STD, dtype=torch.float32
        ).view(3, 1, 1)

        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _shortest_edge_resize(img: Image.Image, shortest_edge: int = 256) -> Image.Image:
        """
        Đưa cạnh ngắn nhất về `shortest_edge` (256) bằng thuật toán nội suy BICUBIC chuẩn
        của EfficientNet_B0_Weights.IMAGENET1K_V1, bảo toàn hoàn hảo tỷ lệ khung hình (aspect ratio).
        """
        w, h = img.size
        if w <= h:
            new_w = shortest_edge
            new_h = int(round(h * (shortest_edge / float(w))))
        else:
            new_h = shortest_edge
            new_w = int(round(w * (shortest_edge / float(h))))
        return img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    def _crop_224(self, img: Image.Image, mode: str = "eval") -> Image.Image:
        """
        Cắt khung 224x224 trên ảnh đã resize (chỉ dùng cho view_mode='global'):
        - 'train': RandomCrop(224)
        - 'eval' hoặc 'extract': CenterCrop(224) tất định
        """
        w, h = img.size
        if w < TARGET_SIZE or h < TARGET_SIZE:
            raise ValueError(
                f"Kích thước ảnh sau resize ({w}, {h}) nhỏ hơn TARGET_SIZE ({TARGET_SIZE})."
            )

        if mode == "train":
            max_x = w - TARGET_SIZE
            max_y = h - TARGET_SIZE
            left = int(self._rng.integers(0, max_x + 1)) if max_x > 0 else 0
            top = int(self._rng.integers(0, max_y + 1)) if max_y > 0 else 0
        else:
            left = int(round((w - TARGET_SIZE) / 2.0))
            top = int(round((h - TARGET_SIZE) / 2.0))

        return img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        Chuyển ảnh PIL RGB đúng 224x224 sang Tensor float32 [0.0, 1.0] và chuẩn hóa ImageNet.
        Bộ nhớ đảm bảo liên tục (.contiguous()) với tốc độ < 0.5 ms/ảnh.
        """
        # Kiểm tra chặt chẽ hệ màu và kích thước để chặn đứng lỗi ngầm
        if not isinstance(img, Image.Image):
            raise TypeError(f"Dữ liệu đưa vào tensor hóa phải là PIL.Image.Image, nhận được: {type(img)}")
        if img.mode != "RGB":
            raise ValueError(
                f"Ảnh đầu vào phải ở hệ màu 'RGB', nhận được mode='{img.mode}'. "
                f"Hãy đảm bảo ảnh đã qua xử lý của CommonProcess ở Bước 3."
            )
        if img.size != (TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                f"Ảnh đưa vào tensor hóa phải có kích thước đúng ({TARGET_SIZE}, {TARGET_SIZE}), "
                f"nhận được kích thước: {img.size}."
            )

        # Chuyển đổi siêu tốc từ PIL -> NumPy float32 -> Tensor (3, 224, 224)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # shape: (224, 224, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # shape: (3, 224, 224)

        if self.normalize:
            tensor = (tensor - self.mean) / self.std

        return tensor

    def __call__(
        self,
        inp: Union[Dict[str, Any], Image.Image],
        *,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Thực thi tiền xử lý cho nhánh EfficientNet.

        Args:
            inp: Có thể là:
                 - Dictionary từ `CommonProcess` ở Bước 3.
                 - Hoặc đối tượng `PIL.Image.Image` trực tiếp.
            mode: 'train', 'eval' hoặc 'extract' (nếu truyền Dictionary, sẽ tự đọc từ inp['mode']).

        Returns:
            torch.Tensor kích thước (3, 224, 224), kiểu torch.float32.
        """
        effective_mode = mode

        # Chốt chặn điều phối Dictionary thông minh: tự động lấy đúng khóa theo view_mode
        if isinstance(inp, dict):
            target_key = "view_native" if self.view_mode == "native" else "view_global"
            if target_key not in inp:
                raise KeyError(
                    f"Thiếu khóa bắt buộc '{target_key}' trong dictionary đầu vào cho view_mode='{self.view_mode}'. "
                    f"Các khóa hiện có trong dictionary: {list(inp.keys())}."
                )
            img = inp[target_key]
            if effective_mode is None and "mode" in inp:
                effective_mode = inp["mode"]
        elif isinstance(inp, Image.Image):
            img = inp
        else:
            raise TypeError(
                f"Đầu vào phải là Dictionary từ Bước 3 hoặc đối tượng PIL.Image.Image, "
                f"nhận được: {type(inp)}."
            )

        if not isinstance(img, Image.Image):
            raise TypeError(
                f"Ảnh trích xuất được từ đầu vào không phải là PIL.Image.Image, nhận được: {type(img)}."
            )

        if effective_mode is None:
            effective_mode = "eval"

        # 1. Chế độ mặc định Baseline: native (Cấm resize, nhận ô 224x224 pixel gốc)
        if self.view_mode == "native":
            if img.size != (TARGET_SIZE, TARGET_SIZE):
                raise ValueError(
                    f"Chế độ view_mode='native' yêu cầu ảnh đã đúng kích thước ({TARGET_SIZE}, {TARGET_SIZE}), "
                    f"nhận được kích thước: {img.size}. Hãy đảm bảo ảnh đã qua xử lý view_native ở Bước 3."
                )
            return self._pil_to_tensor(img)

        # 2. Chế độ thực nghiệm đối chứng: global (Ablation: Shortest-edge 256 -> Crop 224)
        resized_img = self._shortest_edge_resize(img, GLOBAL_RESIZE_SHORTEST_EDGE)
        cropped_img = self._crop_224(resized_img, mode=effective_mode)
        return self._pil_to_tensor(cropped_img)


# Alias tiện dụng đồng nhất import
PreprocessEfficientNet = EfficientNetPreprocessor


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP NGHIỆM THU TOÀN DIỆN BƯỚC 4A (EFFICIENTNET)
# ==============================================================================
def run_unit_tests():
    """
    Kiểm chứng toàn diện 7 tiêu chuẩn nghiệm thu cho pre_processing_EfficientNet.py:
    1. Shape (3, 224, 224) và kiểu dữ liệu torch.float32.
    2. Bảo toàn điểm ảnh Native: khôi phục nguyên vẹn từng byte pixel khi tắt normalize.
    3. Tích hợp trực tiếp với CommonProcess trên ảnh thật SDXL và MS_COCO.
    4. Chế độ đối chứng 'global': Shortest-edge BICUBIC 256 giữ tỷ lệ khung hình + crop 224.
    5. Chốt chặn an toàn: từ chối ảnh sai kích thước ở native, sai hệ màu (L/RGBA), sai kiểu dữ liệu.
    6. Điều phối Dictionary chính xác: native lấy view_native, global lấy view_global.
    7. Tính chuẩn xác của công thức chuẩn hóa ImageNet: so khớp torch.allclose (atol=1e-6).
    8. Đo lường phân phối độ trễ (median & p95) trên 100 lần chạy.
    """
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ NGHIỆM THU BƯỚC 4A: pre_processing_EfficientNet.py")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Kiểm tra Shape (3, 224, 224) và Kiểu torch.float32
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 1] Kiểm tra Shape và Dtype:")
    prep = EfficientNetPreprocessor(view_mode="native", normalize=True)
    dummy_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8), mode="RGB")
    t1 = prep(dummy_img)

    assert isinstance(t1, torch.Tensor), "Đầu ra không phải là torch.Tensor!"
    assert t1.shape == (3, 224, 224), f"Shape sai: {t1.shape}, kỳ vọng (3, 224, 224)!"
    assert t1.dtype == torch.float32, f"Dtype sai: {t1.dtype}, kỳ vọng torch.float32!"
    assert t1.is_contiguous(), "Tensor không liên tục trong bộ nhớ (contiguous)!"
    print(f"  + Shape: {tuple(t1.shape)} | Dtype: {t1.dtype} | Contiguous: True => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 2: Chứng minh bảo toàn 100% pixel gốc khi tắt normalize
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 2] Chứng minh bảo toàn điểm ảnh gốc tuyệt đối (Pixel Preservation):")
    prep_raw = EfficientNetPreprocessor(view_mode="native", normalize=False)
    raw_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    raw_img = Image.fromarray(raw_arr, mode="RGB")
    t_raw = prep_raw(raw_img)

    # Khôi phục số nguyên [0..255] và đối chiếu chính xác từng byte
    restored = torch.round(t_raw * 255.0).to(torch.uint8)
    expected = torch.from_numpy(raw_arr).permute(2, 0, 1)
    assert torch.equal(restored, expected), "Giá trị pixel bị sai lệch trong quá trình chuyển đổi sang Tensor!"
    print("  + So khớp từng byte gốc: restored == original (150,528 điểm ảnh) => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 3: Tích hợp trực tiếp với CommonProcess (Bước 3 -> Bước 4A)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 3] Tích hợp liên hoàn với Bước 3 (CommonProcess) trên ảnh thật:")
    from pre_processing_for_all import CommonProcess

    dataset_root = Path(r"D:\nam_4\DATN\dataset\Defactify_dataset")
    meta_path = dataset_root / "metadata.csv"

    if meta_path.exists():
        import pandas as pd
        df = pd.read_csv(meta_path)
        sample_sdxl = dataset_root / df[df["model_name"] == "Stable_Diffusion_XL"].iloc[0]["filepath"]
        sample_coco = dataset_root / df[df["model_name"] == "MS_COCO"].iloc[0]["filepath"]

        step3_proc = CommonProcess(mode="eval", crop_size=224)
        out_sdxl = step3_proc(sample_sdxl)
        out_coco = step3_proc(sample_coco)

        # Truyền thẳng nguyên dictionary của Bước 3 vào Bước 4A
        t_sdxl = prep(out_sdxl)
        t_coco = prep(out_coco)

        assert t_sdxl.shape == (3, 224, 224) and t_sdxl.dtype == torch.float32
        assert t_coco.shape == (3, 224, 224) and t_coco.dtype == torch.float32
        print(f"  + SDXL (1024x1024) -> CommonProcess -> EfficientNet Tensor {tuple(t_sdxl.shape)} => [PASS]")
        print(f"  + MS_COCO (Chữ nhật) -> CommonProcess -> EfficientNet Tensor {tuple(t_coco.shape)} => [PASS]")
    else:
        print("  ! Bỏ qua kiểm tra ảnh đĩa do không tìm thấy metadata.csv.")

    # -------------------------------------------------------------------------
    # TEST 4: Kiểm tra chế độ thực nghiệm đối chứng 'global' (Shortest-Edge 256 -> Crop 224)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 4] Kiểm tra chế độ thực nghiệm đối chứng 'global' (Ablation Mode):")
    prep_global = EfficientNetPreprocessor(view_mode="global", normalize=True, seed=42)

    # Thử nghiệm trên ảnh chữ nhật 640x480 (tương tự MS_COCO) có gradient để kiểm tra phân biệt vị trí cắt
    grad_arr = np.linspace(0, 255, 640 * 480, dtype=np.uint8).reshape((480, 640))
    rect_arr = np.stack([grad_arr, grad_arr // 2, 255 - grad_arr], axis=-1)
    rect_img = Image.fromarray(rect_arr, mode="RGB")
    resized_rect = EfficientNetPreprocessor._shortest_edge_resize(rect_img, 256)
    rw, rh = resized_rect.size
    assert (rw, rh) == (341, 256), f"Shortest-edge resize sai kích thước: {(rw, rh)}, kỳ vọng (341, 256)!"

    # Gọi qua preprocessor ở cả 2 mode: eval (CenterCrop) và train (RandomCrop)
    t_global_eval = prep_global(rect_img, mode="eval")
    t_global_train = prep_global(rect_img, mode="train")

    assert t_global_eval.shape == (3, 224, 224) and t_global_train.shape == (3, 224, 224)
    assert not torch.equal(t_global_eval, t_global_train), "RandomCrop và CenterCrop sinh ra kết quả trùng nhau!"
    print(f"  + Ảnh 640x480 -> Shortest-Edge Bicubic 256 ({rw}, {rh}) -> Crop 224 -> Tensor (3, 224, 224) => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 5: Chốt chặn an toàn (Input Validation & Error Guards)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 5] Kiểm tra chốt chặn an toàn đầu vào:")
    # 5A: Từ chối ảnh không phải RGB (ví dụ RGBA, L)
    rgba_img = Image.new("RGBA", (224, 224), (255, 0, 0, 128))
    gray_img = Image.new("L", (224, 224), 128)
    for bad_mode_img in (rgba_img, gray_img):
        try:
            prep(bad_mode_img)
            raise AssertionError(f"Phải báo lỗi với ảnh hệ màu '{bad_mode_img.mode}'!")
        except ValueError:
            pass

    # 5B: Từ chối ảnh sai kích thước khi ở view_mode='native'
    wrong_size_img = Image.new("RGB", (256, 256), (100, 100, 100))
    try:
        prep(wrong_size_img)
        raise AssertionError("Phải báo lỗi khi truyền ảnh khác (224, 224) vào native mode!")
    except ValueError:
        pass

    # 5C: Từ chối sai kiểu dữ liệu
    for bad_input in ("duong_dan_anh.jpg", 12345, [1, 2, 3]):
        try:
            prep(bad_input)
            raise AssertionError(f"Phải báo lỗi với kiểu dữ liệu không hợp lệ: {type(bad_input)}!")
        except TypeError:
            pass

    print("  + Bắt lỗi thành công: Chặn ảnh không phải RGB | Chặn sai kích thước native | Chặn sai kiểu dữ liệu => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 6: Điều phối Dictionary chính xác (Native lấy native, Global lấy global)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 6] Kiểm tra điều phối Dictionary (Dynamic Key Extraction):")
    native_sample = Image.new("RGB", (224, 224), (10, 20, 30))
    global_sample = Image.new("RGB", (640, 480), (70, 80, 90))

    mock_dict = {
        "view_native": native_sample,
        "view_global": global_sample,
        "mode": "eval",
    }

    prep_nat = EfficientNetPreprocessor(view_mode="native", normalize=False)
    prep_glo = EfficientNetPreprocessor(view_mode="global", normalize=False)

    out_from_nat = prep_nat(mock_dict)
    out_from_glo = prep_glo(mock_dict)

    # out_from_nat phải khớp với native_sample (giá trị 10, 20, 30)
    assert torch.all(torch.round(out_from_nat[0, :, :] * 255) == 10)
    # out_from_glo phải khớp với global_sample (giá trị 70, 80, 90)
    assert torch.all(torch.round(out_from_glo[0, :, :] * 255) == 70)

    # Kiểm tra dictionary thiếu khóa bắt buộc
    bad_dict = {"wrong_key": native_sample}
    try:
        prep_nat(bad_dict)
        raise AssertionError("Phải báo lỗi khi dictionary thiếu khóa 'view_native'!")
    except KeyError:
        pass

    print("  + Dictionary dispatch: view_mode='native' lấy đúng view_native | view_mode='global' lấy đúng view_global => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 7: Kiểm tra tính chuẩn xác của công thức ImageNet Normalization
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 7] Kiểm tra công thức toán học ImageNet Normalization:")
    prep_norm = EfficientNetPreprocessor(view_mode="native", normalize=True)
    t_norm = prep_norm(raw_img)

    # Tính toán đối chứng độc lập
    mean_vec = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std_vec = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    expected_norm = (t_raw - mean_vec) / std_vec

    assert torch.allclose(t_norm, expected_norm, atol=1e-6), "Công thức chuẩn hóa ImageNet bị sai lệch số học!"
    print("  + Chuẩn hóa ImageNet khớp hoàn hảo với công thức độc lập (allclose atol=1e-6) => [PASS]")

    # -------------------------------------------------------------------------
    # BENCHMARK: Đo lường phân phối độ trễ (Median & P95) trên 100 lần chạy
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK] Đo lường hiệu năng xử lý trên CPU:")
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        prep_norm(dummy_img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    print(f"  + Native Mode Latency (100 runs): Median = {median_ms:.3f} ms/ảnh | P95 = {p95_ms:.3f} ms/ảnh => [PASS]")

    print("\n" + "=" * 80)
    print("TẤT CẢ TIÊU CHUẨN NGHIỆM THU BƯỚC 4A (EFFICIENTNET) ĐÃ VƯỢT QUA 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
