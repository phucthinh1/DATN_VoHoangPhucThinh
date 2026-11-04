import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# BỘ HẰNG SỐ CHUẨN CHO NHÁNH TẦN SỐ PHỔ (SPECTRAL / 2D FFT BRANCH)
# ==============================================================================
# Hệ số độ chói biểu kiến (Luma Y') theo khuyến nghị quốc tế ITU-R BT.601
LUMA_WEIGHTS_BT601: Tuple[float, float, float] = (0.299, 0.587, 0.114)
FFT_TARGET_SIZE: int = 224
DEFAULT_STD_EPS: float = 1e-6


class FFTPreprocessor:
    """
    BƯỚC 4C: Tiền xử lý miền tần số phổ 2D FFT cho Nhánh Phổ (Spectral Branch).

    Thiết kế Baseline mặc định (`mode='raw'`, `use_hann=False`, `remove_dc=False`, `normalize=True`):
    1. Tiếp nhận `view_native` (ảnh PIL RGB 224x224 cắt trực tiếp từ pixel gốc ở Bước 3,
       TUYỆT ĐỐI KHÔNG Resize để bảo toàn dải tần số cao tới giới hạn Nyquist).
    2. Chuyển sang 1 kênh độ chói biểu kiến Luma Y' (ITU-R BT.601) trực tiếp trên `float32` [0.0, 1.0]:
           Y' = 0.299 * R + 0.587 * G + 0.114 * B
    3. Chốt chặn phương sai miền không gian trước biến đổi:
       Nếu ô ảnh hoàn toàn đồng màu phẳng (`std < eps`, ví dụ đen 0, xám 128, trắng 255),
       trả về ngay Tensor 0.0 kích thước (1, H, W) khi `normalize=True`, ngăn chặn tuyệt đối
       hiện tượng thành phần DC đơn lẻ tạo xung gai +224.0 hoặc lỗi chia 0 sinh `NaN`.
    4. Biến đổi Fourier rời rạc 2 chiều trên `torch.float32` (vô hiệu hóa `autocast` half/bfloat16
       vì cuFFT không hỗ trợ kích thước 224 = 2^5 * 7 ở độ chính xác bán phần):
           S(u, v) = log(1 + |fftshift(fft2(signal))|)
    5. Chuẩn hóa Instance Z-score trên toàn bộ ma trận phổ với `unbiased=False` (`correction=0`)
       kèm chốt chặn `std < eps` lần 2 sau `log1p`.

    Các công tắc mở rộng phục vụ thực nghiệm đối chứng (Ablation) khi cần:
    - `mode='gaussian_residual'`: Lọc thông cao R = Y' - Gaussian_3x3(Y', sigma=1.0) đệm gương 'reflect'.
    - `use_hann=True`: Nhân cửa sổ Hann 2D đối xứng gương (`periodic=False`) giảm rò rỉ phổ biên.
    - `remove_dc=True`: Khử thành phần trung bình `Y' - mean(Y')` trước cửa sổ / FFT.
    """

    VALID_MODES = ("raw", "gaussian_residual")

    def __init__(
        self,
        mode: str = "raw",
        use_hann: bool = False,
        remove_dc: bool = True,
        normalize: bool = True,
        image_size: int = FFT_TARGET_SIZE,
        gaussian_sigma: float = 1.0,
        eps: float = DEFAULT_STD_EPS,
    ):
        """
        Khởi tạo bộ tiền xử lý phổ 2D FFT chuẩn SOTA cho Baseline.

        Cấu hình Baseline tối ưu độ chính xác cao nhất (Accuracy-Maximized Baseline):
        - mode='raw': Giữ nguyên phổ toàn dải từ Luma gốc.
        - use_hann=False: Không làm mờ mép viền để bảo toàn 100% diện tích ô crop.
        - remove_dc=True (MẶC ĐỊNH MỚI CHO BASELINE): Triệt tiêu hoàn toàn thành phần DC (0 Hz)
          tại tâm (112, 112) mang độ sáng trung bình của ảnh, ngăn chặn triệt để hiện tượng
          học vẹt độ sáng (Shortcut Learning) giữa MS_COCO và AI, tối đa hóa độ chính xác
          tổng quát hóa trên các generator chưa từng thấy (Unseen AI).
        - normalize=True: Chuẩn hóa Instance Z-score (mean=0, std=1) làm nổi bật gai tần số AI.
        """
        if not HAS_TORCH:
            raise ImportError("Module này yêu cầu PyTorch. Hãy kích hoạt môi trường venv đã cài đặt PyTorch.")

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode không hợp lệ: '{mode}'. Chỉ chấp nhận một trong {self.VALID_MODES}."
            )
        if not isinstance(use_hann, bool):
            raise TypeError(f"use_hann phải là kiểu bool, nhận được: {type(use_hann)}")
        if not isinstance(remove_dc, bool):
            raise TypeError(f"remove_dc phải là kiểu bool, nhận được: {type(remove_dc)}")
        if not isinstance(normalize, bool):
            raise TypeError(f"normalize phải là kiểu bool, nhận được: {type(normalize)}")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 4:
            raise ValueError(f"image_size phải là số nguyên >= 4, nhận được: {image_size}")
        if (
            isinstance(gaussian_sigma, bool)
            or not isinstance(gaussian_sigma, (int, float))
            or not math.isfinite(float(gaussian_sigma))
            or gaussian_sigma <= 0.0
        ):
            raise ValueError(f"gaussian_sigma phải là số thực hữu hạn > 0, nhận được: {gaussian_sigma}")
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(float(eps))
            or eps <= 0.0
        ):
            raise ValueError(f"eps phải là số thực hữu hạn > 0, nhận được: {eps}")

        self.mode: str = mode
        self.use_hann: bool = use_hann
        self.remove_dc: bool = remove_dc
        self.normalize: bool = normalize
        self.image_size: int = image_size
        self.gaussian_sigma: float = float(gaussian_sigma)
        self.eps: float = float(eps)

        # Tiền cấp phát các Tensor hằng số tường minh trên CPU ngay trong __init__ (an toàn 100% khi pickle qua DataLoader spawn)
        self._luma_weights: torch.Tensor = torch.tensor(
            LUMA_WEIGHTS_BT601, dtype=torch.float32, device="cpu"
        )
        self._gauss_kernel: torch.Tensor = self._build_gaussian_kernel_3x3(self.gaussian_sigma)
        self._hann_2d: torch.Tensor = self._build_hann_window_2d(self.image_size)

    @staticmethod
    def _build_gaussian_kernel_3x3(sigma: float = 1.0) -> torch.Tensor:
        """
        Tạo nhân lọc Gaussian 2D kích thước (1, 1, 3, 3) đối xứng tâm và có tổng trọng số bằng 1.0 trên CPU.
        """
        coords = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cpu")
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel_2d = torch.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma)))
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.view(1, 1, 3, 3).contiguous()

    @staticmethod
    def _build_hann_window_2d(size: int = FFT_TARGET_SIZE) -> torch.Tensor:
        """
        Tạo cửa sổ Hann 2D kích thước (size, size) bằng tích ngoài của hai cửa sổ 1D trên CPU.
        Bắt buộc đặt `periodic=False` để cửa sổ đối xứng gương hoàn hảo hai bên (w[k] == w[size - 1 - k]),
        đảm bảo tính nhất quán tuyệt đối khi ảnh đã qua lật ngang đồng bộ ở Bước 3.
        """
        w1d = torch.hann_window(size, periodic=False, dtype=torch.float32, device="cpu")
        return torch.outer(w1d, w1d).contiguous()

    def _rgb_pil_to_luminance(self, img: Image.Image) -> torch.Tensor:
        """
        Kiểm tra nghiêm ngặt hệ màu 'RGB' & kích thước `(self.image_size, self.image_size)`,
        sau đó chuyển sang ma trận Luma Y' (H, W) kiểu `torch.float32` trong đoạn [0.0, 1.0].
        Tuyệt đối không Resize và không lượng tử hóa ngược về `uint8`.
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f"Đầu vào phải là PIL.Image.Image, nhận được: {type(img)}")
        if img.mode != "RGB":
            raise ValueError(
                f"Ảnh đầu vào của FFTPreprocessor phải ở hệ màu 'RGB', nhận được mode='{img.mode}'. "
                f"Hãy đảm bảo ảnh đã qua chuẩn hóa ở Bước 3 (CommonProcess)."
            )
        expected_size = (self.image_size, self.image_size)
        if img.size != expected_size:
            raise ValueError(
                f"FFTPreprocessor yêu cầu ảnh đầu vào đúng kích thước {expected_size} (không tự ý Resize), "
                f"nhận được kích thước: {img.size}. Hãy truyền 'view_native' từ Bước 3."
            )

        arr = np.array(img, dtype=np.uint8, copy=True)  # (H, W, 3), writable
        rgb = torch.from_numpy(arr).to(dtype=torch.float32).div_(255.0)
        # Tính tích vô hướng theo trục kênh màu cuối cùng: Y' = 0.299*R + 0.587*G + 0.114*B
        luma = torch.matmul(rgb, self._luma_weights)  # (H, W), float32
        return luma

    def _extract_signal(self, luma: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất tín hiệu không gian 2D trước khi đưa vào FFT2 theo cấu hình `mode`, `remove_dc`, `use_hann`.
        """
        signal = luma.to(dtype=torch.float32)

        if self.mode == "gaussian_residual":
            # Đệm gương 1 pixel (trái, phải, trên, dưới) trên tensor [1, 1, H, W] để tránh viền đen giả ở biên
            x4d = signal.view(1, 1, self.image_size, self.image_size)
            padded = F.pad(x4d, (1, 1, 1, 1), mode="reflect")
            smoothed = F.conv2d(padded, self._gauss_kernel).view(self.image_size, self.image_size)
            signal = signal - smoothed

        # Nếu bật cửa sổ Hann, nhân trước để triệt tiêu bước nhảy không gian ở 4 mép biên
        if self.use_hann:
            signal = signal * self._hann_2d

        # Khử thành phần DC: Trừ trung bình sau khi windowing để bảo đảm tổng các điểm ảnh bằng 0,
        # triệt tiêu hoàn toàn thành phần DC tại tâm phổ F(0, 0)
        if self.remove_dc:
            signal = signal - signal.mean()

        return signal.to(dtype=torch.float32)

    def _compute_log_spectrum(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Tính phổ biên độ logarit 2 chiều dịch tâm:
            S(u, v) = log(1 + |fftshift(fft2(signal))|)
        kèm chuẩn hóa Instance Z-score an toàn (`unbiased=False`).
        """
        # Chốt chặn 1 (trước FFT): Nếu tín hiệu không gian phẳng đồng nhất, trả về ngay ma trận 0.0 khi normalize=True
        if self.normalize and float(signal.std(unbiased=False).item()) < self.eps:
            return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32, device="cpu")

        # Khóa cứng float32 để không bao giờ bị lỗi cuFFT/CPU half-precision với kích thước 224 = 2^5 * 7
        sig_f32 = signal.to(dtype=torch.float32)
        fft_complex = torch.fft.fft2(sig_f32, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        log_mag = torch.log1p(magnitude)

        if self.remove_dc:
            # Khử triệt để sai số làm tròn float32: gán bin DC tại tâm (H//2, W//2) về 0.0
            mid = self.image_size // 2
            log_mag[mid, mid] = 0.0

        if self.normalize:
            mean = log_mag.mean()
            std = log_mag.std(unbiased=False)
            # Chốt chặn 2 (sau log1p): Bảo vệ tuyệt đối khỏi phép chia cho 0 -> NaN / Inf
            if float(std.item()) < self.eps:
                return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32, device="cpu")
            log_mag = (log_mag - mean) / std

        return log_mag.unsqueeze(0).to(dtype=torch.float32, device="cpu").contiguous()

    def __call__(self, inp: Union[Dict[str, Any], Image.Image]) -> torch.Tensor:
        """
        Thực thi tiền xử lý miền tần số phổ 2D FFT cho 1 mẫu ảnh.

        Args:
            inp: Có thể là:
                 - Dictionary đầu ra từ `CommonProcess` ở Bước 3 (tự động lấy `inp['view_native']`).
                 - Hoặc đối tượng `PIL.Image.Image` hệ màu 'RGB' kích thước (224, 224) trực tiếp.

        Returns:
            torch.Tensor kích thước (1, 224, 224), kiểu `torch.float32`, bộ nhớ liên tục (`contiguous`).
        """
        if isinstance(inp, dict):
            if "view_native" not in inp:
                raise KeyError(
                    f"Thiếu khóa bắt buộc 'view_native' trong dictionary đầu vào từ Bước 3. "
                    f"Các khóa hiện có: {list(inp.keys())}."
                )
            img = inp["view_native"]
        elif isinstance(inp, Image.Image):
            img = inp
        else:
            raise TypeError(
                f"Đầu vào của FFTPreprocessor phải là Dictionary từ Bước 3 hoặc PIL.Image.Image, "
                f"nhận được: {type(inp)}."
            )

        # Khóa tắt autocast trên CPU nếu bên ngoài vô tình bọc `torch.autocast` (bảo vệ F.conv2d & fft2 giữ float32)
        with torch.amp.autocast(device_type="cpu", enabled=False):
            luma = self._rgb_pil_to_luminance(img)

            # Chốt chặn 0 (ngay trên ảnh Luma gốc trước cả cửa sổ Hann):
            # Nếu ô ảnh hoàn toàn đồng màu phẳng (đen, xám, trắng, hoặc màu đơn sắc bất kỳ),
            # không có cấu trúc tần số nào để học -> trả về thẳng ma trận 0.0 ngay lập tức.
            if self.normalize and float(luma.std(unbiased=False).item()) < self.eps:
                return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32, device="cpu")

            signal = self._extract_signal(luma)
            return self._compute_log_spectrum(signal)


# Alias đồng nhất quy ước đặt tên giữa các module Bước 4
PreprocessFFT = FFTPreprocessor


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP NGHIỆM THU TOÀN DIỆN BƯỚC 4C (2D FFT SPECTRAL BRANCH)
# ==============================================================================
def run_unit_tests():
    """
    Kiểm chứng nghiệm thu độc lập 7 tiêu chuẩn toán học & hệ thống cho `pre_processing_FFT.py`:
    1. Shape (1, 224, 224), Dtype torch.float32, Contiguous, Mean ≈ 0.0, Std == 1.0, không NaN/Inf,
       và an toàn ngay cả khi bọc trong `torch.autocast('cpu', dtype=torch.bfloat16)`.
    2. Định vị chính xác tọa độ đỉnh phổ tuần hoàn (2px Nyquist tại (0, 0), sóng chu kỳ 4px tại
       (112, 56)/(112, 168), và sóng chu kỳ 8px tại (84, 112)/(140, 112)).
    3. Bất biến dịch chuyển tuần hoàn (`torch.roll`) & Quy tắc đối xứng gương chính xác trên lưới chẵn
       224x224 sau `fftshift` khi lật ngang ảnh (cột 0 Nyquist đứng yên, cột 1..223 đảo qua tâm 112).
    4. Chốt chặn ảnh phẳng đồng màu: Đen (0,0,0), Xám (128,128,128), Trắng (255,255,255) và Màu phẳng
       (30,140,220) đều trả về ma trận toàn 0.0 ở mọi chế độ (không sinh đỉnh DC +224.0, không NaN).
    5. Đối chiếu độc lập công thức toán học của `gaussian_residual`, `use_hann=True` (`periodic=False`)
       và `remove_dc=True`.
    6. Tích hợp liên hoàn với Bước 3 (`CommonProcess`) trên ảnh thật SDXL, MS_COCO chữ nhật, MS_COCO < 224px
       (đảm bảo không làm thay đổi dữ liệu gốc của nhánh EfficientNet) + kiểm tra chốt chặn bắt lỗi đầu vào.
    7. Đo hiệu năng thực tế trên CPU (Warm-up + Median & P95 ms/ảnh + Thông lượng ảnh/giây).
    """
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ NGHIỆM THU BƯỚC 4C: pre_processing_FFT.py (2D FFT Baseline)")
    print("=" * 80)

    rng = np.random.default_rng(42)
    prep = FFTPreprocessor(mode="raw", use_hann=False, normalize=True)

    # -------------------------------------------------------------------------
    # TEST 0: Kiểm tra chốt chặn cấu hình không hợp lệ (Chặn triệt để NaN, Inf, âm, sai kiểu)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 0] Kiểm tra chốt chặn cấu hình không hợp lệ (Chặn triệt để NaN, Inf, âm, sai kiểu):")
    invalid_configs = [
        {"eps": float("nan")},
        {"eps": float("inf")},
        {"gaussian_sigma": float("nan")},
        {"gaussian_sigma": float("inf")},
        {"eps": -1e-6},
        {"gaussian_sigma": 0.0},
        {"image_size": 2},
    ]
    for cfg in invalid_configs:
        try:
            FFTPreprocessor(**cfg)
            raise AssertionError(f"Không chặn được cấu hình sai: {cfg}")
        except ValueError:
            pass
    print("  + Chặn thành công eps/gaussian_sigma là NaN, Inf, âm, <= 0 và image_size < 4 => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 1: Kiểm tra Shape, Dtype, Contiguous, Z-score (mean=0, std=1) & AMP Safety
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 1] Kiểm tra Shape, Dtype, Z-score chuẩn xác và an toàn Autocast:")
    rand_arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    dummy_img = Image.fromarray(rand_arr, mode="RGB")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t1 = prep(dummy_img)

    assert isinstance(t1, torch.Tensor), "Đầu ra không phải là torch.Tensor!"
    assert t1.shape == (1, 224, 224), f"Shape sai: {t1.shape}, kỳ vọng (1, 224, 224)!"
    assert t1.dtype == torch.float32, f"Dtype sai: {t1.dtype}, kỳ vọng torch.float32!"
    assert t1.is_contiguous(), "Tensor không liên tục trong bộ nhớ (contiguous)!"
    assert torch.isfinite(t1).all(), "Tensor chứa NaN hoặc Inf!"
    assert abs(float(t1.mean().item())) < 1e-5, f"Mean sau Z-score lệch 0: {t1.mean().item()}"
    assert abs(float(t1.std(unbiased=False).item()) - 1.0) < 1e-5, (
        f"Std(unbiased=False) sau Z-score lệch 1.0: {t1.std(unbiased=False).item()}"
    )

    # Kiểm tra an toàn khi bên ngoài bật autocast bfloat16 trên CPU
    prep_res_amp = FFTPreprocessor(mode="gaussian_residual", use_hann=True, normalize=True)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        t_amp = prep_res_amp(dummy_img)
    assert t_amp.dtype == torch.float32 and torch.isfinite(t_amp).all()
    print(
        f"  + Shape: {tuple(t1.shape)} | Dtype: {t1.dtype} | Mean={t1.mean().item():.2e}, "
        f"Std={t1.std(unbiased=False).item():.6f} | Autocast-Safe => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 2: Định vị chính xác tọa độ đỉnh sóng tuần hoàn chu kỳ 2px, 4px, 8px
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 2] Kiểm chứng định vị tọa độ đỉnh tần số chu kỳ 2px, 4px và 8px:")
    prep_zero_dc = FFTPreprocessor(mode="raw", remove_dc=True, normalize=True)
    yy, xx = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")

    # 2A: Bàn cờ chu kỳ 2px theo cả 2 chiều: (-1)^(x + y) -> Đỉnh Nyquist phải nằm đúng tại (0, 0)
    checker_val = np.where(((xx + yy) % 2) == 0, 200, 50).astype(np.uint8)
    checker_img = Image.fromarray(np.stack([checker_val] * 3, axis=-1), mode="RGB")
    spec_2px = prep_zero_dc(checker_img)[0]
    peak_2px = divmod(int(torch.argmax(spec_2px).item()), 224)
    assert peak_2px == (0, 0), f"Đỉnh bàn cờ 2px nằm sai vị trí: {peak_2px}, kỳ vọng (0, 0)!"

    # 2B: Sóng sọc dọc chu kỳ 4px (k_x = ±224/4 = ±56) -> Hai đỉnh phải tại (112, 56) và (112, 168)
    wave_4px = np.rint(128.0 + 100.0 * np.cos(2.0 * np.pi * xx / 4.0)).astype(np.uint8)
    img_4px = Image.fromarray(np.stack([wave_4px] * 3, axis=-1), mode="RGB")
    spec_4px = prep_zero_dc(img_4px)[0]
    top2_4px = torch.topk(spec_4px.flatten(), k=2).indices.tolist()
    coords_4px = {divmod(idx, 224) for idx in top2_4px}
    assert coords_4px == {(112, 56), (112, 168)}, (
        f"Đỉnh sóng chu kỳ 4px sai tọa độ: {coords_4px}, kỳ vọng {{(112, 56), (112, 168)}}!"
    )

    # 2C: Sóng sọc ngang chu kỳ 8px (k_y = ±224/8 = ±28) -> Hai đỉnh phải tại (84, 112) và (140, 112)
    wave_8px = np.rint(128.0 + 100.0 * np.cos(2.0 * np.pi * yy / 8.0)).astype(np.uint8)
    img_8px = Image.fromarray(np.stack([wave_8px] * 3, axis=-1), mode="RGB")
    spec_8px = prep_zero_dc(img_8px)[0]
    top2_8px = torch.topk(spec_8px.flatten(), k=2).indices.tolist()
    coords_8px = {divmod(idx, 224) for idx in top2_8px}
    assert coords_8px == {(84, 112), (140, 112)}, (
        f"Đỉnh sóng chu kỳ 8px sai tọa độ: {coords_8px}, kỳ vọng {{(84, 112), (140, 112)}}!"
    )
    print(
        "  + Chu kỳ 2px -> Đỉnh (0, 0) | Chu kỳ 4px -> Đỉnh (112, 56) & (112, 168) | "
        "Chu kỳ 8px -> Đỉnh (84, 112) & (140, 112) => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 3: Bất biến dịch chuyển tuần hoàn (torch.roll) & Đối xứng gương lật ngang lưới chẵn 224
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 3] Kiểm tra bất biến dịch chuyển tuần hoàn & đối xứng lật ngang trên lưới chẵn 224:")
    rolled_arr = np.roll(rand_arr, shift=(37, -53), axis=(0, 1))
    rolled_img = Image.fromarray(rolled_arr, mode="RGB")
    spec_orig = prep(dummy_img)
    spec_rolled = prep(rolled_img)
    assert torch.allclose(spec_orig, spec_rolled, atol=1e-4, rtol=1e-4), (
        f"Phổ biên độ không bất biến với dịch chuyển tuần hoàn! Max diff = {(spec_orig - spec_rolled).abs().max()}"
    )

    # Kiểm tra đối xứng khi lật ngang ảnh (kể cả khi bật use_hann=True nhờ periodic=False!)
    for hann_flag in (False, True):
        p_sym = FFTPreprocessor(mode="raw", use_hann=hann_flag, normalize=True)
        s_orig = p_sym(dummy_img)
        flipped_pil = dummy_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        s_flip = p_sym(flipped_pil)

        # Trên lưới chẵn 224 sau fftshift: cột 0 (Nyquist -112) giữ nguyên, cột 1..223 đảo qua tâm 112
        expected_flip_spec = torch.empty_like(s_orig)
        expected_flip_spec[:, :, 0] = s_orig[:, :, 0]
        expected_flip_spec[:, :, 1:] = s_orig[:, :, 1:].flip(-1)
        assert torch.allclose(s_flip, expected_flip_spec, atol=1e-4, rtol=1e-4), (
            f"Sai quy tắc đối xứng lật ngang (use_hann={hann_flag})! "
            f"Max diff = {(s_flip - expected_flip_spec).abs().max()}"
        )
    print(
        "  + Dịch chuyển tuần hoàn khớp (atol=1e-4) | Lật ngang đối xứng chuẩn qua trục k_x=0 "
        "(cột 0 Nyquist cố định, cả use_hann=False & True) => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 4: Chốt chặn ảnh phẳng đồng màu (Đen, Xám, Trắng, Màu đơn sắc) -> Ma trận 0.0
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 4] Kiểm tra chốt chặn ảnh phẳng đồng màu (Chống xung DC +224 & chống NaN):")
    flat_colors = [
        ("Đen (0,0,0)", (0, 0, 0)),
        ("Xám (128,128,128)", (128, 128, 128)),
        ("Trắng (255,255,255)", (255, 255, 255)),
        ("Màu đơn sắc (30,140,220)", (30, 140, 220)),
    ]
    for mode_name in ("raw", "gaussian_residual"):
        for hann_flag in (False, True):
            p_flat = FFTPreprocessor(mode=mode_name, use_hann=hann_flag, normalize=True)
            for label_str, rgb_color in flat_colors:
                flat_im = Image.new("RGB", (224, 224), color=rgb_color)
                out_flat = p_flat(flat_im)
                assert torch.isfinite(out_flat).all(), f"Sinh NaN/Inf trên ảnh {label_str}!"
                assert torch.all(out_flat == 0.0), (
                    f"Ảnh phẳng {label_str} (mode={mode_name}, hann={hann_flag}) không trả về toàn 0.0! "
                    f"Max={out_flat.max().item()}"
                )
    print(
        "  + Đen (0) | Xám (128) | Trắng (255) | Màu phẳng -> Trả về toàn 0.0 sạch tuyệt đối "
        "ở mọi tổ hợp mode & Hann => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 5: Đối chiếu công thức toán học độc lập cho Gaussian Residual, Hann & Remove DC
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 5] Đối chiếu tham chiếu toán học cho Gaussian Residual, Hann Window & Remove DC:")
    # 5A: Tính độc lập từng bước từ đầu để kiểm chứng Gaussian Residual + Hann (remove_dc=False)
    p_unnorm_res = FFTPreprocessor(mode="gaussian_residual", use_hann=True, remove_dc=False, normalize=False)
    out_res = p_unnorm_res(dummy_img)
    ref_rgb = torch.from_numpy(rand_arr.astype(np.float32) / 255.0)
    ref_luma = 0.299 * ref_rgb[..., 0] + 0.587 * ref_rgb[..., 1] + 0.114 * ref_rgb[..., 2]
    c1d = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cpu")
    gy, gx = torch.meshgrid(c1d, c1d, indexing="ij")
    gk = torch.exp(-(gx ** 2 + gy ** 2) / 2.0)
    gk = (gk / gk.sum()).view(1, 1, 3, 3)
    ref_smooth = F.conv2d(F.pad(ref_luma.view(1, 1, 224, 224), (1, 1, 1, 1), mode="reflect"), gk).view(224, 224)
    hw1d = torch.hann_window(224, periodic=False, dtype=torch.float32, device="cpu")
    assert torch.allclose(hw1d, hw1d.flip(0)), "Cửa sổ Hann 1D không đối xứng!"
    ref_sig = (ref_luma - ref_smooth) * torch.outer(hw1d, hw1d)
    ref_spec = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(ref_sig)))).unsqueeze(0)

    assert torch.allclose(out_res, ref_spec, atol=1e-5), (
        f"Sai lệch công thức Gaussian Residual + Hann! Max diff = {(out_res - ref_spec).abs().max()}"
    )

    # 5B: Kiểm chứng toán học độc lập cho remove_dc=True (cả khi không Hann và có Hann)
    p_dc_raw = FFTPreprocessor(mode="raw", use_hann=False, remove_dc=True, normalize=False)
    out_dc_raw = p_dc_raw(dummy_img)
    dc_bin_raw = float(out_dc_raw[0, 112, 112].item())
    assert dc_bin_raw < 1e-4, f"remove_dc=True (không Hann) chưa triệt tiêu DC: {dc_bin_raw}"

    p_dc_hann = FFTPreprocessor(mode="raw", use_hann=True, remove_dc=True, normalize=False)
    out_dc_hann = p_dc_hann(dummy_img)
    dc_bin_hann = float(out_dc_hann[0, 112, 112].item())
    assert dc_bin_hann < 1e-4, f"remove_dc=True (kết hợp Hann) chưa triệt tiêu DC: {dc_bin_hann}"

    print(
        f"  + Gaussian 3x3 + Hann 2D khớp công thức (atol=1e-5) | "
        f"remove_dc=True triệt tiêu DC tuyệt đối: Raw={dc_bin_raw:.2e}, Hann={dc_bin_hann:.2e} => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 6: Tích hợp liên hoàn với Bước 3 (CommonProcess) & Chốt chặn an toàn đầu vào
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 6] Tích hợp liên hoàn với Bước 3 (CommonProcess) & Kiểm tra chốt chặn đầu vào:")
    from pre_processing_for_all import CommonProcess
    from pre_processing_EfficientNet import EfficientNetPreprocessor

    import os
    dataset_env = os.environ.get("DEFACTIFY_DATASET_ROOT")
    dataset_root = Path(dataset_env) if dataset_env else Path(r"D:\nam_4\DATN\dataset\Defactify_dataset")
    meta_path = dataset_root / "metadata.csv"
    disk_test_status = "SKIP (Không tìm thấy metadata.csv)"

    if meta_path.exists():
        import pandas as pd

        df = pd.read_csv(meta_path)
        sample_sdxl = dataset_root / df[df["model_name"] == "Stable_Diffusion_XL"].iloc[0]["filepath"]
        coco_df = df[df["model_name"] == "MS_COCO"]
        sample_coco_rect = None
        sample_coco_small = None
        for rel_p in coco_df["filepath"]:
            fp = dataset_root / rel_p
            with Image.open(fp) as im:
                w0, h0 = im.size
                if sample_coco_rect is None and w0 != h0 and min(w0, h0) >= 224:
                    sample_coco_rect = fp
                if sample_coco_small is None and min(w0, h0) < 224:
                    sample_coco_small = fp
                if sample_coco_rect is not None and sample_coco_small is not None:
                    break

        step3 = CommonProcess(mode="eval", crop_size=224)
        eff_prep = EfficientNetPreprocessor(view_mode="native", normalize=True)

        for tag, spath in [
            ("SDXL (1024x1024)", sample_sdxl),
            ("MS_COCO chữ nhật", sample_coco_rect),
            ("MS_COCO nhỏ (<224px)", sample_coco_small),
        ]:
            views = step3(spath)
            native_copy_before = np.array(views["view_native"], copy=True)

            t_fft = prep(views)
            t_eff = eff_prep(views)

            # Đảm bảo FFTPreprocessor không làm biến đổi view_native dùng chung với EfficientNet
            assert np.array_equal(native_copy_before, np.asarray(views["view_native"]))
            assert t_fft.shape == (1, 224, 224) and t_fft.dtype == torch.float32
            assert torch.isfinite(t_fft).all()
            assert t_eff.shape == (3, 224, 224)
            print(f"  + {tag:<22} -> CommonProcess -> FFT Tensor {tuple(t_fft.shape)} (Đồng bộ Native) => [PASS]")
        disk_test_status = "PASS (3/3 định dạng ảnh thật)"
    else:
        print(f"  ! Bỏ qua kiểm tra ảnh đĩa thực tế vì không tìm thấy: {meta_path}")

    # Chốt chặn bắt lỗi đầu vào: sai hệ màu, sai kích thước (chống truyền nhầm view_global chưa cắt), thiếu khóa
    for bad_img in (
        Image.new("RGBA", (224, 224), (255, 0, 0, 128)),
        Image.new("L", (224, 224), 128),
        Image.new("RGB", (640, 480), (100, 100, 100)),  # Sai kích thước 224x224 -> Phải chặn ngay!
    ):
        try:
            prep(bad_img)
            raise AssertionError("Phải ném ValueError với ảnh sai hệ màu hoặc sai kích thước!")
        except ValueError:
            pass

    for bad_inp in ("path.jpg", 123, [1, 2], None):
        try:
            prep(bad_inp)  # type: ignore[arg-type]
            raise AssertionError(f"Phải ném TypeError với kiểu {type(bad_inp)}!")
        except TypeError:
            pass

    try:
        prep({"view_global": Image.new("RGB", (640, 480))})
        raise AssertionError("Phải ném KeyError khi dictionary thiếu 'view_native'!")
    except KeyError:
        pass

    # Kiểm tra đồng bộ tham số image_size tùy chỉnh (ví dụ 128x128)
    prep_128 = FFTPreprocessor(image_size=128)
    t_128 = prep_128(Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8), mode="RGB"))
    assert t_128.shape == (1, 128, 128)
    print("  + Chốt chặn an toàn: Chặn ngoài RGB | Chặn ảnh khác 224x224 | Chặn thiếu 'view_native' | Đồng bộ image_size => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 7 (BENCHMARK): Warm-up + Median, P95 & Thông lượng xử lý trên CPU
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK] Đo hiệu năng xử lý trên CPU (20 warm-up + 100 runs):")
    for _ in range(20):
        prep(dummy_img)

    latencies_raw = []
    latencies_res = []
    prep_res = FFTPreprocessor(mode="gaussian_residual", use_hann=False, normalize=True)
    for _ in range(20):
        prep_res(dummy_img)

    for _ in range(100):
        t0 = time.perf_counter()
        prep(dummy_img)
        latencies_raw.append((time.perf_counter() - t0) * 1000.0)

        t1_start = time.perf_counter()
        prep_res(dummy_img)
        latencies_res.append((time.perf_counter() - t1_start) * 1000.0)

    med_raw = float(np.median(latencies_raw))
    p95_raw = float(np.percentile(latencies_raw, 95))
    fps_raw = 1000.0 / float(np.mean(latencies_raw))
    med_res = float(np.median(latencies_res))
    p95_res = float(np.percentile(latencies_res, 95))
    print(
        f"  + Mode 'raw' (Baseline)      : Median = {med_raw:.3f} ms/ảnh | P95 = {p95_raw:.3f} ms/ảnh "
        f"(~{fps_raw:,.0f} ảnh/giây)"
    )
    print(
        f"  + Mode 'gaussian_residual'   : Median = {med_res:.3f} ms/ảnh | P95 = {p95_res:.3f} ms/ảnh => [PASS]"
    )

    print("\n" + "=" * 80)
    print("TỔNG KẾT NGHIỆM THU BƯỚC 4C (2D FFT SPECTRAL):")
    print("  + Kiểm tra độc lập toán học (Test 0 -> Test 5): 100% PASS")
    print(f"  + Kiểm tra tích hợp ảnh đĩa (Test 6)         : {disk_test_status}")
    print("  + Kiểm tra Benchmark CPU (Test 7)           : PASS")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
