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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# BỘ HẰNG SỐ CHUẨN GỐC OPENAI CLIP (WIT-400M) CHO NHÁNH NGỮ NGHĨA (ViT-B/16)
# ==============================================================================
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
CLIP_TARGET_SIZE: int = 224


class CLIPPreprocessor:
    """
    BƯỚC 4B: Tiền xử lý Baseline chuẩn gốc OpenAI CLIP cho Nhánh Ngữ Nghĩa
    (Semantic Branch - Frozen CLIP ViT-B/16 theo phương pháp UnivFD, CVPR 2023).

    Lưu ý kiến trúc cho Bước 5 (`CLIP.py`):
    - Đầu vào mạng ViT-B/16: Tensor (3, 224, 224) chia thành 14x14 = 196 patch (16x16) + 1 CLS token.
    - Chiều ẩn nội bộ Transformer: 768, nhưng vector đặc trưng đầu ra chuẩn từ
      `clip_model.encode_image(images)` (sau tầng Linear Projection) có kích thước [B, 512].

    Quy trình toán học Baseline chuẩn OpenAI (`clip._transform(224)`):
    1. Tiếp nhận `view_global` (ảnh PIL RGB giữ nguyên kích thước gốc WxH, đã chuẩn hóa EXIF
       và đồng bộ trạng thái lật ngang ở Bước 3 `CommonProcess`).
    2. Shortest-Edge Resize về 224 bằng thuật toán nội suy `BICUBIC` (bảo toàn 100% tỷ lệ
       hình học tự nhiên của vật thể, không ép vuông gây méo hình).
    3. Cắt chính tâm tất định `CenterCrop(224, 224)` ở mọi chế độ (`train`, `eval`, `extract`)
       nhằm đồng bộ tuyệt đối với backbone CLIP đóng băng (`requires_grad=False`).
    4. Chuyển sang Tensor float32 [0.0, 1.0] (khớp từng bit với `torchvision.transforms.ToTensor`)
       và chuẩn hóa Z-score bằng hằng số gốc `CLIP_MEAN`, `CLIP_STD` của OpenAI.
    """

    def __init__(
        self,
        image_size: int = CLIP_TARGET_SIZE,
        normalize: bool = True,  # chuẩn hóa theo phân phối của openAI CLIP vì 400 triệu ảnh đều
        #được chuẩn hóa bằng các bộ số này
    ):
        """
        Khởi tạo bộ tiền xử lý chuẩn OpenAI CLIP.

        Args:
            image_size: Kích thước khung nhìn vuông đầu ra (mặc định 224 cho CLIP ViT-B/16).
            normalize: True để chuẩn hóa theo CLIP_MEAN / CLIP_STD (chuẩn cho train/eval/extract),
                       False để giữ Tensor trong đoạn [0.0, 1.0] (phục vụ kiểm thử/trực quan hóa).
        """
        if not HAS_TORCH:
            raise ImportError("Module này yêu cầu PyTorch. Hãy kích hoạt môi trường venv đã cài đặt PyTorch.")

        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError(f"image_size phải là số nguyên dương (> 0), nhận được: {image_size}")
        if not isinstance(normalize, bool):
            raise TypeError(f"normalize phải là kiểu bool, nhận được: {type(normalize)}")

        self.image_size: int = image_size
        self.normalize: bool = normalize

        # Tiền cấp phát Tensor mean/std kích thước (3, 1, 1) ngay trong __init__
        # để không cấp phát lại bộ nhớ trên mỗi bức ảnh trong vòng lặp DataLoader
        self.mean: torch.Tensor = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std: torch.Tensor = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)

    @staticmethod
    def _shortest_edge_resize(img: Image.Image, target_size: int = CLIP_TARGET_SIZE) -> Image.Image:
        """
        Thu phóng cạnh ngắn nhất về `target_size` (224) bằng nội suy BICUBIC.
        Sử dụng công thức ép phần nguyên `int(target_size * long_side / short_side)`
        khớp tuyệt đối 100% với `torchvision.transforms.Resize(224, BICUBIC)` của OpenAI CLIP.
        """
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Kích thước ảnh không hợp lệ: {(w, h)}")

        # Tối ưu tốc độ (Short-circuit): Nếu cạnh ngắn đã bằng đúng target_size thì không nội suy lại
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            return img

        if w < h:
            new_w = target_size
            new_h = int(target_size * h / w)
        else:
            new_h = target_size
            new_w = int(target_size * w / h)

        return img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    @staticmethod
    def _center_crop(img: Image.Image, crop_size: int = CLIP_TARGET_SIZE) -> Image.Image:
        """
        Cắt khung vuông `crop_size x crop_size` (224x224) ngay tại chính tâm bức ảnh,
        khớp chính xác công thức làm tròn tọa độ của `torchvision.transforms.CenterCrop`.
        """
        w, h = img.size
        if w < crop_size or h < crop_size:
            raise ValueError(
                f"Kích thước ảnh sau resize ({w}, {h}) nhỏ hơn crop_size ({crop_size})."
            )

        if w == crop_size and h == crop_size:
            return img

        left = int(round((w - crop_size) / 2.0))
        top = int(round((h - crop_size) / 2.0))
        return img.crop((left, top, left + crop_size, top + crop_size))

    @staticmethod
    def compute_kept_area_ratio(
        orig_size: Tuple[int, int], image_size: int = CLIP_TARGET_SIZE
    ) -> float:
        """
        Tiện ích thống kê tỷ lệ diện tích khung hình được giữ lại sau bước
        Shortest-Edge Resize + CenterCrop (ví dụ: 1024x1024 -> 1.0, 640x480 -> ~0.7517).
        """
        w, h = orig_size
        if w <= 0 or h <= 0:
            raise ValueError(f"orig_size không hợp lệ: {orig_size}")
        if w < h:
            rw, rh = image_size, int(image_size * h / w)
        elif h < w:
            rw, rh = int(image_size * w / h), image_size
        else:
            rw, rh = image_size, image_size
        return float(image_size * image_size) / float(rw * rh)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        Chuyển đổi ảnh PIL RGB (224x224) sang Tensor float32 [0.0, 1.0] và chuẩn hóa CLIP.
        Sử dụng mảng uint8 writable + toán tử in-place của PyTorch để triệt tiêu cảnh báo
        Non-writable NumPy array và đạt sai số 0.0 tuyệt đối so với `torchvision`.
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f"Đầu vào tensor hóa phải là PIL.Image.Image, nhận được: {type(img)}")
        if img.mode != "RGB":
            raise ValueError(
                f"Ảnh đầu vào phải ở hệ màu 'RGB', nhận được mode='{img.mode}'. "
                f"Hãy đảm bảo ảnh đã qua chuẩn hóa ở Bước 3 (CommonProcess)."
            )
        if img.size != (self.image_size, self.image_size):
            raise ValueError(
                f"Ảnh trước khi tensor hóa phải có kích thước ({self.image_size}, {self.image_size}), "
                f"nhận được: {img.size}."
            )

        arr = np.array(img, dtype=np.uint8, copy=True)  # shape: (224, 224, 3), writable
        tensor = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )

        if self.normalize:
            tensor.sub_(self.mean).div_(self.std)

        return tensor

    def __call__(self, inp: Union[Dict[str, Any], Image.Image]) -> torch.Tensor:
        """
        Thực thi tiền xử lý Baseline chuẩn OpenAI CLIP cho 1 mẫu ảnh.

        Args:
            inp: Có thể là:
                 - Dictionary đầu ra từ `CommonProcess` ở Bước 3 (tự động trích xuất `inp['view_global']`).
                 - Hoặc đối tượng `PIL.Image.Image` hệ màu 'RGB' trực tiếp.

        Returns:
            torch.Tensor kích thước (3, 224, 224), kiểu `torch.float32`, bộ nhớ liên tục (`contiguous`).
        """
        if isinstance(inp, dict):
            if "view_global" not in inp:
                raise KeyError(
                    f"Thiếu khóa bắt buộc 'view_global' trong dictionary đầu vào từ Bước 3. "
                    f"Các khóa hiện có: {list(inp.keys())}."
                )
            img = inp["view_global"]
        elif isinstance(inp, Image.Image):
            img = inp
        else:
            raise TypeError(
                f"Đầu vào của CLIPPreprocessor phải là Dictionary từ Bước 3 hoặc PIL.Image.Image, "
                f"nhận được: {type(inp)}."
            )

        if not isinstance(img, Image.Image):
            raise TypeError(
                f"Trường 'view_global' phải là đối tượng PIL.Image.Image, nhận được: {type(img)}."
            )
        if img.mode != "RGB":
            raise ValueError(
                f"Ảnh đầu vào của CLIPPreprocessor phải ở hệ màu 'RGB', nhận được mode='{img.mode}'."
            )

        resized_img = self._shortest_edge_resize(img, self.image_size)
        cropped_img = self._center_crop(resized_img, self.image_size)
        return self._pil_to_tensor(cropped_img)


# Alias đồng nhất quy ước đặt tên giữa các module Bước 4
PreprocessCLIP = CLIPPreprocessor


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP NGHIỆM THU TOÀN DIỆN BƯỚC 4B (OPENAI CLIP BASELINE)
# ==============================================================================
def run_unit_tests():
    """
    Kiểm chứng nghiệm thu độc lập 6 tiêu chuẩn cốt lõi cho `pre_processing_CLIP.py`:
    1. Shape (3, 224, 224), Dtype torch.float32 và tính liên tục bộ nhớ (.is_contiguous()).
    2. Khớp tuyệt đối từng bit (sai số 0.0 / torch.equal) với pipeline chính thức của OpenAI CLIP
       dựng bằng `torchvision.transforms` trên mọi tỷ lệ ảnh (vuông, ngang, dọc, ảnh nhỏ < 224px)
       mà không phát sinh bất kỳ UserWarning nào.
    3. Chứng minh bảo toàn 100% tỷ lệ hình học vật thể (vẽ hình tròn tâm ảnh chữ nhật 640x480,
       kiểm chứng sau tiền xử lý đường kính ngang == đường kính dọc, không bị méo thành elip).
    4. Tích hợp liên hoàn với Bước 3 (`CommonProcess`) trên ảnh thật SDXL, MS_COCO chữ nhật,
       ảnh MS_COCO nhỏ < 224px và kiểm tra đồng bộ lật ngang ở `mode='extract'`.
    5. Kiểm tra toàn bộ chốt chặn an toàn đầu vào (chặn RGBA/L, sai kiểu dữ liệu, thiếu khóa dict).
    6. Đo hiệu năng thực tế trên CPU (Median & P95 ms/ảnh).
    """
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ NGHIỆM THU BƯỚC 4B: pre_processing_CLIP.py (OpenAI CLIP Baseline)")
    print("=" * 80)

    prep = CLIPPreprocessor(image_size=224, normalize=True)

    # -------------------------------------------------------------------------
    # TEST 1: Kiểm tra Shape (3, 224, 224), Dtype float32 và Contiguous
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 1] Kiểm tra Shape, Dtype và tính liên tục bộ nhớ:")
    rng = np.random.default_rng(42)
    dummy_img = Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), mode="RGB")
    t1 = prep(dummy_img)

    assert isinstance(t1, torch.Tensor), "Đầu ra không phải là torch.Tensor!"
    assert t1.shape == (3, 224, 224), f"Shape sai: {t1.shape}, kỳ vọng (3, 224, 224)!"
    assert t1.dtype == torch.float32, f"Dtype sai: {t1.dtype}, kỳ vọng torch.float32!"
    assert t1.is_contiguous(), "Tensor đầu ra không liên tục trong bộ nhớ (contiguous)!"
    print(f"  + Shape: {tuple(t1.shape)} | Dtype: {t1.dtype} | Contiguous: True => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 2: Đối chiếu khớp từng bit (0.0 diff) với chuẩn OpenAI / Torchvision
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 2] Đối chiếu sai số từng bit với pipeline chuẩn OpenAI CLIP (Torchvision):")
    official_clip_transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(CLIP_MEAN, CLIP_STD),
    ])

    test_resolutions = [
        (1024, 1024),  # Ảnh AI vuông chuẩn SDXL / SD3 / DALL-E 3
        (640, 480),    # Ảnh Real chữ nhật ngang 4:3 phổ biến của MS_COCO
        (640, 427),    # Ảnh Real chữ nhật ngang 3:2 của MS_COCO
        (427, 640),    # Ảnh Real chữ nhật dọc 2:3
        (500, 333),    # Kích thước lẻ bất kỳ
        (640, 213),    # Ảnh MS_COCO có cạnh ngắn < 224px
        (224, 224),    # Ảnh đã đúng kích thước đích (kiểm tra nhánh short-circuit)
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Bắt chặt mọi cảnh báo ngầm (kể cả Non-writable NumPy warning)
        for w, h in test_resolutions:
            test_im = Image.fromarray(
                rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB"
            )
            t_ours = prep(test_im)
            t_official = official_clip_transform(test_im)
            max_diff = float((t_ours - t_official).abs().max().item())
            assert torch.equal(t_ours, t_official), (
                f"Lệch số học trên kích thước {(w, h)}: max_diff={max_diff}"
            )

    kept_square = CLIPPreprocessor.compute_kept_area_ratio((1024, 1024))
    kept_coco_43 = CLIPPreprocessor.compute_kept_area_ratio((640, 480))
    assert abs(kept_square - 1.0) < 1e-6
    assert abs(kept_coco_43 - (224.0 / 298.0)) < 1e-6
    print(
        f"  + Khớp từng bit (max_diff = 0.0, 0 cảnh báo) trên cả {len(test_resolutions)} tỷ lệ khung hình! => [PASS]"
    )
    print(
        f"  + Tỷ lệ diện tích giữ lại: AI 1024x1024 = {kept_square*100:.1f}% | "
        f"COCO 640x480 = {kept_coco_43*100:.1f}%"
    )

    # -------------------------------------------------------------------------
    # TEST 3: Chứng minh bảo toàn 100% tỷ lệ hình học vật thể (Không méo hình)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 3] Chứng minh hình học không làm bóp méo vật thể trên ảnh chữ nhật 640x480:")
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:480, :640]
    circle_mask = (xx - 320) ** 2 + (yy - 240) ** 2 <= 120 ** 2
    canvas[circle_mask] = 255
    circle_img = Image.fromarray(canvas, mode="RGB")

    prep_unnorm = CLIPPreprocessor(image_size=224, normalize=False)
    t_circle = prep_unnorm(circle_img)  # (3, 224, 224) trong đoạn [0, 1]
    bright_mask = t_circle[0] > 0.5
    ys, xs = torch.where(bright_mask)
    diam_x = int(xs.max().item() - xs.min().item() + 1)
    diam_y = int(ys.max().item() - ys.min().item() + 1)
    assert abs(diam_x - diam_y) <= 1, (
        f"Hình tròn bị méo thành elip! Đường kính ngang={diam_x}, dọc={diam_y}"
    )
    print(
        f"  + Hình tròn bán kính 120px trên khung 640x480 -> Sau CLIP đường kính ngang={diam_x}px, "
        f"dọc={diam_y}px (Tròn đều 100%) => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 4: Tích hợp liên hoàn với Bước 3 (CommonProcess) trên ảnh thật
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 4] Tích hợp liên hoàn với Bước 3 (CommonProcess) trên dữ liệu thật:")
    from pre_processing_for_all import CommonProcess

    dataset_root = Path(r"D:\nam_4\DATN\dataset\Defactify_dataset")
    meta_path = dataset_root / "metadata.csv"

    if meta_path.exists():
        import pandas as pd

        df = pd.read_csv(meta_path)
        sample_sdxl = dataset_root / df[df["model_name"] == "Stable_Diffusion_XL"].iloc[0]["filepath"]

        # Chọn 1 ảnh MS_COCO chữ nhật chuẩn và 1 ảnh MS_COCO có cạnh < 224px
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

        step3_eval = CommonProcess(mode="eval", crop_size=224)
        out_sdxl = step3_eval(sample_sdxl)
        out_coco_rect = step3_eval(sample_coco_rect)
        out_coco_small = step3_eval(sample_coco_small)

        t_sdxl = prep(out_sdxl)
        t_coco_rect = prep(out_coco_rect)
        t_coco_small = prep(out_coco_small)

        for t_check in (t_sdxl, t_coco_rect, t_coco_small):
            assert t_check.shape == (3, 224, 224) and t_check.dtype == torch.float32
            assert torch.isfinite(t_check).all(), "Phát hiện NaN/Inf trong Tensor đầu ra!"

        # Kiểm chứng đồng bộ trạng thái lật ngang ở mode='extract' (cho trích xuất đặc trưng đóng băng)
        step3_ext = CommonProcess(mode="extract", crop_size=224)
        t_no_flip = prep(step3_ext(sample_coco_rect, force_flip=False))
        t_flipped = prep(step3_ext(sample_coco_rect, force_flip=True))
        # Với chiều rộng chẵn 298 -> crop 224 (left=37, right=261; đối xứng hoàn hảo vì 298 - 261 = 37)
        assert torch.allclose(torch.flip(t_no_flip, dims=[2]), t_flipped, atol=1e-5), (
            "Tensor CLIP khi force_flip=True không đối xứng gương ngang với force_flip=False!"
        )

        print(f"  + SDXL {out_sdxl['orig_size']} -> CommonProcess -> CLIP Tensor {tuple(t_sdxl.shape)} => [PASS]")
        print(f"  + MS_COCO chữ nhật {out_coco_rect['orig_size']} -> CommonProcess -> CLIP Tensor {tuple(t_coco_rect.shape)} => [PASS]")
        print(f"  + MS_COCO cạnh nhỏ {out_coco_small['orig_size']} -> CommonProcess -> CLIP Tensor {tuple(t_coco_small.shape)} => [PASS]")
        print("  + Đồng bộ lật ngang ở mode='extract' (force_flip=False/True) khớp đối xứng gương => [PASS]")
    else:
        print("  ! Bỏ qua kiểm tra ảnh đĩa do không tìm thấy metadata.csv.")

    # -------------------------------------------------------------------------
    # TEST 5: Chốt chặn an toàn đầu vào (Input Validation & Error Guards)
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 5] Kiểm tra các chốt chặn an toàn đầu vào:")
    rgba_img = Image.new("RGBA", (640, 480), (255, 0, 0, 128))
    gray_img = Image.new("L", (640, 480), 128)
    for bad_mode_img in (rgba_img, gray_img):
        try:
            prep(bad_mode_img)
            raise AssertionError(f"Phải báo lỗi với ảnh hệ màu '{bad_mode_img.mode}'!")
        except ValueError:
            pass

    for bad_inp in ("anh.jpg", 999, [1, 2, 3], None):
        try:
            prep(bad_inp)  # type: ignore[arg-type]
            raise AssertionError(f"Phải báo lỗi TypeError với đầu vào: {type(bad_inp)}")
        except TypeError:
            pass

    try:
        prep({"view_native": Image.new("RGB", (224, 224))})
        raise AssertionError("Phải báo lỗi KeyError khi thiếu khóa 'view_global'!")
    except KeyError:
        pass

    print("  + Bắt lỗi thành công: Chặn ảnh ngoài RGB | Chặn sai kiểu dữ liệu | Chặn thiếu khóa 'view_global' => [PASS]")

    # -------------------------------------------------------------------------
    # BENCHMARK: Đo hiệu năng xử lý trên CPU (100 lần chạy)
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK] Đo lường tốc độ thực thi trên CPU (100 runs):")
    lat_rect, lat_sq = [], []
    sq_1024 = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8), mode="RGB")
    for _ in range(100):
        t0 = time.perf_counter()
        prep(dummy_img)  # 640x480
        lat_rect.append((time.perf_counter() - t0) * 1000.0)

        t1_start = time.perf_counter()
        prep(sq_1024)    # 1024x1024
        lat_sq.append((time.perf_counter() - t1_start) * 1000.0)

    print(
        f"  + Ảnh 640x480   : Median = {np.median(lat_rect):.3f} ms/ảnh | P95 = {np.percentile(lat_rect, 95):.3f} ms/ảnh"
    )
    print(
        f"  + Ảnh 1024x1024 : Median = {np.median(lat_sq):.3f} ms/ảnh | P95 = {np.percentile(lat_sq, 95):.3f} ms/ảnh => [PASS]"
    )

    print("\n" + "=" * 80)
    print("TẤT CẢ TIÊU CHUẨN NGHIỆM THU BƯỚC 4B (OPENAI CLIP BASELINE) ĐÃ VƯỢT QUA 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()

