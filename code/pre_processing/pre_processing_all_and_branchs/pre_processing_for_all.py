import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

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


class CommonProcess:
    """
    BƯỚC 3: Tầng tiền xử lý chung đồng bộ (Shared Dual-View Preprocessor).

    Quy trình chuẩn hóa thống nhất cho kiến trúc 3 nhánh chữ Y:
    1. Đọc ảnh đúng 1 lần -> sửa hướng hiển thị EXIF -> xử lý ảnh 16-bit / kênh Alpha -> chuyển RGB.
    2. Thực hiện biến đổi đồng bộ trên ảnh nguồn dựa theo chế độ `mode`:
       - 'train'  : Lật ngang ngẫu nhiên (p = flip_prob), nén JPEG giả lập nếu bật (Baseline tắt p = 0.0).
       - 'eval'   : Tắt toàn bộ yếu tố ngẫu nhiên (không lật, không nén JPEG ngẫu nhiên).
       - 'extract': Phục vụ trích xuất đặc trưng cố định cho Fusion MLP; cho phép chỉ định
                    `force_flip=True/False` và giữ vị trí cắt CenterCrop tất định.
    3. Tách thành hai góc nhìn (Dual-View) đồng bộ hướng và trạng thái lật:
       - `view_native` (PIL 224x224): Cắt trực tiếp từ pixel gốc, KHÔNG resize trước khi cắt
         (cấp cho EfficientNet-B0 và FFT CNN). Nếu ảnh có cạnh < 224px (44 ảnh MS_COCO),
         đệm gương viền theo vòng lặp an toàn (Reflect Padding) và bật cờ `was_padded=True`.
       - `view_global` (PIL WxH gốc): Ảnh RGB nguồn đã chuẩn hóa hướng và đồng bộ trạng thái lật,
         giữ nguyên tỷ lệ khung hình gốc để nhánh CLIP áp dụng đúng transform chuẩn của checkpoint.
    """

    VALID_MODES = ("train", "eval", "extract")

    def __init__(
        self,
        mode: Optional[str] = None,
        crop_size: int = 224,
        flip_prob: float = 0.5,
        jpeg_prob: float = 0.0,
        jpeg_quality_range: Tuple[int, int] = (70, 95),
        fixed_jpeg_quality: Optional[int] = None,
        alpha_bg_color: Tuple[int, int, int] = (128, 128, 128),
        seed: Optional[int] = None,
        is_train: Optional[bool] = None,
    ):
        """
        Khởi tạo bộ tiền xử lý chung.

        Args:
            mode: Chế độ hoạt động ('train', 'eval', hoặc 'extract'). Mặc định là 'train'.
            crop_size: Kích thước ô cắt vuông cho `view_native` (mặc định 224).
            flip_prob: Xác suất lật ngang ảnh nguồn khi `mode='train'` (mặc định 0.5).
            jpeg_prob: Xác suất nén JPEG giả lập khi `mode='train'` (Baseline = 0.0).
            jpeg_quality_range: Khoảng chất lượng JPEG (min_q, max_q) khi bật `jpeg_prob > 0`.
            fixed_jpeg_quality: Chất lượng JPEG cố định (1..100) dùng riêng khi kiểm tra độ bền (Robustness).
            alpha_bg_color: Màu nền RGB trung tính khi trộn ảnh có kênh trong suốt Alpha.
            seed: Hạt giống ngẫu nhiên phục vụ tái lập thực nghiệm.
            is_train: Tham số tương thích ngược (chỉ dùng khi không truyền `mode` tường minh).
        """
        # Chặn xung đột ngầm nếu người dùng truyền đồng thời cả `mode` và `is_train` trái ngược nhau
        if mode is not None and is_train is not None:
            implied_mode = "train" if is_train else "eval"
            if mode != implied_mode:
                raise ValueError(
                    f"Xung đột tham số: mode='{mode}' mâu thuẫn với is_train={is_train}. "
                    f"Hãy chỉ truyền `mode` ('train', 'eval', 'extract')."
                )
        if mode is None:
            mode = ("train" if is_train else "eval") if is_train is not None else "train"

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode không hợp lệ: '{mode}'. Chỉ chấp nhận một trong {self.VALID_MODES}."
            )
        if isinstance(crop_size, bool) or crop_size <= 0:
            raise ValueError(f"crop_size phải là số nguyên > 0, nhận được: {crop_size}.")
        if not (0.0 <= flip_prob <= 1.0):
            raise ValueError(f"flip_prob phải thuộc [0.0, 1.0], nhận được: {flip_prob}.")
        if not (0.0 <= jpeg_prob <= 1.0):
            raise ValueError(f"jpeg_prob phải thuộc [0.0, 1.0], nhận được: {jpeg_prob}.")

        q_min, q_max = jpeg_quality_range
        if not (1 <= q_min <= q_max <= 100):
            raise ValueError(
                f"jpeg_quality_range phải thỏa 1 <= min <= max <= 100, nhận được: {jpeg_quality_range}."
            )
        if fixed_jpeg_quality is not None and (
            isinstance(fixed_jpeg_quality, bool) or not (1 <= fixed_jpeg_quality <= 100)
        ):
            raise ValueError(
                f"fixed_jpeg_quality phải thuộc [1, 100], nhận được: {fixed_jpeg_quality}."
            )
        if mode == "train" and fixed_jpeg_quality is not None:
            raise ValueError(
                f"Không được dùng fixed_jpeg_quality={fixed_jpeg_quality} khi mode='train' "
                f"(tránh vô tình làm suy hao ảnh huấn luyện Baseline). "
                f"Chỉ dùng fixed_jpeg_quality ở mode='eval' hoặc 'extract' khi kiểm tra độ bền Robustness."
            )
        if mode != "train" and jpeg_prob > 0.0:
            raise ValueError(
                f"Tham số jpeg_prob={jpeg_prob} chỉ dùng cho mode='train' (hiện tại mode='{mode}'). "
                f"Nếu muốn kiểm tra độ bền nén JPEG khi đánh giá, hãy dùng fixed_jpeg_quality."
            )
        if len(alpha_bg_color) != 3 or any(not (0 <= c <= 255) for c in alpha_bg_color):
            raise ValueError(f"alpha_bg_color phải là bộ 3 giá trị RGB [0..255]: {alpha_bg_color}.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError(f"seed phải là số nguyên không âm (>= 0) hoặc None, nhận được: {seed}.")

        self.mode = mode
        self.is_train = (mode == "train")
        self.crop_size = int(crop_size)
        self.flip_prob = float(flip_prob) if self.mode == "train" else 0.0
        self.jpeg_prob = float(jpeg_prob) if self.mode == "train" else 0.0
        self.jpeg_quality_range = (int(q_min), int(q_max))
        self.fixed_jpeg_quality = fixed_jpeg_quality
        self.alpha_bg_color = tuple(int(c) for c in alpha_bg_color)
        self.seed = seed

        # Lưu `epoch` trong bộ nhớ chia sẻ liên tiến trình (shared memory) để lệnh `set_epoch()`
        # ở tiến trình chính tự động đồng bộ sang mọi DataLoader worker (kể cả persistent_workers=True/False)
        if HAS_TORCH:
            self._shared_epoch: Optional["torch.Tensor"] = torch.zeros(1, dtype=torch.int64).share_memory_()
            self._local_epoch = 0
        else:
            self._shared_epoch = None
            self._local_epoch = 0

        # Quản lý bộ sinh số ngẫu nhiên an toàn cho cả tiến trình chính lẫn multi-worker DataLoader
        self._rng: Optional[np.random.Generator] = None
        self._rng_state_key: Optional[Tuple[int, int, Optional[int], Optional[int]]] = None

    @property
    def epoch(self) -> int:
        if self._shared_epoch is not None:
            return int(self._shared_epoch[0].item())
        return self._local_epoch

    def set_epoch(self, epoch: int) -> None:
        """
        Đồng bộ số thứ tự epoch (tương tự BalancedGroupSampler.set_epoch) qua bộ nhớ chia sẻ
        để mỗi epoch có chuỗi augmentation mới nhưng vẫn tái lập được 100% theo `seed`.
        Lưu ý: Luôn gọi `set_epoch(epoch)` ở đầu mỗi epoch, trước `for batch in train_loader:`.
        """
        if isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch phải là số nguyên >= 0, nhận được: {epoch}.")
        if self._shared_epoch is not None:
            self._shared_epoch[0] = int(epoch)
        self._local_epoch = int(epoch)
        self._rng = None
        self._rng_state_key = None

    def _get_rng(self) -> np.random.Generator:
        """
        Lấy bộ sinh số ngẫu nhiên an toàn trên đa tiến trình (DataLoader num_workers >= 0)
        và đa epoch (kể cả khi persistent_workers=False hoặc True).
        """
        current_pid = os.getpid()
        worker_torch_seed: Optional[int] = None
        worker_id: Optional[int] = None
        if HAS_TORCH:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker_torch_seed = int(worker_info.seed)
                worker_id = int(worker_info.id)

        state_key = (current_pid, self.epoch, worker_torch_seed, worker_id)
        if self._rng is None or self._rng_state_key != state_key:
            if worker_torch_seed is not None and worker_id is not None:
                # Trong DataLoader worker: kết hợp worker_info.seed (tự đổi mỗi epoch), worker_id, epoch và self.seed
                entropy = [worker_torch_seed, worker_id, self.epoch]
                if self.seed is not None:
                    entropy.insert(0, int(self.seed))
                seq = np.random.SeedSequence(entropy)
                self._rng = np.random.default_rng(seq)
            elif self.seed is not None:
                # Trong tiến trình chính có đặt seed tái lập
                seq = np.random.SeedSequence([int(self.seed), self.epoch])
                self._rng = np.random.default_rng(seq)
            else:
                self._rng = np.random.default_rng()
            self._rng_state_key = state_key

        return self._rng

    def _load_and_sanitize_rgb(
        self, img_input: Union[str, os.PathLike, Image.Image]
    ) -> Image.Image:
        """
        Bước 1: Đọc ảnh 1 lần, chuẩn hóa hướng EXIF, xử lý ảnh 16-bit / kênh Alpha và chuyển sang RGB.
        """
        if isinstance(img_input, (str, os.PathLike)):
            img_path = Path(img_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {img_path}")
            try:
                with Image.open(img_path) as raw_img:
                    raw_img.load()
                    img = self._normalize_pil_image(raw_img)
                return img
            except (UnidentifiedImageError, OSError) as exc:
                raise OSError(f"File ảnh bị hỏng hoặc không thể giải mã tại '{img_path}': {exc}") from exc

        if isinstance(img_input, Image.Image):
            return self._normalize_pil_image(img_input)

        raise TypeError(
            f"img_input phải là đường dẫn file hoặc PIL.Image.Image, nhận được: {type(img_input)}"
        )

    def _normalize_pil_image(self, raw_img: Image.Image) -> Image.Image:
        """
        Áp dụng EXIF orientation, xử lý ảnh xám 16-bit / float và trộn nền Alpha trước khi sang RGB.
        """
        # 1. Xoay/lật đúng hướng hiển thị theo thẻ EXIF (nếu có) và gỡ thẻ orientation cũ
        img = ImageOps.exif_transpose(raw_img)

        # 2. Bảo vệ trường hợp ảnh scan / RAW 16-bit ('I;16', 'I') hoặc ảnh thực ('F')
        # (Pillow .convert('RGB') mặc định cắt cụt giá trị > 255 thành trắng xóa 255)
        if img.mode.startswith("I;16"):
            arr16 = np.asarray(img, dtype=np.float32)
            arr8 = np.clip(np.rint(arr16 / 257.0), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr8, mode="L").convert("RGB")
            return img
        if img.mode in ("I", "F"):
            arr_f = np.asarray(img, dtype=np.float32)
            vmin, vmax = float(arr_f.min()), float(arr_f.max())
            if vmax > 255.0 or vmin < 0.0:
                scale = (vmax - vmin) if vmax > vmin else 1.0
                arr8 = np.clip(np.rint((arr_f - vmin) / scale * 255.0), 0, 255).astype(np.uint8)
            else:
                arr8 = np.clip(np.rint(arr_f), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr8, mode="L").convert("RGB")
            return img

        # 3. Kiểm tra kênh Alpha / Transparency để tránh pixel rác dưới vùng trong suốt
        has_alpha = (
            img.mode in ("RGBA", "LA", "RGBa", "La")
            or (img.mode == "P" and "transparency" in img.info)
        )
        if has_alpha:
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, self.alpha_bg_color + (255,))
            img = Image.alpha_composite(background, rgba).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        elif img is raw_img:
            # Chỉ sao chép nếu exif_transpose trả lại cùng tham chiếu gốc
            img = img.copy()

        return img

    @staticmethod
    def _apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
        """
        Nén JPEG trong bộ nhớ RAM (phục vụ thực nghiệm độ bền nén mạng xã hội khi được bật).
        """
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            compressed.load()
            return compressed.convert("RGB")

    @staticmethod
    def _reflect_pad_if_needed(
        img: Image.Image, target_size: int
    ) -> Tuple[Image.Image, bool, Tuple[int, int, int, int]]:
        """
        Đệm gương viền (Reflect Padding) có kiểm soát nếu cạnh ảnh nhỏ hơn `target_size`.
        Dùng vòng lặp `while` an toàn kể cả với ảnh cực nhỏ (< target_size / 2).
        Nếu một chiều bất kỳ chỉ có đúng 1 pixel, riêng chiều đó dùng 'edge' còn chiều > 1 vẫn giữ 'reflect'.
        Không chèn viền đen 0 để tránh tạo bước nhảy biên độ giả trên phổ FFT.

        Returns:
            (padded_img, was_padded, (pad_left, pad_top, pad_right, pad_bottom))
        """
        w, h = img.size
        if w >= target_size and h >= target_size:
            return img, False, (0, 0, 0, 0)

        arr = np.asarray(img)  # shape: (H, W, 3), uint8
        cur_h, cur_w = arr.shape[:2]
        total_left, total_top, total_right, total_bottom = 0, 0, 0, 0

        while cur_h < target_size or cur_w < target_size:
            need_h = max(0, target_size - cur_h)
            need_w = max(0, target_size - cur_w)

            # Mỗi bước reflect tối đa bằng (kích_thước_hiện_tại - 1) để tương thích mọi bản NumPy
            max_step_h = max(1, cur_h - 1) if cur_h > 1 else need_h
            max_step_w = max(1, cur_w - 1) if cur_w > 1 else need_w

            pad_top = min(need_h // 2, max_step_h)
            pad_bottom = min(need_h - pad_top, max_step_h)
            pad_left = min(need_w // 2, max_step_w)
            pad_right = min(need_w - pad_left, max_step_w)

            if cur_h > 1 and cur_w > 1:
                arr = np.pad(
                    arr,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="reflect",
                )
            else:
                # Tách riêng từng trục để chiều có kích thước > 1 vẫn được reflect chuẩn xác
                if pad_top > 0 or pad_bottom > 0:
                    mode_h = "reflect" if cur_h > 1 else "edge"
                    arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode=mode_h)
                if pad_left > 0 or pad_right > 0:
                    mode_w = "reflect" if cur_w > 1 else "edge"
                    arr = np.pad(arr, ((0, 0), (pad_left, pad_right), (0, 0)), mode=mode_w)

            total_left += pad_left
            total_top += pad_top
            total_right += pad_right
            total_bottom += pad_bottom
            cur_h, cur_w = arr.shape[:2]

        padding_info = (total_left, total_top, total_right, total_bottom)
        return Image.fromarray(arr, mode="RGB"), True, padding_info

    # TẠO DUAL VIEW
    def _create_native_view(
        self, img: Image.Image, rng: Optional[np.random.Generator]
    ) -> Tuple[Image.Image, bool, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Bước 3A: Tạo `view_native` (224x224) cắt trực tiếp từ pixel gốc, không resize trước khi cắt.
        Lưu ý: Nếu `was_padded=True`, `crop_box` nằm trên hệ tọa độ của ảnh sau khi đã đệm gương;
        kết hợp với `padding=(pad_left, pad_top, pad_right, pad_bottom)` để đối chiếu ngược về `view_global`.
        """
        padded_img, was_padded, padding_info = self._reflect_pad_if_needed(img, self.crop_size)
        w, h = padded_img.size

        if self.mode == "train":
            assert rng is not None
            max_x = w - self.crop_size
            max_y = h - self.crop_size
            left = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
            top = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        else:
            # 'eval' và 'extract': luôn dùng CenterCrop tất định (khớp chuẩn torchvision)
            left = int(round((w - self.crop_size) / 2.0))
            top = int(round((h - self.crop_size) / 2.0))

        crop_box = (left, top, left + self.crop_size, top + self.crop_size)
        view_native = padded_img.crop(crop_box)
        return view_native, was_padded, padding_info, crop_box

    def __call__(
        self,
        img_input: Union[str, os.PathLike, Image.Image],
        *,
        force_flip: Optional[bool] = None,
        sample_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Thực thi tiền xử lý tầng chung cho 1 mẫu ảnh.

        Args:
            img_input: Đường dẫn tới file ảnh hoặc đối tượng `PIL.Image.Image`.
            force_flip: Chỉ dùng ở `mode='extract'` (`True` để ép lật ngang, `False`/`None` để giữ nguyên).
            sample_id: Mã định danh mẫu ổn định (ví dụ index hoặc filepath) phục vụ đối khớp đặc trưng.

        Returns:
            Dict chứa:
                - 'view_native'     : PIL.Image (RGB, 224x224, không nội suy resize)
                - 'view_global'     : PIL.Image (RGB, WxH gốc, đã sửa EXIF & đồng bộ trạng thái lật)
                - 'mode'            : str ('train', 'eval', hoặc 'extract')
                - 'is_flipped'      : bool
                - 'was_padded'      : bool
                - 'padding'         : Tuple[int, int, int, int] (pad_left, pad_top, pad_right, pad_bottom)
                - 'jpeg_quality'    : Optional[int] (None nếu không nén JPEG bổ sung)
                - 'orig_size'       : Tuple[int, int] (W, H) sau chuẩn hóa EXIF
                - 'native_crop_box' : Tuple[int, int, int, int] (left, top, right, bottom trên ảnh sau pad)
                - 'sample_id'       : Optional[Union[str, int]]
        """
        if force_flip is not None and self.mode != "extract":
            raise ValueError(
                f"Tham số force_flip chỉ được phép sử dụng khi mode='extract' (hiện tại mode='{self.mode}')."
            )

        # 1. Đọc ảnh 1 lần & chuẩn hóa EXIF + 16-bit + Alpha + RGB
        img = self._load_and_sanitize_rgb(img_input)
        orig_size = img.size

        rng = self._get_rng() if self.mode == "train" else None

        # 2. Đồng bộ phép lật ngang ngay trên ảnh RGB nguồn
        if self.mode == "train":
            assert rng is not None
            is_flipped = bool(rng.random() < self.flip_prob)
        elif self.mode == "extract":
            is_flipped = bool(force_flip) if force_flip is not None else False
        else:
            # mode == 'eval'
            is_flipped = False

        if is_flipped:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 3. Công tắc nén JPEG đồng bộ trên ảnh nguồn (Baseline tắt: jpeg_quality = None)
        applied_jpeg_q: Optional[int] = None
        if self.fixed_jpeg_quality is not None:
            applied_jpeg_q = int(self.fixed_jpeg_quality)
            img = self._apply_jpeg_compression(img, applied_jpeg_q)
        elif self.mode == "train" and self.jpeg_prob > 0.0:
            assert rng is not None
            if rng.random() < self.jpeg_prob:
                q_min, q_max = self.jpeg_quality_range
                applied_jpeg_q = int(rng.integers(q_min, q_max + 1))
                img = self._apply_jpeg_compression(img, applied_jpeg_q)

        # 4A. Sinh `view_native` (224x224 pixel gốc, không resize trước khi cắt)
        view_native, was_padded, padding_info, native_crop_box = self._create_native_view(img, rng)

        # 4B. Sinh `view_global` (giữ nguyên tỷ lệ và kích thước nguồn đã đồng bộ hướng & lật)
        view_global = img

        return {
            "view_native": view_native,
            "view_global": view_global,
            "mode": self.mode,
            "is_flipped": is_flipped,
            "was_padded": was_padded,
            "padding": padding_info,
            "jpeg_quality": applied_jpeg_q,
            "orig_size": orig_size,
            "native_crop_box": native_crop_box,
            "sample_id": sample_id,
        }

    @staticmethod
    def extract_batch_meta(out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tách phần metadata an toàn cho `torch.utils.data.default_collate` trong `Dataset.__getitem__`:
        - Loại bỏ 2 đối tượng `PIL.Image` (`view_native`, `view_global`) sau khi đã đưa qua Bước 4.
        - Đổi các giá trị `None` (`jpeg_quality`, `sample_id`) sang kiểu nguyên thủy (`-1`, `""`).
        - Đóng gói tọa độ/kích thước sang mảng số nguyên `int64` để DataLoader gom thành Tensor [B, K].
        """
        return {
            "mode": str(out["mode"]),
            "is_flipped": bool(out["is_flipped"]),
            "was_padded": bool(out["was_padded"]),
            "padding": np.asarray(out["padding"], dtype=np.int64),
            "jpeg_quality": -1 if out["jpeg_quality"] is None else int(out["jpeg_quality"]),
            "orig_size": np.asarray(out["orig_size"], dtype=np.int64),
            "native_crop_box": np.asarray(out["native_crop_box"], dtype=np.int64),
            "sample_id": "" if out["sample_id"] is None else str(out["sample_id"]),
        }


# Alias tiện dụng để import đồng nhất giữa các module
SharedDualViewPreprocessor = CommonProcess


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP NGHIỆM THU TOÀN DIỆN BƯỚC 3 (CHẠY TRÊN MÁY CÁ NHÂN)
# ==============================================================================
DATASET_ROOT = Path(r"D:\nam_4\DATN\dataset\Defactify_dataset")


if HAS_TORCH:
    class _PreprocessorAuditDataset(torch.utils.data.Dataset):
        """
        Dataset mức module để tương thích cơ chế spawn pickling trên Windows
        khi kiểm thử thực tế với PyTorch DataLoader(num_workers > 0).
        """

        def __init__(self, image_paths, preprocessor: CommonProcess):
            self.image_paths = [str(p) for p in image_paths]
            self.preprocessor = preprocessor

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int):
            out = self.preprocessor(self.image_paths[idx], sample_id=idx)
            l, t, r, b = out["native_crop_box"]
            return torch.tensor(
                [int(out["is_flipped"]), l, t, r, b, self.preprocessor.epoch],
                dtype=torch.int64,
            )


def run_unit_tests():
    """
    Kiểm chứng độc lập toàn bộ các tiêu chuẩn nghiệm thu của Bước 3 trên máy cá nhân:
    1. Chạy thực tế trên ảnh đại diện của MS_COCO, DALL-E_3, SD_2.1, SDXL; mở file đúng 1 lần.
    2. Bảo toàn 100% giá trị pixel gốc trong `view_native` (không bị nội suy resize) & giữ nguyên tỷ lệ `view_global`.
    3. Chứng minh toán học đối xứng gương từng pixel của Reflect Padding (4 cạnh + ảnh 1x100 + ảnh COCO < 224px thực tế + truy vết `padding`).
    4. Kiểm tra chuẩn hóa hướng EXIF (Orientation = 6), kênh Alpha (RGBA), và ảnh scan 16-bit (`I;16`).
    5. Kiểm tra `mode='train'` trên cả ảnh vuông SDXL (1024x1024) và chữ nhật MS_COCO (640x480) + `DataLoader` thật (`num_workers=0` & `num_workers=2`) qua nhiều Epoch.
    6. Kiểm tra tính bất biến 100% của cả `view_native` & `view_global` ở `mode='eval'` + chặn xung đột cấu hình JPEG / `mode`.
    7. Kiểm tra `mode='extract'` (`force_flip=False/True`) trên cả ảnh vuông SDXL và ảnh chữ nhật MS_COCO.
    """
    import time
    from unittest.mock import patch
    import pandas as pd

    metadata_path = DATASET_ROOT / "metadata.csv"

    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ NGHIỆM THU BƯỚC 3: pre_processing_for_all.py (CommonProcess)")
    print(f"Thư mục dữ liệu: {DATASET_ROOT}")
    print("=" * 80)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata tại: {metadata_path}")

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # TEST 1 & 2: Kiểm tra trên 4 nguồn ảnh thực tế + Đọc file 1 lần + Không nội suy pixel gốc
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 1 & 2] Thử nghiệm trên ảnh thật (MS_COCO, DALL-E_3, SD_2.1, SDXL):")
    eval_proc = CommonProcess(mode="eval", crop_size=224)
    target_models = ["MS_COCO", "DALL-E_3", "Stable_Diffusion_2.1", "Stable_Diffusion_XL"]
    chosen_paths: Dict[str, Path] = {}

    for m_name in target_models:
        candidates = train_df[train_df["model_name"] == m_name]
        chosen_rel = None
        for rel_p in candidates["filepath"].iloc[:30]:
            full_p = DATASET_ROOT / rel_p
            with Image.open(full_p) as tmp_im:
                w0, h0 = tmp_im.size
                if min(w0, h0) >= 224 and (m_name != "MS_COCO" or w0 != h0):
                    chosen_rel = rel_p
                    break
        assert chosen_rel is not None, f"Không tìm thấy ảnh mẫu cho {m_name}"
        full_img_path = DATASET_ROOT / chosen_rel
        chosen_paths[m_name] = full_img_path

        real_open = Image.open
        open_count = 0

        def counting_open(*args, **kwargs):
            nonlocal open_count
            open_count += 1
            return real_open(*args, **kwargs)

        with patch("PIL.Image.open", side_effect=counting_open):
            out = eval_proc(full_img_path, sample_id=chosen_rel)

        assert open_count == 1, f"Ảnh bị mở {open_count} lần (yêu cầu đúng 1 lần)!"
        assert out["mode"] == "eval"
        assert out["view_native"].size == (224, 224), "view_native sai kích thước 224x224!"
        assert out["view_global"].size == out["orig_size"], "view_global bị thay đổi kích thước gốc!"
        assert out["is_flipped"] is False and out["was_padded"] is False
        assert out["padding"] == (0, 0, 0, 0)
        assert out["jpeg_quality"] is None

        box = out["native_crop_box"]
        expected_patch = np.asarray(out["view_global"].crop(box))
        actual_patch = np.asarray(out["view_native"])
        assert np.array_equal(expected_patch, actual_patch), (
            f"Pixel của view_native bị sai lệch trên {m_name}!"
        )
        print(
            f"  + {m_name:<22} | Gốc: {str(out['orig_size']):<12} "
            f"| Native: {out['view_native'].size} | Mở file: {open_count} lần | Pixel gốc khớp 100% => [PASS]"
        )

    # -------------------------------------------------------------------------
    # TEST 3: Chứng minh toán học đối xứng gương Reflect Padding + truy vết `padding`
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 3] Kiểm tra toán học đối xứng gương (Reflect Padding) & truy vết tọa độ:")
    synth_orig = np.arange(180 * 200 * 3, dtype=np.uint32).reshape(180, 200, 3) % 251
    synth_orig = synth_orig.astype(np.uint8)
    synth_out = eval_proc(Image.fromarray(synth_orig, mode="RGB"))
    pad_l, pad_t, pad_r, pad_b = synth_out["padding"]
    assert synth_out["was_padded"] is True
    assert (pad_l, pad_t, pad_r, pad_b) == (12, 22, 12, 22), f"Sai thông số padding: {synth_out['padding']}"
    vn_synth = np.asarray(synth_out["view_native"])
    assert np.array_equal(vn_synth[pad_t : pad_t + 180, pad_l : pad_l + 200], synth_orig)
    assert np.array_equal(vn_synth[:pad_t, pad_l : pad_l + 200], synth_orig[pad_t:0:-1, :])
    assert np.array_equal(vn_synth[pad_t + 180 :, pad_l : pad_l + 200], synth_orig[-2 : -2 - pad_b : -1, :])
    assert np.array_equal(vn_synth[pad_t : pad_t + 180, :pad_l], synth_orig[:, pad_l:0:-1])
    assert np.array_equal(vn_synth[pad_t : pad_t + 180, pad_l + 200 :], synth_orig[:, -2 : -2 - pad_r : -1])

    line_arr = np.arange(100, dtype=np.uint8).reshape(1, 100, 1).repeat(3, axis=2)
    padded_line, line_was_padded, line_pad = CommonProcess._reflect_pad_if_needed(
        Image.fromarray(line_arr, mode="RGB"), 120
    )
    pl, pt, pr, pb = line_pad
    padded_line_arr = np.asarray(padded_line)
    assert line_was_padded is True and (pl, pr) == (10, 10)
    assert np.array_equal(padded_line_arr[0, :pl, 0], line_arr[0, pl:0:-1, 0]), "Chiều W>1 phải được reflect!"

    coco_rows = df[df["model_name"] == "MS_COCO"]
    small_real_path = None
    small_size = None
    for rel_p in coco_rows["filepath"]:
        fp = DATASET_ROOT / rel_p
        with Image.open(fp) as im:
            if min(im.size) < 224:
                small_real_path = fp
                small_size = im.size
                break

    assert small_real_path is not None, "Không tìm thấy ảnh COCO < 224px để test!"
    small_out = eval_proc(small_real_path)
    sl, st, sr, sb = small_out["padding"]
    cl, ct, cr, cb = small_out["native_crop_box"]
    assert small_out["was_padded"] is True
    assert small_out["view_native"].size == (224, 224)
    assert small_out["view_global"].size == small_size

    vg_small = np.asarray(small_out["view_global"])
    vn_small = np.asarray(small_out["view_native"])
    orig_w, orig_h = small_size
    inter_x0 = max(cl, sl)
    inter_x1 = min(cr, sl + orig_w)
    inter_y0 = max(ct, st)
    inter_y1 = min(cb, st + orig_h)
    assert np.array_equal(
        vn_small[inter_y0 - ct : inter_y1 - ct, inter_x0 - cl : inter_x1 - cl],
        vg_small[inter_y0 - st : inter_y1 - st, inter_x0 - sl : inter_x1 - sl],
    ), "Truy vết tọa độ qua `padding` và `native_crop_box` không khớp với `view_global`!"

    tiny_arr = np.random.default_rng(0).integers(20, 230, size=(80, 64, 3), dtype=np.uint8)
    tiny_out = eval_proc(Image.fromarray(tiny_arr, mode="RGB"))
    assert tiny_out["was_padded"] is True and tiny_out["view_native"].size == (224, 224)
    assert np.asarray(tiny_out["view_native"]).min() >= 20
    print(
        f"  + Đối xứng gương 4 cạnh khớp 100% | Ảnh COCO nhỏ {small_size} padding={small_out['padding']} "
        f"truy vết khớp `view_global` => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 4: Kiểm tra EXIF Orientation, Kênh Alpha (RGBA) và Ảnh Scan 16-bit ('I;16')
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 4] Kiểm tra hướng EXIF, trộn nền Alpha (RGBA) và ảnh 16-bit ('I;16'):")
    raw_rect = Image.new("RGB", (300, 240), color=(60, 120, 180))
    exif = raw_rect.getexif()
    exif[274] = 6  # Rotate 90 CW
    exif_buf = io.BytesIO()
    raw_rect.save(exif_buf, format="JPEG", exif=exif)
    exif_buf.seek(0)
    with Image.open(exif_buf) as exif_loaded:
        exif_loaded.load()
        exif_out = eval_proc(exif_loaded)
    assert exif_out["orig_size"] == (240, 300)

    rgba_arr = np.zeros((250, 250, 4), dtype=np.uint8)
    rgba_arr[..., 0] = 255
    rgba_arr[..., 3] = 0
    rgba_img = Image.fromarray(rgba_arr, mode="RGBA")
    alpha_out = eval_proc(rgba_img)
    assert np.all(np.asarray(alpha_out["view_native"]) == 128)

    grad16 = np.linspace(0, 65535, 250 * 250, dtype=np.uint16).reshape(250, 250)
    out16 = eval_proc(Image.fromarray(grad16))
    assert len(np.unique(np.asarray(out16["view_global"]))) > 200
    print("  + EXIF xoay chuẩn (240, 300) | RGBA sạch nền | Ảnh 16-bit bảo toàn >200 mức xám => [PASS]")

    # -------------------------------------------------------------------------
    # TEST 5: Kiểm tra 'train' trên cả SDXL vuông & MS_COCO chữ nhật + DataLoader thực tế
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 5] Kiểm tra 'train' (SDXL vuông + MS_COCO chữ nhật + DataLoader đa worker/epoch):")
    sample_sdxl_path = chosen_paths["Stable_Diffusion_XL"]
    sample_coco_path = chosen_paths["MS_COCO"]

    train_proc = CommonProcess(mode="train", crop_size=224, flip_prob=0.5, seed=123)
    for tag, spath in [("SDXL_square", sample_sdxl_path), ("COCO_rect", sample_coco_path)]:
        with Image.open(spath) as orig_im:
            orig_rgb = ImageOps.exif_transpose(orig_im).convert("RGB")
            orig_arr = np.asarray(orig_rgb)
            mirrored_arr = np.asarray(orig_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

        for _ in range(12):
            res = train_proc(spath)
            vg_arr = np.asarray(res["view_global"])
            vn_arr = np.asarray(res["view_native"])
            l, t, r, b = res["native_crop_box"]
            if res["is_flipped"]:
                assert np.array_equal(vg_arr, mirrored_arr), f"Sai view_global khi lật trên {tag}"
                assert np.array_equal(vn_arr, mirrored_arr[t:b, l:r]), f"Sai view_native khi lật trên {tag}"
            else:
                assert np.array_equal(vg_arr, orig_arr), f"Sai view_global khi giữ nguyên trên {tag}"
                assert np.array_equal(vn_arr, orig_arr[t:b, l:r]), f"Sai view_native khi giữ nguyên trên {tag}"

    if HAS_TORCH:
        from torch.utils.data import DataLoader

        audit_paths = [sample_sdxl_path, sample_coco_path] * 6

        # 5A. num_workers = 0
        proc_nw0 = CommonProcess(mode="train", crop_size=224, flip_prob=0.5, seed=2026)
        ds_nw0 = _PreprocessorAuditDataset(audit_paths, proc_nw0)
        loader_nw0 = DataLoader(ds_nw0, batch_size=4, shuffle=False, num_workers=0)
        proc_nw0.set_epoch(0)
        ep0_nw0 = torch.cat(list(loader_nw0), dim=0)
        proc_nw0.set_epoch(1)
        ep1_nw0 = torch.cat(list(loader_nw0), dim=0)
        assert not torch.equal(ep0_nw0, ep1_nw0), "DataLoader num_workers=0 bị trùng trạng thái qua Epoch!"

        # 5B. num_workers = 2 (kiểm chứng cả persistent_workers=True đồng bộ set_epoch qua shared memory)
        def _collect_two_epochs_mw(gen_seed: int, persistent: bool):
            g = torch.Generator()
            g.manual_seed(gen_seed)
            proc_mw = CommonProcess(mode="train", crop_size=224, flip_prob=0.5, seed=2026)
            ds_mw = _PreprocessorAuditDataset(audit_paths, proc_mw)
            loader_mw = DataLoader(
                ds_mw,
                batch_size=3,
                shuffle=False,
                num_workers=2,
                persistent_workers=persistent,
                generator=g,
            )
            proc_mw.set_epoch(0)
            e0 = torch.cat(list(loader_mw), dim=0)
            proc_mw.set_epoch(1)
            e1 = torch.cat(list(loader_mw), dim=0)
            return e0, e1

        mw_ep0_a, mw_ep1_a = _collect_two_epochs_mw(999, persistent=False)
        mw_ep0_b, mw_ep1_b = _collect_two_epochs_mw(999, persistent=False)
        assert not torch.equal(mw_ep0_a, mw_ep1_a), (
            "DataLoader num_workers=2 bị lặp lại cùng vị trí crop/flip giữa Epoch 0 và Epoch 1!"
        )
        assert torch.equal(mw_ep0_a, mw_ep0_b) and torch.equal(mw_ep1_a, mw_ep1_b), (
            "DataLoader num_workers=2 không tái lập được kết quả khi cố định generator!"
        )

        pw_ep0, pw_ep1 = _collect_two_epochs_mw(999, persistent=True)
        assert torch.all(pw_ep0[:, 5] == 0) and torch.all(pw_ep1[:, 5] == 1), (
            "persistent_workers=True không đồng bộ được set_epoch() sang các worker đang sống!"
        )
        assert not torch.equal(pw_ep0[:, :5], pw_ep1[:, :5]), (
            "persistent_workers=True bị trùng trạng thái crop/flip qua Epoch!"
        )
    print(
        "  + SDXL vuông & MS_COCO chữ nhật đồng bộ 100% | DataLoader thật (num_workers=0 & 2, "
        "persistent=False/True) đồng bộ set_epoch chuẩn xác => [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 6 & 7: 'eval' bất biến, chặn xung đột JPEG, extract_batch_meta & 'extract'
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 6 & 7] Kiểm tra 'eval', chặn JPEG khi train, default_collate & 'extract':")
    for bad_kwargs in [
        {"mode": "extract", "is_train": False},
        {"mode": "train", "fixed_jpeg_quality": 85, "jpeg_prob": 0.0},
        {"mode": "train", "fixed_jpeg_quality": 85, "jpeg_prob": 0.5},
        {"mode": "eval", "jpeg_prob": 0.3},
    ]:
        try:
            CommonProcess(**bad_kwargs)
            raise AssertionError(f"Phải báo lỗi với cấu hình xung đột: {bad_kwargs}")
        except ValueError:
            pass

    if HAS_TORCH:
        from torch.utils.data._utils.collate import default_collate

        raw_m1 = eval_proc(sample_coco_path)  # jpeg_quality=None, sample_id=None
        raw_m2 = eval_proc(sample_sdxl_path, sample_id="sdxl_01")
        collated_meta = default_collate(
            [CommonProcess.extract_batch_meta(raw_m1), CommonProcess.extract_batch_meta(raw_m2)]
        )
        assert collated_meta["orig_size"].shape == (2, 2)
        assert collated_meta["padding"].shape == (2, 4)
        assert collated_meta["native_crop_box"].shape == (2, 4)
        assert collated_meta["jpeg_quality"].tolist() == [-1, -1]
        assert collated_meta["sample_id"] == ["", "sdxl_01"]

    eval_ref_out = eval_proc(sample_coco_path)
    eval_ref_native = np.asarray(eval_ref_out["view_native"])
    eval_ref_global = np.asarray(eval_ref_out["view_global"])
    for _ in range(5):
        cur_eval = eval_proc(sample_coco_path)
        assert np.array_equal(eval_ref_native, np.asarray(cur_eval["view_native"]))
        assert np.array_equal(eval_ref_global, np.asarray(cur_eval["view_global"]))

    ext_proc = CommonProcess(mode="extract", crop_size=224)
    for tag, spath in [("SDXL_square", sample_sdxl_path), ("COCO_rect", sample_coco_path)]:
        ext_no_flip = ext_proc(spath, force_flip=False, sample_id=tag)
        ext_flipped = ext_proc(spath, force_flip=True, sample_id=tag)

        assert ext_no_flip["mode"] == "extract"
        assert ext_no_flip["is_flipped"] is False and ext_flipped["is_flipped"] is True
        assert ext_no_flip["sample_id"] == tag
        vg_nf = np.asarray(ext_no_flip["view_global"])
        vg_fl = np.asarray(ext_flipped["view_global"])
        assert np.array_equal(vg_nf[:, ::-1, :], vg_fl), f"Sai đối xứng view_global trên {tag}!"
        fl_box = ext_flipped["native_crop_box"]
        assert np.array_equal(
            np.asarray(ext_flipped["view_native"]),
            vg_fl[fl_box[1] : fl_box[3], fl_box[0] : fl_box[2]],
        ), f"Sai CenterCrop của view_native khi force_flip=True trên {tag}!"

    t0 = time.perf_counter()
    for _ in range(50):
        eval_proc(sample_sdxl_path)
    avg_ms = (time.perf_counter() - t0) * 1000 / 50
    print(
        f"  + Chặn xung đột JPEG/mode OK | 'eval' bất biến cả 2 view | 'extract' chuẩn trên cả COCO chữ nhật | "
        f"Tốc độ TB: {avg_ms:.2f} ms/ảnh => [PASS]"
    )

    print("\n" + "=" * 80)
    print("TẤT CẢ TIÊU CHUẨN NGHIỆM THU BƯỚC 3 ĐÃ CHẶT CHẼ TUYỆT ĐỐI VÀ VƯỢT QUA 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()