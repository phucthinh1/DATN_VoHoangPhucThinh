"""
Pipeline Trích Xuất và Lưu Đệm Đặc Trưng (Feature Caching) — Bước 5B / Giai Đoạn 2.
=====================================================================================

Mục tiêu:
---------
Chạy một lượt (Single-Pass Multi-Branch Extraction) qua các tập dữ liệu tuần tự
(`train`, `val`, `test`) bằng `build_extract_loader` (tắt `BalancedGroupSampler`)
để trích xuất vector đặc trưng từ 3 nhánh đã khóa cứng thân mạng (`.eval()` +
`torch.inference_mode()`):
  1. `spatial`  (EfficientNet-B0): `f_spatial`  in R^{1280} — nạp từ `best_spatial.pth`
  2. `semantic` (CLIP ViT-B/16)  : `f_semantic` in R^{512}  — nạp từ OpenAI Pretrained (L2-norm = 1)
  3. `spectral` (FFT + ResNet18) : `f_spectral` in R^{256}  — nạp từ `best_spectral.pth`

Đặc điểm kiến trúc then chốt (theo `Chot_Cau_Hinh_Buoc_5B_3_Nhanh.md`):
-----------------------------------------------------------------------
1. **Đồng bộ hóa Horizontal Flip trên tập Train (Mục 3, 5, 8.1 #6, #7)**:
   - Với `split == "train"`, trích xuất cả 2 góc nhìn trên từng mẫu `i`:
     * `features`      (`force_flip=False`): Đặc trưng ảnh gốc `(N, D)`
     * `features_flip` (`force_flip=True`) : Đặc trưng ảnh lật ngang `(N, D)`
   - Nhờ đó ở Giai đoạn 2 (`train_fusion.py`), `CachedFeatureDataset` có thể tung đồng xu
     `p = 0.5` cho mỗi mẫu `i` để chọn đồng bộ góc nhìn gốc hoặc lật ngang cho cả 3 nhánh
     mà không phải chạy lại backbone nặng nề.
2. **Bảo toàn 100% căn chỉnh mẫu giữa 3 nhánh (Mục 8.1 #5)**:
   - Mỗi file `.pt` lưu đầy đủ: `features`, `features_flip`, `sample_idx`, `filepaths`,
     `label_a`, `label_b`, `group_id`, `model_names`, `manifest_hash`, `jpeg_quality`
     và `checkpoint_info`.
   - Hàm `verify_cross_branch_alignment` đối soát nghiêm ngặt `sample_idx == 0..N-1`,
     `filepaths`, `label_a`, `label_b`, `group_id` và `manifest_hash` khớp 100% giữa
     `spatial`, `semantic` và `spectral`.
3. **Tối ưu I/O một lượt đọc ảnh (Single-Pass Multi-Branch)**:
   - Khi trích xuất đồng thời nhiều nhánh (`--branches spatial semantic spectral`),
     `build_extract_loader` chỉ đọc và giải mã mỗi ảnh từ ổ cứng đúng 1 lần trong
     `CommonProcess`, sau đó phân phối cả 3 tensor đầu vào cho 3 nhánh trên GPU.
4. **Hỗ trợ kiểm thử độ bền nén JPEG cố định trên tập Test (Mục 8.1 #14)**:
   - Tham số `--jpeg-qualities` (ví dụ `90,70,50,30`) cho phép trích xuất thêm các bộ
     cache `test` ở từng mức nén JPEG cố định (`{branch}_test_jpeg{q}.pt`) phục vụ đánh giá
     độ bền cuối kỳ.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo console Windows hiển thị tiếng Việt UTF-8 không lỗi font
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# THIẾT LẬP ĐƯỜNG DẪN NỘI BỘ (CROSS-DIRECTORY PATH RESOLVER)
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = CURRENT_DIR.parent
BRANCH_DIR = SYSTEM_DIR / "branch"
DATA_DIR = SYSTEM_DIR / "prepare_data"
REPO_ROOT = SYSTEM_DIR.parent.parent

for _p in (CURRENT_DIR, SYSTEM_DIR, BRANCH_DIR, DATA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from base import BranchBase  # noqa: E402
from CLIP import SemanticBranch  # noqa: E402
from dataset import DEFAULT_DATASET_ROOT, build_extract_loader  # noqa: E402
from EfficientNet import SpatialBranch  # noqa: E402
from engine import get_optimal_amp_config, load_checkpoint  # noqa: E402
from FFT import SpectralBranch  # noqa: E402
from metrics import (  # noqa: E402
    ALL_AI_GENERATORS,
    KNOWN_GENERATORS,
    REAL_MODEL_NAME,
    UNSEEN_GENERATORS,
)
from split_dataset import get_split_dataframes  # noqa: E402


# ==============================================================================
# HẰNG SỐ CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
VALID_BRANCHES: Tuple[str, ...] = ("spatial", "semantic", "spectral")
VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")

BRANCH_FEATURE_DIMS: Dict[str, int] = {
    "spatial": 1280,
    "semantic": 512,
    "spectral": 256,
}

# Quy định nguồn dữ liệu hợp lệ theo từng split (chặn rò rỉ Unseen vào train/val)
ALLOWED_MODELS_BY_SPLIT: Dict[str, frozenset] = {
    "train": frozenset({REAL_MODEL_NAME, *KNOWN_GENERATORS}),
    "val": frozenset({REAL_MODEL_NAME, *KNOWN_GENERATORS}),
    "test": frozenset({REAL_MODEL_NAME, *ALL_AI_GENERATORS}),
}

# Bảng ánh xạ bắt buộc giữa tên nguồn sinh (model_name) và nhãn chi tiết label_b (0..5)
MODEL_NAME_TO_LABEL_B: Dict[str, int] = {
    "MS_COCO": 0,
    "Stable_Diffusion_2.1": 1,
    "Stable_Diffusion_XL": 2,
    "Stable_Diffusion_3": 3,
    "DALL-E_3": 4,
    "Midjourney_V6": 5,
}

# Hồ sơ cấu hình tiền xử lý chuẩn của từng nhánh khi chạy ở mode='extract'
BRANCH_PREPROCESS_CONFIG: Dict[str, Dict[str, Any]] = {
    "spatial": {
        "view_mode": "native",
        "crop_size": 224,
        "resize": False,
        "normalization": "imagenet",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "semantic": {
        "view_mode": "global",
        "image_size": 224,
        "resize": "shortest_edge_bicubic_224",
        "center_crop": 224,
        "normalization": "openai_clip",
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
    },
    "spectral": {
        "view_mode": "native",
        "crop_size": 224,
        "fft_mode": "raw",
        "use_hann": False,
        "remove_dc": True,
        "normalization": "instance_zscore",
        "coord_conv": True,
    },
}

DEFAULT_CACHE_DIR: Path = SYSTEM_DIR / "checkpoints" / "feature_cache"
DEFAULT_SPATIAL_CKPT: Path = SYSTEM_DIR / "checkpoints" / "spatial_baseline" / "best_spatial.pth"
DEFAULT_SPECTRAL_CKPT: Path = SYSTEM_DIR / "checkpoints" / "spectral_baseline" / "best_spectral.pth"

DEFAULT_CLIP_MODEL_NAME: str = "ViT-B-16"
DEFAULT_CLIP_PRETRAINED: str = "openai"

REQUIRED_CACHE_KEYS: Tuple[str, ...] = (
    "branch",
    "split",
    "feature_dim",
    "num_samples",
    "features",
    "features_flip",
    "has_flip",
    "sample_idx",
    "filepaths",
    "label_a",
    "label_b",
    "group_id",
    "model_names",
    "manifest_hash",
    "jpeg_quality",
    "checkpoint_info",
    "preprocess_config",
    "extraction_signature",
    "created_at",
)


# ==============================================================================
# TIỆN ÍCH ĐƯỜNG DẪN, MANIFEST HASH & DỌN DẸP DATALOADER
# ==============================================================================
def _shutdown_dataloader_workers(loader: Any) -> None:
    """Đóng tường minh các tiến trình worker của DataLoader (an toàn trên Windows)."""
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        try:
            iterator._shutdown_workers()
        except Exception:
            pass
    loader._iterator = None


def get_cache_filepath(
    cache_dir: Union[str, Path],
    branch: str,
    split: str,
    jpeg_quality: Optional[int] = None,
) -> Path:
    """
    Trả về đường dẫn chuẩn hóa của file `.pt` lưu đệm đặc trưng.

    Quy ước đặt tên:
      - Ảnh chuẩn (`jpeg_quality=None`) : `{branch}_{split}.pt`
        (VD: `spatial_train.pt`, `semantic_val.pt`, `spectral_test.pt`)
      - Ảnh nén JPEG cố định (`jpeg_quality=q`): `{branch}_{split}_jpeg{q}.pt`
        (VD: `semantic_test_jpeg70.pt`)
    """
    branch_norm = str(branch).strip().lower()
    split_norm = str(split).strip().lower()

    if branch_norm not in VALID_BRANCHES:
        raise ValueError(
            f"[extract_features] branch='{branch}' không hợp lệ. Chọn trong {VALID_BRANCHES}."
        )
    if split_norm not in VALID_SPLITS:
        raise ValueError(
            f"[extract_features] split='{split}' không hợp lệ. Chọn trong {VALID_SPLITS}."
        )

    cache_root = Path(cache_dir)
    if jpeg_quality is None:
        filename = f"{branch_norm}_{split_norm}.pt"
    else:
        q = int(jpeg_quality)
        if not (1 <= q <= 100):
            raise ValueError(
                f"[extract_features] jpeg_quality phải nằm trong [1, 100], nhận được: {jpeg_quality}"
            )
        filename = f"{branch_norm}_{split_norm}_jpeg{q}.pt"

    return cache_root / filename


def compute_manifest_hash(
    filepaths: Sequence[str],
    labels_a: Union[Sequence[float], torch.Tensor],
    model_names: Sequence[str],
    sample_indices: Optional[Union[Sequence[int], torch.Tensor]] = None,
    labels_b: Optional[Union[Sequence[int], torch.Tensor]] = None,
    group_ids: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> str:
    """
    Tính mã băm SHA-256 đầy đủ (64 ký tự hex) đại diện cho toàn bộ bảng thông tin mẫu theo đúng thứ tự.

    Khi truyền đủ 6 trường (`sample_idx`, `filepath`, `label_a`, `label_b`, `group_id`, `model_name`),
    mọi sai lệch dù nhỏ ở thứ tự dòng, đường dẫn, nhãn nhị phân, mã generator hay nhóm ảnh gốc
    đều làm thay đổi `manifest_hash` ngay lập tức.
    """
    n_samples = len(filepaths)

    def _to_int_list(vals: Union[Sequence[Any], torch.Tensor], is_float_label: bool = False) -> List[int]:
        raw = vals.detach().cpu().tolist() if isinstance(vals, torch.Tensor) else list(vals)
        if is_float_label:
            return [int(round(float(v))) for v in raw]
        return [int(v) for v in raw]

    labels_a_list = _to_int_list(labels_a, is_float_label=True)
    idx_list = (
        list(range(n_samples))
        if sample_indices is None
        else _to_int_list(sample_indices)
    )
    labels_b_list = (
        [MODEL_NAME_TO_LABEL_B.get(str(m), -1) for m in model_names]
        if labels_b is None
        else _to_int_list(labels_b)
    )
    group_id_list = (
        list(range(n_samples))
        if group_ids is None
        else _to_int_list(group_ids)
    )

    if not (
        n_samples
        == len(labels_a_list)
        == len(model_names)
        == len(idx_list)
        == len(labels_b_list)
        == len(group_id_list)
    ):
        raise ValueError(
            f"[extract_features] Độ dài không khớp khi tính manifest_hash: "
            f"filepaths={n_samples}, labels_a={len(labels_a_list)}, model_names={len(model_names)}, "
            f"sample_indices={len(idx_list)}, labels_b={len(labels_b_list)}, group_ids={len(group_id_list)}"
        )

    hasher = hashlib.sha256()
    for s_idx, fp, la, lb, gid, mname in zip(
        idx_list, filepaths, labels_a_list, labels_b_list, group_id_list, model_names
    ):
        norm_fp = str(fp).replace("\\", "/")
        row_str = f"{s_idx}|{norm_fp}|{la}|{lb}|{gid}|{mname}\n"
        hasher.update(row_str.encode("utf-8"))
    return hasher.hexdigest()


def _compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Tính SHA-256 (64 ký tự hex) của file checkpoint để nhận diện chính xác phiên bản trọng số."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"[_compute_file_sha256] Không tìm thấy file: '{path}'.")
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_json_dict(data: Mapping[str, Any]) -> str:
    """Tính SHA-256 (64 ký tự hex) ổn định từ dict cấu hình đã sắp xếp khóa."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_preprocess_config(
    branch: str,
    include_flip: bool,
    fixed_jpeg_quality: Optional[int],
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    """Tạo từ điển `preprocess_config` chuẩn cho 1 nhánh ở chế độ trích xuất."""
    cfg = copy.deepcopy(BRANCH_PREPROCESS_CONFIG[branch])
    cfg.update(
        {
            "mode": "extract",
            "has_flip": bool(include_flip),
            "fixed_jpeg_quality": int(fixed_jpeg_quality) if fixed_jpeg_quality is not None else None,
            "use_amp": bool(use_amp),
            "amp_dtype": str(amp_dtype) if use_amp else "torch.float32",
        }
    )
    return cfg


def build_extraction_signature(
    branch: str,
    split: str,
    checkpoint_info: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
    include_flip: bool,
    fixed_jpeg_quality: Optional[int],
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Tạo chữ ký cấu hình (`extraction_signature`) dùng để kiểm tra tính tương thích
    khi cân nhắc bỏ qua (`[SKIP]`) các file cache `.pt` đã tồn tại trên đĩa.
    """
    return {
        "branch": str(branch).strip().lower(),
        "split": str(split).strip().lower(),
        "source": checkpoint_info.get("source"),
        "checkpoint_sha256": checkpoint_info.get("checkpoint_sha256"),
        "clip_model_name": checkpoint_info.get("model_name"),
        "clip_pretrained": checkpoint_info.get("pretrained"),
        "clip_quick_gelu": checkpoint_info.get("quick_gelu"),
        "clip_l2_normalize": checkpoint_info.get("l2_normalize"),
        "open_clip_version": checkpoint_info.get("open_clip_version"),
        "preprocess_config_hash": _hash_json_dict(preprocess_cfg),
        "has_flip": bool(include_flip),
        "jpeg_quality": int(fixed_jpeg_quality) if fixed_jpeg_quality is not None else None,
        "use_amp": bool(use_amp),
        "amp_dtype": str(amp_dtype) if use_amp else "torch.float32",
    }


# ==============================================================================
# KHỞI TẠO & NẠP TRỌNG SỐ CÁC NHÁNH TRÍCH XUẤT (BRANCH EXTRACTOR LOADER)
# ==============================================================================
def _normalize_clip_pretrained_arg(
    clip_pretrained: Optional[Union[str, bool]],
    allow_untrained: bool,
) -> Optional[Union[str, bool]]:
    """Chuẩn hóa tham số `clip_pretrained` và chặn giá trị rỗng/None khi trích xuất chính thức."""
    is_empty_pretrained = (
        clip_pretrained is None
        or clip_pretrained is False
        or (
            isinstance(clip_pretrained, str)
            and clip_pretrained.strip().lower() in ("", "none", "false", "null")
        )
    )
    resolved: Optional[Union[str, bool]] = None if is_empty_pretrained else clip_pretrained
    if resolved is None and not allow_untrained:
        raise ValueError(
            "[load_branch_extractor] Không được bỏ trống `clip_pretrained` (hoặc truyền 'none') "
            "khi trích xuất chính thức nhánh Semantic. Hãy dùng OpenAI pretrained (`--clip-pretrained openai`)."
        )
    return resolved


def _get_open_clip_runtime_meta(
    clip_model_name: str,
    resolved_clip_pretrained: Optional[Union[str, bool]],
) -> Dict[str, Any]:
    """Đọc phiên bản `open_clip` và cờ `quick_gelu` mà không cần nạp nặng trọng số ViT."""
    open_clip_mod = SemanticBranch._import_open_clip()
    pretrained_arg = SemanticBranch._normalize_pretrained_arg(resolved_clip_pretrained)
    quick_gelu = SemanticBranch._resolve_quick_gelu_flag(
        model_name=clip_model_name,
        pretrained=pretrained_arg,
        force_quick_gelu=None,
    )
    return {
        "source": "open_clip_pretrained" if pretrained_arg is not None else "untrained_fallback",
        "model_name": str(clip_model_name).strip(),
        "pretrained": str(pretrained_arg),
        "quick_gelu": bool(quick_gelu),
        "l2_normalize": True,
        "open_clip_version": str(getattr(open_clip_mod, "__version__", "unknown")),
    }


def resolve_expected_checkpoint_info(
    branch: str,
    spatial_ckpt: Union[str, Path] = DEFAULT_SPATIAL_CKPT,
    spectral_ckpt: Union[str, Path] = DEFAULT_SPECTRAL_CKPT,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
    clip_pretrained: Optional[Union[str, bool]] = DEFAULT_CLIP_PRETRAINED,
    allow_untrained: bool = False,
) -> Dict[str, Any]:
    """
    Tính dấu vân tay cấu hình kỳ vọng (`checkpoint_sha256` / CLIP config) ở mức nhẹ trên CPU
    để kiểm chứng cache cũ trước khi tốn thời gian nạp mô hình lên GPU.
    """
    br = str(branch).strip().lower()
    if br in ("spatial", "spectral"):
        ckpt_path = Path(spatial_ckpt if br == "spatial" else spectral_ckpt)
        if ckpt_path.is_file():
            return {
                "source": "checkpoint",
                "checkpoint_path": str(ckpt_path.resolve()).replace("\\", "/"),
                "checkpoint_sha256": _compute_file_sha256(ckpt_path),
            }
        if allow_untrained:
            return {
                "source": "untrained_fallback",
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": None,
            }
        raise FileNotFoundError(
            f"[resolve_expected_checkpoint_info] Không tìm thấy checkpoint nhánh '{br}' tại: '{ckpt_path}'."
        )

    resolved_clip = _normalize_clip_pretrained_arg(clip_pretrained, allow_untrained=allow_untrained)
    return _get_open_clip_runtime_meta(clip_model_name, resolved_clip)


def _verify_dataset_preprocess_alignment(dataset: Any, active_branches: Sequence[str]) -> None:
    """
    Đối chiếu trực tiếp cấu hình tiền xử lý thực tế trong `DefactifyDataset`
    với hằng số mô tả `BRANCH_PREPROCESS_CONFIG` để không bao giờ lệch ngầm.
    """
    crop_size = getattr(dataset, "crop_size", None)
    if crop_size != 224:
        raise RuntimeError(
            f"[_verify_dataset_preprocess_alignment] crop_size={crop_size} != 224!"
        )
    if "spatial" in active_branches and getattr(dataset, "spatial_proc", None) is not None:
        sp_proc = dataset.spatial_proc
        exp_view = BRANCH_PREPROCESS_CONFIG["spatial"]["view_mode"]
        if getattr(sp_proc, "view_mode", None) != exp_view or not bool(getattr(sp_proc, "normalize", False)):
            raise RuntimeError(
                f"[_verify_dataset_preprocess_alignment] Spatial preprocessor lệch cấu hình chuẩn ({exp_view})!"
            )
    if "semantic" in active_branches and getattr(dataset, "semantic_proc", None) is not None:
        sem_proc = dataset.semantic_proc
        exp_sz = BRANCH_PREPROCESS_CONFIG["semantic"]["image_size"]
        if getattr(sem_proc, "image_size", None) != exp_sz or not bool(getattr(sem_proc, "normalize", False)):
            raise RuntimeError(
                f"[_verify_dataset_preprocess_alignment] Semantic preprocessor lệch cấu hình chuẩn ({exp_sz})!"
            )
    if "spectral" in active_branches and getattr(dataset, "spectral_proc", None) is not None:
        fft_proc = dataset.spectral_proc
        exp_fft = BRANCH_PREPROCESS_CONFIG["spectral"]
        if (
            getattr(fft_proc, "mode", None) != exp_fft["fft_mode"]
            or bool(getattr(fft_proc, "use_hann", True)) != bool(exp_fft["use_hann"])
            or bool(getattr(fft_proc, "remove_dc", False)) != bool(exp_fft["remove_dc"])
            or not bool(getattr(fft_proc, "normalize", False))
            or getattr(fft_proc, "image_size", None) != exp_fft["crop_size"]
        ):
            raise RuntimeError(
                "[_verify_dataset_preprocess_alignment] Spectral FFT preprocessor lệch cấu hình chuẩn!"
            )


def load_branch_extractor(
    branch: str,
    device: Union[str, torch.device],
    spatial_ckpt: Union[str, Path] = DEFAULT_SPATIAL_CKPT,
    spectral_ckpt: Union[str, Path] = DEFAULT_SPECTRAL_CKPT,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
    clip_pretrained: Optional[Union[str, bool]] = DEFAULT_CLIP_PRETRAINED,
    allow_untrained: bool = False,
) -> Tuple[BranchBase, Dict[str, Any]]:
    """
    Khởi tạo nhánh trích xuất đặc trưng, nạp trọng số chuẩn xác và khóa cứng ở chế độ `.eval()`.

    Chi tiết nguồn trọng số:
      - `spatial`  : Nạp checkpoint `best_spatial.pth` đã huấn luyện ở Giai đoạn 1.
      - `spectral` : Nạp checkpoint `best_spectral.pth` đã huấn luyện ở Giai đoạn 1.
      - `semantic` : Nạp trực tiếp trọng số tiền huấn luyện OpenAI CLIP `ViT-B-16`
                     (thân mạng luôn khóa cứng 100%, chuẩn hóa L2 đầu ra `512-D`).
    """
    branch_norm = str(branch).strip().lower()
    if branch_norm not in VALID_BRANCHES:
        raise ValueError(
            f"[load_branch_extractor] branch='{branch}' không hợp lệ. Chọn trong {VALID_BRANCHES}."
        )

    dev = torch.device(device)

    if branch_norm == "spatial":
        ckpt_path = Path(spatial_ckpt)
        model: BranchBase = SpatialBranch(
            pretrained=False,
            dropout=0.2,
        )
        if ckpt_path.is_file():
            meta = load_checkpoint(
                filepath=ckpt_path,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=dev,
                strict=True,
                restore_rng=False,
            )
            checkpoint_info: Dict[str, Any] = {
                "source": "checkpoint",
                "checkpoint_path": str(ckpt_path.resolve()).replace("\\", "/"),
                "checkpoint_sha256": _compute_file_sha256(ckpt_path),
                "epoch": int(meta.get("epoch", 0)),
                "val_metrics": copy.deepcopy(meta.get("metrics", meta.get("best_metrics", {}))),
                "best_metrics": copy.deepcopy(meta.get("best_metrics", {})),
            }
        elif allow_untrained:
            checkpoint_info = {
                "source": "untrained_fallback",
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": None,
                "epoch": 0,
                "val_metrics": {},
            }
        else:
            raise FileNotFoundError(
                f"[load_branch_extractor] Không tìm thấy checkpoint nhánh Spatial tại: '{ckpt_path}'. "
                f"Hãy huấn luyện nhánh Spatial trước bằng lệnh: "
                f"`python code/system/train/train_branch.py --branch spatial`"
            )

    elif branch_norm == "spectral":
        ckpt_path = Path(spectral_ckpt)
        model = SpectralBranch(
            dropout=0.3,
        )
        if ckpt_path.is_file():
            meta = load_checkpoint(
                filepath=ckpt_path,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=dev,
                strict=True,
                restore_rng=False,
            )
            checkpoint_info = {
                "source": "checkpoint",
                "checkpoint_path": str(ckpt_path.resolve()).replace("\\", "/"),
                "checkpoint_sha256": _compute_file_sha256(ckpt_path),
                "epoch": int(meta.get("epoch", 0)),
                "val_metrics": copy.deepcopy(meta.get("metrics", meta.get("best_metrics", {}))),
                "best_metrics": copy.deepcopy(meta.get("best_metrics", {})),
            }
        elif allow_untrained:
            checkpoint_info = {
                "source": "untrained_fallback",
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": None,
                "epoch": 0,
                "val_metrics": {},
            }
        else:
            raise FileNotFoundError(
                f"[load_branch_extractor] Không tìm thấy checkpoint nhánh Spectral tại: '{ckpt_path}'. "
                f"Hãy huấn luyện nhánh Spectral trước bằng lệnh: "
                f"`python code/system/train/train_branch.py --branch spectral`"
            )

    else:  # branch_norm == "semantic"
        resolved_clip_pretrained = _normalize_clip_pretrained_arg(
            clip_pretrained, allow_untrained=allow_untrained
        )
        clip_meta = _get_open_clip_runtime_meta(clip_model_name, resolved_clip_pretrained)

        model = SemanticBranch(
            model_name=clip_model_name,
            pretrained=resolved_clip_pretrained,
            dropout=0.0,
            l2_normalize=True,
        )
        checkpoint_info = {
            "source": clip_meta["source"],
            "model_name": clip_meta["model_name"],
            "pretrained": clip_meta["pretrained"],
            "quick_gelu": bool(getattr(model, "use_quick_gelu", clip_meta["quick_gelu"])),
            "l2_normalize": bool(getattr(model, "l2_normalize", True)),
            "open_clip_version": clip_meta["open_clip_version"],
            "epoch": 0,
            "val_metrics": {},
        }

    # Chuyển lên thiết bị, đóng băng toàn bộ thân mạng + head và đặt chế độ .eval()
    model = model.to(dev)
    model.freeze_backbone(True)
    for param in model.parameters():
        param.requires_grad = False
        param.grad = None
    model.eval()

    expected_dim = BRANCH_FEATURE_DIMS[branch_norm]
    if int(model.feature_dim) != expected_dim:
        raise RuntimeError(
            f"[load_branch_extractor] Sai lệch feature_dim của nhánh '{branch_norm}': "
            f"kỳ vọng {expected_dim}, nhận được {model.feature_dim}."
        )

    return model, checkpoint_info


# ==============================================================================
# LÕI TRÍCH XUẤT ĐẶC TRƯNG ĐƠN GÓC NHÌN & ĐA GÓC NHÌN (SINGLE-PASS ENGINE)
# ==============================================================================
@torch.inference_mode()
def _run_single_view_pass(
    models: Mapping[str, BranchBase],
    split: str,
    dataset_root: Union[str, Path],
    metadata_path: Optional[Union[str, Path]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    force_flip: bool,
    fixed_jpeg_quality: Optional[int] = None,
    max_batches: Optional[int] = None,
    log_interval: int = 50,
) -> Dict[str, Any]:
    """
    Chạy đúng 1 lượt tuần tự qua `build_extract_loader` cho tất cả các nhánh trong `models`.

    Mỗi ảnh chỉ được đọc từ ổ cứng đúng 1 lần trong `CommonProcess`, sau đó trích xuất
    song song `forward_features` cho tất cả các nhánh đang kích hoạt.
    """
    active_branches = tuple(models.keys())
    if len(active_branches) == 0:
        raise ValueError("[_run_single_view_pass] `models` không được rỗng.")

    for br, m in models.items():
        m.eval()

    if max_batches is not None and (isinstance(max_batches, bool) or int(max_batches) < 1):
        raise ValueError(
            f"[_run_single_view_pass] `max_batches` phải là số nguyên >= 1 hoặc None, nhận được: {max_batches!r}"
        )

    pin_memory = device.type == "cuda"
    loader = build_extract_loader(
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_root=dataset_root,
        metadata_path=metadata_path,
        pin_memory=pin_memory,
        force_flip=force_flip,
        active_branches=active_branches,
        fixed_jpeg_quality=fixed_jpeg_quality,
    )
    _verify_dataset_preprocess_alignment(loader.dataset, active_branches)

    total_batches = len(loader)
    target_batches = min(total_batches, int(max_batches)) if max_batches is not None else total_batches

    feat_chunks: Dict[str, List[torch.Tensor]] = {br: [] for br in active_branches}
    sample_idx_chunks: List[torch.Tensor] = []
    label_a_chunks: List[torch.Tensor] = []
    label_b_chunks: List[torch.Tensor] = []
    group_id_chunks: List[torch.Tensor] = []
    filepaths_all: List[str] = []
    model_names_all: List[str] = []

    view_tag = "FLIP (force_flip=True)" if force_flip else "ORIG (force_flip=False)"
    jpeg_tag = f", jpeg={fixed_jpeg_quality}" if fixed_jpeg_quality is not None else ""
    print(
        f"  -> Bắt đầu lượt trích xuất [{split.upper()} | {view_tag}{jpeg_tag}] "
        f"cho nhánh {list(active_branches)} ({target_batches}/{total_batches} batches)..."
    )

    start_time = time.perf_counter()
    processed_samples = 0

    try:
        for batch_idx, batch in enumerate(itertools.islice(loader, target_batches)):

            # 1. Thu thập metadata của batch
            s_idx = batch["sample_idx"].detach().to(device="cpu", dtype=torch.int64)
            lbl_a = batch["label_a"].detach().to(device="cpu", dtype=torch.float32)
            lbl_b = batch["label_b"].detach().to(device="cpu", dtype=torch.int64)
            grp_id = batch["group_id"].detach().to(device="cpu", dtype=torch.int64)

            fps_batch = [str(fp).replace("\\", "/") for fp in batch["filepath"]]
            mnames_batch = [str(mn) for mn in batch["model_name"]]

            batch_len = int(s_idx.shape[0])
            processed_samples += batch_len

            sample_idx_chunks.append(s_idx)
            label_a_chunks.append(lbl_a)
            label_b_chunks.append(lbl_b)
            group_id_chunks.append(grp_id)
            filepaths_all.extend(fps_batch)
            model_names_all.extend(mnames_batch)

            # 2. Trích xuất forward_features cho từng nhánh từ cùng batch đã đọc
            for br in active_branches:
                if br not in batch:
                    raise KeyError(
                        f"[_run_single_view_pass] Batch không chứa khóa '{br}'. "
                        f"Các khóa hiện có: {list(batch.keys())}"
                    )
                x = batch[br].to(device=device, non_blocking=pin_memory)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    feats = models[br].forward_features(x)

                if feats.requires_grad or feats.grad_fn is not None:
                    raise RuntimeError(
                        f"[_run_single_view_pass] Phát hiện computation graph còn bật trên đặc trưng nhánh '{br}'!"
                    )

                # Luôn ép về float32 trên CPU trước khi lưu đệm để tránh tràn/mất độ chính xác
                feats_cpu = feats.detach().to(device="cpu", dtype=torch.float32).contiguous()

                expected_dim = BRANCH_FEATURE_DIMS[br]
                if feats_cpu.ndim != 2 or feats_cpu.shape != (batch_len, expected_dim):
                    raise RuntimeError(
                        f"[_run_single_view_pass] Shape đặc trưng nhánh '{br}' sai lệch tại batch {batch_idx}: "
                        f"kỳ vọng ({batch_len}, {expected_dim}), nhận được {tuple(feats_cpu.shape)}"
                    )
                if not torch.isfinite(feats_cpu).all():
                    raise FloatingPointError(
                        f"[_run_single_view_pass] Phát hiện NaN/Inf trong đặc trưng nhánh '{br}' "
                        f"tại batch {batch_idx} (split='{split}', force_flip={force_flip})!"
                    )

                # Kiểm tra chuẩn hóa L2 = 1.0 đối với nhánh semantic (CLIP)
                if br == "semantic" and getattr(models[br], "l2_normalize", True):
                    norms = torch.norm(feats_cpu, p=2, dim=-1)
                    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
                        max_err = float((norms - 1.0).abs().max().item())
                        raise RuntimeError(
                            f"[_run_single_view_pass] Đặc trưng CLIP không thỏa chuẩn L2=1.0 "
                            f"(sai số cực đại={max_err:.6f}) tại batch {batch_idx}."
                        )

                feat_chunks[br].append(feats_cpu)

            if log_interval > 0 and (
                (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == target_batches
            ):
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                speed = processed_samples / elapsed
                print(
                    f"     [Batch {batch_idx + 1:>4d}/{target_batches:<4d}] "
                    f"Samples: {processed_samples:>6d} | "
                    f"Speed: {speed:>6.1f} img/s | "
                    f"Elapsed: {elapsed:>6.1f}s"
                )
    finally:
        _shutdown_dataloader_workers(loader)

    if len(sample_idx_chunks) == 0:
        raise RuntimeError(f"[_run_single_view_pass] Không trích xuất được mẫu nào cho split='{split}'.")

    sample_idx_all = torch.cat(sample_idx_chunks, dim=0).contiguous()
    label_a_all = torch.cat(label_a_chunks, dim=0).contiguous()
    label_b_all = torch.cat(label_b_chunks, dim=0).contiguous()
    group_id_all = torch.cat(group_id_chunks, dim=0).contiguous()

    num_samples = int(sample_idx_all.shape[0])
    expected_indices = torch.arange(num_samples, dtype=torch.int64)
    if not torch.equal(sample_idx_all, expected_indices):
        raise RuntimeError(
            f"[_run_single_view_pass] Thứ tự mẫu (sample_idx) bị xáo trộn hoặc thiếu hụt trên "
            f"split='{split}', force_flip={force_flip}! Đảm bảo shuffle=False trong build_extract_loader."
        )

    if max_batches is None:
        expected_total = len(loader.dataset)
        if num_samples != expected_total:
            raise RuntimeError(
                f"[_run_single_view_pass] Thiếu mẫu khi trích xuất toàn phần split='{split}': "
                f"trích được {num_samples} != kỳ vọng {expected_total}."
            )
        if filepaths_all != list(loader.dataset.filepaths):
            raise RuntimeError(
                f"[_run_single_view_pass] Thứ tự filepaths trích xuất không khớp với dataset gốc trên split='{split}'!"
            )

    features_by_branch: Dict[str, torch.Tensor] = {}
    for br in active_branches:
        features_by_branch[br] = torch.cat(feat_chunks[br], dim=0).contiguous()

    return {
        "num_samples": num_samples,
        "features_by_branch": features_by_branch,
        "sample_idx": sample_idx_all,
        "filepaths": filepaths_all,
        "label_a": label_a_all,
        "label_b": label_b_all,
        "group_id": group_id_all,
        "model_names": model_names_all,
    }


def extract_split_features(
    models: Mapping[str, BranchBase],
    checkpoint_infos: Mapping[str, Dict[str, Any]],
    split: str,
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    metadata_path: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    device: Union[str, torch.device] = "cuda",
    use_amp: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
    include_flip: Optional[bool] = None,
    fixed_jpeg_quality: Optional[int] = None,
    max_batches: Optional[int] = None,
    log_interval: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """
    Trích xuất toàn diện đặc trưng cho 1 `split` (`train`, `val`, hoặc `test`) trên các nhánh chỉ định.

    Quy tắc góc nhìn lật ngang (`include_flip`):
      - Mặc định `include_flip = (split == "train" and fixed_jpeg_quality is None)`:
        * Tập `train`: Tự động trích xuất cả `features` (`force_flip=False`) lẫn
          `features_flip` (`force_flip=True`) và đối soát 100% `sample_idx`/`filepaths` giữa 2 lượt.
        * Tập `val` và `test`: Chỉ trích xuất `features` (`force_flip=False`), `features_flip = None`.

    Returns:
        Dict ánh xạ `{branch_name: payload_dict}` sẵn sàng để lưu xuống file `.pt`.
    """
    split_norm = str(split).strip().lower()
    if split_norm not in VALID_SPLITS:
        raise ValueError(
            f"[extract_split_features] split='{split}' không hợp lệ. Chọn trong {VALID_SPLITS}."
        )

    dev = torch.device(device)
    if use_amp is None or amp_dtype is None:
        auto_amp, auto_dtype, _ = get_optimal_amp_config(dev)
        if use_amp is None:
            use_amp = auto_amp
        if amp_dtype is None:
            amp_dtype = auto_dtype

    if include_flip is None:
        include_flip = split_norm == "train" and fixed_jpeg_quality is None

    # Lượt 1: Ảnh gốc (force_flip = False)
    orig_pass = _run_single_view_pass(
        models=models,
        split=split_norm,
        dataset_root=dataset_root,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=dev,
        use_amp=bool(use_amp),
        amp_dtype=amp_dtype,
        force_flip=False,
        fixed_jpeg_quality=fixed_jpeg_quality,
        max_batches=max_batches,
        log_interval=log_interval,
    )

    # Lượt 2 (nếu bật include_flip): Ảnh lật ngang (force_flip = True)
    flip_pass: Optional[Dict[str, Any]] = None
    if include_flip:
        flip_pass = _run_single_view_pass(
            models=models,
            split=split_norm,
            dataset_root=dataset_root,
            metadata_path=metadata_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=dev,
            use_amp=bool(use_amp),
            amp_dtype=amp_dtype,
            force_flip=True,
            fixed_jpeg_quality=fixed_jpeg_quality,
            max_batches=max_batches,
            log_interval=log_interval,
        )

        # Đối soát nghiêm ngặt lượt gốc và lượt lật ngang phải khớp 100% từng mẫu
        if not torch.equal(orig_pass["sample_idx"], flip_pass["sample_idx"]):
            raise RuntimeError(
                "[extract_split_features] sample_idx giữa lượt gốc (force_flip=False) "
                "và lượt lật ngang (force_flip=True) không khớp!"
            )
        if orig_pass["filepaths"] != flip_pass["filepaths"]:
            raise RuntimeError(
                "[extract_split_features] Danh sách filepaths giữa lượt gốc và lượt lật ngang không khớp!"
            )
        if not torch.equal(orig_pass["label_a"], flip_pass["label_a"]):
            raise RuntimeError(
                "[extract_split_features] label_a giữa lượt gốc và lượt lật ngang không khớp!"
            )
        if not torch.equal(orig_pass["label_b"], flip_pass["label_b"]):
            raise RuntimeError(
                "[extract_split_features] label_b giữa lượt gốc và lượt lật ngang không khớp!"
            )
        if not torch.equal(orig_pass["group_id"], flip_pass["group_id"]):
            raise RuntimeError(
                "[extract_split_features] group_id giữa lượt gốc và lượt lật ngang không khớp!"
            )
        if orig_pass["model_names"] != flip_pass["model_names"]:
            raise RuntimeError(
                "[extract_split_features] model_names giữa lượt gốc và lượt lật ngang không khớp!"
            )

    manifest_hash = compute_manifest_hash(
        filepaths=orig_pass["filepaths"],
        labels_a=orig_pass["label_a"],
        model_names=orig_pass["model_names"],
        sample_indices=orig_pass["sample_idx"],
        labels_b=orig_pass["label_b"],
        group_ids=orig_pass["group_id"],
    )
    created_at = datetime.now().isoformat(timespec="seconds")

    payloads_by_branch: Dict[str, Dict[str, Any]] = {}
    for br in models.keys():
        feats_orig = orig_pass["features_by_branch"][br]
        feats_flip = flip_pass["features_by_branch"][br] if flip_pass is not None else None

        preprocess_cfg = build_preprocess_config(
            branch=br,
            include_flip=bool(include_flip),
            fixed_jpeg_quality=fixed_jpeg_quality,
            use_amp=bool(use_amp),
            amp_dtype=amp_dtype,
        )
        ckpt_info_copy = copy.deepcopy(checkpoint_infos.get(br, {}))
        extraction_sig = build_extraction_signature(
            branch=br,
            split=split_norm,
            checkpoint_info=ckpt_info_copy,
            preprocess_cfg=preprocess_cfg,
            include_flip=bool(include_flip),
            fixed_jpeg_quality=fixed_jpeg_quality,
            use_amp=bool(use_amp),
            amp_dtype=amp_dtype,
        )

        payload: Dict[str, Any] = {
            "branch": br,
            "split": split_norm,
            "feature_dim": BRANCH_FEATURE_DIMS[br],
            "num_samples": int(orig_pass["num_samples"]),
            "features": feats_orig,
            "features_flip": feats_flip,
            "has_flip": bool(include_flip),
            "sample_idx": orig_pass["sample_idx"].clone(),
            "filepaths": list(orig_pass["filepaths"]),
            "label_a": orig_pass["label_a"].clone(),
            "label_b": orig_pass["label_b"].clone(),
            "group_id": orig_pass["group_id"].clone(),
            "model_names": list(orig_pass["model_names"]),
            "manifest_hash": manifest_hash,
            "jpeg_quality": int(fixed_jpeg_quality) if fixed_jpeg_quality is not None else None,
            "checkpoint_info": ckpt_info_copy,
            "preprocess_config": preprocess_cfg,
            "extraction_signature": extraction_sig,
            "created_at": created_at,
        }
        validate_feature_cache_payload(payload, expected_branch=br, expected_split=split_norm)
        payloads_by_branch[br] = payload

    # Nếu trích xuất >= 2 nhánh cùng lúc, kiểm tra căn chỉnh chéo ngay tại chỗ
    if len(payloads_by_branch) > 1:
        verify_cross_branch_alignment(payloads_by_branch)

    return payloads_by_branch


# ==============================================================================
# KIỂM ĐỊNH TOÀN VẸN, LƯU ATOMIC & ĐỐI SOÁT CĂN CHỈNH 3 NHÁNH
# ==============================================================================
def validate_feature_cache_payload(
    payload: Mapping[str, Any],
    expected_branch: Optional[str] = None,
    expected_split: Optional[str] = None,
) -> None:
    """
    Kiểm định toàn diện cấu trúc, kiểu dữ liệu, kích thước tensor và tính toàn vẹn của 1 bộ cache.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"[validate_feature_cache_payload] Payload phải là dict/Mapping, nhận được {type(payload)}."
        )

    missing_keys = [k for k in REQUIRED_CACHE_KEYS if k not in payload]
    if missing_keys:
        raise KeyError(
            f"[validate_feature_cache_payload] Thiếu các khóa bắt buộc trong cache: {missing_keys}"
        )

    branch = str(payload["branch"]).strip().lower()
    split = str(payload["split"]).strip().lower()

    if branch not in VALID_BRANCHES:
        raise ValueError(f"[validate_feature_cache_payload] branch='{branch}' không hợp lệ.")
    if split not in VALID_SPLITS:
        raise ValueError(f"[validate_feature_cache_payload] split='{split}' không hợp lệ.")

    if expected_branch is not None and branch != str(expected_branch).strip().lower():
        raise ValueError(
            f"[validate_feature_cache_payload] Kỳ vọng branch='{expected_branch}', "
            f"nhưng file cache ghi branch='{branch}'."
        )
    if expected_split is not None and split != str(expected_split).strip().lower():
        raise ValueError(
            f"[validate_feature_cache_payload] Kỳ vọng split='{expected_split}', "
            f"nhưng file cache ghi split='{split}'."
        )

    expected_dim = BRANCH_FEATURE_DIMS[branch]
    feat_dim = int(payload["feature_dim"])
    if feat_dim != expected_dim:
        raise ValueError(
            f"[validate_feature_cache_payload] feature_dim={feat_dim} không khớp "
            f"với chuẩn của nhánh '{branch}' ({expected_dim})."
        )

    num_samples = int(payload["num_samples"])
    if num_samples <= 0:
        raise ValueError(
            f"[validate_feature_cache_payload] num_samples phải > 0, nhận được {num_samples}."
        )

    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        raise TypeError("[validate_feature_cache_payload] `features` phải là torch.Tensor.")
    if features.dtype != torch.float32:
        raise TypeError(
            f"[validate_feature_cache_payload] `features` phải có dtype=torch.float32, "
            f"nhận được {features.dtype}."
        )
    if features.ndim != 2 or tuple(features.shape) != (num_samples, expected_dim):
        raise ValueError(
            f"[validate_feature_cache_payload] `features` có shape {tuple(features.shape)}, "
            f"kỳ vọng ({num_samples}, {expected_dim})."
        )
    if not torch.isfinite(features).all():
        raise FloatingPointError(
            f"[validate_feature_cache_payload] Phát hiện NaN/Inf trong `features` của {branch}_{split}!"
        )

    has_flip = bool(payload["has_flip"])
    features_flip = payload["features_flip"]
    if has_flip:
        if not isinstance(features_flip, torch.Tensor):
            raise TypeError(
                "[validate_feature_cache_payload] Khi has_flip=True, `features_flip` phải là torch.Tensor."
            )
        if features_flip.dtype != torch.float32:
            raise TypeError(
                f"[validate_feature_cache_payload] `features_flip` phải có dtype=torch.float32, "
                f"nhận được {features_flip.dtype}."
            )
        if features_flip.ndim != 2 or tuple(features_flip.shape) != (num_samples, expected_dim):
            raise ValueError(
                f"[validate_feature_cache_payload] `features_flip` có shape {tuple(features_flip.shape)}, "
                f"kỳ vọng ({num_samples}, {expected_dim})."
            )
        if not torch.isfinite(features_flip).all():
            raise FloatingPointError(
                f"[validate_feature_cache_payload] Phát hiện NaN/Inf trong `features_flip` của {branch}_{split}!"
            )
    else:
        if features_flip is not None:
            raise ValueError(
                "[validate_feature_cache_payload] Khi has_flip=False, `features_flip` phải là None."
            )

    # Kiểm tra chuẩn hóa L2 = 1.0 đối với nhánh semantic (CLIP)
    if branch == "semantic":
        norms = torch.norm(features, p=2, dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
            raise ValueError(
                f"[validate_feature_cache_payload] `features` của nhánh semantic không có L2-norm = 1.0."
            )
        if features_flip is not None:
            norms_flip = torch.norm(features_flip, p=2, dim=-1)
            if not torch.allclose(norms_flip, torch.ones_like(norms_flip), atol=1e-3):
                raise ValueError(
                    "[validate_feature_cache_payload] `features_flip` của nhánh semantic không có L2-norm = 1.0."
                )

    # Kiểm tra các tensor metadata & thứ tự sample_idx
    sample_idx = payload["sample_idx"]
    label_a = payload["label_a"]
    label_b = payload["label_b"]
    group_id = payload["group_id"]
    filepaths = payload["filepaths"]
    model_names = payload["model_names"]

    if not isinstance(sample_idx, torch.Tensor) or sample_idx.dtype != torch.int64:
        raise TypeError("[validate_feature_cache_payload] `sample_idx` phải là Tensor int64.")
    if tuple(sample_idx.shape) != (num_samples,):
        raise ValueError(
            f"[validate_feature_cache_payload] `sample_idx` shape {tuple(sample_idx.shape)} != ({num_samples},)."
        )
    if not torch.equal(sample_idx.cpu(), torch.arange(num_samples, dtype=torch.int64)):
        raise ValueError(
            "[validate_feature_cache_payload] `sample_idx` không liên tục chuẩn [0..N-1]!"
        )

    if not isinstance(label_a, torch.Tensor) or label_a.dtype != torch.float32:
        raise TypeError("[validate_feature_cache_payload] `label_a` phải là Tensor float32.")
    if tuple(label_a.shape) != (num_samples,):
        raise ValueError("[validate_feature_cache_payload] `label_a` sai kích thước.")

    if not isinstance(label_b, torch.Tensor) or label_b.dtype != torch.int64:
        raise TypeError("[validate_feature_cache_payload] `label_b` phải là Tensor int64.")
    if tuple(label_b.shape) != (num_samples,):
        raise ValueError("[validate_feature_cache_payload] `label_b` sai kích thước.")

    if not isinstance(group_id, torch.Tensor) or group_id.dtype != torch.int64:
        raise TypeError("[validate_feature_cache_payload] `group_id` phải là Tensor int64.")
    if tuple(group_id.shape) != (num_samples,):
        raise ValueError("[validate_feature_cache_payload] `group_id` sai kích thước.")

    if not isinstance(filepaths, list) or len(filepaths) != num_samples:
        raise ValueError("[validate_feature_cache_payload] `filepaths` phải là list có độ dài num_samples.")
    if any((not isinstance(fp, str) or not fp.strip()) for fp in filepaths):
        raise ValueError("[validate_feature_cache_payload] Phát hiện phần tử rỗng hoặc sai kiểu trong `filepaths`.")
    if len(set(filepaths)) != num_samples:
        raise ValueError(
            f"[validate_feature_cache_payload] Phát hiện đường dẫn ảnh bị trùng lặp trong `filepaths` "
            f"(unique={len(set(filepaths))} < num_samples={num_samples})!"
        )

    if not isinstance(model_names, list) or len(model_names) != num_samples:
        raise ValueError("[validate_feature_cache_payload] `model_names` phải là list có độ dài num_samples.")

    # Chặn rò rỉ mô hình Unseen (SD3, Midjourney_V6) vào tập Train hoặc Val
    allowed_models = ALLOWED_MODELS_BY_SPLIT[split]
    unexpected_models = set(model_names) - allowed_models
    if unexpected_models:
        raise ValueError(
            f"[validate_feature_cache_payload] Phát hiện nguồn ảnh không hợp lệ trên split='{split}': "
            f"{sorted(unexpected_models)}. Chỉ cho phép: {sorted(allowed_models)}."
        )

    # Đối chiếu nhất quán giữa model_name với label_a (MS_COCO <-> 0, AI <-> 1) và label_b (0..5)
    labels_a_list = label_a.detach().cpu().tolist()
    labels_b_list = label_b.detach().cpu().tolist()
    for idx_m, (mname, lbl_val, lbl_b_val) in enumerate(zip(model_names, labels_a_list, labels_b_list)):
        expected_lbl = 0.0 if mname == REAL_MODEL_NAME else 1.0
        if float(lbl_val) != expected_lbl:
            raise ValueError(
                f"[validate_feature_cache_payload] Mâu thuẫn nhãn tại sample {idx_m}: "
                f"model_name='{mname}' nhưng label_a={lbl_val} (kỳ vọng {expected_lbl})."
            )
        if mname not in MODEL_NAME_TO_LABEL_B:
            raise ValueError(
                f"[validate_feature_cache_payload] model_name='{mname}' không nằm trong bảng ánh xạ MODEL_NAME_TO_LABEL_B!"
            )
        expected_lbl_b = MODEL_NAME_TO_LABEL_B[mname]
        if int(lbl_b_val) != expected_lbl_b:
            raise ValueError(
                f"[validate_feature_cache_payload] Mâu thuẫn nhãn chi tiết label_b tại sample {idx_m}: "
                f"model_name='{mname}' nhưng label_b={lbl_b_val} (kỳ vọng {expected_lbl_b})."
            )

    preprocess_cfg = payload["preprocess_config"]
    if not isinstance(preprocess_cfg, Mapping) or len(preprocess_cfg) == 0:
        raise ValueError(
            "[validate_feature_cache_payload] `preprocess_config` phải là dict/Mapping khác rỗng."
        )

    extraction_sig = payload["extraction_signature"]
    if not isinstance(extraction_sig, Mapping) or len(extraction_sig) == 0:
        raise ValueError(
            "[validate_feature_cache_payload] `extraction_signature` phải là dict/Mapping khác rỗng."
        )
    if (
        str(extraction_sig.get("branch", "")).strip().lower() != branch
        or str(extraction_sig.get("split", "")).strip().lower() != split
        or bool(extraction_sig.get("has_flip")) != has_flip
        or extraction_sig.get("jpeg_quality") != payload["jpeg_quality"]
    ):
        raise ValueError(
            f"[validate_feature_cache_payload] `extraction_signature` mâu thuẫn với thông tin top-level của cache '{branch}_{split}'!"
        )

    recomputed_hash = compute_manifest_hash(
        filepaths=filepaths,
        labels_a=label_a,
        model_names=model_names,
        sample_indices=sample_idx,
        labels_b=label_b,
        group_ids=group_id,
    )
    if str(payload["manifest_hash"]) != recomputed_hash:
        raise ValueError(
            f"[validate_feature_cache_payload] `manifest_hash` không khớp dữ liệu nội tại: "
            f"lưu='{payload['manifest_hash']}' vs tính lại='{recomputed_hash}'."
        )


def save_feature_cache_atomic(
    payload: Mapping[str, Any],
    output_path: Union[str, Path],
    verify_readback: bool = True,
) -> Path:
    """
    Kiểm định payload, ghi file tạm, đọc kiểm thử lại toàn vẹn file tạm rồi mới `os.replace`
    (Atomic Write) để bảo đảm không bao giờ tạo ra file `.pt` bị hỏng trên ổ đĩa.
    """
    validate_feature_cache_payload(payload)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".tmp_{out_path.stem}_",
        suffix=".pt",
        dir=str(out_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        torch.save(dict(payload), tmp_path)
        if verify_readback:
            tmp_loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
            validate_feature_cache_payload(
                tmp_loaded,
                expected_branch=str(payload["branch"]),
                expected_split=str(payload["split"]),
            )
            if str(tmp_loaded["manifest_hash"]) != str(payload["manifest_hash"]):
                raise ValueError(
                    "[save_feature_cache_atomic] Read-back `manifest_hash` của file tạm không khớp!"
                )
        os.replace(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return out_path


def export_split_manifest_csv(
    payload: Mapping[str, Any],
    output_csv_path: Union[str, Path],
) -> Path:
    """
    Xuất bảng kê khai mẫu (Manifest CSV) tương ứng với file `.pt` theo cơ chế Atomic Write
    để dễ dàng đối chiếu trực quan bằng mắt hoặc công cụ phân tích bên ngoài.
    """
    validate_feature_cache_payload(payload)
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.DataFrame(
        {
            "sample_idx": payload["sample_idx"].detach().cpu().numpy().tolist(),
            "filepath": list(payload["filepaths"]),
            "group_id": payload["group_id"].detach().cpu().numpy().tolist(),
            "split": [str(payload["split"])] * int(payload["num_samples"]),
            "label_a": [int(round(float(v))) for v in payload["label_a"].detach().cpu().tolist()],
            "label_b": payload["label_b"].detach().cpu().numpy().tolist(),
            "model_name": list(payload["model_names"]),
            "manifest_hash": [str(payload["manifest_hash"])] * int(payload["num_samples"]),
        }
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".tmp_{out_csv.stem}_",
        suffix=".csv",
        dir=str(out_csv.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df_manifest.to_csv(tmp_path, index=False, encoding="utf-8")
        os.replace(str(tmp_path), str(out_csv))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return out_csv


def load_feature_cache(
    filepath: Union[str, Path],
    expected_branch: Optional[str] = None,
    expected_split: Optional[str] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Nạp file `.pt` lưu đệm đặc trưng và kiểm định toàn vẹn cấu trúc trước khi trả về.
    Hàm này được `train_fusion.py` gọi trực tiếp khi khởi tạo `CachedFeatureDataset`.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"[load_feature_cache] Không tìm thấy file cache đặc trưng tại: '{path}'."
        )

    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_feature_cache_payload(
        payload,
        expected_branch=expected_branch,
        expected_split=expected_split,
    )
    return payload


def verify_cross_branch_alignment(
    cache_payloads: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Đối soát nghiêm ngặt sự căn chỉnh mẫu (Sample Alignment) giữa các nhánh (Mục 8.1 #5).

    Đảm bảo mẫu thứ `i` của `spatial`, `semantic`, `spectral` có cùng:
      - `sample_idx[i] == i`
      - `filepaths[i]`
      - `label_a[i]` (0.0 = Real, 1.0 = AI)
      - `label_b[i]` (0..5 = Generator ID)
      - `group_id[i]`
      - `model_names[i]`
      - `manifest_hash`, `split`, `num_samples`, `has_flip`, `jpeg_quality`
    """
    branch_names = list(cache_payloads.keys())
    if len(branch_names) == 0:
        raise ValueError("[verify_cross_branch_alignment] `cache_payloads` không được rỗng.")

    for br, payload in cache_payloads.items():
        validate_feature_cache_payload(payload, expected_branch=br)

    ref_branch = branch_names[0]
    ref = cache_payloads[ref_branch]

    ref_split = ref["split"]
    ref_num_samples = int(ref["num_samples"])
    ref_has_flip = bool(ref["has_flip"])
    ref_jpeg = ref["jpeg_quality"]
    ref_hash = str(ref["manifest_hash"])

    for br in branch_names[1:]:
        cur = cache_payloads[br]
        if cur["split"] != ref_split:
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch split giữa '{ref_branch}' ({ref_split}) "
                f"và '{br}' ({cur['split']})."
            )
        if int(cur["num_samples"]) != ref_num_samples:
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch số lượng mẫu (num_samples) giữa "
                f"'{ref_branch}' ({ref_num_samples}) và '{br}' ({cur['num_samples']})."
            )
        if bool(cur["has_flip"]) != ref_has_flip:
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch trạng thái has_flip giữa "
                f"'{ref_branch}' ({ref_has_flip}) và '{br}' ({cur['has_flip']})."
            )
        if cur["jpeg_quality"] != ref_jpeg:
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch jpeg_quality giữa "
                f"'{ref_branch}' ({ref_jpeg}) và '{br}' ({cur['jpeg_quality']})."
            )
        if str(cur["manifest_hash"]) != ref_hash:
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch manifest_hash giữa "
                f"'{ref_branch}' ({ref_hash}) và '{br}' ({cur['manifest_hash']})."
            )
        if not torch.equal(ref["sample_idx"].cpu(), cur["sample_idx"].cpu()):
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch tensor sample_idx giữa '{ref_branch}' và '{br}'!"
            )
        if ref["filepaths"] != cur["filepaths"]:
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch danh sách filepaths giữa '{ref_branch}' và '{br}'!"
            )
        if not torch.equal(ref["label_a"].cpu(), cur["label_a"].cpu()):
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch tensor label_a giữa '{ref_branch}' và '{br}'!"
            )
        if not torch.equal(ref["label_b"].cpu(), cur["label_b"].cpu()):
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch tensor label_b giữa '{ref_branch}' và '{br}'!"
            )
        if not torch.equal(ref["group_id"].cpu(), cur["group_id"].cpu()):
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch tensor group_id giữa '{ref_branch}' và '{br}'!"
            )
        if ref["model_names"] != cur["model_names"]:
            raise ValueError(
                f"[verify_cross_branch_alignment] Lệch danh sách model_names giữa '{ref_branch}' và '{br}'!"
            )

    return {
        "aligned": True,
        "branches": branch_names,
        "split": ref_split,
        "num_samples": ref_num_samples,
        "has_flip": ref_has_flip,
        "jpeg_quality": ref_jpeg,
        "manifest_hash": ref_hash,
        "feature_dims": {br: int(cache_payloads[br]["feature_dim"]) for br in branch_names},
    }


# ==============================================================================
# ĐIỀU PHỐI TRÍCH XUẤT TOÀN HỆ THỐNG (ORCHESTRATOR)
# ==============================================================================
def run_extraction_pipeline(
    branches: Sequence[str] = VALID_BRANCHES,
    splits: Sequence[str] = VALID_SPLITS,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    metadata_path: Optional[Union[str, Path]] = None,
    spatial_ckpt: Union[str, Path] = DEFAULT_SPATIAL_CKPT,
    spectral_ckpt: Union[str, Path] = DEFAULT_SPECTRAL_CKPT,
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
    clip_pretrained: Optional[Union[str, bool]] = DEFAULT_CLIP_PRETRAINED,
    batch_size: int = 64,
    num_workers: int = 2,
    device: Optional[Union[str, torch.device]] = None,
    use_amp: Optional[bool] = None,
    train_include_flip: bool = True,
    test_jpeg_qualities: Optional[Sequence[int]] = None,
    save_manifests: bool = True,
    overwrite: bool = False,
    allow_untrained: bool = False,
    max_batches: Optional[int] = None,
    log_interval: int = 50,
) -> Dict[str, Any]:
    """
    Hàm điều phối cấp cao:
      1. Kiểm tra những file cache `.pt` nào đã tồn tại (bỏ qua nếu `overwrite=False`).
      2. Đối chiếu trực tiếp danh sách mẫu chuẩn từ `metadata.csv` (`get_split_dataframes`).
      3. Chỉ nạp lên GPU đúng các nhánh cần trích xuất.
      4. Chạy trích xuất một lượt (Single-Pass) cho từng `split`, lưu Atomic kèm Read-back Check.
      5. Đối soát chéo sự căn chỉnh mẫu (`verify_cross_branch_alignment`) và xuất CSV manifest.
    """
    norm_branches = tuple(dict.fromkeys(str(b).strip().lower() for b in branches))
    norm_splits = tuple(dict.fromkeys(str(s).strip().lower() for s in splits))

    if len(norm_branches) == 0:
        raise ValueError("[run_extraction_pipeline] `branches` không được rỗng.")
    if len(norm_splits) == 0:
        raise ValueError("[run_extraction_pipeline] `splits` không được rỗng.")
    if max_batches is not None and (isinstance(max_batches, bool) or int(max_batches) < 1):
        raise ValueError(
            f"[run_extraction_pipeline] `max_batches` phải là số nguyên >= 1 hoặc None, nhận được: {max_batches!r}"
        )

    for b in norm_branches:
        if b not in VALID_BRANCHES:
            raise ValueError(f"[run_extraction_pipeline] Nhánh '{b}' không hợp lệ.")
    for s in norm_splits:
        if s not in VALID_SPLITS:
            raise ValueError(f"[run_extraction_pipeline] Split '{s}' không hợp lệ.")

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    auto_amp, auto_dtype, _ = get_optimal_amp_config(dev)
    if use_amp is False:
        resolved_amp = False
        amp_dtype = torch.float32
    elif use_amp is True:
        if not auto_amp:
            warnings.warn(
                f"[run_extraction_pipeline] Yêu cầu `use_amp=True` nhưng thiết bị '{dev}' không hỗ trợ CUDA AMP -> dùng FP32.",
                RuntimeWarning,
            )
            resolved_amp = False
            amp_dtype = torch.float32
        else:
            resolved_amp = True
            amp_dtype = auto_dtype
    else:
        resolved_amp = auto_amp
        amp_dtype = auto_dtype

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Đọc bảng chuẩn metadata gốc để đối chiếu số lượng mẫu & manifest_hash khi trích xuất toàn phần
    meta_file = Path(metadata_path) if metadata_path else Path(dataset_root) / "metadata.csv"
    expected_counts: Dict[str, int] = {}
    expected_hashes: Dict[str, str] = {}
    if max_batches is None:
        train_df, val_df, test_df = get_split_dataframes(metadata_path=str(meta_file))
        for sp_name, sp_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            if sp_name in norm_splits or (sp_name == "test" and test_jpeg_qualities):
                sp_fps = [str(p).replace("\\", "/") for p in sp_df["filepath"].tolist()]
                sp_lbls_a = sp_df["label_a"].tolist()
                sp_lbls_b = sp_df["label_b"].tolist()
                sp_gids = sp_df["group_id"].tolist()
                sp_mnames = [str(m) for m in sp_df["model_name"].tolist()]
                expected_counts[sp_name] = len(sp_df)
                expected_hashes[sp_name] = compute_manifest_hash(
                    filepaths=sp_fps,
                    labels_a=sp_lbls_a,
                    model_names=sp_mnames,
                    sample_indices=range(len(sp_df)),
                    labels_b=sp_lbls_b,
                    group_ids=sp_gids,
                )

    # Xây dựng danh sách tác vụ (split, jpeg_quality)
    tasks: List[Tuple[str, Optional[int]]] = []
    for sp in norm_splits:
        tasks.append((sp, None))
    if test_jpeg_qualities:
        for q in test_jpeg_qualities:
            tasks.append(("test", int(q)))

    # Tính trước dấu vân tay cấu hình kỳ vọng trên CPU để kiểm định cache cũ trước khi skip
    expected_branch_ckpts: Dict[str, Dict[str, Any]] = {
        br: resolve_expected_checkpoint_info(
            branch=br,
            spatial_ckpt=spatial_ckpt,
            spectral_ckpt=spectral_ckpt,
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            allow_untrained=allow_untrained,
        )
        for br in norm_branches
    }

    # Xác định các nhánh thực sự cần chạy cho từng tác vụ
    tasks_needed_branches: Dict[Tuple[str, Optional[int]], List[str]] = {}
    expected_task_sigs: Dict[Tuple[str, Optional[int], str], Dict[str, Any]] = {}
    all_needed_branches_set = set()

    for sp, jq in tasks:
        needed: List[str] = []
        inc_flip = (sp == "train" and jq is None and bool(train_include_flip))
        for br in norm_branches:
            exp_preproc = build_preprocess_config(
                branch=br,
                include_flip=inc_flip,
                fixed_jpeg_quality=jq,
                use_amp=resolved_amp,
                amp_dtype=amp_dtype,
            )
            exp_sig = build_extraction_signature(
                branch=br,
                split=sp,
                checkpoint_info=expected_branch_ckpts[br],
                preprocess_cfg=exp_preproc,
                include_flip=inc_flip,
                fixed_jpeg_quality=jq,
                use_amp=resolved_amp,
                amp_dtype=amp_dtype,
            )
            expected_task_sigs[(sp, jq, br)] = exp_sig

            out_file = get_cache_filepath(cache_root, br, sp, jpeg_quality=jq)
            if out_file.is_file() and not overwrite:
                try:
                    cached = load_feature_cache(out_file, expected_branch=br, expected_split=sp)
                    if max_batches is None and sp in expected_hashes:
                        if str(cached["manifest_hash"]) != expected_hashes[sp]:
                            raise ValueError(
                                f"manifest_hash cũ '{cached['manifest_hash'][:12]}...' != metadata hiện tại '{expected_hashes[sp][:12]}...'"
                            )
                    cached_sig = cached.get("extraction_signature", {})
                    if cached_sig != exp_sig:
                        diff_keys = [
                            k for k in set(exp_sig.keys()) | set(cached_sig.keys())
                            if cached_sig.get(k) != exp_sig.get(k)
                        ]
                        raise ValueError(
                            f"extraction_signature không tương thích (khác ở: {sorted(diff_keys)})"
                        )
                    print(
                        f"  [SKIP] Đã có sẵn cache hợp lệ & khớp signature: {out_file.name} "
                        f"(dùng --overwrite nếu muốn trích xuất lại)."
                    )
                    continue
                except Exception as exc:
                    print(
                        f"  [WARN] Cache cũ '{out_file.name}' hết hiệu lực ({exc}) -> Sẽ trích xuất lại."
                    )
            needed.append(br)
            all_needed_branches_set.add(br)
        tasks_needed_branches[(sp, jq)] = needed

    # Nạp các mô hình cần thiết lên thiết bị
    loaded_models: Dict[str, BranchBase] = {}
    checkpoint_infos: Dict[str, Dict[str, Any]] = {}

    if all_needed_branches_set:
        print("=" * 80)
        print(
            f"KHỞI TẠO CÁC NHÁNH TRÍCH XUẤT ĐẶC TRƯNG TRÊN [{dev.type.upper()}] "
            f"(AMP={resolved_amp}, dtype={amp_dtype})"
        )
        print("=" * 80)
        for br in norm_branches:
            if br in all_needed_branches_set:
                t0 = time.perf_counter()
                model, ckpt_info = load_branch_extractor(
                    branch=br,
                    device=dev,
                    spatial_ckpt=spatial_ckpt,
                    spectral_ckpt=spectral_ckpt,
                    clip_model_name=clip_model_name,
                    clip_pretrained=clip_pretrained,
                    allow_untrained=allow_untrained,
                )
                loaded_models[br] = model
                checkpoint_infos[br] = ckpt_info
                dt = time.perf_counter() - t0
                print(
                    f"  [+] Đã nạp nhánh '{br:<8s}' | dim={model.feature_dim:<4d} | "
                    f"source={ckpt_info.get('source')} | epoch={ckpt_info.get('epoch')} ({dt:.2f}s)"
                )

    saved_files: List[Path] = []
    manifest_files: List[Path] = []
    alignment_reports: Dict[str, Any] = {}

    try:
        for sp, jq in tasks:
            needed_branches = tasks_needed_branches[(sp, jq)]
            task_label = f"{sp}" if jq is None else f"{sp}_jpeg{jq}"

            if needed_branches:
                print("\n" + "-" * 80)
                print(
                    f"TRÍCH XUẤT SPLIT='{task_label.upper()}' | "
                    f"NHÁNH={needed_branches} | BATCH_SIZE={batch_size} | AMP={resolved_amp}"
                )
                print("-" * 80)

                active_models = {br: loaded_models[br] for br in needed_branches}
                active_ckpt_infos = {br: checkpoint_infos[br] for br in needed_branches}
                inc_flip = (sp == "train" and jq is None and bool(train_include_flip))

                payloads = extract_split_features(
                    models=active_models,
                    checkpoint_infos=active_ckpt_infos,
                    split=sp,
                    dataset_root=dataset_root,
                    metadata_path=metadata_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=dev,
                    use_amp=resolved_amp,
                    amp_dtype=amp_dtype,
                    include_flip=inc_flip,
                    fixed_jpeg_quality=jq,
                    max_batches=max_batches,
                    log_interval=log_interval,
                )

                for br, payload in payloads.items():
                    if max_batches is None and sp in expected_counts:
                        if int(payload["num_samples"]) != expected_counts[sp]:
                            raise RuntimeError(
                                f"[run_extraction_pipeline] Lệch số mẫu '{br}_{sp}': "
                                f"{payload['num_samples']} != {expected_counts[sp]} trong metadata gốc."
                            )
                        if str(payload["manifest_hash"]) != expected_hashes[sp]:
                            raise RuntimeError(
                                f"[run_extraction_pipeline] Lệch manifest_hash '{br}_{sp}' so với metadata gốc!"
                            )

                    out_file = get_cache_filepath(cache_root, br, sp, jpeg_quality=jq)
                    save_feature_cache_atomic(payload, out_file, verify_readback=True)
                    size_mb = out_file.stat().st_size / (1024 * 1024)
                    saved_files.append(out_file)
                    print(
                        f"  [SAVED] {out_file.name:<26s} | "
                        f"N={payload['num_samples']:<6d} | "
                        f"dim={payload['feature_dim']:<4d} | "
                        f"flip={str(payload['has_flip']):<5s} | "
                        f"hash={payload['manifest_hash']} | "
                        f"{size_mb:.2f} MB"
                    )

            # Đối soát căn chỉnh giữa tất cả các nhánh được yêu cầu cho tác vụ này
            loaded_for_verify: Dict[str, Dict[str, Any]] = {}
            for br in norm_branches:
                out_file = get_cache_filepath(cache_root, br, sp, jpeg_quality=jq)
                if out_file.is_file():
                    loaded_for_verify[br] = load_feature_cache(
                        out_file, expected_branch=br, expected_split=sp
                    )

            if len(loaded_for_verify) > 0:
                report = verify_cross_branch_alignment(loaded_for_verify)
                if max_batches is None and sp in expected_hashes:
                    if int(report["num_samples"]) != expected_counts[sp]:
                        raise RuntimeError(
                            f"[run_extraction_pipeline] Lệch tổng số mẫu khi đối soát '{task_label}': "
                            f"{report['num_samples']} != {expected_counts[sp]}."
                        )
                    if str(report["manifest_hash"]) != expected_hashes[sp]:
                        raise RuntimeError(
                            f"[run_extraction_pipeline] Lệch manifest_hash khi đối soát '{task_label}' so với metadata gốc!"
                        )
                alignment_reports[task_label] = report
                print(
                    f"  [ALIGNED] Split '{task_label}' khớp 100% trên {report['branches']} | "
                    f"N={report['num_samples']} | manifest_hash={report['manifest_hash']}"
                )
                if save_manifests:
                    ref_payload = loaded_for_verify[report["branches"][0]]
                    manifest_csv = cache_root / "manifests" / f"{task_label}_manifest.csv"
                    export_split_manifest_csv(ref_payload, manifest_csv)
                    manifest_files.append(manifest_csv)
                    print(f"  [MANIFEST] Đã xuất bảng đối chiếu mẫu: {manifest_csv.name}")
    finally:
        loaded_models.clear()
        gc.collect()
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "cache_dir": str(cache_root.resolve()).replace("\\", "/"),
        "saved_files": [str(p.resolve()).replace("\\", "/") for p in saved_files],
        "manifest_files": [str(p.resolve()).replace("\\", "/") for p in manifest_files],
        "alignment_reports": alignment_reports,
    }


# ==============================================================================
# BỘ KIỂM THỬ ĐƠN VỊ & TÍCH HỢP (SELF-CONTAINED UNIT & INTEGRATION TESTS)
# ==============================================================================
def run_unit_tests() -> None:
    """
    Kiểm thử toàn diện 10 cổng nghiệm thu cho `extract_features.py`:
      [Test 1]  Quy ước đặt tên file cache, `compute_manifest_hash` (64 hex) & chặn `clip_pretrained=None`.
      [Test 2]  Nạp thực tế cả 3 nhánh (`best_spatial.pth`, `best_spectral.pth`, OpenAI `CLIP`)
                và kiểm tra khóa cứng `.eval()` + `requires_grad=False`.
      [Test 3]  Trích xuất một lượt (Single-Pass) cả 3 nhánh trên tập `train` thật (`max_batches=2`,
                `include_flip=True`) — kiểm tra `features`, `features_flip`, `preprocess_config` & CLIP `L2=1`.
      [Test 4]  Lưu Atomic kèm đọc lại file tạm (`verify_readback=True`), `export_split_manifest_csv`
                và đối soát căn chỉnh chéo 3 nhánh (`verify_cross_branch_alignment`).
      [Test 5]  Phát hiện lệch nhãn, lệch thứ tự ảnh, trùng lặp `filepaths` và rò rỉ Unseen (`SD3`) vào `train`.
      [Test 6]  Chứng minh nén JPEG (`fixed_jpeg_quality=70`) thay đổi thực tế pixel đầu vào và đặc trưng Test.
      [Test 7]  Kiểm chứng tính tất định (Determinism) qua 2 lượt chạy FP32 & đối chiếu sai số AMP vs FP32.
      [Test 8]  Cache invalidation khi cấu hình trích xuất đổi (`has_flip`, `use_amp`).
      [Test 9]  Cache invalidation khi trọng số checkpoint `.pth` thay đổi mà đường dẫn giữ nguyên.
      [Test 10] Toàn vẹn metadata mở rộng (`label_b`, `group_id` trong `manifest_hash` & ánh xạ `model_name <-> label_b`).
    """
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ ĐƠN VỊ & TÍCH HỢP (10 CỔNG): extract_features.py")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # [Test 1] Kiểm tra đường dẫn file cache, manifest_hash & chặn CLIP không pretrained
    # ------------------------------------------------------------------
    print("\n[Test 1] Kiểm tra quy ước file cache, compute_manifest_hash (64 hex) & guard CLIP pretrained...")
    p_clean = get_cache_filepath(DEFAULT_CACHE_DIR, "spatial", "train")
    p_jpeg = get_cache_filepath(DEFAULT_CACHE_DIR, "semantic", "test", jpeg_quality=70)
    assert p_clean.name == "spatial_train.pt", f"Tên file sai: {p_clean.name}"
    assert p_jpeg.name == "semantic_test_jpeg70.pt", f"Tên file sai: {p_jpeg.name}"

    h1 = compute_manifest_hash(
        ["train/a.jpg", "train/b.jpg"], [0.0, 1.0], ["MS_COCO", "DALL-E_3"],
        sample_indices=[0, 1], labels_b=[0, 4], group_ids=[10, 10],
    )
    h2 = compute_manifest_hash(
        ["train/a.jpg", "train/b.jpg"], [0.0, 1.0], ["MS_COCO", "DALL-E_3"],
        sample_indices=[0, 1], labels_b=[0, 4], group_ids=[10, 10],
    )
    h3 = compute_manifest_hash(
        ["train/b.jpg", "train/a.jpg"], [0.0, 1.0], ["MS_COCO", "DALL-E_3"],
        sample_indices=[0, 1], labels_b=[0, 4], group_ids=[10, 10],
    )
    assert len(h1) == 64 and h1 == h2 and h1 != h3

    caught_no_pretrained = False
    try:
        load_branch_extractor(branch="semantic", device="cpu", clip_pretrained="none", allow_untrained=False)
    except ValueError:
        caught_no_pretrained = True
    assert caught_no_pretrained, "Phải chặn clip_pretrained='none' khi allow_untrained=False!"
    print(f"  -> manifest_hash={h1[:16]}... (len=64, đổi thứ tự -> {h3[:16]}...), chặn clip_pretrained='none' OK.")
    print("  [PASS] Quy ước tên file, hàm băm manifest 64 ký tự và kiểm soát pretrained chính xác 100%.")

    # ------------------------------------------------------------------
    # [Test 2] Kiểm tra nạp 3 nhánh trích xuất (Spatial, Semantic, Spectral)
    # ------------------------------------------------------------------
    print("\n[Test 2] Kiểm tra nạp 3 nhánh từ best_spatial.pth, best_spectral.pth và OpenAI CLIP...")
    models: Dict[str, BranchBase] = {}
    ckpt_infos: Dict[str, Dict[str, Any]] = {}
    for br in VALID_BRANCHES:
        m, info = load_branch_extractor(
            branch=br,
            device=device,
            spatial_ckpt=DEFAULT_SPATIAL_CKPT,
            spectral_ckpt=DEFAULT_SPECTRAL_CKPT,
            allow_untrained=False,
        )
        assert isinstance(m, BranchBase)
        assert not m.training, f"Nhánh {br} chưa ở chế độ .eval()!"
        assert all(not p.requires_grad for p in m.parameters()), (
            f"Nhánh {br} vẫn còn tham số requires_grad=True!"
        )
        assert m.feature_dim == BRANCH_FEATURE_DIMS[br]
        models[br] = m
        ckpt_infos[br] = info
        print(
            f"  -> Nhánh '{br:<8s}': dim={m.feature_dim:<4d}, "
            f"source={info['source']}, epoch={info.get('epoch')}"
        )
    print("  [PASS] Cả 3 nhánh nạp trọng số thật, khóa cứng tham số và đặt .eval() thành công.")

    # ------------------------------------------------------------------
    # [Test 3] Trích xuất Single-Pass 3 nhánh trên tập Train (có Flip + preprocess_config)
    # ------------------------------------------------------------------
    print("\n[Test 3] Kiểm tra trích xuất Single-Pass 3 nhánh trên tập Train (include_flip=True)...")
    train_payloads = extract_split_features(
        models=models,
        checkpoint_infos=ckpt_infos,
        split="train",
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=8,
        num_workers=0,
        device=device,
        include_flip=True,
        max_batches=2,
        log_interval=1,
    )
    for br in VALID_BRANCHES:
        pl = train_payloads[br]
        expected_dim = BRANCH_FEATURE_DIMS[br]
        assert pl["num_samples"] == 16
        assert pl["has_flip"] is True
        assert pl["features"].shape == (16, expected_dim)
        assert pl["features_flip"] is not None and pl["features_flip"].shape == (16, expected_dim)
        assert pl["features"].dtype == torch.float32
        assert pl["features_flip"].dtype == torch.float32
        assert not pl["features"].requires_grad and pl["features"].grad_fn is None
        assert "preprocess_config" in pl and pl["preprocess_config"]["mode"] == "extract"
        # Ảnh lật ngang phải sinh ra vector đặc trưng khác ảnh gốc
        diff = float((pl["features"] - pl["features_flip"]).abs().max().item())
        assert diff > 1e-5, f"Đặc trưng gốc và lật ngang của nhánh {br} giống hệt nhau bất thường!"
        print(
            f"  -> Train '{br:<8s}': shape={tuple(pl['features'].shape)}, "
            f"flip_shape={tuple(pl['features_flip'].shape)}, max|orig-flip|={diff:.5f}, "
            f"norm={pl['preprocess_config']['normalization']}"
        )
    print("  [PASS] Trích xuất Single-Pass tập Train (Orig + Flip + preprocess_config) chuẩn xác.")

    # ------------------------------------------------------------------
    # [Test 4] Lưu Atomic (kèm Read-back Verify), xuất CSV Manifest & đối soát 3 nhánh
    # ------------------------------------------------------------------
    print("\n[Test 4] Kiểm tra lưu Atomic (.pt + verify_readback), xuất manifest.csv & alignment...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        reloaded: Dict[str, Dict[str, Any]] = {}
        for br in VALID_BRANCHES:
            out_p = get_cache_filepath(tmp_dir, br, "train")
            save_feature_cache_atomic(train_payloads[br], out_p, verify_readback=True)
            assert out_p.is_file()
            reloaded[br] = load_feature_cache(out_p, expected_branch=br, expected_split="train")
            assert torch.equal(reloaded[br]["features"], train_payloads[br]["features"])
            assert torch.equal(reloaded[br]["features_flip"], train_payloads[br]["features_flip"])

        align_report = verify_cross_branch_alignment(reloaded)
        assert align_report["aligned"] is True
        assert align_report["num_samples"] == 16

        csv_p = Path(tmp_dir) / "manifests" / "train_manifest.csv"
        export_split_manifest_csv(reloaded["spatial"], csv_p)
        assert csv_p.is_file()
        df_csv = pd.read_csv(csv_p)
        assert len(df_csv) == 16
        assert list(df_csv["filepath"]) == reloaded["spatial"]["filepaths"]
        print(f"  -> Alignment report: {align_report}")
        print(f"  -> Manifest CSV verified: {csv_p.name} ({len(df_csv)} rows)")
    print("  [PASS] Lưu Atomic, Read-back Verify, CSV Manifest và đối soát 3 nhánh thành công 100%.")

    # ------------------------------------------------------------------
    # [Test 5] Phát hiện lệch nhãn, lệch thứ tự, trùng ảnh & rò rỉ Unseen vào train
    # ------------------------------------------------------------------
    print("\n[Test 5] Kiểm tra chặn lệch nhãn, lệch thứ tự, trùng ảnh & rò rỉ Unseen vào train...")
    # 5a. Đổi label_a nhưng giữ nguyên manifest_hash cũ
    tampered = copy.deepcopy(train_payloads)
    tampered["spectral"]["label_a"][0] = 1.0 - tampered["spectral"]["label_a"][0]
    caught_hash_error = False
    try:
        verify_cross_branch_alignment(tampered)
    except ValueError:
        caught_hash_error = True
    assert caught_hash_error, "Không bắt được lỗi khi label_a bị thay đổi!"

    # 5b. Đảo thứ tự 2 filepaths của spectral (tính lại hash để test check chéo)
    tampered2 = copy.deepcopy(train_payloads)
    fps = tampered2["spectral"]["filepaths"]
    fps[0], fps[1] = fps[1], fps[0]
    tampered2["spectral"]["manifest_hash"] = compute_manifest_hash(
        fps,
        tampered2["spectral"]["label_a"],
        tampered2["spectral"]["model_names"],
        sample_indices=tampered2["spectral"]["sample_idx"],
        labels_b=tampered2["spectral"]["label_b"],
        group_ids=tampered2["spectral"]["group_id"],
    )
    caught_cross_error = False
    try:
        verify_cross_branch_alignment(tampered2)
    except ValueError:
        caught_cross_error = True
    assert caught_cross_error, "Không bắt được lỗi lệch thứ tự filepaths giữa các nhánh!"

    # 5c. Trùng lặp filepath trong cùng 1 cache (giống lỗi BalancedGroupSampler)
    tampered3 = copy.deepcopy(train_payloads["spatial"])
    tampered3["filepaths"][1] = tampered3["filepaths"][0]
    tampered3["label_a"][1] = tampered3["label_a"][0]
    tampered3["label_b"][1] = tampered3["label_b"][0]
    tampered3["model_names"][1] = tampered3["model_names"][0]
    tampered3["manifest_hash"] = compute_manifest_hash(
        tampered3["filepaths"],
        tampered3["label_a"],
        tampered3["model_names"],
        sample_indices=tampered3["sample_idx"],
        labels_b=tampered3["label_b"],
        group_ids=tampered3["group_id"],
    )
    caught_dup_error = False
    try:
        validate_feature_cache_payload(tampered3, expected_branch="spatial", expected_split="train")
    except ValueError as exc:
        if "trùng lặp" in str(exc):
            caught_dup_error = True
    assert caught_dup_error, "Không bắt được lỗi trùng lặp filepath!"

    # 5d. Rò rỉ nguồn Unseen ('Stable_Diffusion_3') vào tập Train
    tampered4 = copy.deepcopy(train_payloads["spatial"])
    unseen_name = UNSEEN_GENERATORS[0]
    tampered4["model_names"][0] = unseen_name
    tampered4["label_a"][0] = 1.0
    tampered4["label_b"][0] = MODEL_NAME_TO_LABEL_B[unseen_name]
    tampered4["manifest_hash"] = compute_manifest_hash(
        tampered4["filepaths"],
        tampered4["label_a"],
        tampered4["model_names"],
        sample_indices=tampered4["sample_idx"],
        labels_b=tampered4["label_b"],
        group_ids=tampered4["group_id"],
    )
    caught_leak_error = False
    try:
        validate_feature_cache_payload(tampered4, expected_branch="spatial", expected_split="train")
    except ValueError as exc:
        if "không hợp lệ trên split='train'" in str(exc):
            caught_leak_error = True
    assert caught_leak_error, "Không bắt được lỗi rò rỉ model Unseen vào tập train!"
    print("  [PASS] Bộ đối soát bắt chính xác 100% lệch nhãn, lệch ảnh, trùng ảnh và rò rỉ Unseen.")

    # ------------------------------------------------------------------
    # [Test 6] Chứng minh nén JPEG cố định thực sự thay đổi đầu vào & đặc trưng Test
    # ------------------------------------------------------------------
    print("\n[Test 6] Kiểm tra nén JPEG thực tế trên tập Test (Clean vs fixed_jpeg_quality=70)...")
    clean_loader = build_extract_loader(
        "test", batch_size=8, num_workers=0, active_branches=("spatial", "spectral"), fixed_jpeg_quality=None
    )
    jpeg_loader = build_extract_loader(
        "test", batch_size=8, num_workers=0, active_branches=("spatial", "spectral"), fixed_jpeg_quality=70
    )
    b_clean = next(iter(clean_loader))
    b_jpeg = next(iter(jpeg_loader))
    _shutdown_dataloader_workers(clean_loader)
    _shutdown_dataloader_workers(jpeg_loader)
    pixel_diff = float((b_clean["spatial"] - b_jpeg["spatial"]).abs().max().item())
    spec_diff = float((b_clean["spectral"] - b_jpeg["spectral"]).abs().max().item())
    assert pixel_diff > 1e-4 and spec_diff > 1e-4, "Ảnh đầu vào JPEG không thay đổi so với ảnh sạch!"

    test_clean_payloads = extract_split_features(
        models=models,
        checkpoint_infos=ckpt_infos,
        split="test",
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=8,
        num_workers=0,
        device=device,
        use_amp=False,
        amp_dtype=torch.float32,
        include_flip=False,
        fixed_jpeg_quality=None,
        max_batches=1,
        log_interval=0,
    )
    test_jpeg_payloads = extract_split_features(
        models=models,
        checkpoint_infos=ckpt_infos,
        split="test",
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=8,
        num_workers=0,
        device=device,
        use_amp=False,
        amp_dtype=torch.float32,
        include_flip=False,
        fixed_jpeg_quality=70,
        max_batches=1,
        log_interval=1,
    )
    for br in VALID_BRANCHES:
        pl = test_jpeg_payloads[br]
        assert pl["num_samples"] == 8
        assert pl["has_flip"] is False
        assert pl["features_flip"] is None
        assert pl["jpeg_quality"] == 70
        feat_diff = float((pl["features"] - test_clean_payloads[br]["features"]).abs().max().item())
        assert feat_diff > 1e-5, f"Feature nhánh {br} sau nén JPEG q=70 giống hệt ảnh sạch!"
    print(f"  -> Pixel delta(q=70)={pixel_diff:.4f} | Spectral delta={spec_diff:.4f} -> [PASS]")

    # ------------------------------------------------------------------
    # [Test 7] Kiểm chứng tính tất định (Determinism) & Đối chiếu sai số AMP vs FP32
    # ------------------------------------------------------------------
    print("\n[Test 7] Kiểm chứng tính tất định (2 lượt FP32) & đối chiếu sai số AMP vs FP32...")
    fp32_run1 = extract_split_features(
        models=models,
        checkpoint_infos=ckpt_infos,
        split="val",
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=8,
        num_workers=0,
        device=device,
        use_amp=False,
        amp_dtype=torch.float32,
        include_flip=False,
        max_batches=1,
        log_interval=0,
    )
    fp32_run2 = extract_split_features(
        models=models,
        checkpoint_infos=ckpt_infos,
        split="val",
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=8,
        num_workers=0,
        device=device,
        use_amp=False,
        amp_dtype=torch.float32,
        include_flip=False,
        max_batches=1,
        log_interval=0,
    )
    auto_amp, auto_dtype, _ = get_optimal_amp_config(device)
    amp_cmp_dtype = auto_dtype if auto_amp else torch.bfloat16
    amp_run = extract_split_features(
        models=models,
        checkpoint_infos=ckpt_infos,
        split="val",
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=8,
        num_workers=0,
        device=device,
        use_amp=True,
        amp_dtype=amp_cmp_dtype,
        include_flip=False,
        max_batches=1,
        log_interval=0,
    )

    for br in VALID_BRANCHES:
        f1 = fp32_run1[br]["features"]
        f2 = fp32_run2[br]["features"]
        fa = amp_run[br]["features"]
        det_diff = float((f1 - f2).abs().max().item())
        assert torch.equal(f1, f2), f"Nhánh {br} không tất định giữa 2 lần chạy (det_diff={det_diff})!"

        cos_min = float(F.cosine_similarity(f1, fa, dim=-1).min().item())
        amp_max_diff = float((f1 - fa).abs().max().item())
        assert cos_min >= 0.995, (
            f"Sai số AMP ({amp_cmp_dtype}) vs FP32 của nhánh {br} quá lớn: cos_min={cos_min:.6f}"
        )
        print(
            f"  -> Nhánh '{br:<8s}': det_diff={det_diff:.1e} | "
            f"AMP({amp_cmp_dtype}) vs FP32 cos_min={cos_min:.6f}, max|diff|={amp_max_diff:.5f}"
        )
    print("  [PASS] Tất định tuyệt đối 100% và AMP khớp cực sát FP32.")

    # ------------------------------------------------------------------
    # [Test 8] Cache invalidation khi cấu hình trích xuất đổi (has_flip / use_amp)
    # ------------------------------------------------------------------
    print("\n[Test 8] Kiểm tra cache invalidation khi thay đổi cấu hình trích xuất...")
    with tempfile.TemporaryDirectory() as tmp_cfg_dir:
        tmp_cfg_path = Path(tmp_cfg_dir)
        tmp_cache = tmp_cfg_path / "cache"
        tmp_ckpt = tmp_cfg_path / "best_spatial_copy.pth"
        tmp_ckpt.write_bytes(Path(DEFAULT_SPATIAL_CKPT).read_bytes())

        # Lần 1: Tạo cache train_include_flip=False, use_amp=False (1 batch cho nhanh)
        s1 = run_extraction_pipeline(
            branches=("spatial",),
            splits=("train",),
            cache_dir=tmp_cache,
            dataset_root=DEFAULT_DATASET_ROOT,
            spatial_ckpt=tmp_ckpt,
            batch_size=8,
            num_workers=0,
            device=device,
            use_amp=False,
            train_include_flip=False,
            save_manifests=False,
            overwrite=False,
            max_batches=1,
            log_interval=0,
        )
        assert len(s1["saved_files"]) == 1, f"Kỳ vọng lưu 1 file cache mới, nhận được {s1['saved_files']}"

        # Lần 2: Chạy lại đúng cấu hình cũ -> phải [SKIP] (saved_files rỗng)
        s2 = run_extraction_pipeline(
            branches=("spatial",),
            splits=("train",),
            cache_dir=tmp_cache,
            dataset_root=DEFAULT_DATASET_ROOT,
            spatial_ckpt=tmp_ckpt,
            batch_size=8,
            num_workers=0,
            device=device,
            use_amp=False,
            train_include_flip=False,
            save_manifests=False,
            overwrite=False,
            max_batches=1,
            log_interval=0,
        )
        assert len(s2["saved_files"]) == 0, f"Cấu hình giống hệt đáng lẽ phải SKIP nhưng vẫn ghi: {s2['saved_files']}"

        # Lần 3: Đổi train_include_flip=True (dù overwrite=False) -> phải trích xuất lại
        s3 = run_extraction_pipeline(
            branches=("spatial",),
            splits=("train",),
            cache_dir=tmp_cache,
            dataset_root=DEFAULT_DATASET_ROOT,
            spatial_ckpt=tmp_ckpt,
            batch_size=8,
            num_workers=0,
            device=device,
            use_amp=False,
            train_include_flip=True,
            save_manifests=False,
            overwrite=False,
            max_batches=1,
            log_interval=0,
        )
        assert len(s3["saved_files"]) == 1, "Đổi train_include_flip=True nhưng pipeline không trích xuất lại cache!"
        cached_flip = load_feature_cache(
            get_cache_filepath(tmp_cache, "spatial", "train"),
            expected_branch="spatial",
            expected_split="train",
        )
        assert cached_flip["has_flip"] is True
        assert cached_flip["extraction_signature"]["has_flip"] is True

        # Kiểm tra cảnh báo RuntimeWarning khi bật use_amp=True trên CPU (fallback FP32 -> khớp cache Lần 3)
        with warnings.catch_warnings(record=True) as caught_warns:
            warnings.simplefilter("always")
            s_cpu = run_extraction_pipeline(
                branches=("spatial",),
                splits=("train",),
                cache_dir=tmp_cache,
                dataset_root=DEFAULT_DATASET_ROOT,
                spatial_ckpt=tmp_ckpt,
                batch_size=8,
                num_workers=0,
                device="cpu",
                use_amp=True,
                train_include_flip=True,
                save_manifests=False,
                overwrite=False,
                max_batches=1,
                log_interval=0,
            )
            assert any("không hỗ trợ CUDA AMP" in str(w.message) for w in caught_warns), (
                "Thiếu cảnh báo RuntimeWarning khi đặt use_amp=True trên CPU!"
            )
            assert len(s_cpu["saved_files"]) == 0, "Fallback FP32 trên CPU phải khớp chữ ký Lần 3 và SKIP!"

        # Lần 4 (khi có CUDA AMP): đổi use_amp=True -> phát hiện lệch extraction_signature & trích lại
        if auto_amp:
            s4 = run_extraction_pipeline(
                branches=("spatial",),
                splits=("train",),
                cache_dir=tmp_cache,
                dataset_root=DEFAULT_DATASET_ROOT,
                spatial_ckpt=tmp_ckpt,
                batch_size=8,
                num_workers=0,
                device=device,
                use_amp=True,
                train_include_flip=True,
                save_manifests=False,
                overwrite=False,
                max_batches=1,
                log_interval=0,
            )
            assert len(s4["saved_files"]) == 1, "Đổi use_amp=True trên CUDA nhưng pipeline không trích xuất lại!"

        # ------------------------------------------------------------------
        # [Test 9] Cache invalidation khi file checkpoint đổi nội dung (cùng đường dẫn)
        # ------------------------------------------------------------------
        print("\n[Test 9] Kiểm tra cache invalidation khi trọng số checkpoint .pth thay đổi (cùng đường dẫn)...")
        before_cache = load_feature_cache(
            get_cache_filepath(tmp_cache, "spatial", "train"),
            expected_branch="spatial",
            expected_split="train",
        )
        old_sha = before_cache["extraction_signature"]["checkpoint_sha256"]

        # Ghi đè nhẹ trọng số vào CHÍNH file tmp_ckpt cũ (cùng path, khác checkpoint_sha256)
        ckpt_obj = torch.load(tmp_ckpt, map_location="cpu", weights_only=False)
        first_float_key = next(
            k for k, v in ckpt_obj["model_state_dict"].items()
            if torch.is_tensor(v) and v.is_floating_point()
        )
        ckpt_obj["model_state_dict"][first_float_key] = ckpt_obj["model_state_dict"][first_float_key] + 1e-4
        ckpt_obj["_cache_invalidation_rev"] = 2
        torch.save(ckpt_obj, tmp_ckpt)

        s_ckpt = run_extraction_pipeline(
            branches=("spatial",),
            splits=("train",),
            cache_dir=tmp_cache,
            dataset_root=DEFAULT_DATASET_ROOT,
            spatial_ckpt=tmp_ckpt,
            batch_size=8,
            num_workers=0,
            device=device,
            use_amp=auto_amp,
            train_include_flip=True,
            save_manifests=False,
            overwrite=False,
            max_batches=1,
            log_interval=0,
        )
        assert len(s_ckpt["saved_files"]) == 1, (
            "Trọng số checkpoint tại cùng đường dẫn đã thay đổi nhưng pipeline vẫn dùng lại cache cũ!"
        )
        after_cache = load_feature_cache(
            get_cache_filepath(tmp_cache, "spatial", "train"),
            expected_branch="spatial",
            expected_split="train",
        )
        new_sha = after_cache["extraction_signature"]["checkpoint_sha256"]
        assert old_sha and new_sha and old_sha != new_sha, (
            f"checkpoint_sha256 phải thay đổi sau khi ghi đè checkpoint (old={old_sha}, new={new_sha})!"
        )
        print(f"  -> Bắt thành công đổi checkpoint cùng path: sha256 {old_sha[:12]}... -> {new_sha[:12]}...")
        print("  [PASS] Cơ chế kiểm tra extraction_signature (has_flip, use_amp, checkpoint_sha256) hoạt động chuẩn 100%.")

    # ------------------------------------------------------------------
    # [Test 10] Toàn vẹn metadata mở rộng: group_id trong manifest_hash & model_name <-> label_b
    # ------------------------------------------------------------------
    print("\n[Test 10] Kiểm tra bảo vệ group_id trong manifest_hash và nhất quán model_name <-> label_b...")
    # 10a. Sửa group_id nhưng giữ nguyên manifest_hash cũ -> bắt buộc báo lỗi hash
    tampered_gid = copy.deepcopy(train_payloads["spatial"])
    tampered_gid["group_id"][0] = tampered_gid["group_id"][0] + 999
    caught_gid_hash = False
    try:
        validate_feature_cache_payload(tampered_gid, expected_branch="spatial", expected_split="train")
    except ValueError as exc:
        if "manifest_hash" in str(exc):
            caught_gid_hash = True
    assert caught_gid_hash, "Không bắt được lỗi sửa group_id khi giữ nguyên manifest_hash!"

    # 10b. Làm lệch label_b so với model_name (dù tự tính lại manifest_hash mới) -> bắt lỗi ánh xạ MODEL_NAME_TO_LABEL_B
    tampered_lblb = copy.deepcopy(train_payloads["spatial"])
    tampered_lblb["label_b"][0] = (int(tampered_lblb["label_b"][0]) + 1) % len(MODEL_NAME_TO_LABEL_B)
    tampered_lblb["manifest_hash"] = compute_manifest_hash(
        tampered_lblb["filepaths"],
        tampered_lblb["label_a"],
        tampered_lblb["model_names"],
        sample_indices=tampered_lblb["sample_idx"],
        labels_b=tampered_lblb["label_b"],
        group_ids=tampered_lblb["group_id"],
    )
    caught_lblb_mismatch = False
    try:
        validate_feature_cache_payload(tampered_lblb, expected_branch="spatial", expected_split="train")
    except ValueError as exc:
        if "label_b" in str(exc):
            caught_lblb_mismatch = True
    assert caught_lblb_mismatch, "Không bắt được lỗi mâu thuẫn model_name <-> label_b!"
    print("  [PASS] manifest_hash bảo vệ cả 6 trường metadata và MODEL_NAME_TO_LABEL_B khóa chặt label_b.")

    # Giải phóng bộ nhớ sau kiểm thử
    models.clear()
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("TẤT CẢ 10 NHÓM KIỂM THỬ (PHỦ ĐỦ TOÀN BỘ CỔNG NGHIỆM THU) ĐỀU PASS 100%!")
    print("=" * 80)


# ==============================================================================
# GIAO DIỆN DÒNG LỆNH (CLI)
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trích xuất và lưu đệm đặc trưng (Feature Caching) cho 3 nhánh Spatial, Semantic, Spectral."
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=list(VALID_BRANCHES),
        choices=list(VALID_BRANCHES),
        help="Danh sách các nhánh cần trích xuất (mặc định: spatial semantic spectral).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(VALID_SPLITS),
        choices=list(VALID_SPLITS),
        help="Danh sách các tập dữ liệu cần trích xuất (mặc định: train val test).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help=f"Thư mục lưu các file cache .pt (mặc định: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Thư mục gốc của bộ dữ liệu Defactify (mặc định: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Đường dẫn tùy chọn tới metadata.csv (mặc định tự suy từ dataset-root).",
    )
    parser.add_argument(
        "--spatial-ckpt",
        type=str,
        default=str(DEFAULT_SPATIAL_CKPT),
        help=f"Đường dẫn tới best_spatial.pth (mặc định: {DEFAULT_SPATIAL_CKPT}).",
    )
    parser.add_argument(
        "--spectral-ckpt",
        type=str,
        default=str(DEFAULT_SPECTRAL_CKPT),
        help=f"Đường dẫn tới best_spectral.pth (mặc định: {DEFAULT_SPECTRAL_CKPT}).",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=DEFAULT_CLIP_PRETRAINED,
        help="Tag trọng số tiền huấn luyện OpenAI CLIP (mặc định: 'openai').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size khi trích xuất đặc trưng (mặc định: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Số tiến trình đọc dữ liệu của DataLoader (mặc định: 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị chạy ('cuda' hoặc 'cpu'; mặc định tự động phát hiện).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Tắt AMP (bfloat16/float16), chạy trích xuất ở độ chính xác FP32 thuần.",
    )
    parser.add_argument(
        "--no-train-flip",
        action="store_true",
        help="Tắt trích xuất góc nhìn lật ngang (force_flip=True) trên tập train.",
    )
    parser.add_argument(
        "--no-manifests",
        action="store_true",
        help="Không xuất file CSV manifest (chỉ lưu .pt đã tích hợp sẵn metadata).",
    )
    parser.add_argument(
        "--jpeg-qualities",
        type=str,
        default=None,
        help="Danh sách chất lượng JPEG trên tập test cách nhau dấu phẩy, VD: '90,70,50,30'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè các file cache .pt đã tồn tại.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Giới hạn số batch trích xuất (dùng khi chạy thử nghiệm nhanh / smoke test).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="In tiến độ sau mỗi N batch (mặc định: 50).",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Chạy bộ kiểm thử đơn vị và tích hợp (7 cổng / 12 tiêu chí) rồi thoát.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_tests:
        run_unit_tests()
        return

    jpeg_list: Optional[List[int]] = None
    if args.jpeg_qualities:
        jpeg_list = [int(x.strip()) for x in args.jpeg_qualities.split(",") if x.strip()]

    summary = run_extraction_pipeline(
        branches=args.branches,
        splits=args.splits,
        cache_dir=args.cache_dir,
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        spatial_ckpt=args.spatial_ckpt,
        spectral_ckpt=args.spectral_ckpt,
        clip_pretrained=args.clip_pretrained,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        use_amp=False if args.no_amp else None,
        train_include_flip=not args.no_train_flip,
        test_jpeg_qualities=jpeg_list,
        save_manifests=not args.no_manifests,
        overwrite=args.overwrite,
        max_batches=args.max_batches,
        log_interval=args.log_interval,
    )

    print("\n" + "=" * 80)
    print("HOÀN TẤT TRÍCH XUẤT & LƯU ĐỆM ĐẶC TRƯNG (FEATURE CACHING)")
    print("=" * 80)
    print(f"  - Thư mục cache : {summary['cache_dir']}")
    print(f"  - Số file .pt mới: {len(summary['saved_files'])}")
    for fp in summary["saved_files"]:
        print(f"      + {fp}")
    if summary.get("manifest_files"):
        print(f"  - Bảng manifest  : {len(summary['manifest_files'])} file CSV")
        for mp in summary["manifest_files"]:
            print(f"      + {mp}")
    print("=" * 80)


if __name__ == "__main__":
    main()

