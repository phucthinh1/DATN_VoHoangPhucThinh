# -*- coding: utf-8 -*-
"""
code/system/train/train_fusion.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Bộ điều phối Huấn luyện & Đánh giá Giai đoạn 2 (Stage 2 Feature-Space Training & Full Evaluation)
hoạt động trực tiếp trên các file lưu đệm đặc trưng (`.pt`) do `extract_features.py` trích xuất:

1. Huấn luyện Nhánh 3 — Semantic Linear Probe (`--mode probe` hoặc `--head linear --branches semantic`):
   - Huấn luyện tầng `Linear(512 -> 1)` (513 tham số) trên đặc trưng CLIP ViT-B/16 đã chuẩn hóa L2.
   - Lưu `best_semantic.pth` tại `code/system/checkpoints/semantic_baseline/best_semantic.pth`
     (đặt ngang hàng với `spatial_baseline/best_spatial.pth` và `spectral_baseline/best_spectral.pth`
     để nạp trực tiếp vào `predict_single.ipynb`), đồng thời lưu đầy đủ `last_semantic.pth`,
     `history.csv`, `history_semantic.json`, `config.json` tại `checkpoints/stage2/semantic_probe/`.

2. Huấn luyện Fusion MLP Baseline (`--mode fusion` hoặc `--head mlp --branches spatial semantic spectral`):
   - Kiến trúc: `LayerNorm` riêng từng nhánh (`1280, 512, 256`) -> `2048 -> 512 -> 128 -> 1`,
     `branch_dropout_p = 0.0`.
   - Lưu tại `code/system/checkpoints/stage2/fusion_baseline/best_fusion.pth`.

3. Thực nghiệm Ablation Study 7 tổ hợp nhánh (`--mode ablation`):
   - Huấn luyện ĐỘC LẬP từng tổ hợp trong 7 tổ hợp (`spatial`, `semantic`, `spectral`,
     `spatial_semantic`, `spatial_spectral`, `semantic_spectral`, `all_branches`) trên tập Train Known,
     lựa chọn checkpoint tốt nhất hoàn toàn theo Validation Known `ROC-AUC` (tie-break `Balanced Accuracy`).
   - Lưu tách biệt tại `code/system/checkpoints/stage2/ablation/<combo_slug>/`.

4. Thực nghiệm Branch Dropout `p = 0.15` (`--mode dropout`):
   - Chỉ áp dụng cho tổ hợp đủ 3 nhánh (`min_active_branches = 2`), chỉ bật khi `training=True`
     và tắt 100% khi Validation/Test.
   - Lưu tại `code/system/checkpoints/stage2/branch_dropout/best_fusion_dropout.pth`.

5. Đánh giá Toàn diện Clean Test (Known / Unseen / Per-Generator) & JPEG Robustness (`90, 70, 50, 30`)
   (`--mode eval` hoặc nằm trong `--mode all`):
   - Chấm điểm đồng thời 3 Checkpoint Đơn nhánh (`best_spatial.pth`, `best_semantic.pth`,
     `best_spectral.pth` thông qua `LinearProbeHead.load_from_branch_checkpoint`) đặt cạnh
     `FusionMLP Baseline`, `FusionMLP Branch Dropout (p=0.15)` và 7 mô hình `Ablation`.
   - Xuất báo cáo chuẩn hóa tại `code/system/checkpoints/stage2/evaluation/`:
     `clean_test.json`, `per_generator.json`, `jpeg_robustness.json`, `ablation_comparison.csv`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# ĐỊNH TUYẾN ĐƯỜNG DẪN HỆ THỐNG
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent               # d:/nam_4/DATN/code/system/train
SYSTEM_DIR = CURRENT_DIR.parent                             # d:/nam_4/DATN/code/system
BRANCH_DIR = SYSTEM_DIR / "branch"                          # d:/nam_4/DATN/code/system/branch
DATA_DIR = SYSTEM_DIR / "prepare_data"                      # d:/nam_4/DATN/code/system/prepare_data
CODE_DIR = SYSTEM_DIR.parent                                # d:/nam_4/DATN/code
PROJECT_ROOT = CODE_DIR.parent                              # d:/nam_4/DATN

for _p in (str(CURRENT_DIR), str(SYSTEM_DIR), str(BRANCH_DIR), str(DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from balance_sample import BalancedGroupSampler  # noqa: E402
from engine import (  # noqa: E402
    EarlyStopping,
    checkpoint_consistency_check,
    evaluate,
    fit,
    get_optimal_amp_config,
    load_checkpoint,
    overfit_single_batch_check,
    sanity_check_gate,
)
from extract_features import (  # noqa: E402
    BRANCH_FEATURE_DIMS,
    DEFAULT_CACHE_DIR,
    DEFAULT_SPATIAL_CKPT,
    DEFAULT_SPECTRAL_CKPT,
    MODEL_NAME_TO_LABEL_B,
    VALID_BRANCHES,
    VALID_SPLITS,
    build_extraction_signature,
    build_preprocess_config,
    compute_manifest_hash,
    get_cache_filepath,
    load_feature_cache,
    save_feature_cache_atomic,
    verify_cross_branch_alignment,
)
from Fusion_mlp import (  # noqa: E402
    ALL_ABLATION_COMBINATIONS,
    BRANCH_ORDER,
    FusionMLP,
    LinearProbeHead,
    build_stage2_model,
    normalize_branch_selection,
)
from metrics import (  # noqa: E402
    ALL_AI_GENERATORS,
    KNOWN_GENERATORS,
    METRIC_KEYS,
    REAL_MODEL_NAME,
    UNSEEN_GENERATORS,
    compute_binary_metrics,
    evaluate_by_generator,
    format_generator_report,
    format_metrics_summary,
)


# ==============================================================================
# HẰNG SỐ ĐƯỜNG DẪN CHECKPOINT & SIÊU THAM SỐ MẶC ĐỊNH GIAI ĐOẠN 2
# ==============================================================================
CHECKPOINTS_ROOT: Path = SYSTEM_DIR / "checkpoints"

# Đặt best_semantic.pth ngang hàng với spatial_baseline/best_spatial.pth và spectral_baseline/best_spectral.pth
DEFAULT_SEMANTIC_BASELINE_DIR: Path = CHECKPOINTS_ROOT / "semantic_baseline"
DEFAULT_SEMANTIC_CKPT: Path = DEFAULT_SEMANTIC_BASELINE_DIR / "best_semantic.pth"

# Toàn bộ các tệp thực nghiệm còn lại của Stage 2 nằm gọn trong checkpoints/stage2/
DEFAULT_STAGE2_DIR: Path = CHECKPOINTS_ROOT / "stage2"
DEFAULT_STAGE2_PROBE_DIR: Path = DEFAULT_STAGE2_DIR / "semantic_probe"
DEFAULT_STAGE2_FUSION_DIR: Path = DEFAULT_STAGE2_DIR / "fusion_baseline"
DEFAULT_STAGE2_ABLATION_DIR: Path = DEFAULT_STAGE2_DIR / "ablation"
DEFAULT_STAGE2_DROPOUT_DIR: Path = DEFAULT_STAGE2_DIR / "branch_dropout"
DEFAULT_STAGE2_EVAL_DIR: Path = DEFAULT_STAGE2_DIR / "evaluation"

DEFAULT_FUSION_CKPT: Path = DEFAULT_STAGE2_FUSION_DIR / "best_fusion.pth"
DEFAULT_DROPOUT_CKPT: Path = DEFAULT_STAGE2_DROPOUT_DIR / "best_fusion_dropout.pth"

DEFAULT_JPEG_QUALITIES: Tuple[int, ...] = (90, 70, 50, 30)

STAGE2_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "linear": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,      # Bài toán tuyến tính lồi 513 tham số trên đặc trưng L2-norm cố định
        "epochs": 30,            # 10 chu kỳ 3-epoch (phủ 100% ảnh AI mỗi 3 epochs)
        "patience": 6,           # Đúng 2 chu kỳ 3-epoch hoàn chỉnh
        "min_delta": 1e-4,
        "max_norm": 1.0,
        "batch_size": 32,
        "eval_batch_size": 256,
        "num_workers": 0,        # Cache .pt nằm sẵn trên RAM CPU -> 0 worker tránh IPC spawn trên Windows
        "flip_prob": 0.5,
        "linear_dropout": 0.0,
    },
    "mlp": {
        "lr": 5e-4,
        "weight_decay": 1e-4,    # Chỉ áp cho Linear.weight (2D); LayerNorm & bias được tách weight_decay=0.0
        "warmup_epochs": 1,      # 1 epoch LinearLR (0.1*lr -> lr) ổn định LayerNorm + MLP
        "epochs": 30,            # 10 chu kỳ 3-epoch
        "patience": 6,           # Đúng 2 chu kỳ 3-epoch hoàn chỉnh
        "min_delta": 1e-4,
        "max_norm": 1.0,
        "batch_size": 32,
        "eval_batch_size": 256,
        "num_workers": 0,
        "flip_prob": 0.5,
        "hidden_dims": (512, 128),
        "hidden_dropouts": (0.3, 0.2),
        "branch_dropout_p": 0.0,
        "min_active_branches": 2,
    },
}

ABLATION_COMBO_SLUGS: Dict[Tuple[str, ...], str] = {
    ("spatial",): "spatial",
    ("semantic",): "semantic",
    ("spectral",): "spectral",
    ("spatial", "semantic"): "spatial_semantic",
    ("spatial", "spectral"): "spatial_spectral",
    ("semantic", "spectral"): "semantic_spectral",
    ("spatial", "semantic", "spectral"): "all_branches",
}


# ==============================================================================
# CÁC HÀM TIỆN ÍCH CƠ SỞ (SEED, JSON, CSV, OPTIMIZER VALIDATION)
# ==============================================================================
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Thiết lập hạt giống ngẫu nhiên đồng bộ cho Python, NumPy và PyTorch CPU/CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def json_default(obj: Any) -> Any:
    """Chuyển đổi an toàn các kiểu NumPy / PyTorch / Path sang kiểu JSON."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Không thể tuần tự hóa đối tượng kiểu {type(obj)} sang JSON.")


def save_json(data: Mapping[str, Any], filepath: Union[str, Path]) -> Path:
    """Ghi từ điển ra file JSON định dạng UTF-8 đẹp mắt (atomic write qua file tạm)."""
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = fp.with_name(f".tmp_{fp.name}_{os.getpid()}_{uuid.uuid4().hex[:6]}")
    try:
        with open(tmp_fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)
        os.replace(tmp_fp, fp)
    finally:
        if tmp_fp.exists():
            try:
                tmp_fp.unlink()
            except OSError:
                pass
    return fp


def load_prior_history(history_path: Path, max_epoch_inclusive: int) -> List[Dict[str, Any]]:
    """Nạp lịch sử các epoch trước điểm ngắt khi `--resume` để không mất log cũ."""
    if not history_path.is_file() or max_epoch_inclusive < 0:
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        old_records = payload.get("history", [])
        return [r for r in old_records if int(r.get("epoch", -1)) <= max_epoch_inclusive]
    except Exception:
        return []


def export_history_csv(history: Sequence[Mapping[str, Any]], filepath: Union[str, Path]) -> Path:
    """Xuất lịch sử huấn luyện từng epoch (khớp cấu trúc phẳng của `engine.fit()`) ra file CSV."""
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_auc",
        "train_balanced_acc",
        "val_loss",
        "val_auc",
        "val_balanced_acc",
        "val_ap",
        "val_acc",
        "is_best",
        "elapsed_sec",
    ]
    with open(fp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in history:
            tr_m = rec.get("train_metrics", {}) or {}
            va_m = rec.get("val_metrics", {}) or {}
            lrs = rec.get("lrs", [])
            lr_val = float(lrs[0]) if lrs else float(rec.get("lr", 0.0))

            train_auc = rec.get("train_auc", tr_m.get("auc", 0.0))
            train_bacc = rec.get("train_balanced_acc", tr_m.get("balanced_acc", 0.0))
            val_auc = rec.get("val_auc", va_m.get("auc", 0.0))
            val_bacc = rec.get("val_balanced_acc", va_m.get("balanced_acc", 0.0))
            val_ap = rec.get("val_ap", va_m.get("ap", 0.0))
            val_acc = rec.get("val_acc", va_m.get("acc", 0.0))
            elapsed_val = rec.get("epoch_time", rec.get("elapsed_time", rec.get("elapsed_sec", 0.0)))

            writer.writerow(
                {
                    "epoch": int(rec.get("epoch", 0)),
                    "lr": f"{lr_val:.8e}",
                    "train_loss": f"{float(rec.get('train_loss', 0.0)):.6f}",
                    "train_auc": f"{float(train_auc):.6f}",
                    "train_balanced_acc": f"{float(train_bacc):.6f}",
                    "val_loss": f"{float(rec.get('val_loss', 0.0)):.6f}",
                    "val_auc": f"{float(val_auc):.6f}",
                    "val_balanced_acc": f"{float(val_bacc):.6f}",
                    "val_ap": f"{float(val_ap):.6f}",
                    "val_acc": f"{float(val_acc):.6f}",
                    "is_best": bool(rec.get("is_best", False)),
                    "elapsed_sec": f"{float(elapsed_val):.4f}",
                }
            )
    return fp


def get_ablation_combo_slug(branches: Union[str, Sequence[str]]) -> str:
    """Ánh xạ danh sách nhánh sang tên thư mục chuẩn của Ablation Study."""
    norm = normalize_branch_selection(branches)
    if norm in ABLATION_COMBO_SLUGS:
        return ABLATION_COMBO_SLUGS[norm]
    return "_".join(norm)


def parse_jpeg_qualities_arg(raw: Optional[Union[str, Sequence[int]]]) -> List[int]:
    """Chuẩn hóa danh sách mức nén JPEG (ví dụ `'90,70,50,30'` -> `[90, 70, 50, 30]`)."""
    if raw is None:
        return list(DEFAULT_JPEG_QUALITIES)
    if isinstance(raw, (list, tuple)):
        items = [int(x) for x in raw]
    else:
        txt = str(raw).strip()
        if not txt or txt.lower() in ("none", "off", "false"):
            return []
        items = [int(part.strip()) for part in txt.split(",") if part.strip()]

    dedup: List[int] = []
    for q in items:
        if not (1 <= q <= 100):
            raise ValueError(f"Mức chất lượng JPEG phải nằm trong [1, 100], nhận được: {q}")
        if q not in dedup:
            dedup.append(q)
    return dedup


def assert_optimizer_covers_trainable_params(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    context_label: str = "Stage2",
) -> None:
    """
    Xác minh 100% tham số `requires_grad=True` của mô hình nằm trong Optimizer,
    không trùng lặp giữa các `param_groups` và không bỏ sót bất kỳ tham số nào.
    """
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    opt_ids: List[int] = []
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            opt_ids.append(id(p))

    if len(opt_ids) != len(set(opt_ids)):
        raise RuntimeError(f"[{context_label}] Optimizer chứa tham số trùng lặp giữa các param_groups!")

    opt_set = set(opt_ids)
    missing = trainable_ids - opt_set
    extra = opt_set - trainable_ids
    if missing or extra:
        raise RuntimeError(
            f"[{context_label}] Optimizer không khớp tham số trainable: thiếu={len(missing)}, thừa={len(extra)}!"
        )


def build_stage2_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    total_epochs: int,
    warmup_epochs: int = 0,
    eta_min: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    Khởi tạo `AdamW` (tách riêng `weight_decay = 0.0` cho `LayerNorm` và `bias`)
    cùng bộ lập lịch `CosineAnnealingLR` (hoặc `SequentialLR(LinearLR + CosineAnnealingLR)` khi `warmup_epochs > 0`).

    Lý do tách `param_groups`:
      - Các trọng số 2D (`Linear.weight`, `p.ndim >= 2`) chịu `weight_decay = 1e-4` để chống quá khớp.
      - Các tham số 1D (`LayerNorm.weight`, `LayerNorm.bias`, `Linear.bias`, `p.ndim < 2`) đặt
        `weight_decay = 0.0` để không kéo hệ số chuẩn hóa gamma của `LayerNorm` từng nhánh về 0.
    """
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"Learning rate phải là số dương hữu hạn, nhận được: {lr}")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError(f"weight_decay phải >= 0, nhận được: {weight_decay}")
    if total_epochs <= 0:
        raise ValueError(f"total_epochs phải >= 1, nhận được: {total_epochs}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs phải >= 0, nhận được: {warmup_epochs}")

    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if param.ndim < 2 or name_lower.endswith(".bias") or "norm" in name_lower:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: List[Dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "group_name": "decay",
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "lr": float(lr),
                "weight_decay": 0.0,
                "group_name": "no_decay",
            }
        )

    if not param_groups:
        raise ValueError("Mô hình không có tham số nào mở khóa (`requires_grad=True`) để tối ưu hóa!")

    optimizer = torch.optim.AdamW(param_groups, lr=float(lr), weight_decay=float(weight_decay))
    assert_optimizer_covers_trainable_params(model, optimizer, context_label=type(model).__name__)

    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=int(warmup_epochs),
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs) - int(warmup_epochs)),
            eta_min=float(eta_min),
        )
        scheduler: Any = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[int(warmup_epochs)],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs)),
            eta_min=float(eta_min),
        )

    return optimizer, scheduler


# ==============================================================================
# NẠP CACHE ĐA NHÁNH & CACHED FEATURE DATASET (ĐỒNG BỘ FLIP + SAMPLER)
# ==============================================================================
def load_aligned_branch_caches(
    cache_dir: Union[str, Path],
    branches: Union[str, Sequence[str]] = BRANCH_ORDER,
    split: str = "train",
    jpeg_quality: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Nạp các file `.pt` của tất cả các nhánh được yêu cầu cho một `split` (và `jpeg_quality` nếu có),
    sau đó chạy `verify_cross_branch_alignment` để bảo đảm căn chỉnh mẫu 100% trước khi sử dụng.
    """
    norm_branches = normalize_branch_selection(branches)
    split_norm = str(split).strip().lower()
    if split_norm not in VALID_SPLITS:
        raise ValueError(f"split không hợp lệ: {split!r}. Chọn trong {VALID_SPLITS}.")

    payloads: Dict[str, Dict[str, Any]] = {}
    for br in norm_branches:
        fp = get_cache_filepath(cache_dir, branch=br, split=split_norm, jpeg_quality=jpeg_quality)
        payload = load_feature_cache(fp, expected_branch=br, expected_split=split_norm, map_location="cpu")
        if payload.get("jpeg_quality") != jpeg_quality:
            raise ValueError(
                f"[load_aligned_branch_caches] File '{fp.name}' có jpeg_quality={payload.get('jpeg_quality')} "
                f"không khớp với mức yêu cầu jpeg_quality={jpeg_quality}!"
            )
        payloads[br] = payload

    alignment_info = verify_cross_branch_alignment(payloads)
    return payloads, alignment_info


class _SyncedSamplerSetEpoch:
    """
    Wrapper callable mức module (pickle-safe trên Windows) tạo đồng bộ 2 chiều:
    Dù vòng lặp bên ngoài chỉ gọi `sampler.set_epoch(ep)` hay chỉ gọi `dataset.set_epoch(ep)`,
    cả `BalancedGroupSampler` lẫn `CachedFeatureDataset._epoch_flip_decisions` đều được cập nhật.
    """

    def __init__(self, orig_set_epoch: Callable[[int], None], dataset: "CachedFeatureDataset") -> None:
        self.orig_set_epoch = orig_set_epoch
        self.dataset = dataset

    def __call__(self, epoch: int) -> None:
        ep = int(epoch)
        self.orig_set_epoch(ep)
        if self.dataset.epoch != ep:
            self.dataset.set_epoch(ep, _sync_sampler=False)


class CachedFeatureDataset(Dataset):
    """
    Dataset đặc trưng lưu đệm trên RAM CPU phục vụ huấn luyện và đánh giá Stage 2:
    1. Kiểm định căn chỉnh mẫu liên nhánh (`verify_cross_branch_alignment`) và bảo đảm
       `sample_idx[i] == i` cho mọi `i = 0 .. N-1`.
    2. Đồng bộ góc nhìn Original / Horizontal Flip (`p = 0.5`) cho CẢ 3 NHÁNH:
       - Tại mỗi epoch `e`, `set_epoch(e)` sinh trước mảng quyết định `self._epoch_flip_decisions`
         kích thước `(N,)` từ `SeedSequence([seed, epoch])`.
       - Khi `__getitem__(idx)` được gọi ở chế độ `mode="train"`, quyết định `use_flip` của mẫu `idx`
         được tra cứu 1 lần duy nhất (`O(1)`) và áp dụng đồng thời cho tất cả các nhánh đang bật.
       - Ở chế độ `val` / `test` / `eval` (hoặc khi `has_flip=False` / `flip_prob=0.0`), luôn ép cứng
         `use_flip = False` (100% dùng đặc trưng ảnh gốc).
    3. Đóng gói sẵn Tensor ghép 2D tại khóa `self.input_key` (tương thích trực tiếp với `engine.py`
       và `FusionMLP.pack_branch_features`), đồng thời trả kèm từng nhánh rời để phục vụ kiểm định.
    4. Khôi phục trực tiếp `DataFrame` cho `BalancedGroupSampler` từ dữ liệu trên RAM mà không cần
       đọc lại `metadata.csv` từ ổ đĩa.
    """

    def __init__(
        self,
        cache_payloads: Mapping[str, Mapping[str, Any]],
        active_branches: Optional[Union[str, Sequence[str]]] = None,
        input_key: str = "fusion",
        mode: str = "train",
        flip_prob: float = 0.5,
        seed: int = 42,
        require_flip_on_train: bool = True,
    ) -> None:
        super().__init__()
        if not cache_payloads:
            raise ValueError("`cache_payloads` không được rỗng.")

        if active_branches is None:
            active_branches = list(cache_payloads.keys())
        self.active_branches: Tuple[str, ...] = normalize_branch_selection(active_branches)

        missing_br = [br for br in self.active_branches if br not in cache_payloads]
        if missing_br:
            raise KeyError(
                f"CachedFeatureDataset thiếu payload của các nhánh {missing_br}. "
                f"Hiện có: {list(cache_payloads.keys())}"
            )

        selected_payloads = {br: cache_payloads[br] for br in self.active_branches}
        self.alignment_info: Dict[str, Any] = verify_cross_branch_alignment(selected_payloads)

        mode_norm = str(mode).strip().lower()
        if mode_norm not in ("train", "val", "test", "eval"):
            raise ValueError(f"mode không hợp lệ: {mode!r}. Chọn trong ('train', 'val', 'test', 'eval').")
        self.mode: str = "train" if mode_norm == "train" else "eval"
        self.split: str = str(self.alignment_info["split"])
        self.input_key: str = str(input_key).strip()
        if not self.input_key:
            raise ValueError("`input_key` không được để trống.")

        if isinstance(flip_prob, bool) or not isinstance(flip_prob, (int, float)):
            raise TypeError(f"`flip_prob` phải là số thực trong [0, 1], nhận được: {flip_prob!r}")
        flip_prob_f = float(flip_prob)
        if not math.isfinite(flip_prob_f) or not (0.0 <= flip_prob_f <= 1.0):
            raise ValueError(f"`flip_prob` phải nằm trong [0.0, 1.0], nhận được: {flip_prob}")

        self.flip_prob: float = flip_prob_f if self.mode == "train" else 0.0
        self.seed: int = int(seed)
        self.epoch: int = 0
        self.sampler: Optional[Any] = None

        ref = selected_payloads[self.active_branches[0]]
        self._num_samples: int = int(ref["num_samples"])
        self.has_flip: bool = bool(ref["has_flip"])
        self.jpeg_quality: Optional[int] = ref.get("jpeg_quality", None)
        self.manifest_hash: str = str(ref["manifest_hash"])

        if self.mode == "train" and self.flip_prob > 0.0 and require_flip_on_train and not self.has_flip:
            raise ValueError(
                "CachedFeatureDataset ở chế độ 'train' với flip_prob > 0 yêu cầu cache phải có "
                "`has_flip=True` và `features_flip` đầy đủ ở mọi nhánh!"
            )

        sample_idx_tensor = ref["sample_idx"].detach().cpu().to(dtype=torch.int64)
        expected_idx = torch.arange(self._num_samples, dtype=torch.int64)
        if not torch.equal(sample_idx_tensor, expected_idx):
            raise ValueError(
                "Tensor `sample_idx` trong cache không khớp với dãy tuần tự `0 .. N-1`! "
                "Không thể ánh xạ trực tiếp với BalancedGroupSampler."
            )

        self.sample_idx: torch.Tensor = sample_idx_tensor
        self.labels_a: torch.Tensor = ref["label_a"].detach().cpu().to(dtype=torch.float32)
        self.labels_int: torch.Tensor = torch.round(self.labels_a).to(dtype=torch.int64)
        self.labels_b: torch.Tensor = ref["label_b"].detach().cpu().to(dtype=torch.int64)
        self.group_ids: torch.Tensor = ref["group_id"].detach().cpu().to(dtype=torch.int64)
        self.filepaths: List[str] = [str(x) for x in ref["filepaths"]]
        self.model_names: List[str] = [str(x) for x in ref["model_names"]]

        self.branch_features_orig: Dict[str, torch.Tensor] = {}
        self.branch_features_flip: Dict[str, torch.Tensor] = {}
        self.extraction_signatures: Dict[str, Any] = {}
        self.checkpoint_infos: Dict[str, Any] = {}

        orig_chunks: List[torch.Tensor] = []
        flip_chunks: List[torch.Tensor] = []

        for br in self.active_branches:
            p = selected_payloads[br]
            f_orig = p["features"].detach().cpu().to(dtype=torch.float32).contiguous()
            if f_orig.shape != (self._num_samples, BRANCH_FEATURE_DIMS[br]):
                raise ValueError(
                    f"Shape features của nhánh '{br}' không hợp lệ: {tuple(f_orig.shape)} != "
                    f"({self._num_samples}, {BRANCH_FEATURE_DIMS[br]})"
                )
            self.branch_features_orig[br] = f_orig
            orig_chunks.append(f_orig)

            if self.has_flip:
                f_flip_raw = p.get("features_flip", None)
                if f_flip_raw is None:
                    raise ValueError(f"Nhánh '{br}' khai báo has_flip=True nhưng thiếu tensor `features_flip`!")
                f_flip = f_flip_raw.detach().cpu().to(dtype=torch.float32).contiguous()
                if f_flip.shape != f_orig.shape:
                    raise ValueError(
                        f"Shape features_flip của nhánh '{br}' ({tuple(f_flip.shape)}) "
                        f"không khớp với features gốc ({tuple(f_orig.shape)})!"
                    )
                self.branch_features_flip[br] = f_flip
                flip_chunks.append(f_flip)

            raw_sig = p.get("extraction_signature", {})
            self.extraction_signatures[br] = dict(raw_sig) if isinstance(raw_sig, Mapping) else str(raw_sig)
            self.checkpoint_infos[br] = p.get("checkpoint_info", {})

        # Đóng gói sẵn Tensor 2D theo đúng chuẩn `FusionMLP.pack_branch_features`
        self._packed_orig: torch.Tensor = (
            orig_chunks[0] if len(orig_chunks) == 1 else torch.cat(orig_chunks, dim=1).contiguous()
        )
        self._packed_flip: Optional[torch.Tensor] = (
            (flip_chunks[0] if len(flip_chunks) == 1 else torch.cat(flip_chunks, dim=1).contiguous())
            if self.has_flip
            else None
        )
        self.total_feature_dim: int = int(self._packed_orig.shape[1])

        self._epoch_flip_decisions: np.ndarray = np.zeros(self._num_samples, dtype=bool)
        self.set_epoch(0)

    def attach_sampler(self, sampler: Any) -> "CachedFeatureDataset":
        """
        Gắn liên kết đồng bộ 2 chiều với `BalancedGroupSampler`:
        - Gọi `dataset.set_epoch(e)` sẽ kéo theo `sampler.set_epoch(e)`.
        - Gọi `sampler.set_epoch(e)` (kể cả trong vòng lặp tự viết ngoài `engine.py`)
          cũng tự kéo theo `dataset.set_epoch(e)` mà không bao giờ lặp đệ quy vô hạn.
        """
        self.sampler = sampler
        if sampler is not None and hasattr(sampler, "set_epoch"):
            raw_orig = sampler.set_epoch
            if isinstance(raw_orig, _SyncedSamplerSetEpoch):
                raw_orig = raw_orig.orig_set_epoch
            sampler.set_epoch = _SyncedSamplerSetEpoch(raw_orig, self)
        return self

    def set_epoch(self, epoch: int, _sync_sampler: bool = True) -> None:
        """
        Cập nhật `epoch` hiện tại và sinh bảng quyết định lật ảnh đồng bộ cho từng mẫu `0..N-1`.
        Quyết định lật của mẫu `idx` tại `epoch` hoàn toàn tất định theo `(self.seed, epoch, idx)`
        và áp dụng chung cho tất cả các nhánh trong `self.active_branches`.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch phải là số nguyên >= 0, nhận được: {epoch!r}")
        self.epoch = int(epoch)

        if self.mode == "train" and self.has_flip and self.flip_prob > 0.0:
            if self.flip_prob >= 1.0:
                self._epoch_flip_decisions = np.ones(self._num_samples, dtype=bool)
            else:
                seq = np.random.SeedSequence([int(self.seed), int(self.epoch)])
                rng = np.random.default_rng(seq)
                self._epoch_flip_decisions = rng.random(self._num_samples) < self.flip_prob
        else:
            self._epoch_flip_decisions = np.zeros(self._num_samples, dtype=bool)

        if _sync_sampler and self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.epoch)

    def build_sampler_dataframe(self) -> pd.DataFrame:
        """
        Khôi phục trực tiếp `pd.DataFrame` từ bộ nhớ đệm trên RAM cho `BalancedGroupSampler`
        (không cần đọc lại `metadata.csv` từ ổ cứng, đảm bảo khớp 100% chỉ số `0 .. N-1`).
        """
        return pd.DataFrame(
            {
                "sample_idx": self.sample_idx.numpy(),
                "group_id": self.group_ids.numpy(),
                "label_a": self.labels_int.numpy(),
                "label_b": self.labels_b.numpy(),
                "model_name": list(self.model_names),
                "filepath": list(self.filepaths),
            }
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Lấy đặc trưng đã chọn góc nhìn đồng bộ của mẫu `idx`.
        Việc chọn Original hay Flip diễn ra ĐÚNG 1 LẦN tại đây cho cả 3 nhánh.
        """
        idx = range(self._num_samples)[idx]
        use_flip = bool(self._epoch_flip_decisions[idx]) and self.has_flip and (self.mode == "train")

        packed_feat = (
            self._packed_flip[idx]
            if (use_flip and self._packed_flip is not None)
            else self._packed_orig[idx]
        )

        item: Dict[str, Any] = {
            self.input_key: packed_feat,
            "label": self.labels_int[idx],
            "label_a": self.labels_a[idx],
            "label_b": self.labels_b[idx],
            "group_id": self.group_ids[idx],
            "sample_idx": int(self.sample_idx[idx].item()),
            "model_name": self.model_names[idx],
            "filepath": self.filepaths[idx],
            "is_flipped": torch.tensor(use_flip, dtype=torch.bool),
        }

        # Đính kèm đặc trưng từng nhánh theo đúng góc nhìn `use_flip` đã chọn
        for br in self.active_branches:
            br_feat = (
                self.branch_features_flip[br][idx]
                if use_flip
                else self.branch_features_orig[br][idx]
            )
            item[br] = br_feat

        return item


def build_stage2_dataloaders(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    active_branches: Union[str, Sequence[str]] = BRANCH_ORDER,
    input_key: str = "fusion",
    batch_size: int = 32,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    flip_prob: float = 0.5,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Khởi tạo cặp `(train_loader, val_loader, cache_meta)` cho Giai đoạn 2:
    - `train_loader`: sử dụng `CachedFeatureDataset(mode='train', flip_prob=flip_prob)` kết hợp
      `BalancedGroupSampler` (bảo đảm 16 Real : 16 AI mỗi batch và phủ 100% ảnh AI sau 3 epochs).
    - `val_loader`: sử dụng `CachedFeatureDataset(mode='val', flip_prob=0.0)` duyệt tuần tự.
    - Mặc định `num_workers = 0` vì toàn bộ tensor `.pt` đã nằm trên RAM CPU.
    """
    norm_branches = normalize_branch_selection(active_branches)
    if batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError(
            f"batch_size của Train phải là số chẵn dương để giữ cân bằng 1:1 theo cặp (Real, AI), "
            f"nhận được: {batch_size}"
        )
    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size phải >= 1, nhận được: {eval_batch_size}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers phải >= 0, nhận được: {num_workers}")
    if int(num_workers) > 0 and float(flip_prob) > 0.0:
        raise ValueError(
            f"Stage 2 giữ toàn bộ feature cache trên RAM CPU nên bắt buộc dùng num_workers=0 khi bật Flip "
            f"(flip_prob={flip_prob}). Nếu để num_workers={num_workers} kèm persistent_workers=True trên Windows, "
            f"worker giữ bản sao _epoch_flip_decisions cũ và không đồng bộ qua các epoch!"
        )

    train_payloads, train_align = load_aligned_branch_caches(
        cache_dir=cache_dir,
        branches=norm_branches,
        split="train",
        jpeg_quality=None,
    )
    val_payloads, val_align = load_aligned_branch_caches(
        cache_dir=cache_dir,
        branches=norm_branches,
        split="val",
        jpeg_quality=None,
    )

    train_ds = CachedFeatureDataset(
        cache_payloads=train_payloads,
        active_branches=norm_branches,
        input_key=input_key,
        mode="train",
        flip_prob=flip_prob,
        seed=seed,
        require_flip_on_train=(flip_prob > 0.0),
    )
    val_ds = CachedFeatureDataset(
        cache_payloads=val_payloads,
        active_branches=norm_branches,
        input_key=input_key,
        mode="val",
        flip_prob=0.0,
        seed=seed,
        require_flip_on_train=False,
    )

    sampler_df = train_ds.build_sampler_dataframe()
    train_sampler = BalancedGroupSampler(sampler_df, seed=seed)
    train_ds.attach_sampler(train_sampler)

    use_pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        sampler=train_sampler,
        num_workers=int(num_workers),
        pin_memory=use_pin,
        drop_last=False,
        generator=dl_gen,
        persistent_workers=(int(num_workers) > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=use_pin,
        drop_last=False,
        persistent_workers=False,
    )

    cache_meta = {
        "cache_dir": str(Path(cache_dir).resolve()),
        "active_branches": list(norm_branches),
        "input_key": input_key,
        "total_feature_dim": int(train_ds.total_feature_dim),
        "train_num_samples": int(len(train_ds)),
        "train_epoch_samples": int(len(train_sampler)),
        "train_num_groups": int(train_sampler.num_groups),
        "val_num_samples": int(len(val_ds)),
        "train_manifest_hash": str(train_ds.manifest_hash),
        "val_manifest_hash": str(val_ds.manifest_hash),
        "train_has_flip": bool(train_ds.has_flip),
        "train_extraction_signatures": dict(train_ds.extraction_signatures),
        "val_extraction_signatures": dict(val_ds.extraction_signatures),
        "train_checkpoint_infos": dict(train_ds.checkpoint_infos),
        "train_alignment": train_align,
        "val_alignment": val_align,
    }
    return train_loader, val_loader, cache_meta


def build_stage2_eval_loader(
    cache_dir: Union[str, Path],
    active_branches: Union[str, Sequence[str]],
    input_key: str,
    split: str = "test",
    jpeg_quality: Optional[int] = None,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, CachedFeatureDataset, Dict[str, Any]]:
    """Tạo DataLoader chế độ đánh giá (`val` hoặc `test`, Clean hoặc JPEG `90/70/50/30`)."""
    norm_branches = normalize_branch_selection(active_branches)
    payloads, align_info = load_aligned_branch_caches(
        cache_dir=cache_dir,
        branches=norm_branches,
        split=split,
        jpeg_quality=jpeg_quality,
    )
    ds = CachedFeatureDataset(
        cache_payloads=payloads,
        active_branches=norm_branches,
        input_key=input_key,
        mode="eval",
        flip_prob=0.0,
        seed=42,
        require_flip_on_train=False,
    )
    use_pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    loader = DataLoader(
        ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=use_pin,
        drop_last=False,
        persistent_workers=False,
    )
    return loader, ds, align_info


# ==============================================================================
# ĐIỀU PHỐI HUẤN LUYỆN 1 THỰC NGHIỆM STAGE 2 (PROBE / FUSION / ABLATION / DROPOUT)
# ==============================================================================
def run_stage2_training_experiment(
    experiment_name: str,
    active_branches: Union[str, Sequence[str]],
    head_type: str,
    output_dir: Union[str, Path],
    best_ckpt_path: Optional[Union[str, Path]] = None,
    last_ckpt_path: Optional[Union[str, Path]] = None,
    mirror_best_ckpt_path: Optional[Union[str, Path]] = None,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    branch_dropout_p: float = 0.0,
    min_active_branches: int = 2,
    hidden_dims: Sequence[int] = (512, 128),
    hidden_dropouts: Sequence[float] = (0.3, 0.2),
    linear_dropout: Optional[float] = 0.0,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    min_delta: Optional[float] = None,
    max_norm: Optional[float] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    flip_prob: Optional[float] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
    resume_from: Optional[Union[str, Path]] = None,
    run_sanity_check: bool = False,
    log_interval: int = 200,
) -> Dict[str, Any]:
    """
    Hàm lõi điều phối huấn luyện một cấu hình Stage 2 bất kỳ trên Feature Cache:
    - Khởi tạo mô hình (`LinearProbeHead` hoặc `FusionMLP`).
    - Nạp `train_loader` (`BalancedGroupSampler` + đồng bộ Flip `p=0.5`) & `val_loader`.
    - Khởi tạo `AdamW` (tách `weight_decay=0.0` cho `LayerNorm` & `bias`) + `CosineAnnealingLR`.
    - Chạy `fit()` của `engine.py`, chọn `best_checkpoint` thuần túy theo Validation Known `ROC-AUC`
      (phân xử hòa bằng `Balanced Accuracy`).
    - Xuất đầy đủ `best_*.pth`, `last_*.pth`, `history.csv`, `history.json`, `config.json`.
    """
    t0 = time.time()
    norm_branches = normalize_branch_selection(active_branches)
    ht = str(head_type).strip().lower()
    if ht not in STAGE2_DEFAULTS:
        raise ValueError(f"head_type không hợp lệ: {head_type!r}. Chọn 'linear' hoặc 'mlp'.")

    defaults = STAGE2_DEFAULTS[ht]
    resolved_lr = float(defaults["lr"] if lr is None else lr)
    resolved_wd = float(defaults["weight_decay"] if weight_decay is None else weight_decay)
    resolved_warmup = int(defaults["warmup_epochs"] if warmup_epochs is None else warmup_epochs)
    resolved_epochs = int(defaults["epochs"] if epochs is None else epochs)
    resolved_patience = int(defaults["patience"] if patience is None else patience)
    resolved_min_delta = float(defaults["min_delta"] if min_delta is None else min_delta)
    resolved_max_norm = float(defaults["max_norm"] if max_norm is None else max_norm)
    resolved_bs = int(defaults["batch_size"] if batch_size is None else batch_size)
    resolved_eval_bs = int(defaults["eval_batch_size"] if eval_batch_size is None else eval_batch_size)
    resolved_workers = int(defaults["num_workers"] if num_workers is None else num_workers)
    resolved_flip_prob = float(defaults["flip_prob"] if flip_prob is None else flip_prob)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    model = build_stage2_model(
        active_branches=norm_branches,
        head_type=ht,
        branch_dropout_p=float(branch_dropout_p) if ht == "mlp" else 0.0,
        min_active_branches=int(min_active_branches),
        hidden_dims=hidden_dims,
        hidden_dropouts=hidden_dropouts,
        linear_dropout=linear_dropout if ht == "linear" else None,
    ).to(device)

    train_loader, val_loader, cache_meta = build_stage2_dataloaders(
        cache_dir=cache_dir,
        active_branches=norm_branches,
        input_key=model.input_key,
        batch_size=resolved_bs,
        eval_batch_size=resolved_eval_bs,
        num_workers=resolved_workers,
        flip_prob=resolved_flip_prob,
        seed=seed,
    )

    auto_amp, auto_dtype, auto_scaler = get_optimal_amp_config(device)
    amp_enabled = auto_amp if use_amp is None else (bool(use_amp) and device.type == "cuda")
    amp_dtype = auto_dtype if amp_enabled else torch.float32
    scaler = auto_scaler if (amp_enabled and amp_dtype == torch.float16 and device.type == "cuda") else None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_fp = Path(best_ckpt_path) if best_ckpt_path is not None else (out_dir / "best_model.pth")
    last_fp = Path(last_ckpt_path) if last_ckpt_path is not None else (out_dir / "last_model.pth")
    best_fp.parent.mkdir(parents=True, exist_ok=True)
    last_fp.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"{experiment_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    config_dict: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "stage": 2,
        "head_type": ht,
        "active_branches": list(norm_branches),
        "input_key": model.input_key,
        "input_feature_dim": int(model.input_feature_dim),
        "output_feature_dim": int(model.output_feature_dim),
        "trainable_params": int(model.count_parameters(trainable_only=True)),
        "total_params": int(model.count_parameters(trainable_only=False)),
        "branch_dropout_p": float(getattr(model, "branch_dropout_p", 0.0)),
        "min_active_branches": int(getattr(model, "min_active_branches", 1)),
        "hidden_dims": list(hidden_dims) if ht == "mlp" else [],
        "hidden_dropouts": list(hidden_dropouts) if ht == "mlp" else [],
        "linear_dropout": float(getattr(model, "dropout_rate", 0.0)) if ht == "linear" else None,
        "optimizer": "AdamW",
        "lr": resolved_lr,
        "weight_decay": resolved_wd,
        "scheduler": (
            f"LinearLR(warmup={resolved_warmup}) + CosineAnnealingLR(T_max={resolved_epochs - resolved_warmup})"
            if (resolved_warmup > 0 and resolved_epochs > resolved_warmup)
            else f"CosineAnnealingLR(T_max={resolved_epochs})"
        ),
        "warmup_epochs": resolved_warmup,
        "total_epochs": resolved_epochs,
        "patience": resolved_patience,
        "min_delta": resolved_min_delta,
        "primary_metric": "auc",
        "secondary_metric": "balanced_acc",
        "max_norm": resolved_max_norm,
        "batch_size": resolved_bs,
        "eval_batch_size": resolved_eval_bs,
        "num_workers": resolved_workers,
        "flip_prob": resolved_flip_prob,
        "seed": int(seed),
        "device": str(device),
        "use_amp": bool(amp_enabled),
        "amp_dtype": str(amp_dtype),
        "best_checkpoint_path": str(best_fp),
        "last_checkpoint_path": str(last_fp),
        "mirror_best_checkpoint_path": str(mirror_best_ckpt_path) if mirror_best_ckpt_path else None,
        "model_config": model.get_config(),
        "cache_meta": cache_meta,
    }

    print("\n" + "=" * 88)
    print(f"BẮT ĐẦU HUẤN LUYỆN STAGE 2: {experiment_name.upper()}")
    print("=" * 88)
    print(
        f"  + Mô hình         : {type(model).__name__} | Nhánh={list(norm_branches)} | "
        f"Chiều đầu vào={model.input_feature_dim} -> Logit (B,) | "
        f"Tham số học={model.count_parameters(trainable_only=True):,}"
    )
    print(
        f"  + Tối ưu hóa      : AdamW(lr={resolved_lr:.1e}, wd={resolved_wd:.1e}, LayerNorm/bias wd=0.0) | "
        f"Warmup={resolved_warmup} ep | Total={resolved_epochs} ep | Patience={resolved_patience}"
    )
    print(
        f"  + Dữ liệu Cache   : Train={cache_meta['train_num_samples']:,} ảnh "
        f"({cache_meta['train_epoch_samples']:,} mẫu cân bằng 1:1/epoch) | "
        f"Val={cache_meta['val_num_samples']:,} ảnh | Flip p={resolved_flip_prob} | Workers={resolved_workers}"
    )
    print(f"  + Lưu Checkpoint  : Best='{best_fp}' | Dir='{out_dir}'")
    print("=" * 88)

    if run_sanity_check:
        if resume_from is not None:
            print("  [Lưu ý] Bỏ qua Sanity Check Gate vì đang ở chế độ --resume từ checkpoint.")
        else:
            # LinearProbeHead trên vector L2-normalized cần bước nhảy lớn hơn trong 100 steps để đẩy margin BCE >= 90%
            gate_overfit_lr = 3e-2 if ht == "linear" else max(resolved_lr * 6.0, 3e-3)
            gate_ok = sanity_check_gate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                overfit_lr=gate_overfit_lr,
                overfit_steps=100,
                smoke_train_batches=50,
                smoke_val_batches=30,
                smoke_lr=resolved_lr,
            )
            if not gate_ok:
                raise RuntimeError(
                    f"Sanity Check Gate THẤT BẠI cho '{experiment_name}' — Dừng huấn luyện Stage 2!"
                )
            set_seed(seed)
            model = build_stage2_model(
                active_branches=norm_branches,
                head_type=ht,
                branch_dropout_p=float(branch_dropout_p) if ht == "mlp" else 0.0,
                min_active_branches=int(min_active_branches),
                hidden_dims=hidden_dims,
                hidden_dropouts=hidden_dropouts,
                linear_dropout=linear_dropout if ht == "linear" else None,
            ).to(device)

    optimizer, scheduler = build_stage2_optimizer_and_scheduler(
        model=model,
        lr=resolved_lr,
        weight_decay=resolved_wd,
        total_epochs=resolved_epochs,
        warmup_epochs=resolved_warmup,
    )
    early_stopping = EarlyStopping(
        patience=resolved_patience,
        min_delta=resolved_min_delta,
        primary_metric="auc",
        secondary_metric="balanced_acc",
        mode="max",
    )
    criterion = nn.BCEWithLogitsLoss()

    fit_result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=resolved_epochs,
        criterion=criterion,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint_path=best_fp,
        last_checkpoint_path=last_fp,
        resume_from=resume_from,
        start_epoch=0,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        max_norm=resolved_max_norm,
        config=config_dict,
        log_interval=log_interval,
        restore_best_weights=True,
    )

    if best_fp.is_file():
        load_checkpoint(best_fp, model, device=device, restore_rng=False)

    if mirror_best_ckpt_path is not None and best_fp.is_file():
        mirror_fp = Path(mirror_best_ckpt_path)
        mirror_fp.parent.mkdir(parents=True, exist_ok=True)
        if mirror_fp.resolve() != best_fp.resolve():
            shutil.copy2(best_fp, mirror_fp)

    val_loss, val_metrics, val_per_gen = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=amp_enabled,
        compute_per_generator=True,
    )

    elapsed_total = time.time() - t0
    new_history_records = fit_result.get("history", [])
    history_json_fp = out_dir / f"history_{experiment_name}.json"
    if resume_from is not None and new_history_records:
        first_new_ep = int(new_history_records[0].get("epoch", 0))
        prior_records = load_prior_history(history_json_fp, first_new_ep - 1)
        history_records = prior_records + list(new_history_records)
    else:
        history_records = list(new_history_records)
    best_epoch = int(fit_result.get("best_epoch", early_stopping.best_epoch))

    config_dict["best_epoch"] = best_epoch
    config_dict["best_val_auc"] = float(val_metrics.get("auc", 0.0))
    config_dict["best_val_balanced_acc"] = float(val_metrics.get("balanced_acc", 0.0))
    config_dict["best_val_loss"] = float(val_loss)
    config_dict["elapsed_total_sec"] = float(elapsed_total)

    history_csv_path = export_history_csv(history_records, out_dir / "history.csv")
    config_json_path = save_json(config_dict, out_dir / "config.json")
    history_json_path = save_json(
        {
            "config": config_dict,
            "best_epoch": best_epoch,
            "best_val_loss": float(val_loss),
            "best_val_metrics": val_metrics,
            "best_val_per_generator": val_per_gen,
            "history": history_records,
        },
        history_json_fp,
    )

    print(
        f"-> Hoàn tất [{experiment_name}] trong {elapsed_total:.2f}s | "
        f"Best Epoch={best_epoch} | Val AUC={val_metrics.get('auc', 0.0):.4f} | "
        f"Val Bal-ACC={val_metrics.get('balanced_acc', 0.0) * 100:.2f}% | "
        f"Val Loss={val_loss:.4f}"
    )

    return {
        "experiment_name": experiment_name,
        "model": model,
        "config": config_dict,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_fp),
        "last_checkpoint_path": str(last_fp),
        "mirror_best_checkpoint_path": str(mirror_best_ckpt_path) if mirror_best_ckpt_path else None,
        "history_csv_path": str(history_csv_path),
        "history_json_path": str(history_json_path),
        "config_json_path": str(config_json_path),
        "val_loss": float(val_loss),
        "val_metrics": val_metrics,
        "val_per_generator": val_per_gen,
        "history": history_records,
        "elapsed_sec": float(elapsed_total),
    }


# ==============================================================================
# CÁC PIPELINE CHUYÊN BIỆT: SEMANTIC PROBE, FUSION BASELINE, DROPOUT, ABLATION
# ==============================================================================
def train_semantic_probe(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    semantic_ckpt_path: Union[str, Path] = DEFAULT_SEMANTIC_CKPT,
    stage2_probe_dir: Union[str, Path] = DEFAULT_STAGE2_PROBE_DIR,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    flip_prob: Optional[float] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
    resume_from: Optional[Union[str, Path]] = None,
    run_sanity_check: bool = False,
) -> Dict[str, Any]:
    """
    Huấn luyện Nhánh 3 — Semantic Linear Probe (`Linear(512 -> 1)`, 513 tham số):
    - Lưu `best_semantic.pth` trực tiếp tại `checkpoints/semantic_baseline/best_semantic.pth`
      (chung chỗ với `spatial_baseline/best_spatial.pth` và `spectral_baseline/best_spectral.pth`).
    - Toàn bộ tệp còn lại (`last_semantic.pth`, `history.csv`, `history_semantic_probe.json`, `config.json`
      kèm bản sao `best_semantic.pth`) được lưu trong `checkpoints/stage2/semantic_probe/`.
    """
    probe_dir = Path(stage2_probe_dir)
    primary_best_fp = Path(semantic_ckpt_path)
    stage2_best_copy_fp = probe_dir / "best_semantic.pth"
    last_fp = probe_dir / "last_semantic.pth"

    return run_stage2_training_experiment(
        experiment_name="semantic_probe",
        active_branches=("semantic",),
        head_type="linear",
        output_dir=probe_dir,
        best_ckpt_path=primary_best_fp,
        last_ckpt_path=last_fp,
        mirror_best_ckpt_path=stage2_best_copy_fp,
        cache_dir=cache_dir,
        linear_dropout=0.0,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=0,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        flip_prob=flip_prob,
        seed=seed,
        device=device,
        use_amp=use_amp,
        resume_from=resume_from,
        run_sanity_check=run_sanity_check,
    )


def train_fusion_baseline(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    fusion_dir: Union[str, Path] = DEFAULT_STAGE2_FUSION_DIR,
    best_ckpt_path: Optional[Union[str, Path]] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    flip_prob: Optional[float] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
    resume_from: Optional[Union[str, Path]] = None,
    run_sanity_check: bool = False,
) -> Dict[str, Any]:
    """
    Huấn luyện Fusion MLP Baseline (3 nhánh `spatial + semantic + spectral`, `2048 -> 512 -> 128 -> 1`,
    `branch_dropout_p = 0.0`) và lưu tại `checkpoints/stage2/fusion_baseline/best_fusion.pth`.
    """
    out_dir = Path(fusion_dir)
    best_fp = Path(best_ckpt_path) if best_ckpt_path is not None else (out_dir / "best_fusion.pth")
    last_fp = out_dir / "last_fusion.pth"

    return run_stage2_training_experiment(
        experiment_name="fusion_baseline",
        active_branches=BRANCH_ORDER,
        head_type="mlp",
        output_dir=out_dir,
        best_ckpt_path=best_fp,
        last_ckpt_path=last_fp,
        cache_dir=cache_dir,
        branch_dropout_p=0.0,
        min_active_branches=2,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        flip_prob=flip_prob,
        seed=seed,
        device=device,
        use_amp=use_amp,
        resume_from=resume_from,
        run_sanity_check=run_sanity_check,
    )


def train_branch_dropout_experiment(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    dropout_dir: Union[str, Path] = DEFAULT_STAGE2_DROPOUT_DIR,
    branch_dropout_p: float = 0.15,
    min_active_branches: int = 2,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    flip_prob: Optional[float] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
    resume_from: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Thí nghiệm Ablation Branch Dropout (`branch_dropout_p = 0.15`, `min_active_branches = 2`):
    Chỉ áp dụng trên mô hình đủ 3 nhánh (`spatial + semantic + spectral`), huấn luyện độc lập
    và lưu tại `checkpoints/stage2/branch_dropout/best_fusion_dropout.pth`.
    """
    if not (0.0 < float(branch_dropout_p) < 1.0):
        raise ValueError(f"Thí nghiệm Branch Dropout yêu cầu 0 < branch_dropout_p < 1, nhận: {branch_dropout_p}")

    out_dir = Path(dropout_dir)
    best_fp = out_dir / "best_fusion_dropout.pth"
    last_fp = out_dir / "last_fusion_dropout.pth"

    return run_stage2_training_experiment(
        experiment_name="branch_dropout",
        active_branches=BRANCH_ORDER,
        head_type="mlp",
        output_dir=out_dir,
        best_ckpt_path=best_fp,
        last_ckpt_path=last_fp,
        cache_dir=cache_dir,
        branch_dropout_p=float(branch_dropout_p),
        min_active_branches=int(min_active_branches),
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        flip_prob=flip_prob,
        seed=seed,
        device=device,
        use_amp=use_amp,
        resume_from=resume_from,
    )


def run_ablation_study(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    ablation_root_dir: Union[str, Path] = DEFAULT_STAGE2_ABLATION_DIR,
    combinations: Sequence[Sequence[str]] = ALL_ABLATION_COMBINATIONS,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    flip_prob: Optional[float] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Thực hiện Ablation Study trên đủ 7 tổ hợp nhánh (`ALL_ABLATION_COMBINATIONS`):
    - Mỗi tổ hợp được huấn luyện ĐỘC LẬP từ đầu trên tập Train Known với cùng kiến trúc MLP head
      (`hidden_dims = (512, 128)`, `branch_dropout_p = 0.0`) và chọn best checkpoint theo Val Known AUC.
    - Tuyệt đối không dùng chung 1 checkpoint 3 nhánh rồi tắt đặc trưng lúc Test.
    - Lưu từng tổ hợp vào thư mục riêng: `checkpoints/stage2/ablation/<slug>/`.
    """
    ablation_root = Path(ablation_root_dir)
    ablation_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    print("\n" + "#" * 88)
    print(f"BẮT ĐẦU ABLATION STUDY ĐỘC LẬP TRÊN {len(combinations)} TỔ HỢP NHÁNH (p=0.0)")
    print("#" * 88)

    for idx, combo in enumerate(combinations, start=1):
        norm_combo = normalize_branch_selection(combo)
        slug = get_ablation_combo_slug(norm_combo)
        combo_dir = ablation_root / slug
        best_fp = combo_dir / f"best_{slug}.pth"
        last_fp = combo_dir / f"last_{slug}.pth"

        print(f"\n---> [Ablation {idx}/{len(combinations)}] Tổ hợp '{slug}': {list(norm_combo)}")
        res = run_stage2_training_experiment(
            experiment_name=f"ablation_{slug}",
            active_branches=norm_combo,
            head_type="mlp",
            output_dir=combo_dir,
            best_ckpt_path=best_fp,
            last_ckpt_path=last_fp,
            cache_dir=cache_dir,
            branch_dropout_p=0.0,
            min_active_branches=2,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            flip_prob=flip_prob,
            seed=seed,
            device=device,
            use_amp=use_amp,
        )
        results[slug] = res

    return results


# ==============================================================================
# ĐÁNH GIÁ TOÀN DIỆN: CLEAN TEST (KNOWN / UNSEEN / PER-GEN) & JPEG ROBUSTNESS
# ==============================================================================
def evaluate_stage2_model_on_cache(
    model: nn.Module,
    cache_dir: Union[str, Path],
    active_branches: Sequence[str],
    split: str = "test",
    jpeg_quality: Optional[int] = None,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Đánh giá một mô hình (`LinearProbeHead` hoặc `FusionMLP`) ở chế độ `.eval()` (tắt 100% Dropout
    và Branch Dropout) trên một tập cache (`val` hoặc `test`, Clean hoặc JPEG `q`):
    - Tính `overall` (gộp toàn bộ mẫu).
    - Tính `per_generator` (cặp 1:1 giữa toàn bộ ảnh Real `MS_COCO` với từng generator AI),
      `macro_avg_known`, `macro_avg_unseen`, `macro_avg_all` thông qua `evaluate_by_generator()`.
    - Tính thêm `pooled_known` (Real `MS_COCO` + 3 Known generators: `SD 2.1`, `SDXL`, `DALL-E 3`)
      và `pooled_unseen` (Real `MS_COCO` + 2 Unseen generators: `SD 3`, `Midjourney V6`) kèm số lượng
      mẫu cụ thể (`num_real`, `num_ai`) để báo cáo minh bạch cả 2 cách ghép.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    auto_amp, auto_dtype, _ = get_optimal_amp_config(device)
    amp_enabled = auto_amp if use_amp is None else (bool(use_amp) and device.type == "cuda")
    amp_dtype = auto_dtype if amp_enabled else torch.float32

    loader, ds, align_info = build_stage2_eval_loader(
        cache_dir=cache_dir,
        active_branches=active_branches,
        input_key=model.input_key,
        split=split,
        jpeg_quality=jpeg_quality,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    criterion = nn.BCEWithLogitsLoss()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_models: List[str] = []
    total_loss = 0.0
    num_samples = 0

    with torch.inference_mode():
        for batch in loader:
            x = batch[model.input_key].to(device, non_blocking=True)
            y = batch["label_a"].to(device, non_blocking=True).float()
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(x)
                loss = criterion(logits.float(), y)
            else:
                logits = model(x)
                loss = criterion(logits.float(), y)

            bs = int(x.size(0))
            total_loss += float(loss.item()) * bs
            num_samples += bs
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(y.detach().float().cpu())
            all_models.extend([str(m) for m in batch["model_name"]])

    avg_loss = total_loss / max(1, num_samples)
    concat_logits = torch.cat(all_logits, dim=0).numpy()
    concat_targets = torch.cat(all_targets, dim=0).numpy()
    models_arr = np.asarray(all_models, dtype=object)

    real_mask = models_arr == REAL_MODEL_NAME
    n_real = int(real_mask.sum())
    gen_counts = {
        g: int((models_arr == g).sum())
        for g in ALL_AI_GENERATORS
        if int((models_arr == g).sum()) > 0
    }
    is_balanced_1to1 = bool(
        n_real > 0
        and len(gen_counts) > 0
        and all(cnt == n_real for cnt in gen_counts.values())
    )

    gen_report = evaluate_by_generator(
        y_true=concat_targets,
        y_pred=concat_logits,
        model_names=all_models,
        from_logits=True,
        require_balanced_pairs=is_balanced_1to1,
        strict_known_generators=True,
    )

    # Tính thêm chỉ số gộp riêng cho tập Test Known (Real + 3 Known) và Test Unseen (Real + 2 Unseen)
    known_mask = real_mask | np.isin(models_arr, KNOWN_GENERATORS)
    unseen_mask = real_mask | np.isin(models_arr, UNSEEN_GENERATORS)

    pooled_known = (
        compute_binary_metrics(concat_targets[known_mask], concat_logits[known_mask], from_logits=True)
        if int(known_mask.sum()) > 0 and int((~real_mask & known_mask).sum()) > 0
        else {}
    )
    pooled_unseen = (
        compute_binary_metrics(concat_targets[unseen_mask], concat_logits[unseen_mask], from_logits=True)
        if int(unseen_mask.sum()) > 0 and int((~real_mask & unseen_mask).sum()) > 0
        else {}
    )

    return {
        "split": split,
        "jpeg_quality": jpeg_quality,
        "num_samples": int(num_samples),
        "num_real": n_real,
        "generator_sample_counts": gen_counts,
        "is_balanced_1to1_per_generator": is_balanced_1to1,
        "loss": float(avg_loss),
        "overall": gen_report["overall"],
        "macro_avg_known": gen_report["macro_avg_known"],
        "macro_avg_unseen": gen_report["macro_avg_unseen"],
        "macro_avg_all": gen_report["macro_avg_all"],
        "pooled_known": pooled_known,
        "pooled_unseen": pooled_unseen,
        "per_generator": gen_report["per_generator"],
        "manifest_hash": str(ds.manifest_hash),
        "extraction_signatures": dict(ds.extraction_signatures),
        "alignment": align_info,
    }


def verify_jpeg_matches_clean_metadata(
    clean_payloads: Mapping[str, Mapping[str, Any]],
    jpeg_payloads: Mapping[str, Mapping[str, Any]],
    expected_jpeg_quality: int,
) -> None:
    """
    Kiểm định nghiêm ngặt tính nhất quán giữa bản Test Clean và bản Test JPEG (Mục 3 Vấn đề 6 ChatGPT):
    1. Cùng tập ảnh Test: `sample_idx`, `filepaths`, `label_a`, `label_b`, `group_id`, `model_names`,
       `manifest_hash` phải trùng khớp 100%.
    2. Mọi nhánh đều có đúng `jpeg_quality == expected_jpeg_quality`.
    3. `extraction_signature` của bản JPEG phải khác bản Clean (vì cấu hình tiền xử lý có `jpeg_quality=q`).
    """
    for br, c_pay in clean_payloads.items():
        if br not in jpeg_payloads:
            raise KeyError(f"Thiếu nhánh '{br}' trong bộ cache JPEG Q={expected_jpeg_quality}!")
        j_pay = jpeg_payloads[br]

        if int(j_pay.get("jpeg_quality", -1)) != int(expected_jpeg_quality):
            raise ValueError(
                f"Nhánh '{br}' có jpeg_quality={j_pay.get('jpeg_quality')} != {expected_jpeg_quality}"
            )
        if str(c_pay["manifest_hash"]) != str(j_pay["manifest_hash"]):
            raise ValueError(
                f"Lệch manifest_hash giữa Clean và JPEG Q={expected_jpeg_quality} ở nhánh '{br}'!"
            )
        if not torch.equal(c_pay["sample_idx"].cpu(), j_pay["sample_idx"].cpu()):
            raise ValueError(f"Lệch sample_idx giữa Clean và JPEG Q={expected_jpeg_quality} ở nhánh '{br}'!")
        if c_pay["filepaths"] != j_pay["filepaths"]:
            raise ValueError(f"Lệch filepaths giữa Clean và JPEG Q={expected_jpeg_quality} ở nhánh '{br}'!")
        if not torch.equal(c_pay["label_a"].cpu(), j_pay["label_a"].cpu()):
            raise ValueError(f"Lệch label_a giữa Clean và JPEG Q={expected_jpeg_quality} ở nhánh '{br}'!")
        if not torch.equal(c_pay["group_id"].cpu(), j_pay["group_id"].cpu()):
            raise ValueError(f"Lệch group_id giữa Clean và JPEG Q={expected_jpeg_quality} ở nhánh '{br}'!")
        if c_pay["model_names"] != j_pay["model_names"]:
            raise ValueError(f"Lệch model_names giữa Clean và JPEG Q={expected_jpeg_quality} ở nhánh '{br}'!")
        c_sig = c_pay.get("extraction_signature", {})
        j_sig = j_pay.get("extraction_signature", {})
        if c_sig == j_sig or (
            isinstance(c_sig, Mapping)
            and isinstance(j_sig, Mapping)
            and c_sig.get("preprocess_config_hash") == j_sig.get("preprocess_config_hash")
        ):
            raise ValueError(
                f"extraction_signature của JPEG Q={expected_jpeg_quality} trùng với Clean ở nhánh '{br}'!"
            )

        # Đảm bảo Clean và JPEG được trích xuất từ CÙNG MỘT checkpoint / backbone & tiền xử lý (chỉ khác JPEG)
        if isinstance(c_sig, Mapping) and isinstance(j_sig, Mapping):
            invariant_keys = (
                "branch",
                "split",
                "source",
                "checkpoint_sha256",
                "clip_model_name",
                "clip_pretrained",
                "clip_quick_gelu",
                "clip_l2_normalize",
                "open_clip_version",
            )
            for key_name in invariant_keys:
                if c_sig.get(key_name) != j_sig.get(key_name):
                    raise ValueError(
                        f"Lệch '{key_name}' giữa Clean ({c_sig.get(key_name)!r}) và JPEG Q={expected_jpeg_quality} "
                        f"({j_sig.get(key_name)!r}) ở nhánh '{br}'! Có nguy cơ dùng nhầm checkpoint khác nhau."
                    )


def _compute_metric_degradation(
    clean_metrics: Mapping[str, Any],
    degraded_metrics: Mapping[str, Any],
) -> Dict[str, float]:
    """Tính độ suy giảm `metric(JPEG_q) - metric(Clean)` trên các chỉ số chính."""
    deltas: Dict[str, float] = {}
    for k in METRIC_KEYS:
        if k in clean_metrics and k in degraded_metrics:
            deltas[f"delta_{k}"] = float(degraded_metrics[k]) - float(clean_metrics[k])
    return deltas


def export_ablation_comparison_csv(
    rows: Sequence[Mapping[str, Any]],
    filepath: Union[str, Path],
) -> Path:
    """
    Xuất bảng đối chiếu kết quả các nhánh đơn gốc (Stage 1 / Linear Probe),
    7 tổ hợp Ablation (`FusionMLP` p=0.0) và biến thể `Branch Dropout (p=0.15)` ra file CSV.
    Ghi chú rõ `model_category` để phân biệt classifier đơn nhánh gốc với MLP 1 nhánh.
    """
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_key",
        "model_category",
        "head_type",
        "active_branches",
        "input_feature_dim",
        "trainable_params",
        "branch_dropout_p",
        "best_epoch",
        "val_auc",
        "val_balanced_acc",
        "test_macro_known_auc",
        "test_macro_known_bal_acc",
        "test_macro_unseen_auc",
        "test_macro_unseen_bal_acc",
        "test_macro_all_auc",
        "test_macro_all_bal_acc",
        "test_sd21_auc",
        "test_sdxl_auc",
        "test_dalle3_auc",
        "test_sd3_auc",
        "test_mjv6_auc",
        "checkpoint_path",
    ]
    with open(fp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return fp


def run_full_evaluation(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    eval_output_dir: Union[str, Path] = DEFAULT_STAGE2_EVAL_DIR,
    spatial_ckpt_path: Optional[Union[str, Path]] = DEFAULT_SPATIAL_CKPT,
    semantic_ckpt_path: Optional[Union[str, Path]] = DEFAULT_SEMANTIC_CKPT,
    spectral_ckpt_path: Optional[Union[str, Path]] = DEFAULT_SPECTRAL_CKPT,
    fusion_ckpt_path: Optional[Union[str, Path]] = DEFAULT_FUSION_CKPT,
    dropout_ckpt_path: Optional[Union[str, Path]] = DEFAULT_DROPOUT_CKPT,
    ablation_root_dir: Optional[Union[str, Path]] = DEFAULT_STAGE2_ABLATION_DIR,
    jpeg_qualities: Sequence[int] = DEFAULT_JPEG_QUALITIES,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
    require_all_models: bool = False,
    strict_jpeg: bool = False,
    include_ablation_in_jpeg: bool = True,
) -> Dict[str, Any]:
    """
    Thực hiện Đánh giá Cuối kỳ toàn diện trên tập Test Clean & JPEG Robustness (`90, 70, 50, 30`):
    1. Chấm điểm 3 mô hình Đơn nhánh gốc (`LinearProbeHead`):
       - `spatial_branch_head` (`best_spatial.pth`)
       - `semantic_probe` (`best_semantic.pth`)
       - `spectral_branch_head` (`best_spectral.pth`)
    2. Chấm điểm `fusion_baseline` (`best_fusion.pth`, `p=0.0`) và `fusion_branch_dropout`
       (`best_fusion_dropout.pth`, `p=0.15`).
    3. Chấm điểm toàn bộ 7 tổ hợp Ablation (`checkpoints/stage2/ablation/<slug>/best_<slug>.pth`).
       (Mặc định `include_ablation_in_jpeg=True` để chấm đủ 12/12 mô hình trên cả Clean & JPEG.)
    4. Xuất các tệp báo cáo chuẩn tại `checkpoints/stage2/evaluation/`:
       - `clean_test.json`, `per_generator.json`, `ablation_comparison.csv`
       - `jpeg_robustness.json` (chỉ tạo khi có dữ liệu JPEG; nếu `strict_jpeg=True` mà thiếu sẽ báo lỗi).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dir = Path(eval_output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Danh sách đủ 12 mô hình kỳ vọng để đối soát thiếu/đủ minh bạch
    expected_model_keys: List[str] = [
        "spatial_branch_head",
        "semantic_probe",
        "spectral_branch_head",
        "fusion_baseline",
        "fusion_branch_dropout",
        *[f"ablation_{get_ablation_combo_slug(c)}" for c in ALL_ABLATION_COMBINATIONS],
    ]

    # Danh sách các mô hình cần đánh giá
    evaluated_specs: List[Dict[str, Any]] = []
    missing_models: List[str] = []

    single_branch_ckpts = [
        ("spatial_branch_head", "single_branch_stage1_head", "spatial", spatial_ckpt_path),
        ("semantic_probe", "single_branch_linear_probe", "semantic", semantic_ckpt_path),
        ("spectral_branch_head", "single_branch_stage1_head", "spectral", spectral_ckpt_path),
    ]
    for model_key, category, br, ckpt_p in single_branch_ckpts:
        if ckpt_p is not None and Path(ckpt_p).is_file():
            probe = LinearProbeHead(branch=br, dropout=0.0)
            ckpt_meta = probe.load_from_branch_checkpoint(ckpt_p, device=device)
            evaluated_specs.append(
                {
                    "model_key": model_key,
                    "model_category": category,
                    "head_type": "linear",
                    "model": probe,
                    "active_branches": (br,),
                    "ckpt_meta": ckpt_meta,
                    "checkpoint_path": str(Path(ckpt_p)),
                    "include_in_jpeg": True,
                }
            )
        else:
            missing_models.append(model_key)

    if fusion_ckpt_path is not None and Path(fusion_ckpt_path).is_file():
        ckpt_raw = torch.load(fusion_ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt_raw.get("config", {})
        branches = tuple(cfg.get("active_branches", BRANCH_ORDER))
        fusion_model = FusionMLP(
            active_branches=branches,
            hidden_dims=tuple(cfg.get("hidden_dims", (512, 128))),
            hidden_dropouts=tuple(cfg.get("hidden_dropouts", (0.3, 0.2))),
            branch_dropout_p=float(cfg.get("branch_dropout_p", 0.0)),
            min_active_branches=int(cfg.get("min_active_branches", 2)),
        ).to(device)
        fusion_model.load_state_dict(ckpt_raw["model_state_dict"], strict=True)
        evaluated_specs.append(
            {
                "model_key": "fusion_baseline",
                "model_category": "fusion_mlp_baseline",
                "head_type": "mlp",
                "model": fusion_model,
                "active_branches": branches,
                "ckpt_meta": ckpt_raw,
                "checkpoint_path": str(Path(fusion_ckpt_path)),
                "include_in_jpeg": True,
            }
        )
    else:
        missing_models.append("fusion_baseline")

    if dropout_ckpt_path is not None and Path(dropout_ckpt_path).is_file():
        ckpt_raw = torch.load(dropout_ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt_raw.get("config", {})
        branches = tuple(cfg.get("active_branches", BRANCH_ORDER))
        drop_model = FusionMLP(
            active_branches=branches,
            hidden_dims=tuple(cfg.get("hidden_dims", (512, 128))),
            hidden_dropouts=tuple(cfg.get("hidden_dropouts", (0.3, 0.2))),
            branch_dropout_p=float(cfg.get("branch_dropout_p", 0.15)),
            min_active_branches=int(cfg.get("min_active_branches", 2)),
        ).to(device)
        drop_model.load_state_dict(ckpt_raw["model_state_dict"], strict=True)
        evaluated_specs.append(
            {
                "model_key": "fusion_branch_dropout",
                "model_category": "fusion_mlp_branch_dropout",
                "head_type": "mlp",
                "model": drop_model,
                "active_branches": branches,
                "ckpt_meta": ckpt_raw,
                "checkpoint_path": str(Path(dropout_ckpt_path)),
                "include_in_jpeg": True,
            }
        )
    else:
        missing_models.append("fusion_branch_dropout")

    abl_root = Path(ablation_root_dir) if ablation_root_dir is not None else None
    for combo in ALL_ABLATION_COMBINATIONS:
        slug = get_ablation_combo_slug(combo)
        m_key = f"ablation_{slug}"
        ckpt_fp = (abl_root / slug / f"best_{slug}.pth") if (abl_root is not None and abl_root.is_dir()) else None
        if ckpt_fp is None or not ckpt_fp.is_file():
            missing_models.append(m_key)
            continue
        ckpt_raw = torch.load(ckpt_fp, map_location=device, weights_only=False)
        cfg = ckpt_raw.get("config", {})
        branches = normalize_branch_selection(cfg.get("active_branches", combo))
        abl_model = FusionMLP(
            active_branches=branches,
            hidden_dims=tuple(cfg.get("hidden_dims", (512, 128))),
            hidden_dropouts=tuple(cfg.get("hidden_dropouts", (0.3, 0.2))),
            branch_dropout_p=float(cfg.get("branch_dropout_p", 0.0)),
            min_active_branches=int(cfg.get("min_active_branches", 2)),
        ).to(device)
        abl_model.load_state_dict(ckpt_raw["model_state_dict"], strict=True)
        evaluated_specs.append(
            {
                "model_key": m_key,
                "model_category": "ablation_fusion_mlp_head",
                "head_type": "mlp",
                "model": abl_model,
                "active_branches": branches,
                "ckpt_meta": ckpt_raw,
                "checkpoint_path": str(ckpt_fp),
                "include_in_jpeg": bool(include_ablation_in_jpeg),
            }
        )

    if require_all_models and missing_models:
        raise FileNotFoundError(
            f"[--strict-eval / require_all_models=True] Thiếu {len(missing_models)}/{len(expected_model_keys)} "
            f"checkpoint bắt buộc để đánh giá toàn diện: {missing_models}"
        )
    if not evaluated_specs:
        raise FileNotFoundError("Không tìm thấy bất kỳ checkpoint nào để thực hiện đánh giá!")
    if missing_models:
        print(f"[!] Cảnh báo: Hiện thiếu {len(missing_models)} mô hình chưa đánh giá: {missing_models}")

    # 1. Đánh giá trên Clean Test
    clean_results: Dict[str, Any] = {}
    per_generator_summary: Dict[str, Any] = {}
    ablation_csv_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 88)
    print(f"ĐÁNH GIÁ TRÊN TẬP CLEAN TEST ({len(evaluated_specs)}/{len(expected_model_keys)} MÔ HÌNH)")
    print("=" * 88)

    for spec in evaluated_specs:
        m_key = spec["model_key"]
        model_obj: nn.Module = spec["model"]
        branches = spec["active_branches"]
        ckpt_meta = spec["ckpt_meta"]
        cfg = ckpt_meta.get("config", {}) or {}
        best_val = ckpt_meta.get("best_metrics", ckpt_meta.get("metrics", {})) or {}

        res_clean = evaluate_stage2_model_on_cache(
            model=model_obj,
            cache_dir=cache_dir,
            active_branches=branches,
            split="test",
            jpeg_quality=None,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            device=device,
            use_amp=use_amp,
        )
        clean_results[m_key] = {
            "model_category": spec["model_category"],
            "head_type": spec["head_type"],
            "active_branches": list(branches),
            "checkpoint_path": spec["checkpoint_path"],
            "best_epoch": ckpt_meta.get("epoch", cfg.get("best_epoch", None)),
            "val_best_metrics": best_val,
            "config": cfg,
            "test_clean": res_clean,
        }
        per_generator_summary[m_key] = {
            "model_category": spec["model_category"],
            "active_branches": list(branches),
            "num_real": res_clean["num_real"],
            "generator_sample_counts": res_clean["generator_sample_counts"],
            "macro_avg_known": res_clean["macro_avg_known"],
            "macro_avg_unseen": res_clean["macro_avg_unseen"],
            "macro_avg_all": res_clean["macro_avg_all"],
            "pooled_known": res_clean["pooled_known"],
            "pooled_unseen": res_clean["pooled_unseen"],
            "per_generator": res_clean["per_generator"],
        }

        per_gen = res_clean["per_generator"]
        mk = res_clean["macro_avg_known"]
        mu = res_clean["macro_avg_unseen"]
        ma = res_clean["macro_avg_all"]

        print(
            f"  * {m_key:<24} | Known AUC={mk.get('auc', 0.0):.4f} (BalACC={mk.get('balanced_acc', 0.0)*100:.2f}%) | "
            f"Unseen AUC={mu.get('auc', 0.0):.4f} (BalACC={mu.get('balanced_acc', 0.0)*100:.2f}%) | "
            f"All AUC={ma.get('auc', 0.0):.4f}"
        )

        ablation_csv_rows.append(
            {
                "model_key": m_key,
                "model_category": spec["model_category"],
                "head_type": spec["head_type"],
                "active_branches": "+".join(branches),
                "input_feature_dim": int(getattr(model_obj, "input_feature_dim", 0)),
                "trainable_params": int(model_obj.count_parameters(trainable_only=False)),
                "branch_dropout_p": float(getattr(model_obj, "branch_dropout_p", 0.0)),
                "best_epoch": ckpt_meta.get("epoch", cfg.get("best_epoch", "")),
                "val_auc": f"{float(best_val.get('auc', cfg.get('best_val_auc', 0.0))):.6f}",
                "val_balanced_acc": f"{float(best_val.get('balanced_acc', cfg.get('best_val_balanced_acc', 0.0))):.6f}",
                "test_macro_known_auc": f"{float(mk.get('auc', 0.0)):.6f}",
                "test_macro_known_bal_acc": f"{float(mk.get('balanced_acc', 0.0)):.6f}",
                "test_macro_unseen_auc": f"{float(mu.get('auc', 0.0)):.6f}",
                "test_macro_unseen_bal_acc": f"{float(mu.get('balanced_acc', 0.0)):.6f}",
                "test_macro_all_auc": f"{float(ma.get('auc', 0.0)):.6f}",
                "test_macro_all_bal_acc": f"{float(ma.get('balanced_acc', 0.0)):.6f}",
                "test_sd21_auc": f"{float(per_gen.get('Stable_Diffusion_2.1', {}).get('auc', 0.0)):.6f}",
                "test_sdxl_auc": f"{float(per_gen.get('Stable_Diffusion_XL', {}).get('auc', 0.0)):.6f}",
                "test_dalle3_auc": f"{float(per_gen.get('DALL-E_3', {}).get('auc', 0.0)):.6f}",
                "test_sd3_auc": f"{float(per_gen.get('Stable_Diffusion_3', {}).get('auc', 0.0)):.6f}",
                "test_mjv6_auc": f"{float(per_gen.get('Midjourney_V6', {}).get('auc', 0.0)):.6f}",
                "checkpoint_path": spec["checkpoint_path"],
            }
        )

    # 2. Đánh giá JPEG Robustness (mặc định đủ cả 12 mô hình ở Q = 90, 70, 50, 30)
    jpeg_specs = [s for s in evaluated_specs if s["include_in_jpeg"]]
    required_jpeg_branches = sorted({br for s in jpeg_specs for br in s["active_branches"]})
    jpeg_report: Optional[Dict[str, Any]] = None
    jpeg_json_path: Optional[Path] = None

    requested_qs = [int(q) for q in jpeg_qualities]
    missing_by_q: Dict[int, List[str]] = {}
    for q in requested_qs:
        miss_q = [
            str(get_cache_filepath(cache_dir, branch=br, split="test", jpeg_quality=q))
            for br in required_jpeg_branches
            if not get_cache_filepath(cache_dir, branch=br, split="test", jpeg_quality=q).is_file()
        ]
        if miss_q:
            missing_by_q[q] = miss_q

    should_run_jpeg = bool(requested_qs)
    if should_run_jpeg and missing_by_q:
        if strict_jpeg:
            raise FileNotFoundError(
                f"Thiếu file cache JPEG cho các mức Q={sorted(missing_by_q.keys())}. "
                f"Hãy chạy `extract_features.py --splits test --jpeg-qualities {','.join(str(q) for q in sorted(missing_by_q.keys()))}` "
                f"trước, hoặc truyền `--jpeg-qualities none` / `--allow-missing-jpeg` nếu chỉ chấm Clean Test. "
                f"Chi tiết file thiếu đầu tiên: {next(iter(missing_by_q.values()))}"
            )
        if len(missing_by_q) < len(requested_qs):
            raise FileNotFoundError(
                f"Phát hiện bộ cache JPEG không đầy đủ (có một số mức Q nhưng thiếu mức Q={sorted(missing_by_q.keys())})! "
                f"Vui lòng trích xuất bù để tránh xuất báo cáo JPEG lệch mức."
            )
        print(
            f"\n[!] Chưa tìm thấy bộ cache JPEG (*_test_jpeg*.pt) cho Q={requested_qs}. "
            f"Chỉ xuất kết quả Clean Test và KHÔNG tạo file `jpeg_robustness.json` rỗng."
        )
        should_run_jpeg = False

    if should_run_jpeg:
        clean_ref_payloads, _ = load_aligned_branch_caches(
            cache_dir=cache_dir,
            branches=required_jpeg_branches,
            split="test",
            jpeg_quality=None,
        )

        jpeg_report = {
            "qualities_requested": requested_qs,
            "qualities_evaluated": [],
            "evaluated_models": [s["model_key"] for s in jpeg_specs],
            "missing_models": list(missing_models),
            "by_quality": {},
            "by_model": {
                s["model_key"]: {
                    "clean": {
                        "macro_avg_known": clean_results[s["model_key"]]["test_clean"]["macro_avg_known"],
                        "macro_avg_unseen": clean_results[s["model_key"]]["test_clean"]["macro_avg_unseen"],
                        "macro_avg_all": clean_results[s["model_key"]]["test_clean"]["macro_avg_all"],
                        "overall": clean_results[s["model_key"]]["test_clean"]["overall"],
                    },
                    "jpeg": {},
                }
                for s in jpeg_specs
            },
        }

        print("\n" + "=" * 88)
        print(f"ĐÁNH GIÁ JPEG ROBUSTNESS ({len(jpeg_specs)} MÔ HÌNH x Q={requested_qs})")
        print("=" * 88)

        for q in requested_qs:
            jpeg_payloads, _ = load_aligned_branch_caches(
                cache_dir=cache_dir,
                branches=required_jpeg_branches,
                split="test",
                jpeg_quality=q,
            )
            verify_jpeg_matches_clean_metadata(clean_ref_payloads, jpeg_payloads, expected_jpeg_quality=q)

            q_key = f"jpeg_{q}"
            jpeg_report["qualities_evaluated"].append(int(q))
            jpeg_report["by_quality"][q_key] = {}

            print(f"\n---> Đánh giá JPEG Robustness [Quality = {q}]:")
            for spec in jpeg_specs:
                m_key = spec["model_key"]
                res_q = evaluate_stage2_model_on_cache(
                    model=spec["model"],
                    cache_dir=cache_dir,
                    active_branches=spec["active_branches"],
                    split="test",
                    jpeg_quality=q,
                    eval_batch_size=eval_batch_size,
                    num_workers=num_workers,
                    device=device,
                    use_amp=use_amp,
                )
                clean_ref = clean_results[m_key]["test_clean"]
                degradation = {
                    "macro_avg_known": _compute_metric_degradation(
                        clean_ref["macro_avg_known"], res_q["macro_avg_known"]
                    ),
                    "macro_avg_unseen": _compute_metric_degradation(
                        clean_ref["macro_avg_unseen"], res_q["macro_avg_unseen"]
                    ),
                    "macro_avg_all": _compute_metric_degradation(
                        clean_ref["macro_avg_all"], res_q["macro_avg_all"]
                    ),
                    "overall": _compute_metric_degradation(clean_ref["overall"], res_q["overall"]),
                }

                entry_q = {
                    "jpeg_quality": int(q),
                    "macro_avg_known": res_q["macro_avg_known"],
                    "macro_avg_unseen": res_q["macro_avg_unseen"],
                    "macro_avg_all": res_q["macro_avg_all"],
                    "pooled_known": res_q["pooled_known"],
                    "pooled_unseen": res_q["pooled_unseen"],
                    "overall": res_q["overall"],
                    "per_generator": res_q["per_generator"],
                    "degradation_from_clean": degradation,
                    "extraction_signatures": res_q["extraction_signatures"],
                }
                jpeg_report["by_quality"][q_key][m_key] = entry_q
                jpeg_report["by_model"][m_key]["jpeg"][q_key] = entry_q

                mk_q = res_q["macro_avg_known"]
                mu_q = res_q["macro_avg_unseen"]
                d_all_auc = degradation["macro_avg_all"].get("delta_auc", 0.0)
                print(
                    f"  * {m_key:<24} | Q={q:<2} | Known AUC={mk_q.get('auc', 0.0):.4f} | "
                    f"Unseen AUC={mu_q.get('auc', 0.0):.4f} | Delta All AUC={d_all_auc:+.4f}"
                )

        jpeg_json_path = save_json(jpeg_report, eval_dir / "jpeg_robustness.json")

    clean_json_path = save_json(clean_results, eval_dir / "clean_test.json")
    per_gen_json_path = save_json(per_generator_summary, eval_dir / "per_generator.json")
    ablation_csv_path = export_ablation_comparison_csv(
        ablation_csv_rows, eval_dir / "ablation_comparison.csv"
    )

    files_out: Dict[str, str] = {
        "clean_test_json": str(clean_json_path),
        "per_generator_json": str(per_gen_json_path),
        "ablation_comparison_csv": str(ablation_csv_path),
    }
    if jpeg_json_path is not None:
        files_out["jpeg_robustness_json"] = str(jpeg_json_path)

    return {
        "evaluated_models": [s["model_key"] for s in evaluated_specs],
        "missing_models": list(missing_models),
        "clean_test": clean_results,
        "per_generator": per_generator_summary,
        "jpeg_robustness": jpeg_report,
        "ablation_rows": ablation_csv_rows,
        "files": files_out,
    }


# ==============================================================================
# BỘ KIỂM THỬ ĐƠN VỊ & LIÊN KẾT TOÀN HỆ THỐNG (8 GATES)
# ==============================================================================
def _create_synthetic_stage2_environment(
    root_dir: Path,
    num_train_groups: int = 35,
    num_val_groups: int = 8,
    num_test_groups: int = 8,
    jpeg_qualities: Sequence[int] = DEFAULT_JPEG_QUALITIES,
) -> Dict[str, Path]:
    """
    Khởi tạo môi trường dữ liệu `.pt` và checkpoint đơn nhánh giả lập có cấu trúc chuẩn 100%:
    - Train / Val: mỗi nhóm có đúng 4 ảnh (`MS_COCO`, `Stable_Diffusion_2.1`, `Stable_Diffusion_XL`, `DALL-E_3`).
      Với `num_train_groups = 35` -> `140` ảnh Train (`35` Real + `105` AI) -> mỗi epoch `BalancedGroupSampler`
      lấy `70` ảnh (`2` batch 32 + `1` batch dư 6 ảnh = `3 Real : 3 AI`).
    - Test (Clean + JPEG `90, 70, 50, 30`): mỗi nhóm có đúng 6 ảnh (`MS_COCO` + 3 Known + 2 Unseen
      `Stable_Diffusion_3`, `Midjourney_V6`) với số lượng ảnh từng generator bằng đúng số ảnh Real
      để kiểm tra cả cặp 1:1 `evaluate_by_generator(require_balanced_pairs=True)`.
    """
    cache_dir = root_dir / "feature_cache"
    ckpt_root = root_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(2026)

    # Tạo 3 vector mẫu phân tách tuyến tính cho 3 nhánh để mô hình học hội tụ nhanh trong test
    true_dirs = {
        br: torch.nn.functional.normalize(torch.randn(dim, generator=g), p=2, dim=0)
        for br, dim in BRANCH_FEATURE_DIMS.items()
    }

    def _build_split_caches(
        split: str,
        num_groups: int,
        include_unseen: bool,
        has_flip: bool,
        jpeg_quality: Optional[int] = None,
        base_records: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ai_Rotation = (
            list(KNOWN_GENERATORS) + list(UNSEEN_GENERATORS)
            if include_unseen
            else list(KNOWN_GENERATORS)
        )
        if base_records is None:
            filepaths: List[str] = []
            model_names: List[str] = []
            labels_a: List[float] = []
            labels_b: List[int] = []
            group_ids: List[int] = []

            model_to_b = dict(MODEL_NAME_TO_LABEL_B)

            if not include_unseen:
                # Cấu trúc chuẩn Train/Val: mỗi group_id có đúng 1 Real + 3 Known AI = 4 ảnh
                for gid in range(num_groups):
                    filepaths.append(f"{split}/real_{gid:04d}.jpg")
                    model_names.append(REAL_MODEL_NAME)
                    labels_a.append(0.0)
                    labels_b.append(0)
                    group_ids.append(gid)
                    for gen_name in sorted(KNOWN_GENERATORS):
                        filepaths.append(f"{split}/{gen_name}_{gid:04d}.jpg")
                        model_names.append(gen_name)
                        labels_a.append(1.0)
                        labels_b.append(model_to_b[gen_name])
                        group_ids.append(gid)
            else:
                # Cấu trúc chuẩn Test 1:1 per-generator: num_groups Real + num_groups cho mỗi trong 5 AI generators
                for gid in range(num_groups):
                    filepaths.append(f"{split}/real_{gid:04d}.jpg")
                    model_names.append(REAL_MODEL_NAME)
                    labels_a.append(0.0)
                    labels_b.append(0)
                    group_ids.append(gid)
                    for gen_name in ai_Rotation:
                        filepaths.append(f"{split}/{gen_name}_{gid:04d}.jpg")
                        model_names.append(gen_name)
                        labels_a.append(1.0)
                        labels_b.append(model_to_b[gen_name])
                        group_ids.append(gid)

            n_total = len(filepaths)
            s_idx = torch.arange(n_total, dtype=torch.int64)
            lbl_a_t = torch.tensor(labels_a, dtype=torch.float32)
            lbl_b_t = torch.tensor(labels_b, dtype=torch.int64)
            grp_t = torch.tensor(group_ids, dtype=torch.int64)
            m_hash = compute_manifest_hash(
                filepaths=filepaths,
                labels_a=lbl_a_t,
                model_names=model_names,
                sample_indices=s_idx,
                labels_b=lbl_b_t,
                group_ids=grp_t,
            )
            base_records = {
                "num_samples": n_total,
                "sample_idx": s_idx,
                "filepaths": filepaths,
                "model_names": model_names,
                "label_a": lbl_a_t,
                "label_b": lbl_b_t,
                "group_id": grp_t,
                "manifest_hash": m_hash,
            }

        n_total = int(base_records["num_samples"])
        y_sign = (base_records["label_a"] * 2.0 - 1.0).unsqueeze(1)  # -1 cho Real, +1 cho AI
        noise_scale = 0.25 if jpeg_quality is None else (0.25 + (100 - int(jpeg_quality)) * 0.005)

        for br, dim in BRANCH_FEATURE_DIMS.items():
            signal = y_sign * true_dirs[br].unsqueeze(0) * 2.5
            feats_orig = signal + torch.randn((n_total, dim), generator=g) * noise_scale
            if br == "semantic":
                feats_orig = torch.nn.functional.normalize(feats_orig, p=2, dim=-1)

            feats_flip: Optional[torch.Tensor] = None
            if has_flip:
                feats_flip = signal + torch.randn((n_total, dim), generator=g) * noise_scale
                if br == "semantic":
                    feats_flip = torch.nn.functional.normalize(feats_flip, p=2, dim=-1)

            ckpt_info = {"source": f"mock_{br}", "checkpoint_sha256": f"mock_sha_{br}"}
            prep_cfg = build_preprocess_config(
                branch=br,
                include_flip=bool(has_flip),
                fixed_jpeg_quality=jpeg_quality,
                use_amp=False,
                amp_dtype=torch.float32,
            )
            ext_sig = build_extraction_signature(
                branch=br,
                split=split,
                checkpoint_info=ckpt_info,
                preprocess_cfg=prep_cfg,
                include_flip=bool(has_flip),
                fixed_jpeg_quality=jpeg_quality,
                use_amp=False,
                amp_dtype=torch.float32,
            )
            payload = {
                "branch": br,
                "split": split,
                "feature_dim": dim,
                "num_samples": n_total,
                "has_flip": bool(has_flip),
                "jpeg_quality": jpeg_quality,
                "manifest_hash": base_records["manifest_hash"],
                "features": feats_orig.to(dtype=torch.float32),
                "features_flip": feats_flip.to(dtype=torch.float32) if feats_flip is not None else None,
                "sample_idx": base_records["sample_idx"].clone(),
                "filepaths": list(base_records["filepaths"]),
                "label_a": base_records["label_a"].clone(),
                "label_b": base_records["label_b"].clone(),
                "group_id": base_records["group_id"].clone(),
                "model_names": list(base_records["model_names"]),
                "checkpoint_info": ckpt_info,
                "preprocess_config": prep_cfg,
                "extraction_signature": ext_sig,
                "created_at": "2026-11-16T00:00:00",
            }
            fp = get_cache_filepath(cache_dir, branch=br, split=split, jpeg_quality=jpeg_quality)
            save_feature_cache_atomic(payload, fp)

        return base_records

    _build_split_caches("train", num_train_groups, include_unseen=False, has_flip=True, jpeg_quality=None)
    _build_split_caches("val", num_val_groups, include_unseen=False, has_flip=False, jpeg_quality=None)
    test_base = _build_split_caches("test", num_test_groups, include_unseen=True, has_flip=False, jpeg_quality=None)
    for q in jpeg_qualities:
        _build_split_caches(
            "test",
            num_test_groups,
            include_unseen=True,
            has_flip=False,
            jpeg_quality=int(q),
            base_records=test_base,
        )

    # Tạo sẵn 2 checkpoint giả lập cho Spatial và Spectral (chứa classifier.1.weight/bias chuẩn)
    spatial_ckpt_fp = ckpt_root / "spatial_baseline" / "best_spatial.pth"
    spectral_ckpt_fp = ckpt_root / "spectral_baseline" / "best_spectral.pth"
    spatial_ckpt_fp.parent.mkdir(parents=True, exist_ok=True)
    spectral_ckpt_fp.parent.mkdir(parents=True, exist_ok=True)

    for br, fp in (("spatial", spatial_ckpt_fp), ("spectral", spectral_ckpt_fp)):
        dim = BRANCH_FEATURE_DIMS[br]
        w = (true_dirs[br].unsqueeze(0) * 3.0).to(dtype=torch.float32)
        b = torch.zeros(1, dtype=torch.float32)
        torch.save(
            {
                "epoch": 10,
                "model_state_dict": {
                    "backbone.dummy": torch.ones(2, 2),
                    "classifier.1.weight": w,
                    "classifier.1.bias": b,
                },
                "metrics": {"auc": 0.99, "balanced_acc": 0.96},
                "best_metrics": {"auc": 0.99, "balanced_acc": 0.96},
                "config": {"branch": br},
            },
            fp,
        )

    return {
        "cache_dir": cache_dir,
        "ckpt_root": ckpt_root,
        "spatial_ckpt": spatial_ckpt_fp,
        "spectral_ckpt": spectral_ckpt_fp,
        "semantic_ckpt": ckpt_root / "semantic_baseline" / "best_semantic.pth",
        "stage2_dir": ckpt_root / "stage2",
    }


def run_unit_tests() -> None:
    """
    Chạy trọn bộ 8 nhóm kiểm thử đơn vị và tích hợp cho `train_fusion.py`:
      [Test 1] Kiểm định `CachedFeatureDataset`, `build_sampler_dataframe` & chặn lệch căn chỉnh mẫu.
      [Test 2] Đồng bộ tuyệt đối góc nhìn Original/Flip (`p=0.5`) trên cả 3 nhánh & khóa chặt ở Val/Test.
      [Test 3] Kiểm chứng `BalancedGroupSampler` trên `CachedFeatureDataset`: cân bằng 16:16 từng batch,
               xử lý batch dư 50:50, bảo toàn cặp `group_id` và phủ ĐÚNG 100.00% ảnh AI sau 3 epochs.
      [Test 4] Kiểm tra tách `weight_decay = 0.0` cho `LayerNorm` & `bias` trong `AdamW` + Scheduler.
      [Test 5] Huấn luyện `train_semantic_probe` -> kiểm tra vị trí lưu `best_semantic.pth` (ngang hàng
               `best_spatial.pth` & `best_spectral.pth`) -> nạp ngược vào `SemanticBranch` (`CLIP.py`)
               và đối chứng logit khớp tuyệt đối (`max_diff == 0.0`).
      [Test 6] Huấn luyện `train_fusion_baseline` (`p=0.0`) + Cổng Overfit 1 batch & Self-Contained Resume.
      [Test 7] Chạy Ablation 7 tổ hợp độc lập + Branch Dropout (`p=0.15`) + Đánh giá Clean & JPEG
               (`90, 70, 50, 30`) xuất đủ 4 tệp báo cáo.
      [Test 8] End-to-End Integration Smoke Test trên dữ liệu ảnh thật (`extract_features` -> `train_fusion`).
    """
    print("=" * 88)
    print("BẮT ĐẦU KIỂM THỬ TOÀN DIỆN: code/system/train/train_fusion.py (8 GATES)")
    print("=" * 88)

    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory(prefix="stage2_train_fusion_test_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        env = _create_synthetic_stage2_environment(tmp_root)
        cache_dir = env["cache_dir"]

        # ------------------------------------------------------------------
        # TEST 1: CachedFeatureDataset & Chặn lệch căn chỉnh mẫu
        # ------------------------------------------------------------------
        print("\n[Test 1] Kiểm tra CachedFeatureDataset, build_sampler_dataframe & chặn lệch mẫu...")
        payloads_tr, align_tr = load_aligned_branch_caches(cache_dir, BRANCH_ORDER, split="train")
        assert align_tr["aligned"] is True
        ds_tr = CachedFeatureDataset(payloads_tr, BRANCH_ORDER, mode="train", flip_prob=0.5, seed=42)
        df_tr = ds_tr.build_sampler_dataframe()
        assert len(df_tr) == len(ds_tr) == 140
        assert list(df_tr.columns) == ["sample_idx", "group_id", "label_a", "label_b", "model_name", "filepath"]

        # Thử làm lệch sample_idx của 1 nhánh -> phải bị chặn ngay
        bad_payloads = {k: dict(v) for k, v in payloads_tr.items()}
        bad_payloads["spectral"]["sample_idx"] = payloads_tr["spectral"]["sample_idx"].clone()
        bad_payloads["spectral"]["sample_idx"][0] = 999
        caught_misalign = False
        try:
            CachedFeatureDataset(bad_payloads, BRANCH_ORDER, mode="train")
        except ValueError:
            caught_misalign = True
        assert caught_misalign, "Không chặn được lỗi lệch sample_idx giữa các nhánh!"

        # Chặn num_workers > 0 khi flip_prob > 0 (tránh persistent_workers giữ bản sao flip cũ)
        caught_workers = False
        try:
            build_stage2_dataloaders(cache_dir=cache_dir, active_branches=BRANCH_ORDER, num_workers=2, flip_prob=0.5)
        except ValueError:
            caught_workers = True
        assert caught_workers, "Phải chặn num_workers > 0 khi flip_prob > 0!"
        print("  -> Nạp 3 nhánh chuẩn xác, dựng DataFrame từ RAM khớp 100%, chặn lệch mẫu & chặn worker lệch flip -> [PASS]")

        # ------------------------------------------------------------------
        # TEST 2: Đồng bộ Original / Flip (p=0.5) giữa cả 3 nhánh
        # ------------------------------------------------------------------
        print("\n[Test 2] Kiểm tra đồng bộ góc nhìn Original/Flip (p=0.5) cho cả 3 nhánh...")
        marker_payloads = {k: dict(v) for k, v in payloads_tr.items()}
        for br, dim in BRANCH_FEATURE_DIMS.items():
            idx_col = torch.arange(140, dtype=torch.float32).unsqueeze(1)
            raw_pos = torch.full((140, dim), 10.0) + idx_col
            raw_neg = -(torch.full((140, dim), 10.0) + idx_col)
            if br == "semantic":
                # Nhánh semantic bắt buộc có L2-norm = 1.0 theo quy ước của validate_feature_cache_payload
                theta = (torch.arange(140, dtype=torch.float32) + 1.0) * 0.005
                sem_pos = torch.zeros((140, dim), dtype=torch.float32)
                sem_pos[:, 0] = torch.cos(theta)
                sem_pos[:, 1] = torch.sin(theta)
                marker_payloads[br]["features"] = sem_pos
                marker_payloads[br]["features_flip"] = -sem_pos
            else:
                marker_payloads[br]["features"] = raw_pos
                marker_payloads[br]["features_flip"] = raw_neg

        ds_sync = CachedFeatureDataset(marker_payloads, BRANCH_ORDER, mode="train", flip_prob=0.5, seed=123)
        flip_flags: List[bool] = []
        for i in range(len(ds_sync)):
            sample_a = ds_sync[i]
            sample_b = ds_sync[i]
            # Gọi 2 lần trong cùng 1 epoch phải trả về đúng cùng 1 góc nhìn
            assert torch.equal(sample_a["fusion"], sample_b["fusion"])
            s_sign = float(sample_a["spatial"][0].item()) > 0
            c_sign = float(sample_a["semantic"][0].item()) > 0
            f_sign = float(sample_a["spectral"][0].item()) > 0
            # Cả 3 nhánh bắt buộc cùng dương (Orig) hoặc cùng âm (Flip), đồng thời đúng chỉ số mẫu i
            assert s_sign == c_sign == f_sign, f"Lệch góc nhìn Orig/Flip giữa các nhánh tại mẫu {i}!"
            expected_mag = 10.0 + float(i)
            assert abs(abs(float(sample_a["spatial"][0].item())) - expected_mag) < 1e-4
            assert bool(sample_a["is_flipped"].item()) == (not s_sign)
            flip_flags.append(bool(sample_a["is_flipped"].item()))

        flip_rate = sum(flip_flags) / len(flip_flags)
        assert 0.30 <= flip_rate <= 0.70, f"Tỷ lệ Flip lệch bất thường: {flip_rate:.3f}"

        # Ở chế độ eval/val/test, 100% mẫu phải dùng bản gốc (is_flipped == False)
        ds_eval_sync = CachedFeatureDataset(marker_payloads, BRANCH_ORDER, mode="val", flip_prob=0.5, seed=123)
        assert all(not bool(ds_eval_sync[i]["is_flipped"].item()) for i in range(len(ds_eval_sync)))
        print(
            f"  -> 140/140 mẫu đồng bộ góc nhìn tuyệt đối cả 3 nhánh (flip_rate={flip_rate:.2%}, eval_flip=0%) -> [PASS]"
        )

        # ------------------------------------------------------------------
        # TEST 3: BalancedGroupSampler 3-Epoch 100% AI Coverage & 16:16 Batch
        # ------------------------------------------------------------------
        print("\n[Test 3] Kiểm tra BalancedGroupSampler chu kỳ 3 Epochs phủ 100% AI & cân bằng 16:16...")
        train_loader_test, _, _ = build_stage2_dataloaders(
            cache_dir=cache_dir,
            active_branches=BRANCH_ORDER,
            batch_size=32,
            eval_batch_size=64,
            num_workers=0,
            flip_prob=0.5,
            seed=42,
        )
        seen_ai_sample_indices: set = set()
        total_ai_in_cache = int((ds_tr.labels_int == 1).sum().item())
        total_real_in_cache = int((ds_tr.labels_int == 0).sum().item())
        assert total_real_in_cache == 35 and total_ai_in_cache == 105

        flip_vectors_by_epoch: List[np.ndarray] = []
        for ep in range(3):
            # Kiểm tra đồng bộ NGƯỢC chiều: chỉ gọi `sampler.set_epoch(ep)` cũng phải tự cập nhật
            # `dataset._epoch_flip_decisions` cho đúng epoch `ep` (chống kẹt flip ở epoch 0).
            train_loader_test.sampler.set_epoch(ep)
            assert train_loader_test.dataset.epoch == ep
            flip_vectors_by_epoch.append(train_loader_test.dataset._epoch_flip_decisions.copy())

            ep_real_indices: List[int] = []
            ep_ai_indices: List[int] = []
            ep_groups: List[int] = []

            for b_idx, batch in enumerate(train_loader_test):
                y_b = batch["label_a"]
                n_real_b = int((y_b == 0.0).sum().item())
                n_ai_b = int((y_b == 1.0).sum().item())
                if len(y_b) == 32:
                    assert n_real_b == 16 and n_ai_b == 16, f"Batch {b_idx} lệch: {n_real_b}:{n_ai_b}"
                else:
                    # Batch dư cuối (6 ảnh = 3 cặp)
                    assert len(y_b) == 6 and n_real_b == 3 and n_ai_b == 3

                s_idxs = [int(x) for x in batch["sample_idx"]]
                for s_i, lbl in zip(s_idxs, y_b.tolist()):
                    if lbl == 0.0:
                        ep_real_indices.append(s_i)
                    else:
                        ep_ai_indices.append(s_i)
                ep_groups.extend([int(g_i) for g_i in batch["group_id"].tolist()])

            # Mỗi epoch duyệt đủ 100% ảnh Real (35/35) và mỗi group_id xuất hiện đúng 2 lần (1 Real, 1 AI)
            assert len(set(ep_real_indices)) == 35
            _, g_counts = np.unique(ep_groups, return_counts=True)
            assert np.all(g_counts == 2)

            # Không được trùng ảnh AI giữa các epoch trong cùng chu kỳ 3 epochs
            overlap_ai = seen_ai_sample_indices.intersection(ep_ai_indices)
            assert len(overlap_ai) == 0, f"Trùng {len(overlap_ai)} ảnh AI ở epoch {ep}!"
            seen_ai_sample_indices.update(ep_ai_indices)

        # Kiểm chứng các quyết định lật ngang thực sự thay đổi qua các epochs (không bị kẹt ở epoch 0)
        assert not np.array_equal(flip_vectors_by_epoch[0], flip_vectors_by_epoch[1])
        assert not np.array_equal(flip_vectors_by_epoch[1], flip_vectors_by_epoch[2])

        # Kiểm tra bổ sung _sync_loader_epoch của engine.py cũng đồng bộ trọn vẹn cả 2 chiều
        from engine import _sync_loader_epoch
        _sync_loader_epoch(train_loader_test, 5)
        assert train_loader_test.dataset.epoch == 5 and train_loader_test.sampler.epoch == 5

        assert len(seen_ai_sample_indices) == total_ai_in_cache == 105
        print(
            f"  -> Batch 16:16 + Batch dư 3:3 | Đồng bộ 2 chiều dataset<->sampler qua mọi epoch | "
            f"Phủ {len(seen_ai_sample_indices)}/{total_ai_in_cache} (100.00%) ảnh AI sau 3 epochs -> [PASS]"
        )

        # ------------------------------------------------------------------
        # TEST 4: Tách weight_decay=0.0 cho LayerNorm & bias trong AdamW
        # ------------------------------------------------------------------
        print("\n[Test 4] Kiểm tra tách weight_decay=0.0 cho LayerNorm & bias trong AdamW...")
        tmp_mlp = FusionMLP()
        opt_mlp, _ = build_stage2_optimizer_and_scheduler(
            tmp_mlp, lr=5e-4, weight_decay=1e-4, total_epochs=30, warmup_epochs=1
        )
        pg_by_name = {pg["group_name"]: pg for pg in opt_mlp.param_groups}
        assert pg_by_name["decay"]["weight_decay"] == 1e-4
        assert pg_by_name["no_decay"]["weight_decay"] == 0.0
        # 3 LayerNorms (1280*2 + 512*2 + 256*2 = 4096) + 3 Linear biases (512 + 128 + 1 = 641) = 4,737 params
        no_decay_count = sum(p.numel() for p in pg_by_name["no_decay"]["params"])
        decay_count = sum(p.numel() for p in pg_by_name["decay"]["params"])
        assert no_decay_count == 4737 and decay_count + no_decay_count == 1118977
        print(
            f"  -> Decay params={decay_count:,} (wd=1e-4) | No-decay params={no_decay_count:,} (wd=0.0) -> [PASS]"
        )

        # ------------------------------------------------------------------
        # TEST 5: Train Semantic Probe -> best_semantic.pth -> SemanticBranch (CLIP.py)
        # ------------------------------------------------------------------
        print("\n[Test 5] Kiểm tra train_semantic_probe & tương thích ngược với SemanticBranch (CLIP.py)...")
        from CLIP import SemanticBranch

        probe_res = train_semantic_probe(
            cache_dir=cache_dir,
            semantic_ckpt_path=env["semantic_ckpt"],
            stage2_probe_dir=env["stage2_dir"] / "semantic_probe",
            epochs=6,
            patience=6,
            batch_size=32,
            seed=42,
            device=test_device,
        )
        # Kiểm tra đúng quy tắc thư mục: best_semantic.pth nằm ở semantic_baseline/, phần còn lại ở stage2/semantic_probe/
        assert env["semantic_ckpt"].is_file()
        assert (env["stage2_dir"] / "semantic_probe" / "last_semantic.pth").is_file()
        assert (env["stage2_dir"] / "semantic_probe" / "history.csv").is_file()
        assert (env["stage2_dir"] / "semantic_probe" / "config.json").is_file()

        # Khởi tạo lại SemanticBranch (CLIP.py) và nạp best_semantic.pth (strict=False như trong predict_single.ipynb)
        sem_branch = SemanticBranch(
            model_name="ViT-B-16",
            pretrained=False,
            dropout=0.0,
            l2_normalize=True,
        ).to(test_device)
        sem_branch.eval()
        saved_sem_ckpt = torch.load(env["semantic_ckpt"], map_location=test_device, weights_only=False)
        missing_keys, unexpected_keys = sem_branch.load_state_dict(
            saved_sem_ckpt["model_state_dict"], strict=False
        )
        assert len(unexpected_keys) == 0
        assert all(k.startswith("visual.") for k in missing_keys)

        trained_probe: LinearProbeHead = probe_res["model"].to(test_device)
        trained_probe.eval()
        sample_clip_feats = torch.nn.functional.normalize(
            torch.randn(16, 512, device=test_device), p=2, dim=-1
        )
        with torch.no_grad():
            logits_probe = trained_probe(sample_clip_feats)
            logits_clip_branch = sem_branch.forward_head(sample_clip_feats)
        max_sem_diff = float((logits_probe - logits_clip_branch).abs().max().item())
        assert max_sem_diff == 0.0, f"Logit lệch giữa LinearProbeHead và SemanticBranch: {max_sem_diff}"
        print(
            f"  -> Lưu chuẩn `{env['semantic_ckpt'].parent.name}/{env['semantic_ckpt'].name}` | "
            f"Khớp SemanticBranch.forward_head (max_diff={max_sem_diff:.1e}, Val AUC={probe_res['val_metrics']['auc']:.4f}) -> [PASS]"
        )

        # ------------------------------------------------------------------
        # TEST 6: Train Fusion Baseline + Sanity Check Gate + Overfit & Real Resume
        # ------------------------------------------------------------------
        print("\n[Test 6] Kiểm tra train_fusion_baseline (p=0.0), Sanity Check Gate & Real Resume Stage 2...")
        sample_batch = next(iter(train_loader_test))
        gate_model = FusionMLP().to(test_device)
        assert overfit_single_batch_check(
            gate_model, sample_batch, test_device, steps=60, lr=5e-3
        )
        assert checkpoint_consistency_check(gate_model, sample_batch, test_device)

        # Kiểm chứng chạy thật `run_sanity_check=True` + kiểm chứng dừng ngay nếu sanity_check_gate trả False
        fusion_res = train_fusion_baseline(
            cache_dir=cache_dir,
            fusion_dir=env["stage2_dir"] / "fusion_baseline",
            epochs=6,
            patience=6,
            batch_size=32,
            seed=42,
            device=test_device,
            run_sanity_check=True,
        )
        assert (env["stage2_dir"] / "fusion_baseline" / "best_fusion.pth").is_file()
        assert fusion_res["val_metrics"]["auc"] > 0.90

        # Kiểm chứng nếu sanity_check_gate() trả về False -> phải ném RuntimeError chặn đứng train
        _g_ns = globals()
        _orig_gate = _g_ns["sanity_check_gate"]
        caught_gate_fail = False
        try:
            _g_ns["sanity_check_gate"] = lambda *a, **kw: False
            try:
                train_fusion_baseline(
                    cache_dir=cache_dir,
                    fusion_dir=tmp_root / "fusion_fail_gate",
                    epochs=2,
                    seed=42,
                    device=test_device,
                    run_sanity_check=True,
                )
            except RuntimeError:
                caught_gate_fail = True
        finally:
            _g_ns["sanity_check_gate"] = _orig_gate
        assert caught_gate_fail, "Phải dừng ngay bằng RuntimeError khi sanity_check_gate() trả về False!"

        # Kiểm chứng luồng Resume thực tế của Stage 2 với cùng lịch Cosine (epochs=6):
        # Chạy liền mạch 0..5 (chụp lại trạng thái tại epoch 2) -> Resume từ epoch 2 -> đối chiếu khớp từng epoch 3..5
        import engine as _eng_mod
        resume_test_dir = tmp_root / "fusion_resume_e2e"
        ep2_last_snap = tmp_root / "snap_ep2_last.pth"
        ep2_best_snap = tmp_root / "snap_ep2_best.pth"
        _orig_save_ckpt = _eng_mod.save_checkpoint

        def _snap_save_ckpt(*args: Any, **kwargs: Any) -> None:
            _orig_save_ckpt(*args, **kwargs)
            ep_num = int(kwargs.get("epoch", -1))
            fp_arg = Path(kwargs.get("filepath", args[0] if args else ""))
            if ep_num == 2 and fp_arg.name == "last_fusion.pth":
                shutil.copy2(fp_arg, ep2_last_snap)
                best_candidate = fp_arg.with_name("best_fusion.pth")
                if best_candidate.is_file():
                    shutil.copy2(best_candidate, ep2_best_snap)

        try:
            _eng_mod.save_checkpoint = _snap_save_ckpt
            cont_res = train_fusion_baseline(
                cache_dir=cache_dir,
                fusion_dir=resume_test_dir,
                epochs=6,
                patience=6,
                batch_size=32,
                seed=42,
                device=test_device,
            )
        finally:
            _eng_mod.save_checkpoint = _orig_save_ckpt

        assert ep2_last_snap.is_file() and ep2_best_snap.is_file()
        shutil.copy2(ep2_best_snap, resume_test_dir / "best_fusion.pth")
        resumed_res = train_fusion_baseline(
            cache_dir=cache_dir,
            fusion_dir=resume_test_dir,
            epochs=6,
            patience=6,
            batch_size=32,
            seed=42,
            device=test_device,
            resume_from=ep2_last_snap,
        )
        resumed_hist_df = pd.read_csv(resumed_res["history_csv_path"])
        assert [int(e) for e in resumed_hist_df["epoch"].tolist()] == [0, 1, 2, 3, 4, 5]
        for ep_idx in range(6):
            assert abs(
                float(resumed_res["history"][ep_idx]["val_loss"])
                - float(cont_res["history"][ep_idx]["val_loss"])
            ) < 1e-4, f"Lệch val_loss sau Resume tại epoch {ep_idx}!"
            assert abs(
                float(resumed_res["history"][ep_idx]["train_loss"])
                - float(cont_res["history"][ep_idx]["train_loss"])
            ) < 1e-4, f"Lệch train_loss sau Resume tại epoch {ep_idx}!"

        # Khóa xác minh khớp chính xác từng trọng số mô hình (state_dict) sau Resume (P1 ChatGPT):
        # Mô hình chạy liên tục vs Mô hình chạy resume từ epoch 2 phải có trọng số khớp nhau 100%
        cont_sd = cont_res["model"].state_dict()
        resumed_sd = resumed_res["model"].state_dict()
        assert cont_sd.keys() == resumed_sd.keys(), "Mâu thuẫn keys trong state_dict sau Resume!"
        for p_name, cont_param in cont_sd.items():
            res_param = resumed_sd[p_name]
            max_p_diff = float((cont_param - res_param).abs().max().item())
            assert max_p_diff < 1e-5, (
                f"Lệch trọng số '{p_name}' sau Resume giữa chạy liên tục và Resume từ epoch 2! "
                f"Max diff = {max_p_diff:.6e}"
            )

        # Kiểm chứng history.csv đọc đúng khóa phẳng từ engine.fit() (không bị ghi 0.000000)
        hist_df = pd.read_csv(fusion_res["history_csv_path"])
        assert len(hist_df) == len(fusion_res["history"]) == 6
        assert float(hist_df["val_auc"].max()) > 0.90
        assert abs(float(hist_df["val_auc"].iloc[-1]) - float(fusion_res["history"][-1]["val_auc"])) < 1e-5
        assert abs(float(hist_df["train_auc"].iloc[-1]) - float(fusion_res["history"][-1]["train_auc"])) < 1e-5

        print(
            f"  -> Fusion Baseline Val AUC={fusion_res['val_metrics']['auc']:.4f} | "
            f"Sanity Gate (PASS+Fail Guard) | Real Resume 3->6ep khớp tuyệt đối | history.csv khớp chuẩn -> [PASS]"
        )

        # ------------------------------------------------------------------
        # TEST 7: Ablation 7 tổ hợp + Branch Dropout (p=0.15) + Clean & JPEG Eval
        # ------------------------------------------------------------------
        print("\n[Test 7] Kiểm tra Ablation 7 tổ hợp, Branch Dropout (p=0.15) & Full Evaluation (Clean + JPEG)...")
        dropout_res = train_branch_dropout_experiment(
            cache_dir=cache_dir,
            dropout_dir=env["stage2_dir"] / "branch_dropout",
            branch_dropout_p=0.15,
            epochs=6,
            patience=6,
            batch_size=32,
            seed=42,
            device=test_device,
        )
        assert (env["stage2_dir"] / "branch_dropout" / "best_fusion_dropout.pth").is_file()

        ablation_res = run_ablation_study(
            cache_dir=cache_dir,
            ablation_root_dir=env["stage2_dir"] / "ablation",
            epochs=4,
            patience=4,
            batch_size=32,
            seed=42,
            device=test_device,
        )
        assert len(ablation_res) == 7
        expected_dims = {
            "spatial": 1280,
            "semantic": 512,
            "spectral": 256,
            "spatial_semantic": 1792,
            "spatial_spectral": 1536,
            "semantic_spectral": 768,
            "all_branches": 2048,
        }
        for slug, exp_dim in expected_dims.items():
            assert slug in ablation_res
            assert ablation_res[slug]["config"]["input_feature_dim"] == exp_dim
            assert Path(ablation_res[slug]["best_checkpoint_path"]).is_file()

        eval_summary = run_full_evaluation(
            cache_dir=cache_dir,
            eval_output_dir=env["stage2_dir"] / "evaluation",
            spatial_ckpt_path=env["spatial_ckpt"],
            semantic_ckpt_path=env["semantic_ckpt"],
            spectral_ckpt_path=env["spectral_ckpt"],
            fusion_ckpt_path=env["stage2_dir"] / "fusion_baseline" / "best_fusion.pth",
            dropout_ckpt_path=env["stage2_dir"] / "branch_dropout" / "best_fusion_dropout.pth",
            ablation_root_dir=env["stage2_dir"] / "ablation",
            jpeg_qualities=DEFAULT_JPEG_QUALITIES,
            device=test_device,
            require_all_models=True,
            strict_jpeg=True,
            include_ablation_in_jpeg=True,
        )
        for f_path in eval_summary["files"].values():
            assert Path(f_path).is_file(), f"Thiếu file báo cáo: {f_path}"
        assert eval_summary["jpeg_robustness"]["qualities_evaluated"] == [90, 70, 50, 30]
        assert len(eval_summary["ablation_rows"]) == 12  # 3 đơn nhánh + 2 Fusion (p=0, p=0.15) + 7 Ablation
        for q_val in DEFAULT_JPEG_QUALITIES:
            q_models = eval_summary["jpeg_robustness"]["by_quality"][f"jpeg_{q_val}"]
            assert len(q_models) == 12, f"JPEG Q={q_val} phải có đủ 12 mô hình, thực tế={len(q_models)}"

        # Kiểm chứng verify_jpeg_matches_clean_metadata bắt được lỗi lệch checkpoint_sha256
        clean_test_pays, _ = load_aligned_branch_caches(cache_dir, BRANCH_ORDER, split="test")
        jpeg90_pays, _ = load_aligned_branch_caches(cache_dir, BRANCH_ORDER, split="test", jpeg_quality=90)
        bad_jpeg90 = {k: dict(v) for k, v in jpeg90_pays.items()}
        bad_jpeg90["spatial"]["extraction_signature"] = dict(jpeg90_pays["spatial"]["extraction_signature"])
        bad_jpeg90["spatial"]["extraction_signature"]["checkpoint_sha256"] = "mismatched_sha256"
        caught_sha_mismatch = False
        try:
            verify_jpeg_matches_clean_metadata(clean_test_pays, bad_jpeg90, expected_jpeg_quality=90)
        except ValueError:
            caught_sha_mismatch = True
        assert caught_sha_mismatch, "Không phát hiện được lệch checkpoint_sha256 giữa Clean và JPEG!"

        # Kiểm chứng strict_jpeg & require_all_models báo lỗi ngay khi thiếu cache JPEG hoặc thiếu checkpoint
        caught_missing_q = False
        try:
            run_full_evaluation(
                cache_dir=cache_dir,
                eval_output_dir=tmp_root / "eval_missing_q",
                spatial_ckpt_path=env["spatial_ckpt"],
                semantic_ckpt_path=env["semantic_ckpt"],
                spectral_ckpt_path=env["spectral_ckpt"],
                fusion_ckpt_path=env["stage2_dir"] / "fusion_baseline" / "best_fusion.pth",
                dropout_ckpt_path=env["stage2_dir"] / "branch_dropout" / "best_fusion_dropout.pth",
                ablation_root_dir=env["stage2_dir"] / "ablation",
                jpeg_qualities=[90, 15],
                device=test_device,
                strict_jpeg=True,
            )
        except FileNotFoundError:
            caught_missing_q = True
        assert caught_missing_q, "strict_jpeg=True phải báo FileNotFoundError khi thiếu cache JPEG!"

        caught_missing_model = False
        try:
            run_full_evaluation(
                cache_dir=cache_dir,
                eval_output_dir=tmp_root / "eval_missing_model",
                spatial_ckpt_path=env["spatial_ckpt"],
                semantic_ckpt_path=env["semantic_ckpt"],
                spectral_ckpt_path=env["spectral_ckpt"],
                fusion_ckpt_path=env["stage2_dir"] / "fusion_baseline" / "best_fusion.pth",
                dropout_ckpt_path=tmp_root / "nonexistent_dropout.pth",
                ablation_root_dir=env["stage2_dir"] / "ablation",
                jpeg_qualities=[],
                device=test_device,
                require_all_models=True,
            )
        except FileNotFoundError:
            caught_missing_model = True
        assert caught_missing_model, "require_all_models=True phải báo FileNotFoundError khi thiếu checkpoint!"

        print(
            "  -> Đủ 12/12 mô hình trên cả Clean & JPEG (90,70,50,30) + Khóa strict checkpoint/JPEG hash -> [PASS]"
        )

        # ------------------------------------------------------------------
        # TEST 8: End-to-End Integration Smoke Test với ảnh thật (Clean + JPEG Q=70)
        # ------------------------------------------------------------------
        print("\n[Test 8] End-to-End Smoke Test (Ảnh thật Clean vs JPEG Q=70 -> 3 Backbones -> FusionMLP)...")
        from extract_features import extract_split_features, load_branch_extractor

        assert DEFAULT_SPATIAL_CKPT.is_file() and DEFAULT_SPECTRAL_CKPT.is_file()
        m_spa, ci_spa = load_branch_extractor("spatial", test_device, spatial_ckpt=DEFAULT_SPATIAL_CKPT)
        m_sem, ci_sem = load_branch_extractor("semantic", test_device, clip_pretrained="openai")
        m_spe, ci_spe = load_branch_extractor("spectral", test_device, spectral_ckpt=DEFAULT_SPECTRAL_CKPT)
        real_models = {"spatial": m_spa, "semantic": m_sem, "spectral": m_spe}
        real_ckpt_infos = {"spatial": ci_spa, "semantic": ci_sem, "spectral": ci_spe}

        # 1. Trích xuất 8 ảnh thật ở chế độ Clean
        real_clean_payloads = extract_split_features(
            models=real_models,
            checkpoint_infos=real_ckpt_infos,
            split="test",
            batch_size=8,
            num_workers=0,
            device=test_device,
            use_amp=(test_device.type == "cuda"),
            include_flip=False,
            fixed_jpeg_quality=None,
            max_batches=1,
            log_interval=10,
        )
        real_clean_ds = CachedFeatureDataset(
            real_clean_payloads,
            active_branches=BRANCH_ORDER,
            input_key="fusion",
            mode="test",
            flip_prob=0.0,
        )
        real_clean_loader = DataLoader(real_clean_ds, batch_size=8, shuffle=False, num_workers=0)
        clean_batch = next(iter(real_clean_loader))

        # 2. Trích xuất đúng 8 ảnh thật đó nhưng qua nén JPEG Q=70 thực thụ trên từng pixel ảnh
        real_jpeg70_payloads = extract_split_features(
            models=real_models,
            checkpoint_infos=real_ckpt_infos,
            split="test",
            batch_size=8,
            num_workers=0,
            device=test_device,
            use_amp=(test_device.type == "cuda"),
            include_flip=False,
            fixed_jpeg_quality=70,
            max_batches=1,
            log_interval=10,
        )
        real_jpeg70_ds = CachedFeatureDataset(
            real_jpeg70_payloads,
            active_branches=BRANCH_ORDER,
            input_key="fusion",
            mode="test",
            flip_prob=0.0,
        )
        real_jpeg70_loader = DataLoader(real_jpeg70_ds, batch_size=8, shuffle=False, num_workers=0)
        jpeg70_batch = next(iter(real_jpeg70_loader))

        # 3. Kiểm chứng thực tế: vector đặc trưng Clean và JPEG Q=70 từ ảnh thật phải có sự dịch chuyển (khác nhau)
        clean_feat = clean_batch["fusion"]
        jpeg70_feat = jpeg70_batch["fusion"]
        feat_diff = float((clean_feat - jpeg70_feat).abs().mean().item())
        assert feat_diff > 1e-4, f"Đặc trưng trích xuất giữa Clean và JPEG Q=70 quá nhỏ ({feat_diff:.6e})!"

        # 4. Kiểm chứng đầu ra logits qua mô hình FusionMLP trên cả Clean và JPEG Q=70 thật
        with torch.no_grad():
            clean_logits = fusion_res["model"](clean_feat.to(test_device))
            jpeg70_logits = fusion_res["model"](jpeg70_feat.to(test_device))

        assert clean_logits.shape == (8,) and torch.isfinite(clean_logits).all()
        assert jpeg70_logits.shape == (8,) and torch.isfinite(jpeg70_logits).all()

        del real_models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"  -> Nối thông suốt 8 ảnh Test thật Clean vs JPEG Q=70 qua 3 nhánh (Diff={feat_diff:.4f}) -> FusionMLP -> [PASS]"
        )

    print("\n" + "=" * 88)
    print("TẤT CẢ 8 CỔNG KIỂM THỬ ĐƠN VỊ & TÍCH Hợp CỦA train_fusion.py ĐỀU PASS 100%!")
    print("=" * 88)


# ==============================================================================
# GIAO DIỆN DÒNG LỆNH (CLI)
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá Stage 2 (Semantic Probe, Fusion MLP, Ablation 7 tổ hợp, JPEG Robustness)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["probe", "fusion", "ablation", "dropout", "eval", "custom", "all"],
        help=(
            "Chế độ thực thi: 'probe' (train best_semantic.pth), 'fusion' (train best_fusion.pth p=0.0), "
            "'ablation' (train 7 tổ hợp), 'dropout' (train 3 nhánh p=0.15), 'eval' (chấm Clean + JPEG Test), "
            "'custom' (tùy chọn --branches & --head), hoặc 'all' (chạy trọn gói Stage 2)."
        ),
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=list(BRANCH_ORDER),
        choices=list(VALID_BRANCHES),
        help="Danh sách nhánh kích hoạt khi dùng --mode custom (mặc định: spatial semantic spectral).",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="mlp",
        choices=["mlp", "linear"],
        help="Loại head khi dùng --mode custom ('mlp' hoặc 'linear').",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help=f"Thư mục chứa các file feature cache .pt (mặc định: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--stage2-dir",
        type=str,
        default=str(DEFAULT_STAGE2_DIR),
        help=f"Thư mục gốc lưu kết quả Stage 2 (mặc định: {DEFAULT_STAGE2_DIR}).",
    )
    parser.add_argument(
        "--semantic-ckpt",
        type=str,
        default=str(DEFAULT_SEMANTIC_CKPT),
        help=f"Đường dẫn lưu/nạp best_semantic.pth (mặc định: {DEFAULT_SEMANTIC_CKPT}).",
    )
    parser.add_argument(
        "--spatial-ckpt",
        type=str,
        default=str(DEFAULT_SPATIAL_CKPT),
        help=f"Đường dẫn tới best_spatial.pth để đối chiếu đánh giá (mặc định: {DEFAULT_SPATIAL_CKPT}).",
    )
    parser.add_argument(
        "--spectral-ckpt",
        type=str,
        default=str(DEFAULT_SPECTRAL_CKPT),
        help=f"Đường dẫn tới best_spectral.pth để đối chiếu đánh giá (mặc định: {DEFAULT_SPECTRAL_CKPT}).",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (mặc định: 1e-3 cho linear, 5e-4 cho mlp).")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay cho Linear.weight (mặc định: 1e-4).")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Số epoch warmup (mặc định: 0 linear, 1 mlp).")
    parser.add_argument("--epochs", type=int, default=None, help="Tổng số epochs (mặc định: 30).")
    parser.add_argument("--patience", type=int, default=None, help="EarlyStopping patience theo Val AUC (mặc định: 6).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size khi train (mặc định: 32).")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Batch size khi đánh giá Val/Test (mặc định: 256).")
    parser.add_argument("--num-workers", type=int, default=0, help="Số worker DataLoader (mặc định: 0 vì cache nằm trên RAM).")
    parser.add_argument("--flip-prob", type=float, default=0.5, help="Xác suất lật ngang đồng bộ khi train (mặc định: 0.5).")
    parser.add_argument(
        "--branch-dropout",
        type=float,
        default=0.15,
        help="Xác suất Branch Dropout cho thí nghiệm --mode dropout (mặc định: 0.15).",
    )
    parser.add_argument(
        "--jpeg-qualities",
        type=str,
        default="90,70,50,30",
        help="Danh sách chất lượng JPEG khi đánh giá độ bền vững (mặc định: '90,70,50,30').",
    )
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên (mặc định: 42).")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị chạy ('cuda' hoặc 'cpu').")
    parser.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision (chạy FP32 thuần).")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint last_*.pth để tiếp tục train.")
    parser.add_argument("--sanity-check", action="store_true", help="Chạy cổng sanity_check_gate trước khi train.")
    parser.add_argument(
        "--strict-eval",
        action="store_true",
        help="Bắt buộc có đủ 12 checkpoint và đủ toàn bộ cache JPEG yêu cầu khi đánh giá (lỗi ngay nếu thiếu).",
    )
    parser.add_argument(
        "--allow-missing-jpeg",
        action="store_true",
        help="Cho phép --mode eval bỏ qua JPEG nếu chưa trích xuất `*_test_jpeg*.pt`.",
    )
    parser.add_argument("--run-tests", action="store_true", help="Chạy bộ kiểm thử 8 cổng rồi thoát.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_tests:
        run_unit_tests()
        return

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = False if args.no_amp else None
    cache_dir = Path(args.cache_dir)
    stage2_dir = Path(args.stage2_dir)
    semantic_ckpt = Path(args.semantic_ckpt)
    spatial_ckpt = Path(args.spatial_ckpt)
    spectral_ckpt = Path(args.spectral_ckpt)
    jpeg_qualities = parse_jpeg_qualities_arg(args.jpeg_qualities)

    mode = args.mode.lower()

    if mode == "custom":
        norm_br = normalize_branch_selection(args.branches)
        ht = args.head.lower()
        if ht == "linear" and norm_br == ("semantic",):
            train_semantic_probe(
                cache_dir=cache_dir,
                semantic_ckpt_path=semantic_ckpt,
                stage2_probe_dir=stage2_dir / "semantic_probe",
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                flip_prob=args.flip_prob,
                seed=args.seed,
                device=device,
                use_amp=use_amp,
                resume_from=args.resume,
                run_sanity_check=args.sanity_check,
            )
        else:
            slug = get_ablation_combo_slug(norm_br)
            out_dir = stage2_dir / f"custom_{ht}_{slug}"
            run_stage2_training_experiment(
                experiment_name=f"custom_{ht}_{slug}",
                active_branches=norm_br,
                head_type=ht,
                output_dir=out_dir,
                best_ckpt_path=out_dir / f"best_{ht}_{slug}.pth",
                last_ckpt_path=out_dir / f"last_{ht}_{slug}.pth",
                cache_dir=cache_dir,
                branch_dropout_p=0.0,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                flip_prob=args.flip_prob,
                seed=args.seed,
                device=device,
                use_amp=use_amp,
                resume_from=args.resume,
                run_sanity_check=args.sanity_check,
            )
        return

    if mode in ("probe", "all"):
        train_semantic_probe(
            cache_dir=cache_dir,
            semantic_ckpt_path=semantic_ckpt,
            stage2_probe_dir=stage2_dir / "semantic_probe",
            lr=args.lr if mode == "probe" else None,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            flip_prob=args.flip_prob,
            seed=args.seed,
            device=device,
            use_amp=use_amp,
            resume_from=args.resume if mode == "probe" else None,
            run_sanity_check=args.sanity_check,
        )

    if mode in ("fusion", "all"):
        train_fusion_baseline(
            cache_dir=cache_dir,
            fusion_dir=stage2_dir / "fusion_baseline",
            lr=args.lr if mode == "fusion" else None,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            flip_prob=args.flip_prob,
            seed=args.seed,
            device=device,
            use_amp=use_amp,
            resume_from=args.resume if mode == "fusion" else None,
            run_sanity_check=args.sanity_check,
        )

    if mode in ("dropout", "all"):
        train_branch_dropout_experiment(
            cache_dir=cache_dir,
            dropout_dir=stage2_dir / "branch_dropout",
            branch_dropout_p=args.branch_dropout,
            lr=args.lr if mode == "dropout" else None,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            flip_prob=args.flip_prob,
            seed=args.seed,
            device=device,
            use_amp=use_amp,
            resume_from=args.resume if mode == "dropout" else None,
        )

    if mode in ("ablation", "all"):
        run_ablation_study(
            cache_dir=cache_dir,
            ablation_root_dir=stage2_dir / "ablation",
            lr=args.lr if mode == "ablation" else None,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            flip_prob=args.flip_prob,
            seed=args.seed,
            device=device,
            use_amp=use_amp,
        )

    if mode in ("eval", "all"):
        # Chỉ chạy đánh giá tập Test khi các file cache *_test.pt đã tồn tại
        test_caches_ready = all(
            get_cache_filepath(cache_dir, branch=br, split="test", jpeg_quality=None).is_file()
            for br in BRANCH_ORDER
        )
        strict_jpeg_flag = (
            (not args.allow_missing_jpeg and bool(jpeg_qualities))
            if mode == "eval"
            else bool(args.strict_eval)
        )
        if test_caches_ready:
            run_full_evaluation(
                cache_dir=cache_dir,
                eval_output_dir=stage2_dir / "evaluation",
                spatial_ckpt_path=spatial_ckpt,
                semantic_ckpt_path=semantic_ckpt,
                spectral_ckpt_path=spectral_ckpt,
                fusion_ckpt_path=stage2_dir / "fusion_baseline" / "best_fusion.pth",
                dropout_ckpt_path=stage2_dir / "branch_dropout" / "best_fusion_dropout.pth",
                ablation_root_dir=stage2_dir / "ablation",
                jpeg_qualities=jpeg_qualities,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                device=device,
                use_amp=use_amp,
                require_all_models=bool(args.strict_eval),
                strict_jpeg=strict_jpeg_flag,
                include_ablation_in_jpeg=True,
            )
        elif mode == "eval":
            raise FileNotFoundError(
                f"Không tìm thấy đủ các file cache `*_test.pt` trong '{cache_dir}'. "
                f"Hãy chạy `extract_features.py` trước khi chạy `--mode eval`."
            )
        else:
            print(
                f"\n[!] Bỏ qua bước đánh giá Test vì chưa có đủ `*_test.pt` trong '{cache_dir}'."
            )


if __name__ == "__main__":
    main()
