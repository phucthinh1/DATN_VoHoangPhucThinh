# -*- coding: utf-8 -*-
"""
code/system/train/train_branch.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Script điều phối huấn luyện chuẩn (CLI) cho các nhánh có học trọng số (`spatial` | `spectral`):
1. Nhánh Không gian (`--branch spatial` — Baseline EfficientNet-B0):
   - Thực hiện Huấn luyện 2 Pha (Two-Phase Differential Fine-Tuning):
     + Pha A (Warmup Head, epoch 0): Khóa 100% backbone (đạo hàm + BatchNorm2d + StochasticDepth ở .eval()),
       chỉ huấn luyện `classifier` (1,281 tham số) với `head_lr = 1e-3`.
     + Pha B (Full Fine-tuning, epochs 1..15): Mở khóa backbone theo `trainable_stages` (mặc định `"all"`),
       khởi tạo mới `optimizer_b` (2 param_groups: `backbone_lr = 1e-4`, `head_lr = 1e-3`) và
       `CosineAnnealingLR(T_max=finetune_epochs)`, giữ liên tục chỉ số `start_epoch` để `BalancedGroupSampler`
       không bị lặp lại chu kỳ 3 epochs, đồng thời dùng chung `EarlyStopping` (AUC chính, Bal-ACC phân xử hòa).
2. Nhánh Phổ (`--branch spectral` — Residual Spectral CNN, gắn sẵn khung chờ `FFT.py`):
   - Huấn luyện từ đầu (`AdamW lr = 1e-3`, `weight_decay = 1e-4`), 1 epoch linear warmup + CosineAnnealingLR.
3. Bảo đảm toàn vẹn thực nghiệm:
   - Hỗ trợ `--sanity-check` / `--sanity-only` chạy Cổng kiểm tra 3 bước trước khi train dài.
   - Hỗ trợ `--resume` phân biệt rõ Pha A / ranh giới Pha A→B / Pha B.
   - Tuyệt đối không dùng tập Test để chọn checkpoint hay siêu tham số (`--eval-test` chỉ chạy khi được bật
     tường minh SAU KHI đã chốt `best_<branch>.pth`).
"""

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Định tuyến đường dẫn liên thư mục chuẩn của hệ thống
CURRENT_DIR = Path(__file__).resolve().parent        # d:/nam_4/DATN/code/system/train
SYSTEM_DIR = CURRENT_DIR.parent                      # d:/nam_4/DATN/code/system
BRANCH_DIR = SYSTEM_DIR / "branch"                   # d:/nam_4/DATN/code/system/branch
DATA_DIR = SYSTEM_DIR / "prepare_data"               # d:/nam_4/DATN/code/system/prepare_data

for _p in (str(CURRENT_DIR), str(SYSTEM_DIR), str(BRANCH_DIR), str(DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import DEFAULT_DATASET_ROOT, build_dataloaders  # noqa: E402
from engine import (  # noqa: E402
    EarlyStopping,
    evaluate,
    fit,
    get_optimal_amp_config,
    load_checkpoint,
    sanity_check_gate,
)
from metrics import (  # noqa: E402
    ALL_AI_GENERATORS,
    evaluate_by_generator,
    format_generator_report,
    format_metrics_summary,
)
from EfficientNet import SpatialBranch  # noqa: E402


# Cấu hình mặc định chính thức cho Baseline từng nhánh (Mục 1, 2, 4 của Chot_Cau_Hinh_Buoc_5B_3_Nhanh.md)
BRANCH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "spatial": {
        "warmup_epochs": 1,
        "epochs": 16,          # 1 epoch Pha A (Warmup Head) + 15 epochs Pha B (Full Fine-tuning)
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "patience": 4,
        "min_delta": 1e-4,
        "max_norm": 1.0,
    },
    "spectral": {
        "warmup_epochs": 1,
        "epochs": 30,          # 1 epoch linear warmup + 29 epochs cosine (train từ đầu)
        "backbone_lr": 1e-3,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "patience": 5,
        "min_delta": 1e-4,
        "max_norm": 1.0,
    },
}


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


def parse_trainable_stages_arg(raw: str) -> Union[str, int, List[int]]:
    """Chuyển chuỗi CLI `--trainable-stages` ('all', 'none', '2', '6,7,8') sang định dạng của SpatialBranch."""
    val = str(raw).strip().lower()
    if val in ("all", "none", "frozen", "head_only"):
        return val
    if "," in val:
        raw_parts = [p.strip() for p in val.split(",")]
        if not raw_parts or any(p == "" for p in raw_parts):
            raise ValueError(
                f"Danh sách stage không hợp lệ (có dấu phẩy thừa hoặc phần tử rỗng): {raw!r}. "
                f"Ví dụ hợp lệ: 'all', 'none', '2', hoặc '6,7,8'."
            )
        return [int(p) for p in raw_parts]
    return int(val)


def _shutdown_dataloader_workers(*loaders: Any) -> None:
    """Đóng tường minh các persistent worker cũ trước khi tạo DataLoader mới (tránh nhân đôi VMS trên Windows)."""
    for loader in loaders:
        if loader is None:
            continue
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown_fn = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception:
                    pass
            try:
                loader._iterator = None
            except Exception:
                pass
    gc.collect()


def json_default(obj: Any) -> Any:
    """Chuyển đổi các kiểu dữ liệu NumPy/Path sang kiểu thuần JSON."""
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
    raise TypeError(f"Không thể tuần tự hóa đối tượng kiểu {type(obj)} sang JSON.")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def load_prior_history(history_path: Path, max_epoch_inclusive: int) -> List[Dict[str, Any]]:
    """Nạp lịch sử các epoch trước điểm ngắt khi `--resume` để file JSON không bị mất log cũ."""
    if not history_path.exists():
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        old_records = payload.get("history", [])
        return [r for r in old_records if int(r.get("epoch", -1)) <= max_epoch_inclusive]
    except Exception:
        return []


def assert_optimizer_covers_trainable_params(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    context_label: str = "",
) -> None:
    """
    Điều kiện kiểm soát chuyển pha (ChatGPT #1):
    Xác nhận 100% tham số có `requires_grad=True` nằm trong Optimizer, không trùng lặp và không bỏ sót.
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


def snapshot_backbone_full(
    model: nn.Module,
    only_frozen: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Chụp toàn bộ 100% tham số (parameters) và bộ đệm (buffers, bao gồm running_mean/var/num_batches_tracked của BatchNorm2d)
    thuộc thân mạng backbone (không thuộc classifier):
    - Khi `only_frozen=False` (Pha A): chụp 100% tham số và bộ đệm của toàn bộ backbone.
    - Khi `only_frozen=True` (Pha B mở khóa một phần stage): chỉ chụp các tham số đang đóng băng (`requires_grad=False`)
      và các bộ đệm thuộc module đang khóa ở chế độ `.eval()` (`module.training == False`).
    """
    classifier_ids = (
        {id(p) for p in model.classifier.parameters()}
        if getattr(model, "classifier", None) is not None
        else set()
    )
    classifier_buf_ids = (
        {id(b) for b in model.classifier.buffers()}
        if getattr(model, "classifier", None) is not None
        else set()
    )

    eval_buf_ids: Optional[set] = None
    if only_frozen:
        eval_buf_ids = set()
        for m in model.modules():
            direct_params = list(m.parameters(recurse=False))
            if direct_params and all(not p.requires_grad for p in direct_params):
                for b in m.buffers(recurse=False):
                    eval_buf_ids.add(id(b))

    params_snap: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if id(p) in classifier_ids:
            continue
        if only_frozen and p.requires_grad:
            continue
        params_snap[name] = p.detach().cpu().clone()

    buffers_snap: Dict[str, torch.Tensor] = {}
    for name, b in model.named_buffers():
        if id(b) in classifier_buf_ids:
            continue
        if only_frozen and eval_buf_ids is not None and id(b) not in eval_buf_ids:
            continue
        buffers_snap[name] = b.detach().cpu().clone()

    return params_snap, buffers_snap


def assert_backbone_unchanged(
    model: nn.Module,
    init_params: Dict[str, torch.Tensor],
    init_buffers: Dict[str, torch.Tensor],
    context_label: str = "Pha A",
) -> None:
    """
    Xác minh nghiêm ngặt 100% tham số và bộ đệm BatchNorm trong snapshot giữ nguyên vẹn tuyệt đối:
    Không bỏ qua bất kỳ khóa nào bị thiếu, và bất kỳ sự thay đổi nào về trọng số hoặc trôi thống kê
    BatchNorm (`running_mean` / `running_var` / `num_batches_tracked`) đều kích hoạt `AssertionError`.
    """
    cur_params = dict(model.named_parameters())
    for name, old_tensor in init_params.items():
        if name not in cur_params:
            raise AssertionError(
                f"[{context_label}] Vi phạm cấu trúc: Tham số backbone '{name}' biến mất khỏi model!"
            )
        cur_tensor = cur_params[name].detach().cpu()
        if not torch.equal(cur_tensor, old_tensor):
            raise AssertionError(
                f"[{context_label}] Vi phạm: Tham số backbone '{name}' bị biến đổi trong quá trình huấn luyện!"
            )

    cur_buffers = dict(model.named_buffers())
    for name, old_tensor in init_buffers.items():
        if name not in cur_buffers:
            raise AssertionError(
                f"[{context_label}] Vi phạm cấu trúc: Bộ đệm backbone '{name}' biến mất khỏi model!"
            )
        cur_tensor = cur_buffers[name].detach().cpu()
        if not torch.equal(cur_tensor, old_tensor):
            raise AssertionError(
                f"[{context_label}] Vi phạm: Bộ đệm backbone '{name}' (BatchNorm running stats) bị trôi!"
            )


def _snapshot_param(
    model: nn.Module,
    in_classifier: bool,
    only_trainable: bool = False,
) -> Optional[Tuple[str, torch.Tensor]]:
    """Lấy bản sao CPU của 1 tham số đại diện (backbone hoặc classifier) để kiểm chứng bất biến / cập nhật."""
    classifier_ids = {id(p) for p in model.classifier.parameters()} if getattr(model, "classifier", None) is not None else set()
    for name, p in model.named_parameters():
        is_cls = id(p) in classifier_ids
        if is_cls != in_classifier:
            continue
        if only_trainable and not p.requires_grad:
            continue
        return name, p.detach().cpu().clone()
    return None


def _current_param_tensor(model: nn.Module, param_name: str) -> torch.Tensor:
    return dict(model.named_parameters())[param_name].detach().cpu()


def build_branch_model(
    branch: str,
    pretrained: bool,
    dropout: float,
    trainable_stages: Union[str, int, Sequence[int]],
) -> nn.Module:
    """Khởi tạo mô hình nhánh tuân thủ hợp đồng BranchBase."""
    if branch == "spatial":
        return SpatialBranch(
            pretrained=pretrained,
            dropout=dropout,
            trainable_stages=trainable_stages,
        )
    if branch == "spectral":
        try:
            from FFT import SpectralBranch  # noqa: WPS433
        except (ImportError, ModuleNotFoundError) as exc:
            raise NotImplementedError(
                "Nhánh 'spectral' yêu cầu class `SpectralBranch` trong `code/system/branch/FFT.py`, "
                f"nhưng chưa sẵn sàng ({exc}). Hãy hoàn thiện `FFT.py` trước khi chạy `--branch spectral`."
            ) from exc
        return SpectralBranch(dropout=dropout)

    raise ValueError(f"train_branch.py chỉ hỗ trợ 'spatial' hoặc 'spectral', nhận được: {branch!r}")


def build_phase_b_components(
    model: SpatialBranch,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    finetune_epochs: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """
    Mở khóa backbone theo `trainable_stages`, tạo MỚI hoàn toàn `optimizer_b` (2 nhóm `backbone` + `head`)
    cùng `CosineAnnealingLR(T_max=finetune_epochs)`, đồng thời kiểm tra khớp 100% tham số trainable.
    """
    model.set_backbone_trainable(True)
    param_groups_b = model.get_param_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    if not param_groups_b:
        raise RuntimeError("Không có tham số trainable nào sau khi mở khóa Pha B!")

    optimizer_b = torch.optim.AdamW(param_groups_b)
    assert_optimizer_covers_trainable_params(model, optimizer_b, context_label="Pha B")

    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b,
        T_max=max(1, int(finetune_epochs)),
        eta_min=1e-6,
    )
    return optimizer_b, scheduler_b


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Gom siêu tham số cuối cùng (ưu tiên tham số CLI người dùng truyền, còn lại dùng Baseline chuẩn)."""
    defaults = BRANCH_DEFAULTS[args.branch]
    warmup_epochs = defaults["warmup_epochs"] if args.warmup_epochs is None else int(args.warmup_epochs)
    total_epochs = defaults["epochs"] if args.epochs is None else int(args.epochs)
    backbone_lr = defaults["backbone_lr"] if args.backbone_lr is None else float(args.backbone_lr)
    head_lr = defaults["head_lr"] if args.head_lr is None else float(args.head_lr)
    weight_decay = defaults["weight_decay"] if args.weight_decay is None else float(args.weight_decay)
    dropout = defaults["dropout"] if args.dropout is None else float(args.dropout)
    patience = defaults["patience"] if args.patience is None else int(args.patience)
    min_delta = defaults["min_delta"] if args.min_delta is None else float(args.min_delta)
    max_norm = defaults["max_norm"] if args.max_norm is None else float(args.max_norm)

    if total_epochs < 1:
        raise ValueError(f"--epochs phải >= 1, nhận được: {total_epochs}")
    if warmup_epochs < 0 or warmup_epochs > total_epochs:
        raise ValueError(
            f"--warmup-epochs ({warmup_epochs}) phải nằm trong [0, total_epochs={total_epochs}]!"
        )
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size phải > 0, nhận được: {args.batch_size}")
    if int(args.eval_batch_size) <= 0:
        raise ValueError(f"--eval-batch-size phải > 0, nhận được: {args.eval_batch_size}")
    if int(args.num_workers) < 0:
        raise ValueError(f"--num-workers phải >= 0, nhận được: {args.num_workers}")
    if not math.isfinite(backbone_lr) or backbone_lr <= 0:
        raise ValueError(f"--backbone-lr phải là số hữu hạn > 0, nhận được: {backbone_lr}")
    if not math.isfinite(head_lr) or head_lr <= 0:
        raise ValueError(f"--head-lr phải là số hữu hạn > 0, nhận được: {head_lr}")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(f"--weight-decay phải là số hữu hạn >= 0, nhận được: {weight_decay}")
    if not math.isfinite(dropout) or not (0.0 <= dropout < 1.0):
        raise ValueError(f"--dropout phải là số hữu hạn trong [0.0, 1.0), nhận được: {dropout}")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError(f"--min-delta phải là số hữu hạn >= 0, nhận được: {min_delta}")
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError(f"--max-norm phải là số hữu hạn >= 0, nhận được: {max_norm}")
    if not (0.0 <= float(args.jpeg_prob) <= 1.0):
        raise ValueError(f"--jpeg-prob phải trong [0.0, 1.0], nhận được: {args.jpeg_prob}")
    if patience <= 0:
        raise ValueError(f"--patience phải > 0, nhận được: {patience}")

    trainable_stages = parse_trainable_stages_arg(args.trainable_stages)
    canonical_stages: Optional[Tuple[int, ...]] = None
    if args.branch == "spatial":
        canonical_stages = SpatialBranch._parse_trainable_stages(trainable_stages)

    pretrained_flag = not bool(args.no_pretrained)
    spatial_view_mode = str(args.spatial_view_mode)
    jpeg_prob = float(args.jpeg_prob)

    if args.run_name:
        run_name = args.run_name
    else:
        ablation_tags: List[str] = []
        if (
            args.branch == "spatial"
            and canonical_stages is not None
            and canonical_stages != tuple(range(SpatialBranch.NUM_STAGES))
        ):
            stage_tag = "none" if not canonical_stages else "_".join(map(str, canonical_stages))
            ablation_tags.append(f"stages_{stage_tag}")
        if args.branch == "spatial" and spatial_view_mode != "native":
            ablation_tags.append(f"view_{spatial_view_mode}")
        if jpeg_prob > 0.0:
            ablation_tags.append(f"jpeg_{jpeg_prob:g}")
        if args.branch == "spatial" and not pretrained_flag:
            ablation_tags.append("scratch")
        if not math.isclose(dropout, defaults["dropout"], rel_tol=1e-9, abs_tol=1e-12):
            ablation_tags.append(f"drop_{dropout:g}")
        if not math.isclose(backbone_lr, defaults["backbone_lr"], rel_tol=1e-9, abs_tol=1e-12):
            ablation_tags.append(f"bblr_{backbone_lr:.0e}")
        if not math.isclose(head_lr, defaults["head_lr"], rel_tol=1e-9, abs_tol=1e-12):
            ablation_tags.append(f"hdlr_{head_lr:.0e}")

        if ablation_tags:
            run_name = f"{args.branch}_{'_'.join(ablation_tags)}"
        else:
            run_name = f"{args.branch}_baseline"
    run_dir = Path(args.output_dir) / run_name
    num_workers = int(args.num_workers)
    is_windows = sys.platform.startswith("win")
    if args.persistent_workers is not None:
        persistent_workers = bool(args.persistent_workers and num_workers > 0)
    else:
        # Trên Windows (cơ chế spawn ~1.25GB VMS/worker), chỉ giữ persistent_workers khi num_workers <= 2
        # để Train + Val không bao giờ chạy đồng thời > 4 worker làm tràn Windows Commit Limit.
        persistent_workers = bool(num_workers > 0 and (not is_windows or num_workers <= 2))

    if args.eval_workers is not None:
        eval_workers = int(args.eval_workers)
        if eval_workers < 0:
            raise ValueError(f"--eval-workers phải >= 0, nhận được: {eval_workers}")
    else:
        eval_workers = min(num_workers, 2) if is_windows else num_workers

    # Siết chặt an toàn bộ nhớ ảo trên Windows ngay cả khi người dùng cố tình bật cờ thủ công (ChatGPT P1)
    if is_windows:
        if num_workers > 2 and persistent_workers:
            print(
                f"[CẢNH BÁO BỘ NHỚ WINDOWS] num_workers={num_workers} với persistent_workers=True "
                f"dễ làm tràn Windows Commit Limit (~{num_workers * 1.25:.1f}GB VMS). "
                f"Tự động chuyển persistent_workers=False để bảo đảm an toàn hệ thống."
            )
            persistent_workers = False
        if eval_workers > 2:
            print(
                f"[CẢNH BÁO BỘ NHỚ WINDOWS] eval_workers={eval_workers} > 2. "
                f"Tự động hạ eval_workers=2 để tránh spawn quá nhiều tiến trình cùng lúc."
            )
            eval_workers = 2

    run_id = f"{run_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    return {
        "run_id": run_id,
        "branch": args.branch,
        "run_name": run_name,
        "run_dir": run_dir,
        "dataset_root": Path(args.dataset_root),
        "metadata_path": Path(args.metadata_path) if args.metadata_path else None,
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": num_workers,
        "eval_workers": eval_workers,
        "persistent_workers": persistent_workers,
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "warmup_epochs": warmup_epochs,
        "total_epochs": total_epochs,
        "finetune_epochs": total_epochs - warmup_epochs,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "patience": patience,
        "min_delta": min_delta,
        "max_norm": max_norm,
        "trainable_stages": trainable_stages,
        "spatial_view_mode": str(args.spatial_view_mode),
        "jpeg_prob": float(args.jpeg_prob),
        "pretrained": not bool(args.no_pretrained),
        "use_amp_flag": not bool(args.no_amp),
    }


def compute_dataset_manifest_hash(train_loader: Any, val_loader: Any) -> str:
    """
    Tính mã băm SHA-256 (manifest hash 16 ký tự) đại diện cho danh tính tập dữ liệu Train + Val
    (danh sách mẫu, nhãn và nguồn sinh ảnh sau lọc) để chống resume nhầm trên tập dữ liệu khác (ChatGPT P1 #1.2).
    """
    hasher = hashlib.sha256()
    for split_tag, loader in (("train", train_loader), ("val", val_loader)):
        ds = getattr(loader, "dataset", None)
        if ds is None:
            continue
        hasher.update(f"{split_tag}:len={len(ds)};".encode("utf-8"))
        filepaths = getattr(ds, "filepaths", None)
        labels_a = getattr(ds, "labels_a", getattr(ds, "y", None))
        model_names = getattr(ds, "model_names", getattr(ds, "models", None))
        if filepaths is not None and labels_a is not None and model_names is not None:
            for fp, lb, mn in zip(filepaths, labels_a, model_names):
                hasher.update(f"{fp}|{float(lb):.1f}|{mn}\n".encode("utf-8"))
        elif labels_a is not None and model_names is not None:
            for idx, (lb, mn) in enumerate(zip(labels_a, model_names)):
                hasher.update(f"{idx}|{float(lb):.1f}|{mn}\n".encode("utf-8"))
    return hasher.hexdigest()[:16]


def _build_branch_dataloaders(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Khởi tạo bộ DataLoader đơn nhánh (Train/Val/Test) tối ưu bộ nhớ Windows và gắn dataset_manifest_hash."""
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        dataset_root=cfg["dataset_root"],
        metadata_path=cfg["metadata_path"],
        batch_size=cfg["batch_size"],
        eval_batch_size=cfg["eval_batch_size"],
        num_workers=cfg["num_workers"],
        active_branches=(cfg["branch"],),
        spatial_view_mode=cfg["spatial_view_mode"],
        jpeg_prob=cfg["jpeg_prob"],
        seed=cfg["seed"],
        persistent_workers=cfg["persistent_workers"],
    )
    if cfg["eval_workers"] != cfg["num_workers"]:
        pin_mem = torch.cuda.is_available()
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=cfg["eval_workers"],
            pin_memory=pin_mem,
        )
        test_loader = DataLoader(
            test_loader.dataset,
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=cfg["eval_workers"],
            pin_memory=pin_mem,
        )
    cfg["dataset_manifest_hash"] = compute_dataset_manifest_hash(train_loader, val_loader)
    return train_loader, val_loader, test_loader


def validate_and_sync_resume(
    resume_path: Path,
    cfg: Dict[str, Any],
    best_ckpt_path: Path,
    history_path: Path,
) -> Tuple[Dict[str, Any], int, str, Path, List[Dict[str, Any]]]:
    """
    Kiểm tra toàn vẹn siêu dữ liệu khi `--resume` (cùng thư mục hoặc khác thư mục):
    1. Đối chiếu đầy đủ 19 tham số cấu hình (bao gồm `total_epochs`, `max_norm`, `use_amp_flag`, `deterministic`
       và `dataset_manifest_hash`).
    2. Kiểm tra tính hợp lệ của `resume_epoch` và sự nhất quán giữa `resume_phase` và `warmup_epochs`.
    3. Bắt buộc `best_<branch>.pth` cùng thư mục với `resume_path` phải tồn tại, khớp `run_id`, khớp `best_epoch`
       và khớp dấu vân tay chỉ số (cho phép dung sai `min_delta` trên `best_primary` khi phân xử hòa bằng Bal-ACC).
    4. Đồng bộ `best_<branch>.pth` và đọc lịch sử `history_<branch>.json` từ đúng thư mục nguồn.
    """
    if not resume_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint để resume: {resume_path}")

    resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    ckpt_cfg = resume_ckpt.get("config", {}) or {}
    ckpt_branch = ckpt_cfg.get("branch")
    if ckpt_branch is not None and ckpt_branch != cfg["branch"]:
        raise ValueError(
            f"Không thể resume checkpoint của nhánh '{ckpt_branch}' vào nhánh '{cfg['branch']}'!"
        )

    float_fields = {"backbone_lr", "head_lr", "weight_decay", "dropout", "min_delta", "max_norm", "jpeg_prob"}
    int_fields = {"warmup_epochs", "total_epochs", "patience", "batch_size", "seed"}
    bool_fields = {"pretrained", "use_amp_flag", "deterministic"}
    resume_check_fields = (
        "branch",
        "warmup_epochs",
        "total_epochs",
        "trainable_stages",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "dropout",
        "patience",
        "min_delta",
        "max_norm",
        "batch_size",
        "seed",
        "deterministic",
        "jpeg_prob",
        "spatial_view_mode",
        "pretrained",
        "use_amp_flag",
        "dataset_manifest_hash",
    )
    for check_k in resume_check_fields:
        if check_k not in ckpt_cfg or cfg.get(check_k) is None:
            continue
        old_val = ckpt_cfg[check_k]
        new_val = cfg.get(check_k)
        if old_val is None:
            continue
        if check_k in float_fields:
            matched = math.isclose(float(old_val), float(new_val), rel_tol=1e-9, abs_tol=1e-12)
        elif check_k in int_fields:
            matched = int(old_val) == int(new_val)
        elif check_k == "trainable_stages" and cfg["branch"] == "spatial":
            matched = (
                SpatialBranch._parse_trainable_stages(old_val)
                == SpatialBranch._parse_trainable_stages(new_val)
            )
        elif check_k in bool_fields:
            matched = bool(old_val) == bool(new_val)
        else:
            matched = str(old_val) == str(new_val)

        if not matched:
            raise ValueError(
                f"[RESUME LỖI CẤU HÌNH] Tham số '{check_k}' trong checkpoint ({old_val!r}) "
                f"khác với cấu hình hiện tại ({new_val!r})! "
                f"Để bảo toàn lịch trình học (đặc biệt T_max của CosineAnnealingLR và danh tính Dataset), "
                f"hãy dùng đúng cấu hình của checkpoint hoặc tạo run mới."
            )

    total_epochs = int(cfg["total_epochs"])
    warmup_epochs = int(cfg["warmup_epochs"])
    resume_epoch = int(resume_ckpt.get("epoch", -1))
    if resume_epoch < 0:
        raise ValueError(f"[RESUME LỖI] Checkpoint '{resume_path}' có epoch={resume_epoch} không hợp lệ (< 0)!")
    if resume_epoch + 1 >= total_epochs:
        raise ValueError(
            f"[RESUME LỖI] Checkpoint '{resume_path}' đã hoàn tất epoch {resume_epoch} "
            f"(total_epochs={total_epochs}), không còn epoch nào để huấn luyện tiếp!"
        )

    if cfg["branch"] == "spatial":
        raw_phase = ckpt_cfg.get("phase")
        resume_phase = str(raw_phase if raw_phase is not None else ("B" if resume_epoch >= warmup_epochs else "A")).upper()
        if resume_phase not in ("A", "B"):
            raise ValueError(f"[RESUME LỖI] Nhánh spatial chỉ nhận phase 'A' hoặc 'B', nhận được: {resume_phase!r}")
        if resume_phase == "A" and resume_epoch >= warmup_epochs:
            raise ValueError(
                f"[RESUME LỖI] Checkpoint ghi phase='A' nhưng epoch={resume_epoch} >= warmup_epochs={warmup_epochs}!"
            )
        if resume_phase == "B" and resume_epoch < warmup_epochs:
            raise ValueError(
                f"[RESUME LỖI] Checkpoint ghi phase='B' nhưng epoch={resume_epoch} < warmup_epochs={warmup_epochs}!"
            )
    else:
        resume_phase = str(ckpt_cfg.get("phase", "spectral_e2e"))

    # Xác thực & đồng bộ best_<branch>.pth từ đúng thư mục nguồn của resume_path (không fallback sang đích)
    source_best_path = resume_path.parent / f"best_{cfg['branch']}.pth"
    es_state = resume_ckpt.get("early_stopping_state_dict") or {}
    expected_best_epoch = int(es_state.get("best_epoch", -1))
    if expected_best_epoch >= 0:
        if not source_best_path.exists():
            raise FileNotFoundError(
                f"[RESUME LỖI] Checkpoint '{resume_path}' ghi nhận best_epoch={expected_best_epoch}, "
                f"nhưng không tìm thấy file Best Checkpoint đi kèm tại '{source_best_path}'! "
                f"Hệ thống từ chối dùng best checkpoint khác thư mục để tránh rò rỉ trọng số sai."
            )
        best_meta = torch.load(source_best_path, map_location="cpu", weights_only=False)
        best_file_epoch = int(best_meta.get("epoch", -2))
        if best_file_epoch != expected_best_epoch:
            raise RuntimeError(
                f"[RESUME LỖI] Không đồng nhất giữa '{resume_path.name}' (early_stopping.best_epoch={expected_best_epoch}) "
                f"và '{source_best_path.name}' (epoch={best_file_epoch})!"
            )

        best_cfg = best_meta.get("config", {}) or {}
        # 1. Kiểm tra tính cùng nguồn (Provenance) bằng run_id giữa last và best (ChatGPT #1):
        #    Nếu MỘT TRONG HAI có run_id thì CẢ HAI phải có cùng run_id (tránh lỗ hổng 1 file thiếu run_id lách qua).
        last_run_id = resume_ckpt.get("run_id") or ckpt_cfg.get("run_id")
        best_run_id = best_meta.get("run_id") or best_cfg.get("run_id")
        if (last_run_id is not None or best_run_id is not None) and str(last_run_id) != str(best_run_id):
            raise RuntimeError(
                f"[RESUME LỖI CÙNG NGUỒN] run_id của checkpoint '{resume_path.name}' ({last_run_id}) "
                f"không khớp với '{source_best_path.name}' ({best_run_id})! Hai checkpoint không thuộc cùng một lần chạy."
            )
        if last_run_id is not None:
            cfg["run_id"] = str(last_run_id)

        # 2. Kiểm tra dấu vân tay chỉ số (Metric Fingerprint) giữa early_stopping, last_best_metrics và best checkpoint:
        #    - Giữa last_best_metrics và best_file_metrics: bắt buộc khớp chặt (1e-6) trên cả 'auc' và 'balanced_acc'.
        #    - Giữa early_stopping.best_primary và best_file_metrics['auc'] (ChatGPT P1 #1.1):
        #      Trong vùng hòa AUC (|curr_auc - best_primary| <= min_delta) mà Bal-ACC thắng tie-break,
        #      EarlyStopping giữ best_primary = max(best_primary, curr_auc) để chống trôi lùi, nên best_primary
        #      có thể cao hơn best_file_metrics['auc'] tối đa một lượng bằng `min_delta` (+ 1e-6 dung sai số thực).
        last_best_metrics = resume_ckpt.get("best_metrics") or es_state.get("best_metrics") or {}
        best_file_metrics = best_meta.get("metrics") or best_meta.get("best_metrics") or {}
        allowed_primary_delta = float(es_state.get("min_delta", cfg.get("min_delta", 1e-4))) + 1e-6
        for metric_k, es_scalar_key in (("auc", "best_primary"), ("balanced_acc", "best_secondary")):
            if metric_k not in best_file_metrics:
                raise RuntimeError(
                    f"[RESUME LỖI CÙNG NGUỒN] File '{source_best_path.name}' thiếu chỉ số '{metric_k}' trong metrics!"
                )
            v_best = float(best_file_metrics[metric_k])
            if metric_k in last_best_metrics:
                v_last = float(last_best_metrics[metric_k])
                if not math.isclose(v_last, v_best, rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(
                        f"[RESUME LỖI CÙNG NGUỒN] Chỉ số '{metric_k}' ghi nhận trong early_stopping ({v_last:.6f}) "
                        f"không khớp với '{source_best_path.name}' ({v_best:.6f})!"
                    )
            if es_scalar_key in es_state and es_state[es_scalar_key] is not None:
                v_es = float(es_state[es_scalar_key])
                if np.isfinite(v_es):
                    if metric_k == "auc":
                        if abs(v_es - v_best) > allowed_primary_delta:
                            raise RuntimeError(
                                f"[RESUME LỖI CÙNG NGUỒN] Giá trị early_stopping.{es_scalar_key} ({v_es:.6f}) "
                                f"lệch quá ngưỡng hòa min_delta ({allowed_primary_delta:.6f}) so với "
                                f"'{source_best_path.name}'[{metric_k}] ({v_best:.6f})!"
                            )
                    elif not math.isclose(v_es, v_best, rel_tol=1e-6, abs_tol=1e-6):
                        raise RuntimeError(
                            f"[RESUME LỖI CÙNG NGUỒN] Giá trị early_stopping.{es_scalar_key} ({v_es:.6f}) "
                            f"không khớp với '{source_best_path.name}'[{metric_k}] ({v_best:.6f})!"
                        )

        # 3. Đối chiếu cấu hình kiến trúc & dataset_manifest_hash giữa best và last
        for arch_k in (
            "branch",
            "seed",
            "batch_size",
            "trainable_stages",
            "spatial_view_mode",
            "pretrained",
            "max_norm",
            "use_amp_flag",
            "deterministic",
            "dataset_manifest_hash",
        ):
            if arch_k in ckpt_cfg and arch_k in best_cfg:
                val_last = ckpt_cfg[arch_k]
                val_best = best_cfg[arch_k]
                if arch_k == "trainable_stages":
                    val_last = SpatialBranch._parse_trainable_stages(val_last)
                    val_best = SpatialBranch._parse_trainable_stages(val_best)
                if str(val_last) != str(val_best):
                    raise RuntimeError(
                        f"[RESUME LỖI CÙNG NGUỒN] Cấu hình '{arch_k}' không khớp giữa '{resume_path.name}' "
                        f"({ckpt_cfg[arch_k]}) và '{source_best_path.name}' ({best_cfg[arch_k]})!"
                    )

        # 4. Nếu best_epoch trùng với resume_epoch, toàn bộ model_state_dict của best và last phải bằng nhau tuyệt đối
        last_sd = resume_ckpt.get("model_state_dict") or {}
        best_sd = best_meta.get("model_state_dict") or {}
        if set(last_sd.keys()) != set(best_sd.keys()):
            raise RuntimeError(
                f"[RESUME LỖI CÙNG NGUỒN] Khóa model_state_dict của '{resume_path.name}' và '{source_best_path.name}' không khớp!"
            )
        if best_file_epoch == resume_epoch:
            for k_tensor, t_last in last_sd.items():
                if not torch.equal(t_last.detach().cpu(), best_sd[k_tensor].detach().cpu()):
                    raise RuntimeError(
                        f"[RESUME LỖI CÙNG NGUỒN] Tại cùng epoch={resume_epoch}, trọng số '{k_tensor}' giữa "
                        f"'{resume_path.name}' và '{source_best_path.name}' bị sai lệch!"
                    )
        elif resume_phase == "A" and str(best_cfg.get("phase", "A")) == "A":
            # Trong Pha A, toàn bộ trọng số/buffer của backbone (không thuộc classifier.*) giữa best và last phải bất biến 100%
            for k_tensor, t_last in last_sd.items():
                if not k_tensor.startswith("classifier."):
                    if not torch.equal(t_last.detach().cpu(), best_sd[k_tensor].detach().cpu()):
                        raise RuntimeError(
                            f"[RESUME LỖI CÙNG NGUỒN] Trong Pha A, trọng số backbone '{k_tensor}' giữa "
                            f"'{resume_path.name}' và '{source_best_path.name}' không khớp!"
                        )

        if source_best_path.resolve() != best_ckpt_path.resolve():
            best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_best_path, best_ckpt_path)
            print(f"[RESUME] Đã xác thực & đồng bộ best checkpoint (epoch={best_file_epoch}) từ '{source_best_path}' sang '{best_ckpt_path}'.")

    # Chỉ đọc lịch sử từ thư mục nguồn của checkpoint resume (không đọc nhầm history cũ ở thư mục đích mới)
    source_history_path = resume_path.parent / f"history_{cfg['branch']}.json"
    all_history = load_prior_history(source_history_path, max_epoch_inclusive=resume_epoch)
    print(
        f"[RESUME] Đã xác thực '{resume_path}': epoch={resume_epoch}, "
        f"phase={resume_phase}, best_epoch={expected_best_epoch}, prior_history_len={len(all_history)}"
    )
    return resume_ckpt, resume_epoch, resume_phase, source_best_path, all_history


def train_spatial_two_phase(
    model: SpatialBranch,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    cfg: Dict[str, Any],
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: Optional[torch.amp.GradScaler],
    resume_path: Optional[Path],
    best_ckpt_path: Path,
    last_ckpt_path: Path,
    history_path: Path,
) -> Tuple[List[Dict[str, Any]], EarlyStopping]:
    """
    Điều phối Huấn luyện 2 Pha cho nhánh Spatial (EfficientNet-B0):
    - Pha A (Warmup Head): epoch `0 .. warmup_epochs - 1`, `freeze_backbone(True)`, `AdamW` head_lr=1e-3.
    - Pha B (Fine-tuning): `start_epoch = warmup_epochs`, `epochs = finetune_epochs`,
      tạo mới `optimizer_b` (`backbone_lr=1e-4`, `head_lr=1e-3`) + `CosineAnnealingLR`.
    - Bảo đảm liên tục `start_epoch` (không đứt chu kỳ 3 epochs của BalancedGroupSampler), dùng chung
      `EarlyStopping` và xử lý an toàn mọi tình huống `--resume` (trong Pha A / ngay ranh giới A→B / trong Pha B).
    """
    cfg.setdefault("dataset_manifest_hash", compute_dataset_manifest_hash(train_loader, val_loader))
    warmup_epochs = int(cfg["warmup_epochs"])
    finetune_epochs = int(cfg["finetune_epochs"])
    stop_after_epoch: Optional[int] = cfg.get("_stop_after_epoch")
    early_stopping = EarlyStopping(
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        primary_metric="auc",
        secondary_metric="balanced_acc",
        mode="max",
    )
    criterion = nn.BCEWithLogitsLoss()
    all_history: List[Dict[str, Any]] = []

    # Kiểm tra trạng thái checkpoint & đồng bộ best_ckpt_path nếu đang chạy `--resume`
    resume_ckpt: Optional[Dict[str, Any]] = None
    resume_phase: str = ""
    resume_epoch: int = -1
    if resume_path is not None:
        resume_ckpt, resume_epoch, resume_phase, _, all_history = validate_and_sync_resume(
            resume_path=resume_path,
            cfg=cfg,
            best_ckpt_path=best_ckpt_path,
            history_path=history_path,
        )

    # -------------------------------------------------------------------------
    # Xác định luồng chạy dựa trên trạng thái mới hay đang Resume
    # -------------------------------------------------------------------------
    should_run_phase_a = False
    phase_a_resume_arg: Optional[Path] = None
    phase_b_resume_arg: Optional[Path] = None

    if resume_ckpt is None:
        should_run_phase_a = warmup_epochs > 0
    else:
        next_epoch = resume_epoch + 1
        if resume_phase == "A" and next_epoch < warmup_epochs:
            # Đang dở giữa Pha A (ví dụ warmup_epochs >= 2)
            should_run_phase_a = True
            phase_a_resume_arg = resume_path
        elif resume_phase == "A" and next_epoch >= warmup_epochs:
            # Pha A đã xong trọn vẹn -> giữ nguyên trọng số tốt nhất Pha A + trạng thái EarlyStopping, đi thẳng vào Pha B
            dl_gen = getattr(train_loader, "generator", None)
            if not isinstance(dl_gen, torch.Generator):
                dl_gen = None
            elif getattr(train_loader, "_base_seed_gen_state", None) is None and getattr(train_loader, "_iterator", None) is None:
                train_loader._base_seed_gen_state = dl_gen.get_state().detach().cpu().clone()
            _ = load_checkpoint(
                filepath=resume_path,
                model=model,
                early_stopping=early_stopping,
                device=device,
                restore_rng=True,
                dataloader_generator=dl_gen,
                train_loader=train_loader,
            )
            if not best_ckpt_path.exists():
                raise FileNotFoundError(
                    f"[LỖI RESUME] Không tìm thấy Best Checkpoint Pha A tại '{best_ckpt_path}' "
                    f"để chuyển giao trọng số sang Pha B!"
                )
            _ = load_checkpoint(best_ckpt_path, model, device=device, restore_rng=False)
            print(
                f"  -> Pha A đã hoàn tất tại epoch {resume_epoch}. Chuyển thẳng sang Pha B từ best weights Pha A "
                f"(start_epoch={warmup_epochs})."
            )
        else:
            # Đang ở trong Pha B
            phase_b_resume_arg = resume_path

    # -------------------------------------------------------------------------
    # PHA A: WARMUP HEAD (Đóng băng 100% backbone, chỉ cập nhật classifier)
    # -------------------------------------------------------------------------
    if should_run_phase_a:
        phase_a_target_epochs = (
            warmup_epochs
            if stop_after_epoch is None
            else min(warmup_epochs, max(0, int(stop_after_epoch) + 1))
        )
        print("\n" + "=" * 80)
        print(
            f"PHA A — WARMUP HEAD (epochs 0..{warmup_epochs - 1}, head_lr={cfg['head_lr']:.1e}, "
            f"backbone=FROZEN)"
        )
        print("=" * 80)

        model.freeze_backbone(True)
        param_groups_a = model.get_param_groups(
            backbone_lr=cfg["backbone_lr"],
            head_lr=cfg["head_lr"],
            weight_decay=cfg["weight_decay"],
        )
        assert len(param_groups_a) == 1 and param_groups_a[0]["group_name"] == "head", (
            f"Pha A chỉ được phép chứa nhóm 'head', nhận được: {[g['group_name'] for g in param_groups_a]}"
        )
        optimizer_a = torch.optim.AdamW(param_groups_a)
        assert_optimizer_covers_trainable_params(model, optimizer_a, context_label="Pha A")

        summary_a = model.stage_summary()
        print(
            f"  + Tham số học Pha A : {summary_a['trainable_params']:,} / {summary_a['total_params']:,} | "
            f"BN eval={summary_a['backbone_bn_in_eval']}/{summary_a['backbone_bn_total']}, "
            f"SD eval={summary_a['backbone_sd_in_eval']}/{summary_a['backbone_sd_total']}"
        )

        bb_params_snap, bb_buffers_snap = snapshot_backbone_full(model, only_frozen=False)
        if phase_a_resume_arg is not None and resume_ckpt is not None:
            ckpt_sd = resume_ckpt.get("model_state_dict", {})
            bb_params_snap = {k: ckpt_sd[k].detach().cpu().clone() for k in bb_params_snap if k in ckpt_sd}
            bb_buffers_snap = {k: ckpt_sd[k].detach().cpu().clone() for k in bb_buffers_snap if k in ckpt_sd}
        head_snap_a = _snapshot_param(model, in_classifier=True, only_trainable=True)

        phase_a_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        phase_a_cfg["phase"] = "A"
        res_a = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer_a,
            device=device,
            epochs=phase_a_target_epochs,
            criterion=criterion,
            scheduler=None,
            early_stopping=early_stopping,
            checkpoint_path=best_ckpt_path,
            last_checkpoint_path=last_ckpt_path,
            resume_from=phase_a_resume_arg,
            start_epoch=0,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_norm=cfg["max_norm"],
            config=phase_a_cfg,
            log_interval=100,
            restore_best_weights=True,  # Điều kiện #5: chuyển sang Pha B bằng trọng số classifier tốt nhất của Pha A
        )
        for rec in res_a["history"]:
            rec["phase"] = "A"
            all_history.append(rec)

        # Kiểm chứng Điều kiện #2 & ChatGPT #4: Xác nhận 100% tham số backbone và toàn bộ BatchNorm buffers không đổi
        assert_backbone_unchanged(model, bb_params_snap, bb_buffers_snap, context_label="Pha A")
        if head_snap_a is not None and len(res_a["history"]) > 0 and phase_a_resume_arg is None:
            assert not torch.equal(_current_param_tensor(model, head_snap_a[0]), head_snap_a[1]), (
                f"Cảnh báo Pha A: trọng số head '{head_snap_a[0]}' không cập nhật!"
            )
        print(
            f"  + [XÁC NHẬN PHA A] Toàn bộ 100% tham số ({len(bb_params_snap):,}) & bộ đệm BatchNorm ({len(bb_buffers_snap):,}) "
            f"bất biến tuyệt đối, chỉ head classifier được cập nhật."
        )
        save_json({"config": phase_a_cfg, "history": all_history}, history_path)

        if stop_after_epoch is not None and int(stop_after_epoch) < warmup_epochs:
            return all_history, early_stopping

    # -------------------------------------------------------------------------
    # PHA B: FULL FINE-TUNING (Mở khóa backbone theo trainable_stages + Differential LR)
    # -------------------------------------------------------------------------
    phase_b_target_epochs = (
        finetune_epochs
        if stop_after_epoch is None
        else max(0, min(finetune_epochs, int(stop_after_epoch) + 1 - warmup_epochs))
    )
    if phase_b_target_epochs > 0 and not early_stopping.early_stop:
        print("\n" + "=" * 80)
        print(
            f"PHA B — FINE-TUNING (start_epoch={warmup_epochs}, epochs={finetune_epochs}, "
            f"backbone_lr={cfg['backbone_lr']:.1e}, head_lr={cfg['head_lr']:.1e}, "
            f"stages={cfg['trainable_stages']})"
        )
        print("=" * 80)

        # Tạo MỚI hoàn toàn optimizer_b & scheduler_b (với đúng T_max=finetune_epochs) TRƯỚC KHI gọi fit(resume_from=...)
        optimizer_b, scheduler_b = build_phase_b_components(
            model=model,
            backbone_lr=cfg["backbone_lr"],
            head_lr=cfg["head_lr"],
            weight_decay=cfg["weight_decay"],
            finetune_epochs=finetune_epochs,
        )
        summary_b = model.stage_summary()
        print(
            f"  + Tham số học Pha B : {summary_b['trainable_params']:,} / {summary_b['total_params']:,} | "
            f"Active stages={summary_b['currently_unlocked_stages']} | "
            f"Optimizer groups={[pg.get('group_name') for pg in optimizer_b.param_groups]}"
        )

        # Chụp 100% vùng backbone còn đóng băng (nếu trainable_stages != 'all') + 1 tham số vùng mở khóa
        frozen_b_params, frozen_b_buffers = snapshot_backbone_full(model, only_frozen=True)
        bb_snap_b = _snapshot_param(model, in_classifier=False, only_trainable=True)
        if phase_b_resume_arg is not None and resume_ckpt is not None:
            ckpt_sd = resume_ckpt.get("model_state_dict", {})
            frozen_b_params = {k: ckpt_sd[k].detach().cpu().clone() for k in frozen_b_params if k in ckpt_sd}
            frozen_b_buffers = {k: ckpt_sd[k].detach().cpu().clone() for k in frozen_b_buffers if k in ckpt_sd}
            if bb_snap_b is not None and bb_snap_b[0] in ckpt_sd:
                bb_snap_b = (bb_snap_b[0], ckpt_sd[bb_snap_b[0]].detach().cpu().clone())

        phase_b_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        phase_b_cfg["phase"] = "B"
        res_b = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer_b,
            device=device,
            epochs=phase_b_target_epochs,
            criterion=criterion,
            scheduler=scheduler_b,
            early_stopping=early_stopping,
            checkpoint_path=best_ckpt_path,
            last_checkpoint_path=last_ckpt_path,
            resume_from=phase_b_resume_arg,
            start_epoch=warmup_epochs,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_norm=cfg["max_norm"],
            config=phase_b_cfg,
            log_interval=100,
            restore_best_weights=False,  # Tự nạp best weights ngay dưới để kiểm chứng xong bước cập nhật cuối
        )
        for rec in res_b["history"]:
            rec["phase"] = "B"
            all_history.append(rec)

        if len(frozen_b_params) > 0 or len(frozen_b_buffers) > 0:
            assert_backbone_unchanged(model, frozen_b_params, frozen_b_buffers, context_label="Pha B (Frozen Stages)")
        if bb_snap_b is not None and len(res_b["history"]) > 0:
            assert not torch.equal(_current_param_tensor(model, bb_snap_b[0]), bb_snap_b[1]), (
                f"Vi phạm Pha B: trọng số backbone '{bb_snap_b[0]}' không thay đổi sau fine-tune!"
            )
            print(
                f"  + [XÁC NHẬN PHA B] Trọng số vùng mở khóa ('{bb_snap_b[0]}') đã cập nhật qua Optimizer B; "
                f"vùng khóa ({len(frozen_b_params):,} params, {len(frozen_b_buffers):,} buffers) bất biến 100%."
            )
        save_json({"config": phase_b_cfg, "history": all_history}, history_path)

    # Luôn khôi phục trọng số tốt nhất từ `best_ckpt_path` trước khi tổng kết Val / Test
    if best_ckpt_path.exists():
        best_ckpt = load_checkpoint(best_ckpt_path, model, device=device, restore_rng=False)
        best_ep = best_ckpt.get("epoch", early_stopping.best_epoch)
        print(
            f"\n-> Đã khôi phục best checkpoint '{best_ckpt_path.name}' "
            f"(Epoch {best_ep}: Val AUC={early_stopping.best_metrics.get('auc', 0.0):.4f}, "
            f"Bal-ACC={early_stopping.best_metrics.get('balanced_acc', 0.0) * 100:.2f}%)."
        )

    return all_history, early_stopping


def train_spectral_e2e(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    cfg: Dict[str, Any],
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: Optional[torch.amp.GradScaler],
    resume_path: Optional[Path],
    best_ckpt_path: Path,
    last_ckpt_path: Path,
    history_path: Path,
) -> Tuple[List[Dict[str, Any]], EarlyStopping]:
    """
    Điều phối huấn luyện từ đầu (End-to-End) cho nhánh Spectral (`FFT.py`):
    1 epoch Linear Warmup (0.1*lr -> lr) + CosineAnnealingLR cho các epoch còn lại.
    """
    cfg.setdefault("dataset_manifest_hash", compute_dataset_manifest_hash(train_loader, val_loader))
    total_epochs = int(cfg["total_epochs"])
    warmup_epochs = int(cfg["warmup_epochs"])
    early_stopping = EarlyStopping(
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        primary_metric="auc",
        secondary_metric="balanced_acc",
        mode="max",
    )
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["head_lr"],
        weight_decay=cfg["weight_decay"],
    )
    assert_optimizer_covers_trainable_params(model, optimizer, context_label="Spectral E2E")

    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        scheduler: Any = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=1e-6,
        )

    all_history: List[Dict[str, Any]] = []
    if resume_path is not None:
        _, _, _, _, all_history = validate_and_sync_resume(
            resume_path=resume_path,
            cfg=cfg,
            best_ckpt_path=best_ckpt_path,
            history_path=history_path,
        )

    spectral_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    spectral_cfg["phase"] = "spectral_e2e"
    stop_after_epoch: Optional[int] = cfg.get("_stop_after_epoch")
    spectral_target_epochs = (
        total_epochs
        if stop_after_epoch is None
        else min(total_epochs, max(0, int(stop_after_epoch) + 1))
    )
    res = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=spectral_target_epochs,
        criterion=criterion,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint_path=best_ckpt_path,
        last_checkpoint_path=last_ckpt_path,
        resume_from=resume_path,
        start_epoch=0,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=scaler,
        max_norm=cfg["max_norm"],
        config=spectral_cfg,
        log_interval=100,
        restore_best_weights=True,
    )
    for rec in res["history"]:
        rec["phase"] = "spectral_e2e"
        all_history.append(rec)
    save_json({"config": spectral_cfg, "history": all_history}, history_path)

    return all_history, early_stopping


def run_test_evaluation(
    model: nn.Module,
    test_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    branch: str,
) -> Dict[str, Any]:
    """
    Đánh giá cuối cùng trên toàn bộ 43,800 ảnh Test trong 1 lượt duyệt duy nhất
    (chỉ gọi SAU KHI đã chốt xong best checkpoint).
    Kiểm tra nghiêm ngặt cân bằng cặp 1:1 (7,300 Real vs 7,300 mỗi generator) cho cả 5 generators.
    """
    print("\n" + "=" * 85)
    print(f"ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST (43,800 ẢNH KNOWN + UNSEEN) — NHÁNH '{branch.upper()}'")
    print("=" * 85)

    model.eval()
    input_key = model.input_key
    amp_enabled = bool(use_amp and device.type == "cuda")
    total_loss = 0.0
    num_samples = 0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_models: List[str] = []

    with torch.inference_mode():
        for batch in test_loader:
            x = batch[input_key].to(device, non_blocking=True)
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

    test_loss = total_loss / max(num_samples, 1)
    y_true = torch.cat(all_targets, dim=0).numpy()
    y_logits = torch.cat(all_logits, dim=0).numpy()
    test_gen_eval = evaluate_by_generator(
        y_true=y_true,
        y_pred=y_logits,
        model_names=all_models,
        from_logits=True,
        require_balanced_pairs=True,
        strict_known_generators=True,
    )
    test_overall = test_gen_eval["overall"]
    assert test_gen_eval["shared_real_anchor_count"] == 7300, (
        f"Tập Test phải có đúng 7,300 ảnh Real MS_COCO, nhận được: {test_gen_eval['shared_real_anchor_count']}"
    )
    assert len(test_gen_eval["per_generator"]) == len(ALL_AI_GENERATORS), (
        f"Tập Test phải đủ 5 AI generators, nhận được: {list(test_gen_eval['per_generator'].keys())}"
    )

    print(f"  + Test BCE Loss : {test_loss:.4f}")
    print(format_metrics_summary(test_overall, f"TỔNG QUÁT TEST OVERALL (POOLED 43,800 ẢNH) — {branch.upper()}"))
    print("\n" + format_generator_report(test_gen_eval, f"BÁO CÁO TEST CẶP 1:1 THEO GENERATOR — {branch.upper()}"))

    return {
        "test_loss": test_loss,
        "overall": test_overall,
        "per_generator_report": test_gen_eval,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện Baseline cho các nhánh đơn lẻ (Spatial EfficientNet-B0 / Spectral FFT CNN)."
    )
    parser.add_argument("--branch", type=str, default="spatial", choices=("spatial", "spectral"))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(SYSTEM_DIR / "checkpoints"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2, help="Số worker Train (mặc định 2: an toàn RAM Windows & theo kịp GPU).")
    parser.add_argument("--eval-workers", type=int, default=None, help="Số worker Val/Test (mặc định min(num_workers, 2) trên Windows).")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--epochs", type=int, default=None, help="Tổng số epoch (mặc định: 16 spatial, 30 spectral).")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Số epoch khởi động (mặc định: 1).")
    parser.add_argument("--backbone-lr", type=float, default=None, help="LR cho backbone ở Pha B (mặc định: 1e-4).")
    parser.add_argument("--head-lr", type=float, default=None, help="LR cho head / nhánh spectral (mặc định: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay AdamW (mặc định: 1e-4).")
    parser.add_argument("--dropout", type=float, default=None, help="Tỉ lệ Dropout (mặc định: 0.2 spatial, 0.3 spectral).")
    parser.add_argument("--patience", type=int, default=None, help="EarlyStopping patience (mặc định: 4 spatial, 5 spectral).")
    parser.add_argument("--min-delta", type=float, default=None, help="Ngưỡng cải thiện tối thiểu (mặc định: 1e-4).")
    parser.add_argument("--max-norm", type=float, default=None, help="Gradient clipping max_norm (mặc định: 1.0).")

    parser.add_argument("--trainable-stages", type=str, default="all", help="Mặc định Baseline='all'. Chỉ dùng ablation khi cần.")
    parser.add_argument("--spatial-view-mode", type=str, default="native", choices=("native", "global"))
    parser.add_argument("--jpeg-prob", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--sanity-check", action="store_true", help="Chạy Cổng kiểm tra 3 bước trước khi train.")
    parser.add_argument("--sanity-only", action="store_true", help="Chỉ chạy Cổng kiểm tra 3 bước rồi thoát.")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn tới file last_<branch>.pth để tiếp tục train.")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá checkpoint đã có trên Val (và Test nếu bật --eval-test).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint dùng cho --eval-only.")
    parser.add_argument("--eval-test", action="store_true", help="Đánh giá tập Test sau khi hoàn tất huấn luyện.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sanity_only and args.resume is not None:
        raise ValueError("--sanity-only không được kết hợp với --resume! Hãy chọn 1 trong 2 chế độ.")
    if args.sanity_only and args.eval_test:
        raise ValueError("--sanity-only chỉ chạy Cổng kiểm tra rồi thoát, không thể đi cùng --eval-test!")
    if args.eval_only and args.resume is not None:
        raise ValueError("--eval-only dùng để đánh giá mô hình, không thể đi cùng --resume!")
    if not args.eval_only and args.checkpoint is not None:
        raise ValueError("--checkpoint chỉ được dùng khi bật cờ --eval-only!")
    cfg = resolve_config(args)
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])

    run_dir: Path = cfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / f"best_{cfg['branch']}.pth"
    last_ckpt_path = run_dir / f"last_{cfg['branch']}.pth"
    history_path = run_dir / f"history_{cfg['branch']}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_amp, amp_dtype, scaler = get_optimal_amp_config(device)
    use_amp = bool(auto_amp and cfg["use_amp_flag"])
    if not use_amp:
        amp_dtype = torch.float32
        scaler = None

    print("=" * 85)
    print(f"HUẤN LUYỆN NHÁNH '{cfg['branch'].upper()}' — RUN: {cfg['run_name']}")
    print("=" * 85)
    dev_desc = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  + Thiết bị    : {device.type.upper()} ({dev_desc}) | AMP: use_amp={use_amp}, dtype={amp_dtype}")
    print(f"  + Thư mục lưu : {run_dir}")
    print(
        f"  + Cấu hình    : total_epochs={cfg['total_epochs']} (warmup={cfg['warmup_epochs']}, finetune={cfg['finetune_epochs']}) | "
        f"batch={cfg['batch_size']}/{cfg['eval_batch_size']} | workers={cfg['num_workers']} "
        f"(eval_w={cfg['eval_workers']}, persist={cfg['persistent_workers']}) | "
        f"patience={cfg['patience']} | stages={cfg['trainable_stages']}"
    )

    # Nạp DataLoader CHỈ kích hoạt đúng 1 nhánh để tiết kiệm tải CPU
    t0_all = time.perf_counter()
    train_loader, val_loader, test_loader = _build_branch_dataloaders(cfg)

    model = build_branch_model(
        branch=cfg["branch"],
        pretrained=cfg["pretrained"],
        dropout=cfg["dropout"],
        trainable_stages=cfg["trainable_stages"],
    ).to(device)
    print(
        f"  + Mô hình     : {model.__class__.__name__} | input_key='{model.input_key}' | "
        f"feature_dim={model.feature_dim} | Tổng params={model.count_parameters(trainable_only=False):,}"
    )

    # Cổng kiểm tra 3 bước (Sanity Check Gate) nếu được yêu cầu
    if args.sanity_check or args.sanity_only:
        if args.resume is not None:
            print("  [Lưu ý] Bỏ qua Sanity Check Gate vì đang ở chế độ --resume từ checkpoint.")
        else:
            gate_ok = sanity_check_gate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                overfit_lr=3e-3,
                overfit_steps=100,
                smoke_train_batches=50,
                smoke_val_batches=30,
                smoke_lr=cfg["backbone_lr"],
            )
            if not gate_ok:
                _shutdown_dataloader_workers(train_loader, val_loader, test_loader)
                raise SystemExit("Sanity Check Gate THẤT BẠI — Dừng quy trình huấn luyện chính thức!")
            if args.sanity_only:
                _shutdown_dataloader_workers(train_loader, val_loader, test_loader)
                print("\n-> Hoàn tất `--sanity-only` thành công. Thoát sạch theo yêu cầu.")
                return

            # Đóng tường minh persistent workers cũ trước khi tái tạo DataLoader sạch để không nhân đôi VMS trên Windows
            _shutdown_dataloader_workers(train_loader, val_loader, test_loader)
            set_seed(cfg["seed"], deterministic=cfg["deterministic"])
            train_loader, val_loader, test_loader = _build_branch_dataloaders(cfg)

    criterion = nn.BCEWithLogitsLoss()

    if args.eval_only:
        eval_ckpt = Path(args.checkpoint) if args.checkpoint else best_ckpt_path
        load_checkpoint(eval_ckpt, model, device=device, restore_rng=False)
        print(f"\n[EVAL-ONLY] Đã nạp checkpoint: {eval_ckpt}")
        all_history: List[Dict[str, Any]] = []
        early_stopping = None
    else:
        resume_path = Path(args.resume) if args.resume else None
        if cfg["branch"] == "spatial":
            all_history, early_stopping = train_spatial_two_phase(
                model=model,  # type: ignore[arg-type]
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                cfg=cfg,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                resume_path=resume_path,
                best_ckpt_path=best_ckpt_path,
                last_ckpt_path=last_ckpt_path,
                history_path=history_path,
            )
        else:
            all_history, early_stopping = train_spectral_e2e(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                cfg=cfg,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                resume_path=resume_path,
                best_ckpt_path=best_ckpt_path,
                last_ckpt_path=last_ckpt_path,
                history_path=history_path,
            )

    # Báo cáo chi tiết cuối cùng trên Val (Known) từ Best Checkpoint
    val_loss, val_metrics, val_gen_eval = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        compute_per_generator=True,
    )
    print("\n" + format_metrics_summary(val_metrics, f"ĐÁNH GIÁ BEST CHECKPOINT TRÊN VAL (KNOWN) — {cfg['branch'].upper()}"))
    if val_gen_eval is not None:
        print("\n" + format_generator_report(val_gen_eval, f"BÁO CÁO VAL CẶP 1:1 THEO GENERATOR — {cfg['branch'].upper()}"))

    elapsed_total = time.perf_counter() - t0_all
    if not args.eval_only:
        summary_payload = {
            "config": cfg,
            "best_epoch": early_stopping.best_epoch if early_stopping is not None else -1,
            "best_val_metrics": early_stopping.best_metrics if early_stopping is not None else val_metrics,
            "final_val_loss": val_loss,
            "final_val_metrics": val_metrics,
            "final_val_per_generator": val_gen_eval,
            "history": all_history,
            "total_wall_time_sec": elapsed_total,
        }
        save_json(summary_payload, history_path)
        print(f"\n-> Đã lưu lịch sử huấn luyện & chỉ số Val tại: {history_path}")
        print(f"-> Best checkpoint : {best_ckpt_path}")
        print(f"-> Last checkpoint : {last_ckpt_path}")

    # Chỉ đánh giá Test nếu người dùng bật tường minh `--eval-test` (SAU KHI đã chốt xong mô hình)
    if args.eval_test:
        test_payload = run_test_evaluation(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            branch=cfg["branch"],
        )
        test_out_path = run_dir / f"test_metrics_{cfg['branch']}.json"
        save_json(test_payload, test_out_path)
        print(f"\n-> Đã lưu kết quả Test chính thức tại: {test_out_path}")

    print(f"\nHOÀN TẤT TOÀN BỘ QUY TRÌNH TRONG {elapsed_total / 60.0:.2f} PHÚT!")


if __name__ == "__main__":
    main()