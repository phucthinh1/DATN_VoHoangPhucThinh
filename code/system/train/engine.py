# -*- coding: utf-8 -*-
"""
code/system/train/engine.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Bộ máy huấn luyện & kiểm định dùng chung (Unified Training & Evaluation Engine):
1. `get_optimal_amp_config()`      : Tự động chọn bf16 (không cần GradScaler) hoặc fp16 + GradScaler.
2. `train_one_epoch()`             : Huấn luyện 1 epoch với AMP, gradient clipping, set_epoch tường minh.
3. `evaluate()`                    : Đánh giá toàn tập (an toàn với bfloat16 -> float32 -> numpy).
4. `EarlyStopping`                 : Chọn best checkpoint theo AUC, phân xử hòa bằng Balanced ACC (chống trôi mốc).
5. `save_checkpoint` / `load_...`  : Lưu/khôi phục đầy đủ model, optimizer, scheduler, scaler (None khi bf16),
                                     epoch, best_metrics và config.
6. `fit()`                         : Điều phối trọn vòng đời huấn luyện đa epoch + khôi phục best weights.
7. `overfit_single_batch_check()`  : Cổng kiểm tra 1 — giảm ≥ 90% loss & đạt 100% ACC trong ≤ 100 bước
                                     (tự động sao lưu & khôi phục trọng số gốc sau khi kiểm tra).
8. `checkpoint_consistency_check()`: Cổng kiểm tra 3 — Save -> Mutate -> Load -> allclose(logits) ở eval().
9. `sanity_check_gate()`           : Cổng kiểm tra tổng hợp 3 bước (Overfit 1 batch + 50 batch Train/Val + Checkpoint).
"""

import copy
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Định tuyến đường dẫn liên thư mục để import thông suốt dù gọi từ bất kỳ thư mục làm việc nào
CURRENT_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = CURRENT_DIR.parent
BRANCH_DIR = SYSTEM_DIR / "branch"

for _p in (str(CURRENT_DIR), str(BRANCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import compute_binary_metrics, evaluate_by_generator  # noqa: E402


def _get_model_input_key(model: nn.Module) -> str:
    """Lấy và kiểm tra chặt thuộc tính `input_key` của mô hình nhánh (không fallback ngầm)."""
    input_key = getattr(model, "input_key", None)
    if not isinstance(input_key, str) or not input_key:
        raise AttributeError(
            f"Mô hình {model.__class__.__name__} thiếu thuộc tính bắt buộc `input_key` "
            f"('spatial' | 'semantic' | 'spectral'). Hãy kế thừa `BranchBase`."
        )
    return input_key


def get_optimal_amp_config(device: torch.device) -> Tuple[bool, torch.dtype, Optional[torch.amp.GradScaler]]:
    """
    Tự động xác định cấu hình Mixed Precision (AMP) tối ưu theo phần cứng (Mục 8.1 #9):
    - GPU hỗ trợ bf16 (RTX 3050 / Ampere / L4...): `(True, torch.bfloat16, None)`.
    - GPU cũ hơn chỉ hỗ trợ fp16 (T4 / P100...)   : `(True, torch.float16, GradScaler('cuda'))`.
    - CPU                                        : `(False, torch.float32, None)`.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, None
        return True, torch.float16, torch.amp.GradScaler("cuda")
    return False, torch.float32, None


def _sync_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """
    Gọi tường minh `set_epoch(epoch)` trên cả Dataset lẫn Sampler (Mục 8.1 #3).
    Cả hai hàm đều idempotent nên gọi tường minh an toàn tuyệt đối kể cả khi đã attach_sampler.
    """
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)


def _check_and_clip_gradients(
    trainable_params: List[torch.Tensor],
    max_norm: float,
    epoch: int,
    batch_idx: int,
) -> float:
    """
    Kiểm tra toàn bộ gradient hữu hạn (không NaN/Inf) TRƯỚC khi clip và gọi `optimizer.step()`.
    Đặc biệt quan trọng với chế độ `bfloat16` / `float32` không dùng `GradScaler`, ngăn chặn triệt để
    việc gradient NaN/Inf cập nhật làm hỏng trọng số mô hình.
    """
    grads = [p.grad for p in trainable_params if p.grad is not None]
    if not grads:
        raise RuntimeError(
            f"Không có gradient nào được sinh ra sau backward() tại epoch {epoch}, batch {batch_idx}!"
        )

    for g in grads:
        if not torch.isfinite(g).all():
            raise FloatingPointError(
                f"Gradient bị NaN/Inf TRƯỚC optimizer.step() tại epoch {epoch}, batch {batch_idx}!"
            )

    if max_norm > 0.0:
        total_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=max_norm, error_if_nonfinite=True
        )
        return float(total_norm.item())

    total_norm_sq = sum(float(g.detach().float().norm(2).item()) ** 2 for g in grads)
    return float(total_norm_sq ** 0.5)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    max_norm: float = 1.0,
    max_batches: Optional[int] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    """
    Huấn luyện mô hình trong 1 epoch (hoặc `max_batches` batch đầu tiên ở Cổng kiểm tra).
    Kiểm tra tính hữu hạn của `loss` TRƯỚC `backward()` và kiểm tra gradient TRƯỚC `optimizer.step()`.
    """
    model.train()
    _sync_loader_epoch(loader, epoch)

    input_key = _get_model_input_key(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            f"Mô hình {model.__class__.__name__} không có tham số nào mở khóa đạo hàm (requires_grad=True)!"
        )

    total_loss = 0.0
    num_samples = 0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    amp_enabled = bool(use_amp and device.type == "cuda")
    t0 = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch[input_key].to(device, non_blocking=True)
        y = batch["label_a"].to(device, non_blocking=True).float()  # Shape (B,)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(x)  # Shape (B,)
                loss = criterion(logits.float(), y)

            if not torch.isfinite(loss).all():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Train loss bị NaN/Inf TRƯỚC backward() tại epoch {epoch}, batch {batch_idx}: loss={loss.item()}"
                )

            if scaler is not None:
                # Ở chế độ fp16 + GradScaler, scaler.step() tự động bỏ qua cập nhật nếu tràn gradient tạm thời
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                try:
                    _check_and_clip_gradients(trainable_params, max_norm=max_norm, epoch=epoch, batch_idx=batch_idx)
                except Exception:
                    optimizer.zero_grad(set_to_none=True)
                    raise
                optimizer.step()
        else:
            logits = model(x)
            loss = criterion(logits.float(), y)

            if not torch.isfinite(loss).all():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Train loss bị NaN/Inf TRƯỚC backward() tại epoch {epoch}, batch {batch_idx}: loss={loss.item()}"
                )

            loss.backward()
            try:
                _check_and_clip_gradients(trainable_params, max_norm=max_norm, epoch=epoch, batch_idx=batch_idx)
            except Exception:
                optimizer.zero_grad(set_to_none=True)
                raise
            optimizer.step()

        batch_size = int(x.size(0))
        total_loss += float(loss.item()) * batch_size
        num_samples += batch_size

        # Ép về float32 trước khi lưu CPU để tương thích 100% với NumPy (tránh lỗi BFloat16)
        all_logits.append(logits.detach().float().cpu())
        all_targets.append(y.detach().float().cpu())

        if log_interval > 0 and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches):
            current_avg_loss = total_loss / num_samples
            elapsed = time.perf_counter() - t0
            print(
                f"  [Epoch {epoch:02d}] Batch {batch_idx + 1:03d}/{total_batches:03d} | "
                f"Train Loss: {current_avg_loss:.4f} | {elapsed:.1f}s"
            )

    epoch_time = time.perf_counter() - t0
    avg_loss = total_loss / max(num_samples, 1)

    train_auc = 0.0
    train_bacc = 0.0
    if num_samples > 0:
        concat_logits = torch.cat(all_logits, dim=0).numpy()
        concat_targets = torch.cat(all_targets, dim=0).numpy()
        tr_metrics = compute_binary_metrics(concat_targets, concat_logits, from_logits=True)
        train_auc = float(tr_metrics["auc"])
        train_bacc = float(tr_metrics["balanced_acc"])

    return {
        "train_loss": avg_loss,
        "train_auc": train_auc,
        "train_balanced_acc": train_bacc,
        "epoch_time": epoch_time,
        "num_samples": num_samples,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
    compute_per_generator: bool = True,
) -> Tuple[float, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Đánh giá mô hình trên toàn bộ tập dữ liệu Validation hoặc Test (Mục 8.1 #13):
    Gom toàn bộ logits trước khi tính chỉ số, tuyệt đối không lấy trung bình theo batch.
    """
    model.eval()
    input_key = _get_model_input_key(model)
    amp_enabled = bool(use_amp and device.type == "cuda")

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_models: List[str] = []
    total_loss = 0.0
    num_samples = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = batch[input_key].to(device, non_blocking=True)
            y = batch["label_a"].to(device, non_blocking=True).float()

            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(x)
                loss = criterion(logits.float(), y)
            else:
                logits = model(x)
                loss = criterion(logits.float(), y)

            batch_size = int(x.size(0))
            total_loss += float(loss.item()) * batch_size
            num_samples += batch_size

            # BẮT BUỘC .float() trước .cpu() vì NumPy không hỗ trợ BFloat16
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(y.detach().float().cpu())

            if "model_name" in batch:
                m_names = batch["model_name"]
                if isinstance(m_names, (list, tuple)):
                    all_models.extend([str(m) for m in m_names])
                elif isinstance(m_names, np.ndarray):
                    all_models.extend([str(m) for m in m_names.tolist()])
                elif isinstance(m_names, str):
                    all_models.append(m_names)

    if num_samples == 0:
        return 0.0, {}, None

    avg_loss = total_loss / num_samples
    concat_logits = torch.cat(all_logits, dim=0).numpy()
    concat_targets = torch.cat(all_targets, dim=0).numpy()

    metrics = compute_binary_metrics(concat_targets, concat_logits, from_logits=True)

    per_gen_eval: Optional[Dict[str, Any]] = None
    if compute_per_generator and len(all_models) == len(concat_targets):
        per_gen_eval = evaluate_by_generator(
            y_true=concat_targets,
            y_pred=concat_logits,
            model_names=all_models,
            from_logits=True,
        )

    return avg_loss, metrics, per_gen_eval


class EarlyStopping:
    """
    Theo dõi dừng sớm và chọn checkpoint tốt nhất (Mục 1 & Mục 8.2):
    - Tiêu chí chính: `auc` (`min_delta=1e-4`).
    - Phân xử khi hòa (`|curr_auc - anchor_auc| <= min_delta`): xét `balanced_acc`.
    - Chống trôi mốc (anti-drift): khi phân xử hòa thắng, `anchor_primary = max(anchor_primary, curr_primary)`
      để các epoch hòa liên tiếp không bao giờ kéo tụt mốc AUC xuống dần.
    """

    def __init__(
        self,
        patience: int = 4,
        min_delta: float = 1e-4,
        primary_metric: str = "auc",
        secondary_metric: str = "balanced_acc",
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode phải là 'max' hoặc 'min', nhận được: {mode}")
        self.patience = patience
        self.min_delta = min_delta
        self.primary_metric = primary_metric
        self.secondary_metric = secondary_metric
        self.mode = mode

        self.best_primary: float = -float("inf") if mode == "max" else float("inf")
        self.best_secondary: float = -float("inf") if mode == "max" else float("inf")
        self.best_metrics: Dict[str, Any] = {}
        self.best_epoch: int = -1
        self.wait_count: int = 0
        self.early_stop: bool = False
        self.is_improved: bool = False

    def step(self, metrics: Dict[str, Any], epoch: int) -> bool:
        curr_primary = float(metrics.get(self.primary_metric, 0.0))
        curr_secondary = float(metrics.get(self.secondary_metric, 0.0))
        self.is_improved = False

        if self.mode == "max":
            primary_diff = curr_primary - self.best_primary
            if primary_diff > self.min_delta:
                self.best_primary = curr_primary
                self.best_secondary = curr_secondary
                self.best_metrics = dict(metrics)
                self.best_epoch = epoch
                self.wait_count = 0
                self.is_improved = True
            elif abs(primary_diff) <= self.min_delta:
                if curr_secondary > self.best_secondary + self.min_delta:
                    # Giữ max để vừa nâng chuẩn nếu AUC nhỉnh hơn, vừa chống trôi lùi nếu AUC thấp hơn trong ngưỡng hòa
                    self.best_primary = max(self.best_primary, curr_primary)
                    self.best_secondary = curr_secondary
                    self.best_metrics = dict(metrics)
                    self.best_epoch = epoch
                    self.wait_count = 0
                    self.is_improved = True
                else:
                    self.wait_count += 1
            else:
                self.wait_count += 1
        else:
            primary_diff = self.best_primary - curr_primary
            if primary_diff > self.min_delta:
                self.best_primary = curr_primary
                self.best_secondary = curr_secondary
                self.best_metrics = dict(metrics)
                self.best_epoch = epoch
                self.wait_count = 0
                self.is_improved = True
            elif abs(primary_diff) <= self.min_delta and curr_secondary < self.best_secondary - self.min_delta:
                self.best_primary = min(self.best_primary, curr_primary)
                self.best_secondary = curr_secondary
                self.best_metrics = dict(metrics)
                self.best_epoch = epoch
                self.wait_count = 0
                self.is_improved = True
            else:
                self.wait_count += 1

        if self.wait_count >= self.patience:
            self.early_stop = True

        return self.is_improved

    def state_dict(self) -> Dict[str, Any]:
        """Xuất toàn bộ trạng thái dừng sớm để đóng gói trực tiếp vào checkpoint Resume."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "primary_metric": self.primary_metric,
            "secondary_metric": self.secondary_metric,
            "mode": self.mode,
            "best_primary": self.best_primary,
            "best_secondary": self.best_secondary,
            "best_metrics": copy.deepcopy(self.best_metrics),
            "best_epoch": self.best_epoch,
            "wait_count": self.wait_count,
            "early_stop": self.early_stop,
            "is_improved": self.is_improved,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Khôi phục nguyên vẹn mốc AUC tốt nhất và bộ đếm `wait_count` khi Resume giữa chừng."""
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.primary_metric = str(state.get("primary_metric", self.primary_metric))
        self.secondary_metric = str(state.get("secondary_metric", self.secondary_metric))
        self.mode = str(state.get("mode", self.mode))
        default_init = -float("inf") if self.mode == "max" else float("inf")
        self.best_primary = float(state.get("best_primary", default_init))
        self.best_secondary = float(state.get("best_secondary", default_init))
        self.best_metrics = copy.deepcopy(state.get("best_metrics", {}))
        self.best_epoch = int(state.get("best_epoch", -1))
        self.wait_count = int(state.get("wait_count", 0))
        self.early_stop = bool(state.get("early_stop", False))
        self.is_improved = bool(state.get("is_improved", False))


def _capture_rng_state(
    dataloader_generator: Optional[torch.Generator] = None,
    train_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Thu thập trọn bộ trạng thái ngẫu nhiên (Python, NumPy, PyTorch CPU/CUDA & DataLoader generator)."""
    cuda_states: Optional[List[torch.Tensor]] = None
    if torch.cuda.is_available():
        cuda_states = [s.detach().cpu().clone() for s in torch.cuda.get_rng_state_all()]

    dl_gen_state: Optional[torch.Tensor] = None
    sampler_gen = getattr(getattr(train_loader, "sampler", None), "generator", None) if train_loader is not None else None
    active_gen = sampler_gen if isinstance(sampler_gen, torch.Generator) else dataloader_generator
    if active_gen is not None:
        dl_gen_state = active_gen.get_state().detach().cpu().clone()

    base_seed_gen_state: Optional[torch.Tensor] = None
    if train_loader is not None and getattr(train_loader, "_base_seed_gen_state", None) is not None:
        base_seed_gen_state = train_loader._base_seed_gen_state.detach().cpu().clone()

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_states,
        "dataloader_generator": dl_gen_state,
        "dataloader_base_seed_gen_state": base_seed_gen_state,
    }


def _restore_rng_state(
    rng_state: Optional[Dict[str, Any]],
    dataloader_generator: Optional[torch.Generator] = None,
    train_loader: Optional[DataLoader] = None,
) -> None:
    """
    Khôi phục trạng thái ngẫu nhiên từ checkpoint.
    Lưu ý quan trọng: `torch.load(..., map_location='cuda')` chuyển mọi Tensor sang GPU,
    trong khi `torch.set_rng_state` bắt buộc nhận `uint8 ByteTensor` trên CPU -> ép `.to('cpu', torch.uint8)`.
    Đặc biệt với `persistent_workers=True` (khi `train_loader._iterator is None` lúc mới Resume):
    - `_MultiProcessingDataLoaderIter.__init__` sẽ rút 1 số ngẫu nhiên từ `train_loader.generator` để tạo
      `_base_seed` (`worker_info.seed`) cho các worker. Do đó `train_loader.generator` phải được khôi phục về
      `dataloader_base_seed_gen_state` (trạng thái trước khi spawn worker ở Epoch 0) để `worker_info.seed`
      trùng khớp 100% với tiến trình chạy liền mạch.
    - Nếu `train_loader.sampler` dùng `generator` (ví dụ `RandomSampler`), tách riêng `sampler.generator`
      và khôi phục về `dataloader_generator` (trạng thái cuối epoch trước) để thứ tự mẫu không bị lệch 1 bước.
    """
    if not rng_state:
        return
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch_cpu") is not None:
        cpu_st = rng_state["torch_cpu"].detach().to(device="cpu", dtype=torch.uint8)
        torch.set_rng_state(cpu_st)
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        cuda_list = rng_state["torch_cuda"]
        n_devs = min(len(cuda_list), torch.cuda.device_count())
        for dev_idx in range(n_devs):
            st = cuda_list[dev_idx].detach().to(device="cpu", dtype=torch.uint8)
            torch.cuda.set_rng_state(st, device=dev_idx)

    if dataloader_generator is not None and rng_state.get("dataloader_generator") is not None:
        gen_st = rng_state["dataloader_generator"].detach().to(device="cpu", dtype=torch.uint8)
        base_st_raw = rng_state.get("dataloader_base_seed_gen_state")
        is_unspawned_persistent = (
            train_loader is not None
            and getattr(train_loader, "persistent_workers", False)
            and int(getattr(train_loader, "num_workers", 0)) > 0
            and getattr(train_loader, "_iterator", None) is None
            and base_st_raw is not None
        )
        if is_unspawned_persistent:
            base_st = base_st_raw.detach().to(device="cpu", dtype=torch.uint8)
            dataloader_generator.set_state(base_st)
            train_loader._base_seed_gen_state = base_st.clone()
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and getattr(sampler, "generator", None) is not None:
                sampler_gen = torch.Generator()
                sampler_gen.set_state(gen_st)
                sampler.generator = sampler_gen
        else:
            dataloader_generator.set_state(gen_st)
            if train_loader is not None:
                if base_st_raw is not None:
                    train_loader._base_seed_gen_state = base_st_raw.detach().to(device="cpu", dtype=torch.uint8).clone()
                sampler = getattr(train_loader, "sampler", None)
                if sampler is not None and isinstance(getattr(sampler, "generator", None), torch.Generator):
                    sampler.generator.set_state(gen_st)


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    best_metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    dataloader_generator: Optional[torch.Generator] = None,
    save_rng_state: bool = True,
    train_loader: Optional[DataLoader] = None,
) -> None:
    """
    Lưu đầy đủ trạng thái huấn luyện tự thân (self-contained) ra file .pth (Mục 8.1 #8):
    Ghi đủ model, optimizer, scheduler, scaler (`None` khi bf16), early_stopping, rng_state,
    epoch, metrics, best_metrics và config.
    """
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)

    metrics_dict = dict(metrics) if metrics is not None else {}
    best_dict = dict(best_metrics) if best_metrics is not None else dict(metrics_dict)
    run_id = config.get("run_id") if config is not None else None

    state: Dict[str, Any] = {
        "run_id": run_id,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": (
            scheduler.state_dict() if (scheduler is not None and hasattr(scheduler, "state_dict")) else None
        ),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "early_stopping_state_dict": early_stopping.state_dict() if early_stopping is not None else None,
        "rng_state": (
            _capture_rng_state(dataloader_generator, train_loader=train_loader) if save_rng_state else None
        ),
        "metrics": metrics_dict,
        "best_metrics": best_dict,
        "config": dict(config) if config is not None else {},
    }
    torch.save(state, fp)


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    restore_rng: bool = False,
    dataloader_generator: Optional[torch.Generator] = None,
    train_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """
    Khôi phục trạng thái mô hình, optimizer, scheduler, scaler, EarlyStopping và tùy chọn RNG từ file .pth.
    - Khi nạp để suy luận/đánh giá: để `restore_rng=False` (mặc định).
    - Khi nạp để Resume huấn luyện tiếp: bật `restore_rng=True` và truyền `early_stopping`.
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {fp}")

    map_loc = device if device is not None else torch.device("cpu")
    checkpoint: Dict[str, Any] = torch.load(fp, map_location=map_loc, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
        and hasattr(scheduler, "load_state_dict")
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if early_stopping is not None and checkpoint.get("early_stopping_state_dict") is not None:
        early_stopping.load_state_dict(checkpoint["early_stopping_state_dict"])
    if restore_rng and checkpoint.get("rng_state") is not None:
        _restore_rng_state(
            checkpoint["rng_state"],
            dataloader_generator=dataloader_generator,
            train_loader=train_loader,
        )

    return checkpoint


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    last_checkpoint_path: Optional[Union[str, Path]] = None,
    resume_from: Optional[Union[str, Path]] = None,
    start_epoch: int = 0,
    use_amp: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_norm: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
    log_interval: int = 50,
    restore_best_weights: bool = True,
) -> Dict[str, Any]:
    """
    Điều phối trọn gói quá trình huấn luyện nhiều epoch, đánh giá Val Known,
    cập nhật LR scheduler, dừng sớm, hỗ trợ Resume tự thân (`resume_from`) và khôi phục best weights.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    auto_amp, auto_dtype, auto_scaler = get_optimal_amp_config(device)
    if use_amp is None:
        use_amp = auto_amp
    if amp_dtype is None:
        amp_dtype = auto_dtype
    if scaler is None and use_amp and amp_dtype == torch.float16 and device.type == "cuda":
        scaler = auto_scaler

    if early_stopping is None:
        early_stopping = EarlyStopping(patience=4, min_delta=1e-4)

    dl_gen = getattr(train_loader, "generator", None)
    if not isinstance(dl_gen, torch.Generator):
        dl_gen = None
    elif getattr(train_loader, "_base_seed_gen_state", None) is None and getattr(train_loader, "_iterator", None) is None:
        train_loader._base_seed_gen_state = dl_gen.get_state().detach().cpu().clone()

    epoch_end = start_epoch + epochs
    epoch_begin = start_epoch

    if resume_from is not None:
        resumed_ckpt = load_checkpoint(
            filepath=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            device=device,
            restore_rng=True,
            dataloader_generator=dl_gen,
            train_loader=train_loader,
        )
        epoch_begin = int(resumed_ckpt["epoch"]) + 1
        print(
            f"-> Đã Resume huấn luyện từ '{resume_from}' (tiếp tục tại epoch {epoch_begin}/{epoch_end - 1}, "
            f"best_epoch={early_stopping.best_epoch}, wait={early_stopping.wait_count}/{early_stopping.patience})."
        )

    history: List[Dict[str, Any]] = []
    best_state_in_ram: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(epoch_begin, epoch_end):
        tr_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            max_norm=max_norm,
            log_interval=log_interval,
        )

        val_loss, val_metrics, val_gen_eval = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            compute_per_generator=True,
        )

        # Chặn cứng nếu tập Validation rỗng hoặc thiếu 1 trong 2 lớp (Real / AI)
        if val_metrics.get("num_real", 0) <= 0 or val_metrics.get("num_ai", 0) <= 0:
            raise RuntimeError(
                f"Tập Validation không hợp lệ tại epoch {epoch}: "
                f"Real={val_metrics.get('num_real', 0)}, AI={val_metrics.get('num_ai', 0)}. "
                f"Bắt buộc có đủ cả 2 lớp (Real > 0 và AI > 0)!"
            )
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Val loss bị NaN/Inf tại epoch {epoch}: val_loss={val_loss}")

        # Ghi nhận đúng LR đã thực sự dùng để huấn luyện epoch hiện tại TRƯỚC KHI gọi scheduler.step() (ChatGPT P2 #3.1)
        current_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]

        # Cập nhật scheduler (hỗ trợ cả ReduceLROnPlateau lẫn CosineAnnealingLR / SequentialLR)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched_metric = (
                    val_metrics.get(early_stopping.primary_metric, 0.0)
                    if getattr(scheduler, "mode", "max") == "max"
                    else val_loss
                )
                scheduler.step(sched_metric)
            else:
                scheduler.step()

        is_best = early_stopping.step(val_metrics, epoch=epoch)

        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": tr_stats["train_loss"],
            "train_auc": tr_stats["train_auc"],
            "train_balanced_acc": tr_stats["train_balanced_acc"],
            "val_loss": val_loss,
            "val_auc": float(val_metrics.get("auc", 0.0)),
            "val_balanced_acc": float(val_metrics.get("balanced_acc", 0.0)),
            "val_ap": float(val_metrics.get("ap", 0.0)),
            "val_acc": float(val_metrics.get("acc", 0.0)),
            "lrs": current_lrs,
            "epoch_time": tr_stats["epoch_time"],
            "is_best": is_best,
        }
        if val_gen_eval is not None:
            record["val_per_generator"] = val_gen_eval.get("per_generator", {})
        history.append(record)

        best_tag = " ★ BEST" if is_best else f" (wait {early_stopping.wait_count}/{early_stopping.patience})"
        print(
            f"[Epoch {epoch:02d}] Train Loss={tr_stats['train_loss']:.4f}, AUC={tr_stats['train_auc']:.4f} | "
            f"Val Loss={val_loss:.4f}, AUC={record['val_auc']:.4f}, "
            f"Bal-ACC={record['val_balanced_acc'] * 100:.2f}%, AP={record['val_ap']:.4f}{best_tag}"
        )

        if is_best:
            if checkpoint_path is not None:
                save_checkpoint(
                    filepath=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    early_stopping=early_stopping,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_metrics=early_stopping.best_metrics,
                    config=config,
                    dataloader_generator=dl_gen,
                    train_loader=train_loader,
                )
            elif restore_best_weights:
                best_state_in_ram = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if last_checkpoint_path is not None:
            save_checkpoint(
                filepath=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopping=early_stopping,
                epoch=epoch,
                metrics=val_metrics,
                best_metrics=early_stopping.best_metrics,
                config=config,
                dataloader_generator=dl_gen,
                train_loader=train_loader,
            )

        if early_stopping.early_stop:
            print(
                f"-> EarlyStopping kích hoạt tại epoch {epoch} "
                f"(Best Epoch {early_stopping.best_epoch}: AUC={early_stopping.best_metrics.get('auc', 0.0):.4f}, "
                f"Bal-ACC={early_stopping.best_metrics.get('balanced_acc', 0.0) * 100:.2f}%)."
            )
            break

    # Khôi phục trọng số tốt nhất vào model trước khi trả về
    if restore_best_weights:
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            load_checkpoint(checkpoint_path, model, device=device, restore_rng=False)
        elif best_state_in_ram is not None:
            model.load_state_dict(best_state_in_ram)

    return {
        "history": history,
        "best_epoch": early_stopping.best_epoch,
        "best_metrics": early_stopping.best_metrics,
    }


# ==============================================================================
# CỔNG KIỂM TRA KỸ THUẬT 10 PHÚT (SANITY CHECK GATE - MỤC 6 & MỤC 8.1 #1, #2, #10)
# ==============================================================================
def overfit_single_batch_check(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    lr: float = 3e-3,
    steps: int = 100,
    loss_reduction_threshold: float = 0.90,
    use_amp: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
    restore_weights: bool = True,
) -> bool:
    """
    Cổng kiểm tra 1 (Mục 8.1 #1): Ép overfit trên đúng 1 batch (16 Real : 16 AI).
    - Điều kiện đạt chuẩn: Loss giảm ≥ 90% so với bước đầu VÀ Accuracy trên chính batch đạt 100%
      trong tối đa `steps=100` bước.
    - Đánh giá `final_loss` & `final_acc` trên trọng số ĐÃ CẬP NHẬT sau `optimizer.step()`.
    - Tự động sao lưu và khôi phục nguyên trạng trọng số gốc (`restore_weights=True`).
    """
    was_training = model.training
    backup_state = copy.deepcopy(model.state_dict()) if restore_weights else None

    auto_amp, auto_dtype, auto_scaler = get_optimal_amp_config(device)
    amp_enabled = (auto_amp if use_amp is None else use_amp) and device.type == "cuda"
    dtype = auto_dtype if amp_dtype is None else amp_dtype
    scaler = auto_scaler if (amp_enabled and dtype == torch.float16) else None

    def _eval_post_update(m: nn.Module, inp: torch.Tensor, target: torch.Tensor, crit: nn.Module) -> Tuple[float, float]:
        m.eval()
        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=dtype):
                    post_logits = m(inp)
                    post_loss = crit(post_logits.float(), target)
            else:
                post_logits = m(inp)
                post_loss = crit(post_logits.float(), target)
            preds = (torch.sigmoid(post_logits.float()) >= 0.5).long()
            acc = float((preds == target.long()).float().mean().item())
            return float(post_loss.item()), acc

    try:
        input_key = _get_model_input_key(model)
        x = batch[input_key].to(device)
        y = batch["label_a"].to(device).float()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Không có tham số requires_grad=True để chạy overfit_single_batch_check!")

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)
        criterion = nn.BCEWithLogitsLoss()

        # Đo initial_loss trước bước cập nhật đầu tiên
        init_loss_val, _ = _eval_post_update(model, x, y, criterion)
        if not np.isfinite(init_loss_val):
            raise FloatingPointError(f"Initial loss bị NaN/Inf trong overfit_single_batch_check: {init_loss_val}")
        initial_loss = max(init_loss_val, 1e-12)

        final_loss = initial_loss
        final_acc = 0.0
        steps_used = 0

        for step in range(1, steps + 1):
            steps_used = step
            model.train()
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits = model(x)
                    loss = criterion(logits.float(), y)
                if not torch.isfinite(loss).all():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"Overfit loss bị NaN/Inf tại step {step}: {loss.item()}")
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    _check_and_clip_gradients(trainable_params, max_norm=0.0, epoch=0, batch_idx=step)
                    optimizer.step()
            else:
                logits = model(x)
                loss = criterion(logits.float(), y)
                if not torch.isfinite(loss).all():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"Overfit loss bị NaN/Inf tại step {step}: {loss.item()}")
                loss.backward()
                _check_and_clip_gradients(trainable_params, max_norm=0.0, epoch=0, batch_idx=step)
                optimizer.step()

            # Đo lại loss và accuracy SAU khi optimizer.step() đã cập nhật trọng số
            final_loss, final_acc = _eval_post_update(model, x, y, criterion)
            loss_drop = (initial_loss - final_loss) / initial_loss
            if loss_drop >= loss_reduction_threshold and final_acc == 1.0:
                break

        loss_drop = (initial_loss - final_loss) / initial_loss
        passed = bool(loss_drop >= loss_reduction_threshold and final_acc == 1.0)

        print(
            f"  + Overfit 1 batch ({steps_used}/{steps} steps): Loss {initial_loss:.4f} -> {final_loss:.4f} "
            f"(Giảm {loss_drop * 100:.1f}% ≥ {loss_reduction_threshold * 100:.0f}%), "
            f"Post-step ACC = {final_acc * 100:.1f}% => [{'PASS' if passed else 'FAIL'}]"
        )
        return passed
    finally:
        if backup_state is not None:
            model.load_state_dict(backup_state)
        model.train(was_training)


def checkpoint_consistency_check(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    tmp_path: Optional[Union[str, Path]] = None,
    use_amp: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> bool:
    """
    Cổng kiểm tra 3 (Mục 8.1 #2 & #8 + Self-Contained Resume Training Equivalence):
    1. Chế độ `eval()`: Tính `logits_before`, lưu checkpoint, làm nhiễu trọng số, nạp lại và
       kiểm tra `torch.allclose(logits_before, logits_after)`.
    2. Chế độ Resume tự thân (Self-contained, KHÔNG khôi phục RNG thủ công từ bên ngoài):
       Lưu trọn gói Model + Optimizer + Scheduler + Scaler + EarlyStopping + RNG vào file `.pth`,
       sau đó xóa trắng trọng số, làm xáo trộn RNG toàn cục, khởi tạo lại từ đầu các đối tượng mới
       và nạp lại từ `.pth` (`restore_rng=True`) -> chứng minh bước train tiếp theo khớp tuyệt đối.
    """
    was_training = model.training
    backup_state = copy.deepcopy(model.state_dict())
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    auto_amp, auto_dtype, auto_scaler = get_optimal_amp_config(device)
    amp_enabled = (auto_amp if use_amp is None else use_amp) and device.type == "cuda"
    dtype = auto_dtype if amp_dtype is None else amp_dtype

    created_temp = False
    if tmp_path is None:
        fd, raw_tmp = tempfile.mkstemp(suffix="_ckpt_check.pth")
        import os
        os.close(fd)
        p = Path(raw_tmp)
        created_temp = True
    else:
        p = Path(tmp_path)
        created_temp = True

    def _run_single_train_step(
        m: nn.Module,
        inp: torch.Tensor,
        target: torch.Tensor,
        opt: torch.optim.Optimizer,
        sched: Any,
        sc: Optional[torch.amp.GradScaler],
        crit: nn.Module,
    ) -> None:
        opt.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = m(inp)
                loss_val = crit(out.float(), target)
            if sc is not None:
                sc.scale(loss_val).backward()
                sc.step(opt)
                sc.update()
            else:
                loss_val.backward()
                opt.step()
        else:
            out = m(inp)
            loss_val = crit(out.float(), target)
            loss_val.backward()
            opt.step()
        if sched is not None:
            sched.step()

    def _eval_logits(m: nn.Module, inp: torch.Tensor) -> torch.Tensor:
        m.eval()
        with torch.inference_mode():
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=dtype):
                    return m(inp).float().detach().cpu()
            return m(inp).float().detach().cpu()

    try:
        input_key = _get_model_input_key(model)
        x = batch[input_key].to(device)
        y = batch["label_a"].to(device).float()
        criterion = nn.BCEWithLogitsLoss()

        # --- PHẦN A: Kiểm tra khôi phục trọng số suy luận ở eval() ---
        logits_before = _eval_logits(model, x)

        dummy_opt = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=1e-4)
        dummy_sched = torch.optim.lr_scheduler.CosineAnnealingLR(dummy_opt, T_max=10)
        save_checkpoint(
            filepath=p,
            model=model,
            optimizer=dummy_opt,
            scheduler=dummy_sched,
            scaler=auto_scaler,
            epoch=1,
            metrics={"auc": 0.95, "balanced_acc": 0.90},
            config={"branch": input_key},
        )

        # Làm nhiễu mạnh trọng số mô hình để đảm bảo load_checkpoint thực sự khôi phục đúng
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.25)

        loaded = load_checkpoint(p, model, optimizer=dummy_opt, scheduler=dummy_sched, scaler=auto_scaler, device=device)
        assert (
            "scaler_state_dict" in loaded
            and "early_stopping_state_dict" in loaded
            and "rng_state" in loaded
            and "best_metrics" in loaded
            and "config" in loaded
        )

        logits_after = _eval_logits(model, x)
        max_diff = float((logits_before - logits_after).abs().max().item())
        is_eval_identical = bool(torch.allclose(logits_before, logits_after, atol=1e-5, rtol=1e-5))

        print(
            f"  + Checkpoint Save -> Mutate -> Load -> eval() allclose: "
            f"Sai số tối đa = {max_diff:.2e} => [{'PASS' if is_eval_identical else 'FAIL'}]"
        )

        # --- PHẦN B: Kiểm tra Resume tự thân 100% từ file .pth (Model + Opt + Sched + Scaler + EarlyStopping + RNG) ---
        model.load_state_dict(backup_state)
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        is_resume_identical = True

        if trainable_params:
            model.train()
            opt_a = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
            sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=10)
            es_a = EarlyStopping(patience=4, min_delta=1e-4)
            es_a.step({"auc": 0.9620, "balanced_acc": 0.9150}, epoch=1)
            es_a.step({"auc": 0.9580, "balanced_acc": 0.9100}, epoch=2)  # wait_count = 1
            scaler_a = (
                torch.amp.GradScaler("cuda", enabled=True)
                if (amp_enabled and dtype == torch.float16)
                else None
            )

            # Chạy 2 bước mồi để sinh động lượng AdamW & đẩy scheduler sang epoch=2
            for _ in range(2):
                _run_single_train_step(model, x, y, opt_a, sched_a, scaler_a, criterion)

            # Lưu checkpoint tự thân (tự đóng gói cả EarlyStopping & RNG ngay tại thời điểm cuối epoch 2)
            save_checkpoint(
                filepath=p,
                model=model,
                optimizer=opt_a,
                scheduler=sched_a,
                scaler=scaler_a,
                early_stopping=es_a,
                epoch=2,
                metrics={"auc": 0.9580, "balanced_acc": 0.9100},
                best_metrics=es_a.best_metrics,
                config={"branch": input_key, "resume_test": True},
                save_rng_state=True,
            )

            # Nhánh 1: Tiếp tục train liên tục bước thứ 3 không gián đoạn (có cả Dropout ngẫu nhiên!)
            model.train()
            _run_single_train_step(model, x, y, opt_a, sched_a, scaler_a, criterion)
            es_a.step({"auc": 0.9600, "balanced_acc": 0.9120}, epoch=3)  # Vẫn thấp hơn 0.9620 -> wait_count = 2
            lr_uninterrupted = list(sched_a.get_last_lr())
            logits_uninterrupted = _eval_logits(model, x)

            # Phá hủy trạng thái hiện tại: xóa trọng số, xáo trộn mạnh RNG toàn cục, tạo mới Opt/Sched/EarlyStopping
            with torch.no_grad():
                for param in model.parameters():
                    param.zero_()
            _ = torch.randn(777, device=device)
            _ = np.random.uniform(size=333)
            _ = random.random()

            opt_b = torch.optim.AdamW(trainable_params, lr=9e-2, weight_decay=0.5)
            sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=99)
            es_b = EarlyStopping(patience=4, min_delta=1e-4)  # Mới tạo -> best_primary = -inf, wait_count = 0
            scaler_b = (
                torch.amp.GradScaler("cuda", init_scale=128.0, enabled=True)
                if (amp_enabled and dtype == torch.float16)
                else None
            )

            # Nhánh 2: Nạp lại hoàn toàn từ file .pth (restore_rng=True, KHÔNG can thiệp RNG thủ công từ ngoài!)
            resumed_ckpt = load_checkpoint(
                p,
                model,
                optimizer=opt_b,
                scheduler=sched_b,
                scaler=scaler_b,
                early_stopping=es_b,
                device=device,
                restore_rng=True,
            )
            epoch_ok = int(resumed_ckpt.get("epoch", -1)) == 2
            es_before_step_ok = (es_b.best_primary == 0.9620 and es_b.wait_count == 1 and es_b.best_epoch == 1)

            model.train()
            _run_single_train_step(model, x, y, opt_b, sched_b, scaler_b, criterion)
            es_improved_step3 = es_b.step({"auc": 0.9600, "balanced_acc": 0.9120}, epoch=3)
            es_after_step_ok = (
                es_improved_step3 is False
                and es_b.best_primary == es_a.best_primary
                and es_b.wait_count == es_a.wait_count == 2
            )

            lr_resumed = list(sched_b.get_last_lr())
            logits_resumed = _eval_logits(model, x)

            lr_ok = bool(np.allclose(lr_uninterrupted, lr_resumed, atol=1e-12, rtol=1e-9))
            resume_diff = float((logits_uninterrupted - logits_resumed).abs().max().item())
            weights_ok = bool(torch.allclose(logits_uninterrupted, logits_resumed, atol=1e-5, rtol=1e-5))

            # Kiểm tra thêm khả năng lưu/khôi phục GradScaler (dành cho máy fp16 như Colab T4)
            scaler_roundtrip_ok = True
            if device.type == "cuda":
                test_scaler_in = torch.amp.GradScaler("cuda", init_scale=4096.0, enabled=True)
                save_checkpoint(filepath=p, model=model, optimizer=opt_b, scheduler=sched_b, scaler=test_scaler_in, epoch=2)
                test_scaler_out = torch.amp.GradScaler("cuda", init_scale=65536.0, enabled=True)
                load_checkpoint(p, model, scaler=test_scaler_out, device=device)
                scaler_roundtrip_ok = (float(test_scaler_out.get_scale()) == 4096.0)

            is_resume_identical = bool(
                epoch_ok and es_before_step_ok and es_after_step_ok and lr_ok and weights_ok and scaler_roundtrip_ok
            )
            print(
                f"  + Checkpoint Self-Contained Resume (epoch={resumed_ckpt.get('epoch')}, Opt/Sched/Scaler/EarlyStop/RNG): "
                f"LR khớp={lr_ok}, EarlyStop khớp={es_after_step_ok} (wait={es_b.wait_count}), "
                f"Sai số bước tiếp theo = {resume_diff:.2e} => [{'PASS' if is_resume_identical else 'FAIL'}]"
            )

        return bool(is_eval_identical and is_resume_identical)
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = prev_cudnn_det
            torch.backends.cudnn.benchmark = prev_cudnn_bench
        if created_temp and p.exists():
            p.unlink()
        model.load_state_dict(backup_state)
        model.train(was_training)


def sanity_check_gate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    overfit_lr: float = 3e-3,
    overfit_steps: int = 100,
    smoke_train_batches: int = 50,
    smoke_val_batches: Optional[int] = 30,
    smoke_lr: float = 1e-4,
) -> bool:
    """
    Cổng kiểm tra tổng hợp 10 phút trước khi huấn luyện dài hạn (Mục 6 sơ đồ luồng):
    1. Overfit 1 batch thật: giảm ≥ 90% loss & đạt 100% ACC sau cập nhật cuối trong ≤ 100 bước.
    2. Chạy thử 50 batch `train_one_epoch()` + `evaluate()` trên Val: bắt buộc tập Train/Val không rỗng,
       Val có đủ cả 2 lớp (`Real > 0` và `AI > 0`), loss hữu hạn và tính toán toàn bộ metrics không lỗi.
    3. Kiểm tra lưu/nạp checkpoint tự thân (`eval()` allclose + Resume trọn gói Opt/Sched/Scaler/EarlyStop/RNG).
    Sau khi hoàn tất, khôi phục 100% trọng số ban đầu để sẵn sàng vào phiên train chính thức.
    """
    print("=" * 80)
    print(f"BẮT ĐẦU CỔNG KIỂM TRA KỸ THUẬT (SANITY CHECK GATE) — nhánh '{_get_model_input_key(model)}'")
    print("=" * 80)

    was_training = model.training
    pristine_state = copy.deepcopy(model.state_dict())
    use_amp, amp_dtype, scaler = get_optimal_amp_config(device)

    try:
        _sync_loader_epoch(train_loader, 0)
        first_batch = next(iter(train_loader))

        # Bước 1: Overfit 1 batch
        print("[CỔNG 1/3] Kiểm tra khả năng hội tụ trên 1 batch đơn lẻ:")
        ok_overfit = overfit_single_batch_check(
            model=model,
            batch=first_batch,
            device=device,
            lr=overfit_lr,
            steps=overfit_steps,
            loss_reduction_threshold=0.90,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            restore_weights=True,
        )
        if not ok_overfit:
            return False

        # Bước 2: Chạy thử 50 batch train + đánh giá Val (ràng buộc bắt buộc đủ 2 lớp Real > 0 & AI > 0)
        print(f"[CỔNG 2/3] Chạy thử vòng lặp thật ({smoke_train_batches} batch Train + Val):")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=smoke_lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        tr_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=0,
            scaler=scaler,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            max_norm=1.0,
            max_batches=smoke_train_batches,
            log_interval=max(1, smoke_train_batches // 2),
        )
        val_loss, val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            max_batches=smoke_val_batches,
            compute_per_generator=True,
        )
        ok_smoke = (
            tr_stats.get("num_samples", 0) > 0
            and np.isfinite(tr_stats["train_loss"])
            and np.isfinite(val_loss)
            and val_metrics.get("num_real", 0) > 0
            and val_metrics.get("num_ai", 0) > 0
            and 0.0 <= float(val_metrics.get("auc", -1.0)) <= 1.0
        )
        print(
            f"  + Smoke Train/Val: Train Loss={tr_stats['train_loss']:.4f} | "
            f"Val Loss={val_loss:.4f}, Val AUC={val_metrics.get('auc', 0.0):.4f}, "
            f"Bal-ACC={val_metrics.get('balanced_acc', 0.0) * 100:.2f}% => "
            f"[{'PASS' if ok_smoke else 'FAIL'}]"
        )
        print(
            f"  + Phân bố & Xác suất Val: Real={val_metrics.get('num_real', 0):,}, AI={val_metrics.get('num_ai', 0):,} | "
            f"P(AI|Real)={val_metrics.get('mean_prob_real', 0.0):.4f}, "
            f"P(AI|AI)={val_metrics.get('mean_prob_ai', 0.0):.4f}"
        )
        print(
            f"  + Ma trận nhầm lẫn Val  : TN={val_metrics.get('tn', 0):,}, FP={val_metrics.get('fp', 0):,}, "
            f"FN={val_metrics.get('fn', 0):,}, TP={val_metrics.get('tp', 0):,}"
        )
        if not ok_smoke:
            return False

        # Bước 3: Lưu/nạp checkpoint trên trạng thái vừa train
        print("[CỔNG 3/3] Kiểm tra lưu và nạp lại Checkpoint:")
        ok_ckpt = checkpoint_consistency_check(
            model=model,
            batch=first_batch,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        if not ok_ckpt:
            return False

        print("=" * 80)
        print("VƯỢT QUA TOÀN BỘ CỔNG KIỂM TRA 3/3 — ĐỦ ĐIỀU KIỆN HUẤN LUYỆN CHÍNH THỨC!")
        print("=" * 80)
        return True
    finally:
        model.load_state_dict(pristine_state)
        model.train(was_training)


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP CHO ENGINE (engine.py Unit Test)
# ==============================================================================
def run_unit_tests() -> None:
    print("=" * 80)
    print("KIỂM THỬ BỘ MÁY HUẤN LUYỆN: code/system/train/engine.py")
    print("=" * 80)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  + Thiết bị thử nghiệm: {device.type.upper()} ({dev_name})")

    use_amp, amp_dtype, scaler = get_optimal_amp_config(device)
    print(f"  + Tự động chọn AMP: use_amp={use_amp}, dtype={amp_dtype}, scaler={scaler is not None}")

    from base import BranchBase

    class ToyBranch(BranchBase):
        input_key = "spatial"
        feature_dim = 64

        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((8, 8))
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 8 * 8, self.feature_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.15),
                nn.Linear(self.feature_dim, 1),
            )

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(self.pool(x))

    # Dataset giả lập trả về đúng cấu trúc dictionary của DefactifyDataset (tín hiệu tách lớp rõ ràng)
    class MockDictDataset(torch.utils.data.Dataset):
        def __init__(self, n_real: int, n_ai: int) -> None:
            self.epochs_seen: List[int] = []
            real_x = torch.randn(n_real, 3, 32, 32) - 0.6
            ai_x = torch.randn(n_ai, 3, 32, 32) + 0.6
            self.x = torch.cat([real_x, ai_x], dim=0)
            self.y = torch.tensor([0.0] * n_real + [1.0] * n_ai, dtype=torch.float32)
            self.models = ["MS_COCO"] * n_real + ["Stable_Diffusion_2.1"] * n_ai

        def set_epoch(self, epoch: int) -> None:
            self.epochs_seen.append(epoch)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return {
                "spatial": self.x[idx],
                "label_a": self.y[idx],
                "model_name": self.models[idx],
            }

    train_ds = MockDictDataset(64, 64)
    val_ds = MockDictDataset(16, 48)  # Giả lập tỉ lệ lệch 1:3 giống tập Val thực tế
    gen_train = torch.Generator().manual_seed(123)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=gen_train)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = ToyBranch().to(device)
    initial_weight_snapshot = copy.deepcopy(model.state_dict())

    # 1. Kiểm tra toàn diện Cổng kiểm tra 3 bước (Overfit post-step + Smoke Train/Val + Self-Contained Checkpoint Resume)
    print("\n[KIỂM TRA 1] Sanity Check Gate 3 bước (Post-step Overfit + bfloat16 Val + Self-Contained Resume):")
    gate_ok = sanity_check_gate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        overfit_lr=3e-3,
        overfit_steps=100,
        smoke_train_batches=4,
        smoke_val_batches=2,
        smoke_lr=3e-3,
    )
    assert gate_ok, "Sanity Check Gate thất bại!"
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weight_snapshot[k]), f"Trọng số '{k}' không được khôi phục nguyên trạng sau Sanity Gate!"
    assert 0 in train_ds.epochs_seen, "train_dataset.set_epoch() chưa được gọi!"
    print("  + Trọng số gốc được bảo toàn 100% sau Sanity Check Gate => [PASS]")

    # 2. Kiểm tra `fit()` chạy liên tục 4 epochs vs gián đoạn tại epoch 2 rồi `resume_from=last_ckpt` (tự khôi phục RNG & EarlyStopping)
    print("\n[KIỂM TRA 2] Đối chứng `fit()` liên tục 4 epochs vs dừng giữa chừng & Resume tự thân từ `last_ckpt`:")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Nhánh chuẩn: Chạy liên tục 4 epochs từ đầu đến cuối
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        g_full = torch.Generator().manual_seed(555)
        loader_full = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g_full)
        model_full = ToyBranch().to(device)
        model_full.load_state_dict(initial_weight_snapshot)
        opt_full = torch.optim.AdamW(model_full.parameters(), lr=4e-3, weight_decay=1e-4)
        sched_full = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=4)
        es_full = EarlyStopping(patience=4, min_delta=1e-4)
        best_full_path = Path(tmpdir) / "best_full.pth"

        res_full = fit(
            model=model_full,
            train_loader=loader_full,
            val_loader=val_loader,
            optimizer=opt_full,
            device=device,
            epochs=4,
            scheduler=sched_full,
            early_stopping=es_full,
            checkpoint_path=best_full_path,
            log_interval=0,
            restore_best_weights=False,
        )

        # Nhánh gián đoạn: Chạy 2 epochs (0, 1) -> lưu last_ckpt -> xóa sạch & xáo trộn RNG -> Resume 2 epochs cuối (2, 3)
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        g_part1 = torch.Generator().manual_seed(555)
        loader_part1 = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g_part1)
        model_part = ToyBranch().to(device)
        model_part.load_state_dict(initial_weight_snapshot)
        opt_part1 = torch.optim.AdamW(model_part.parameters(), lr=4e-3, weight_decay=1e-4)
        sched_part1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_part1, T_max=4)
        es_part1 = EarlyStopping(patience=4, min_delta=1e-4)
        best_part_path = Path(tmpdir) / "best_part.pth"
        last_part_path = Path(tmpdir) / "last_part.pth"

        _ = fit(
            model=model_part,
            train_loader=loader_part1,
            val_loader=val_loader,
            optimizer=opt_part1,
            device=device,
            epochs=2,
            scheduler=sched_part1,
            early_stopping=es_part1,
            checkpoint_path=best_part_path,
            last_checkpoint_path=last_part_path,
            log_interval=0,
            restore_best_weights=False,
        )

        # Mô phỏng tắt máy & khởi động lại tiến trình mới hoàn toàn
        _ = torch.randn(1024, device=device)
        g_part2 = torch.Generator().manual_seed(11111)  # Seed sai lệch cố ý, sẽ được load_checkpoint ghi đè từ last_ckpt
        loader_part2 = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g_part2)
        model_resumed = ToyBranch().to(device)
        opt_part2 = torch.optim.AdamW(model_resumed.parameters(), lr=9e-2, weight_decay=0.9)
        sched_part2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_part2, T_max=99)
        es_part2 = EarlyStopping(patience=4, min_delta=1e-4)

        res_resumed = fit(
            model=model_resumed,
            train_loader=loader_part2,
            val_loader=val_loader,
            optimizer=opt_part2,
            device=device,
            epochs=4,
            scheduler=sched_part2,
            early_stopping=es_part2,
            checkpoint_path=best_part_path,
            last_checkpoint_path=last_part_path,
            resume_from=last_part_path,
            log_interval=0,
            restore_best_weights=False,
        )

        assert es_part2.best_epoch == es_full.best_epoch, f"Lệch best_epoch: {es_part2.best_epoch} vs {es_full.best_epoch}"
        assert es_part2.wait_count == es_full.wait_count, f"Lệch wait_count: {es_part2.wait_count} vs {es_full.wait_count}"
        assert abs(es_part2.best_primary - es_full.best_primary) < 1e-6
        assert len(res_resumed["history"]) == 2 and res_full["best_metrics"]["auc"] > 0.95

        model_full.eval()
        model_resumed.eval()
        sample_x = next(iter(val_loader))["spatial"].to(device)
        with torch.inference_mode():
            diff_fit_resume = float((model_full(sample_x).float() - model_resumed(sample_x).float()).abs().max().item())
        assert diff_fit_resume < 1e-5, f"Trọng số sau Resume lệch so với train liên tục: {diff_fit_resume}"
        print(
            f"  + fit() Resume 2+2 epochs khớp 100% với chạy liên tục 4 epochs "
            f"(sai số logits={diff_fit_resume:.2e}, wait={es_part2.wait_count}, best_epoch={es_part2.best_epoch}) => [PASS]"
        )

    # 3. Kiểm tra EarlyStopping (phân xử hòa bằng Balanced ACC & chống trôi mốc AUC)
    print("\n[KIỂM TRA 3] EarlyStopping logic (AUC chính, Balanced ACC phụ khi hòa, chống trôi mốc):")
    es = EarlyStopping(patience=3, min_delta=1e-4)
    assert es.step({"auc": 0.99900, "balanced_acc": 0.9500}, epoch=0) is True
    assert es.step({"auc": 0.99893, "balanced_acc": 0.9650}, epoch=1) is True
    assert es.best_primary == 0.99900, f"Bị trôi mốc AUC: {es.best_primary}"
    assert es.step({"auc": 0.99885, "balanced_acc": 0.9800}, epoch=2) is False
    es.step({"auc": 0.99800, "balanced_acc": 0.9400}, epoch=3)
    es.step({"auc": 0.99800, "balanced_acc": 0.9400}, epoch=4)
    assert es.early_stop is True
    print("  + Phân xử hòa chuẩn xác, chống trôi mốc AUC thành công và dừng sớm đúng hạn => [PASS]")

    # 4. Kiểm tra P0 Numerical Stability: Chặn NaN/Inf TRƯỚC backward()/optimizer.step(), bảo toàn 100% trọng số
    print("\n[KIỂM TRA 4] Bảo vệ số học P0 (chặn Loss/Gradient NaN & Inf TRƯỚC `optimizer.step()`):")
    model_nan = ToyBranch().to(device)
    clean_weights = copy.deepcopy(model_nan.state_dict())
    opt_nan = torch.optim.AdamW(model_nan.parameters(), lr=1e-2)
    crit_nan = nn.BCEWithLogitsLoss()

    # 4a. Tiêm NaN vào đầu vào -> Loss bị NaN -> Phải bắt trước backward()/step(), trọng số không đổi
    bad_ds = MockDictDataset(8, 8)
    bad_ds.x[0, 0, 0, 0] = float("nan")
    bad_loader = DataLoader(bad_ds, batch_size=16, shuffle=False)
    try:
        train_one_epoch(
            model=model_nan, loader=bad_loader, optimizer=opt_nan, criterion=crit_nan,
            device=device, epoch=0, scaler=None, amp_dtype=amp_dtype, use_amp=use_amp, log_interval=0,
        )
        raise AssertionError("Phải quăng FloatingPointError khi Loss là NaN!")
    except FloatingPointError:
        pass
    for k, v in model_nan.state_dict().items():
        assert torch.equal(v, clean_weights[k]), f"Trọng số '{k}' bị hỏng khi Loss=NaN!"

    # 4b. Tiêm Inf vào gradient qua backward hook -> Phải bắt trước optimizer.step(), trọng số không đổi
    good_loader = DataLoader(MockDictDataset(8, 8), batch_size=16, shuffle=False)
    hook_handle = model_nan.classifier[-1].weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        train_one_epoch(
            model=model_nan, loader=good_loader, optimizer=opt_nan, criterion=crit_nan,
            device=device, epoch=0, scaler=None, amp_dtype=amp_dtype, use_amp=use_amp, log_interval=0,
        )
        raise AssertionError("Phải quăng FloatingPointError khi Gradient là Inf!")
    except FloatingPointError:
        pass
    finally:
        hook_handle.remove()
    for k, v in model_nan.state_dict().items():
        assert torch.equal(v, clean_weights[k]), f"Trọng số '{k}' bị hỏng khi Gradient=Inf!"
    print("  + Đã chặn đứng cả Loss=NaN lẫn Gradient=Inf trước `optimizer.step()`, trọng số nguyên vẹn 100% => [PASS]")

    # 5. Kiểm tra P0 Validation Gate: Từ chối tập Val chỉ có 1 lớp hoặc rỗng
    print("\n[KIỂM TRA 5] Cổng bảo vệ Validation P0 (chặn tập Val đơn lớp / rỗng):")
    single_class_val = DataLoader(MockDictDataset(0, 32), batch_size=32, shuffle=False)  # 0 Real, 32 AI
    gate_single_ok = sanity_check_gate(
        model=model_nan,
        train_loader=good_loader,
        val_loader=single_class_val,
        device=device,
        smoke_train_batches=1,
        smoke_val_batches=1,
    )
    assert gate_single_ok is False, "Sanity Check Gate KHÔNG ĐƯỢC PASS khi tập Val thiếu lớp Real!"
    try:
        fit(
            model=model_nan, train_loader=good_loader, val_loader=single_class_val,
            optimizer=opt_nan, device=device, epochs=1, log_interval=0,
        )
        raise AssertionError("fit() phải báo lỗi RuntimeError khi tập Val chỉ có 1 lớp!")
    except RuntimeError:
        pass
    print("  + Đã chặn thành công tập Validation thiếu lớp Real ở cả `sanity_check_gate()` và `fit()` => [PASS]")

    print("\n" + "=" * 80)
    print("TẤT CẢ UNIT TESTS CỦA engine.py ĐÃ VƯỢT QUA 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
