# -*- coding: utf-8 -*-
"""
code/system/test_spatial_end_to_end.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Script kiểm tra tích hợp toàn diện (Full Integration Sanity Check) cho nhánh Spatial (EfficientNet-B0):
1. Kết nối Dataset: Nạp DefactifyDataset thật với active_branches=('spatial',) và num_workers=4.
2. Nạp Mô hình: Khởi tạo SpatialBranch với trọng số tiền huấn luyện ImageNet-1K (IMAGENET1K_V1).
3. Động cơ huấn luyện: Chạy Cổng kiểm tra Sanity Check Gate 3 bước của engine.py trên RTX 3050 (bfloat16).
4. Huấn luyện 2 Pha: Kiểm tra chuyển đổi Pha A (Warmup Head) sang Pha B (Full Fine-tuning) với Differential LR.
5. Thước đo đánh giá: Chạy evaluate() và evaluate_by_generator() kiểm tra tính toàn vẹn số liệu.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
BRANCH_DIR = CURRENT_DIR / "branch"
TRAIN_DIR = CURRENT_DIR / "train"
DATA_DIR = CURRENT_DIR / "prepare_data"

for p in (str(CURRENT_DIR), str(BRANCH_DIR), str(TRAIN_DIR), str(DATA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from EfficientNet import SpatialBranch
from dataset import build_dataloaders
from engine import get_optimal_amp_config, sanity_check_gate, train_one_epoch, evaluate
from metrics import format_metrics_summary, format_generator_report


def run_integration_verification():
    print("=" * 85)
    print("   KIỂM TRA LIÊN KẾT HỆ THỐNG: EfficientNet.py <-> dataset.py <-> engine.py <-> metrics.py")
    print("=" * 85)
    t_start = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  + Phần cứng thực thi     : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    use_amp, amp_dtype, scaler = get_optimal_amp_config(device)
    print(f"  + Cấu hình AMP tự động   : use_amp={use_amp}, dtype={amp_dtype}, scaler={scaler is not None}")

    # -------------------------------------------------------------------------
    # BƯỚC 1: Khởi tạo SpatialBranch với trọng số IMAGENET1K_V1
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 1/5] Khởi tạo SpatialBranch (EfficientNet-B0 IMAGENET1K_V1):")
    model = SpatialBranch(pretrained=True, dropout=0.2, trainable_stages="all").to(device)
    assert model.input_key == "spatial", "input_key không phải 'spatial'!"
    assert model.feature_dim == 1280, "feature_dim không phải 1280!"

    summary = model.stage_summary()
    print(f"  + Trọng số tiền huấn luyện: Đã nạp thành công IMAGENET1K_V1!")
    print(f"  + Tổng tham số mô hình   : {summary['total_params']:,} (Backbone=4,007,548 | Head=1,281)")
    print(f"  + Số stage hoạt động     : {summary['configured_stages']}")

    # -------------------------------------------------------------------------
    # BƯỚC 2: Kết nối DataLoader từ dataset.py (chỉ bật active_branches=('spatial',))
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 2/5] Nạp DataLoader thật từ DefactifyDataset (active_branches=('spatial',), num_workers=4):")
    train_loader, val_loader, _, _ = build_dataloaders(
        batch_size=32,
        num_workers=4,
        active_branches=("spatial",),
        persistent_workers=False,
    )
    first_train_batch = next(iter(train_loader))
    assert model.input_key in first_train_batch, f"Batch thiếu khóa {model.input_key}!"
    assert "label_a" in first_train_batch, "Batch thiếu khóa 'label_a'!"
    assert "model_name" in first_train_batch, "Batch thiếu khóa 'model_name'!"

    x_sample = first_train_batch[model.input_key].to(device)
    y_sample = first_train_batch["label_a"].to(device).float()
    print(f"  + Input tensor batch['spatial'] : shape={tuple(x_sample.shape)}, dtype={x_sample.dtype}")
    print(f"  + Target tensor batch['label_a'] : shape={tuple(y_sample.shape)}, Real={(y_sample==0).sum().item()}, AI={(y_sample==1).sum().item()}")

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                f_sample = model.forward_features(x_sample)
                l_sample = model(x_sample)
        else:
            f_sample = model.forward_features(x_sample)
            l_sample = model(x_sample)

    assert f_sample.shape == (32, 1280), f"Lỗi shape features: {f_sample.shape}"
    assert l_sample.shape == (32,), f"Lỗi shape logits: {l_sample.shape}"
    print(f"  + forward_features(x) trả về   : shape={tuple(f_sample.shape)} (chuẩn bị ghép Fusion MLP)")
    print(f"  + forward(x) trả về            : shape={tuple(l_sample.shape)} (chuẩn bị tính BCE loss)")

    # -------------------------------------------------------------------------
    # BƯỚC 3: Chạy Cổng kiểm tra Sanity Check Gate 10 phút của engine.py
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 3/5] Chạy Cổng kiểm tra kỹ thuật (Sanity Check Gate) qua engine.py:")
    gate_passed = sanity_check_gate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        overfit_steps=100,
        smoke_train_batches=10,
        smoke_val_batches=10,
        smoke_lr=1e-3,
    )
    assert gate_passed, "Sanity Check Gate của SpatialBranch thất bại!"

    # -------------------------------------------------------------------------
    # BƯỚC 4: Kiểm tra cơ chế Huấn luyện 2 Pha (Pha A: Warmup Head -> Pha B: Fine-tune)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 4/5] Kiểm tra chuyển đổi 2 Pha huấn luyện (Pha A -> Pha B):")
    # Pha A: Khóa backbone, chỉ học head
    model.freeze_backbone(True)
    model.train()
    summary_a = model.stage_summary()
    param_groups_a = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3)
    assert summary_a["trainable_params"] == 1281
    assert len(param_groups_a) == 1 and param_groups_a[0]["group_name"] == "head"
    opt_a = torch.optim.AdamW(param_groups_a)
    crit = nn.BCEWithLogitsLoss()

    print(f"  + Pha A (Warmup Head): Trainable={summary_a['trainable_params']:,} params, BN ở eval={summary_a['backbone_bn_in_eval']}/{summary_a['backbone_bn_total']}")
    stats_a = train_one_epoch(
        model=model, loader=train_loader, optimizer=opt_a, criterion=crit,
        device=device, epoch=0, scaler=scaler, amp_dtype=amp_dtype, use_amp=use_amp, max_batches=2, log_interval=0
    )
    print(f"    -> Pha A chạy thử 2 batch: Train Loss={stats_a['train_loss']:.4f} => [PASS]")

    # Pha B: Mở khóa toàn bộ backbone với Differential LR
    model.freeze_backbone(False)
    model.train()
    summary_b = model.stage_summary()
    param_groups_b = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3)
    assert summary_b["trainable_params"] == 4_008_829
    assert len(param_groups_b) == 2
    assert param_groups_b[0]["group_name"] == "backbone" and param_groups_b[0]["lr"] == 1e-4
    assert param_groups_b[1]["group_name"] == "head" and param_groups_b[1]["lr"] == 1e-3
    opt_b = torch.optim.AdamW(param_groups_b)

    print(f"  + Pha B (Full Fine-tune): Trainable={summary_b['trainable_params']:,} params (Backbone=1e-4, Head=1e-3)")
    stats_b = train_one_epoch(
        model=model, loader=train_loader, optimizer=opt_b, criterion=crit,
        device=device, epoch=1, scaler=scaler, amp_dtype=amp_dtype, use_amp=use_amp, max_batches=2, log_interval=0
    )
    print(f"    -> Pha B chạy thử 2 batch: Train Loss={stats_b['train_loss']:.4f} => [PASS]")

    # -------------------------------------------------------------------------
    # BƯỚC 5: Đánh giá phân rã theo Generator qua metrics.py
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 5/5] Kiểm tra liên kết đánh giá phân rã theo Generator (metrics.py):")
    val_loss, val_metrics, val_gen_eval = evaluate(
        model=model,
        loader=val_loader,
        criterion=crit,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        max_batches=15,
        compute_per_generator=True,
    )
    assert val_gen_eval is not None, "Không tạo được val_gen_eval!"
    print(format_metrics_summary(val_metrics, "TỔNG HỢP METRICS VAL (15 BATCH THẬT)"))
    print("\n" + format_generator_report(val_gen_eval, "BÁO CÁO CẶP 1:1 THEO GENERATOR TRÊN TẬP VAL"))

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 85)
    print(f"  KẾT LUẬN: SpatialBranch (EfficientNet.py) HOÀN TOÀN TƯƠNG THÍCH VÀ KẾT NỐI TRƠN TRU 100%")
    print(f"  VỚI TẤT CẢ CÁC MODULE TRONG HỆ THỐNG TRONG {elapsed:.1f} GIÂY!")
    print("  ĐÃ SẴN SÀNG ĐỂ VIẾT SCRIPT HUẤN LUYỆN CHÍNH THỨC: train_branch.py")
    print("=" * 85)


if __name__ == "__main__":
    run_integration_verification()

