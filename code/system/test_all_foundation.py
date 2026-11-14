# -*- coding: utf-8 -*-
"""
code/system/test_all_foundation.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Script nghiệm thu tổng hợp toàn bộ các module nền tảng trước khi bước vào huấn luyện:
1. Module Giao diện nhánh: code/system/branch/base.py
2. Module Thước đo đánh giá: code/system/train/metrics.py
3. Module Động cơ huấn luyện: code/system/train/engine.py
4. Kiểm thử tích hợp End-to-End trên dữ liệu thật Defactify qua GPU RTX 3050 (bfloat16).
"""

import sys
import time
from pathlib import Path

# Cấu hình UTF-8 cho Windows Terminal
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from base import BranchBase, run_unit_tests as run_base_tests
from metrics import (
    ALL_AI_GENERATORS,
    REAL_MODEL_NAME,
    compute_binary_metrics,
    run_unit_tests as run_metrics_tests,
    sigmoid,
)
from engine import evaluate, get_optimal_amp_config, sanity_check_gate, run_unit_tests as run_engine_tests
from dataset import build_dataloaders


class IntegrationBranch(BranchBase):
    """Nhánh thử nghiệm kết nối trực tiếp đầu ra 'spatial' (3x224x224) của DefactifyDataset."""

    input_key = "spatial"
    feature_dim = 1280

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, self.feature_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


def verify_label_and_logit_alignment(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    """
    Kiểm chứng tường minh 4 câu hỏi đối soát nhãn & chiều dự đoán trên dữ liệu thật:
    1. label_a: 0 == Real (MS_COCO), 1 == AI (5 bộ sinh)?
    2. Logit dương (> 0) có được hiểu là AI (P(AI) > 0.5) trong BCEWithLogitsLoss & sigmoid không?
    3. ROC-AUC có nhận xác suất P(AI) và nhãn dương AI = 1 không?
    4. evaluate() có thu thập nhãn và logits khớp thứ tự từng mẫu 100% không?
    """
    print("\n  [ĐỐI SOÁT 4 ĐIỂM NHÃN - LOGIT - THỨ TỰ MẪU TRÊN DỮ LIỆU THẬT]")
    meta_csv = CURRENT_DIR.parents[1] / "dataset" / "Defactify_dataset" / "metadata.csv"
    df = pd.read_csv(meta_csv)

    # 1. Đối soát quy ước nhãn trên toàn bộ 94,272 dòng metadata.csv
    real_rows = df[df["model_name"] == REAL_MODEL_NAME]
    ai_rows = df[df["model_name"].isin(ALL_AI_GENERATORS)]
    assert len(real_rows) + len(ai_rows) == len(df), "Phát hiện model_name lạ ngoài MS_COCO và 5 AI generators!"
    assert (real_rows["label_a"] == 0).all(), "LỖI ĐẢO NHÃN: Có ảnh MS_COCO mang label_a != 0!"
    assert (ai_rows["label_a"] == 1).all(), "LỖI ĐẢO NHÃN: Có ảnh AI mang label_a != 1!"
    print(
        f"  + [1/4] Quy ước label_a (94,272 ảnh): MS_COCO ({len(real_rows):,}) -> 100% label_a=0 | "
        f"5 AI Generators ({len(ai_rows):,}) -> 100% label_a=1 => [PASS]"
    )

    # 2. Đối soát chiều Logit dương = AI trong BCEWithLogitsLoss & Sigmoid
    bce = nn.BCEWithLogitsLoss()
    pos_logit = torch.tensor([4.0])
    loss_when_ai = float(bce(pos_logit, torch.tensor([1.0])).item())
    loss_when_real = float(bce(pos_logit, torch.tensor([0.0])).item())
    prob_ai = float(sigmoid(np.array([4.0]))[0])
    assert prob_ai > 0.98 and loss_when_ai < 0.02 and loss_when_real > 4.0
    print(
        f"  + [2/4] Chiều Logit dương (+4.0): Sigmoid P(AI)={prob_ai:.4f}, "
        f"BCE(y=1)={loss_when_ai:.4f} << BCE(y=0)={loss_when_real:.4f} => [PASS]"
    )

    # 3. Đối soát chiều ROC-AUC nhận P(AI) và nhãn dương AI = 1
    m_dir = compute_binary_metrics(y_true=[0, 0, 1, 1], y_pred=[-3.0, -1.0, 1.0, 3.0], from_logits=True)
    m_inv = compute_binary_metrics(y_true=[0, 0, 1, 1], y_pred=[3.0, 1.0, -1.0, -3.0], from_logits=True)
    assert m_dir["auc"] == 1.0 and m_inv["auc"] == 0.0
    print(
        f"  + [3/4] Chiều ROC-AUC: Logit(AI)>Logit(Real) -> AUC={m_dir['auc']:.4f} | "
        f"Ngược dấu -> AUC={m_inv['auc']:.4f} => [PASS]"
    )

    # 4. Đối soát thứ tự mẫu giữa evaluate() và vòng lặp suy luận thủ công trực tiếp trên val_loader
    model.eval()
    manual_logits, manual_labels = [], []
    with torch.inference_mode():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= 4:
                break
            x = batch["spatial"].to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(x)
            else:
                out = model(x)
            manual_logits.append(out.float().cpu().numpy().ravel())
            manual_labels.append(batch["label_a"].float().cpu().numpy().ravel())
            # Kiểm tra ngay trong batch thực tế: MS_COCO == 0, AI == 1
            for m_name, lbl in zip(batch["model_name"], batch["label_a"].tolist()):
                expected = 0 if m_name == REAL_MODEL_NAME else 1
                assert int(lbl) == expected, f"Lệch nhãn trực tiếp trong batch: {m_name} vs {lbl}"

    manual_metrics = compute_binary_metrics(
        np.concatenate(manual_labels), np.concatenate(manual_logits), from_logits=True
    )
    _, engine_metrics, _ = evaluate(
        model=model,
        loader=val_loader,
        criterion=bce,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        max_batches=4,
        compute_per_generator=False,
    )
    assert abs(float(manual_metrics["auc"]) - float(engine_metrics["auc"])) < 1e-12
    assert manual_metrics["tn"] == engine_metrics["tn"] and manual_metrics["tp"] == engine_metrics["tp"]
    print(
        f"  + [4/4] Khớp thứ tự mẫu evaluate() (128 ảnh Val thật): "
        f"Sai số AUC = {abs(float(manual_metrics['auc']) - float(engine_metrics['auc'])):.1e} => [PASS]\n"
    )


def run_all_tests():
    print("=" * 85)
    print("      TỔNG HỢP KIỂM THỬ NGHIỆM THU HỆ THỐNG NỀN TẢNG (DATN AI DETECTION)")
    print("=" * 85)
    t_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # BƯỚC 1: Kiểm thử giao diện BranchBase
    # -------------------------------------------------------------------------
    print("\n>>> [PHẦN 1/4] Chạy Unit Test: code/system/branch/base.py")
    run_base_tests()

    # -------------------------------------------------------------------------
    # BƯỚC 2: Kiểm thử toán học & thước đo metrics.py (bao gồm 43,800 mẫu Test thật)
    # -------------------------------------------------------------------------
    print("\n>>> [PHẦN 2/4] Chạy Unit Test: code/system/train/metrics.py")
    run_metrics_tests()

    # -------------------------------------------------------------------------
    # BƯỚC 3: Kiểm thử động cơ huấn luyện engine.py (bao gồm Resume Optimizer/Scheduler/Scaler)
    # -------------------------------------------------------------------------
    print("\n>>> [PHẦN 3/4] Chạy Unit Test: code/system/train/engine.py")
    run_engine_tests()

    # -------------------------------------------------------------------------
    # BƯỚC 4: Kiểm thử tích hợp End-to-End 50 batch thật trên DefactifyDataset + GPU RTX 3050
    # -------------------------------------------------------------------------
    print("\n>>> [PHẦN 4/4] Kiểm thử tích hợp End-to-End 50 Batch Thật (DefactifyDataset + GPU RTX 3050 + bf16)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  + Thiết bị phần cứng : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    use_amp, amp_dtype, scaler = get_optimal_amp_config(device)
    print(f"  + Cấu hình AMP GPU   : use_amp={use_amp}, dtype={amp_dtype}, scaler={scaler is not None}")

    print("  + Đang khởi tạo DataLoader thực tế (batch_size=32, num_workers=4, active_branches=('spatial',))...")
    train_loader, val_loader, _, _ = build_dataloaders(
        batch_size=32,
        num_workers=4,
        active_branches=("spatial",),
        persistent_workers=False,
    )

    torch.manual_seed(42)
    model = IntegrationBranch().to(device)

    # Đối soát 4 câu hỏi về nhãn, chiều logit, ROC-AUC và thứ tự mẫu trong evaluate()
    verify_label_and_logit_alignment(model, val_loader, device, use_amp, amp_dtype)

    # Chạy Cổng kiểm tra 3 bước đủ 50 batch Train (1,600 ảnh) + 50 batch Val (1,600 ảnh) trên dữ liệu thật
    ok = sanity_check_gate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        overfit_steps=100,
        smoke_train_batches=50,
        smoke_val_batches=50,
        smoke_lr=1e-3,
    )
    assert ok, "Kiểm thử tích hợp 50 batch dữ liệu thật thất bại!"

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 85)
    print(f"  KẾT QUẢ: TOÀN BỘ 4 PHẦN NỀN TẢNG ĐÃ VƯỢT QUA 100% TRONG {total_time:.1f} GIÂY!")
    print("  HỆ THỐNG ĐÃ SẴN SÀNG 100% ĐỂ BẮT ĐẦU HUẤN LUYỆN CÁC NHÁNH!")
    print("=" * 85)


if __name__ == "__main__":
    run_all_tests()

