# -*- coding: utf-8 -*-
"""
predict_single.py
Kiểm thử dự đoán 1 ảnh độc lập qua 2 nhánh:
- Nhánh 1: Spatial Branch (EfficientNet-B0)
- Nhánh 2: Spectral Branch (2D FFT Residual CNN)

Tách biệt hoàn toàn bên ngoài thư mục code/ để phục vụ kiểm thử nhanh.
Cách dùng từ dòng lệnh (Terminal):
    python predict_single.py --image "duong_dan_anh.jpg"
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image

# ------------------------------------------------------------------------------
# 1. Thiết lập đường dẫn liên kết tới thư mục code/
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
BRANCH_DIR = ROOT_DIR / "code" / "system" / "branch"
PREPROC_DIR = ROOT_DIR / "code" / "pre_processing" / "pre_processing_all_and_branchs"
CHECKPOINT_DIR = ROOT_DIR / "code" / "system" / "checkpoints"

for p in (str(BRANCH_DIR), str(PREPROC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from EfficientNet import SpatialBranch
from FFT import SpectralBranch
from pre_processing_for_all import CommonProcess
from pre_processing_EfficientNet import EfficientNetPreprocessor
from pre_processing_FFT import FFTPreprocessor


def load_models(device: torch.device):
    """Nạp trọng số tốt nhất (best checkpoint) của cả 2 nhánh."""
    spatial_ckpt_path = CHECKPOINT_DIR / "spatial_baseline" / "best_spatial.pth"
    spectral_ckpt_path = CHECKPOINT_DIR / "spectral_baseline" / "best_spectral.pth"

    if not spatial_ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint Spatial tại: {spatial_ckpt_path}")
    if not spectral_ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint Spectral tại: {spectral_ckpt_path}")

    # 1. Nhánh Spatial (EfficientNet-B0)
    spatial_model = SpatialBranch(pretrained=False, dropout=0.2)
    s_ckpt = torch.load(spatial_ckpt_path, map_location="cpu", weights_only=False)
    spatial_model.load_state_dict(s_ckpt["model_state_dict"])
    spatial_model.to(device).eval()

    # 2. Nhánh Spectral (FFT Residual CNN)
    spectral_model = SpectralBranch(dropout=0.3)
    f_ckpt = torch.load(spectral_ckpt_path, map_location="cpu", weights_only=False)
    spectral_model.load_state_dict(f_ckpt["model_state_dict"])
    spectral_model.to(device).eval()

    return spatial_model, spectral_model


def predict_single_image(
    image_path: str,
    spatial_model: SpatialBranch,
    spectral_model: SpectralBranch,
    device: torch.device,
    threshold: float = 0.5,
):
    """
    Tiền xử lý và suy luận dự đoán cho một ảnh duy nhất.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

    # Khởi tạo các bộ tiền xử lý chuẩn eval
    prep_common = CommonProcess(mode="eval")
    prep_spatial = EfficientNetPreprocessor(view_mode="native", normalize=True)
    prep_spectral = FFTPreprocessor(mode="raw", use_hann=False, remove_dc=True, normalize=True)

    # 1. Tiền xử lý chung (lấy view_native 224x224 pixel gốc)
    common_out = prep_common(img_path)

    # 2. Tiền xử lý riêng cho từng nhánh
    tensor_spatial = prep_spatial(common_out).unsqueeze(0).to(device)    # (1, 3, 224, 224)
    tensor_spectral = prep_spectral(common_out).unsqueeze(0).to(device)  # (1, 1, 224, 224)

    # 3. Suy luận qua mô hình
    with torch.no_grad():
        logit_spatial = spatial_model(tensor_spatial).item()
        prob_ai_spatial = float(torch.sigmoid(torch.tensor(logit_spatial)).item())

        logit_spectral = spectral_model(tensor_spectral).item()
        prob_ai_spectral = float(torch.sigmoid(torch.tensor(logit_spectral)).item())

    prob_ai_ensemble = (prob_ai_spatial + prob_ai_spectral) / 2.0

    return {
        "image_path": str(img_path),
        "orig_size": common_out["orig_size"],
        "was_padded": common_out["was_padded"],
        "native_crop_box": common_out["native_crop_box"],
        "spatial": {
            "logit": logit_spatial,
            "prob_ai": prob_ai_spatial,
            "prob_real": 1.0 - prob_ai_spatial,
            "label": "AI GENERATED" if prob_ai_spatial >= threshold else "REAL (MS_COCO)",
        },
        "spectral": {
            "logit": logit_spectral,
            "prob_ai": prob_ai_spectral,
            "prob_real": 1.0 - prob_ai_spectral,
            "label": "AI GENERATED" if prob_ai_spectral >= threshold else "REAL (MS_COCO)",
        },
        "ensemble": {
            "prob_ai": prob_ai_ensemble,
            "prob_real": 1.0 - prob_ai_ensemble,
            "label": "AI GENERATED" if prob_ai_ensemble >= threshold else "REAL (MS_COCO)",
        },
    }


def print_prediction_report(res: dict):
    """In bảng kết quả định dạng trực quan ra console."""
    print("\n" + "=" * 70)
    print("           KẾT QUẢ KIỂM THỬ DỰ ĐOÁN ẢNH ĐƠN LẺ (SINGLE INFERENCE)")
    print("=" * 70)
    print(f"File ảnh         : {res['image_path']}")
    print(f"Kích thước gốc   : {res['orig_size'][0]}x{res['orig_size'][1]} px")
    print(f"Vùng cắt 224x224 : {res['native_crop_box']} (Pad viền: {res['was_padded']})")
    print("-" * 70)

    # Nhánh Spatial
    sp = res["spatial"]
    print(f"[1] Nhánh Không gian (Spatial - EfficientNet-B0):")
    print(f"    - Xác suất AI sinh : {sp['prob_ai'] * 100:6.2f}%")
    print(f"    - Xác suất Ảnh thật: {sp['prob_real'] * 100:6.2f}%")
    print(f"    - Logit            : {sp['logit']:+.4f}")
    print(f"    -> Đánh giá        : [ {sp['label']} ]")
    print("-" * 70)

    # Nhánh Spectral
    spec = res["spectral"]
    print(f"[2] Nhánh Phổ tần số (Spectral - 2D FFT CNN):")
    print(f"    - Xác suất AI sinh : {spec['prob_ai'] * 100:6.2f}%")
    print(f"    - Xác suất Ảnh thật: {spec['prob_real'] * 100:6.2f}%")
    print(f"    - Logit            : {spec['logit']:+.4f}")
    print(f"    -> Đánh giá        : [ {spec['label']} ]")
    print("-" * 70)

    # Kết hợp Ensemble
    ens = res["ensemble"]
    print(f"[3] Kết hợp 2 nhánh (Trung bình xác suất Ensemble):")
    print(f"    - Xác suất AI sinh : {ens['prob_ai'] * 100:6.2f}%")
    print(f"    - Xác suất Ảnh thật: {ens['prob_real'] * 100:6.2f}%")
    print(f"    ==================================================")
    print(f"    ==> KẾT LUẬN CHUNG : >>> {ens['label']} <<<")
    print(f"    ==================================================")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử dự đoán ảnh AI / Real qua 2 nhánh.")
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=r"dataset\Defactify_dataset\train\real\group_00001_MS_COCO.jpg",
        help="Đường dẫn tới file ảnh cần kiểm tra (mặc định lấy 1 ảnh mẫu trong dataset).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Ngưỡng xác định nhãn AI (mặc định 0.5).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Thiết bị tính toán: {device}")
    print("-> Đang tải trọng số mô hình...")
    spatial_model, spectral_model = load_models(device)
    print("-> Nạp mô hình hoàn tất!")

    target_path = Path(args.image)
    if not target_path.is_absolute():
        target_path = ROOT_DIR / target_path

    results = predict_single_image(
        image_path=str(target_path),
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        device=device,
        threshold=args.threshold,
    )
    print_prediction_report(results)

