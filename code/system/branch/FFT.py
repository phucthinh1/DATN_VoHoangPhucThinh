# -*- coding: utf-8 -*-
"""
code/system/branch/FFT.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Nhánh Tần số Phổ (Spectral Branch) — Residual Spectral CNN (Train from scratch):
- Vai trò chuyên môn: Phân tích bản đồ phổ 2D Fourier (từ `view_native` 224x224 qua Luma Y' -> 2D FFT -> log1p -> Z-score)
  để phát hiện các vết lưới bàn cờ (checkerboard artifacts), đỉnh xung gai tần số cao (periodic spectral spikes)
  và độ dốc suy giảm năng lượng 1/f^alpha do các giải thuật khuếch tán (Diffusion) và bộ giải mã VAE/Upsampling sinh ra.
- Tuân thủ hợp đồng giao diện `BranchBase` (Mục 1, 4 & 7 của Chot_Cau_Hinh_Buoc_5B_3_Nhanh.md):
  + `input_key = "spectral"`        : Nhận `batch["spectral"]` shape `(B, 1, 224, 224)` từ DataLoader (`remove_dc=True`).
  + `feature_dim = 256`             : Trích xuất đặc trưng qua Dual Spectral Pooling (GAP || GMP) -> Linear -> LayerNorm -> GELU.
  + `classifier`                    : `Dropout(0.3) -> Linear(256, 1)` xuất 1 logit `(B,)` phục vụ `BCEWithLogitsLoss`.
- 4 Điểm Đột phá Kiến trúc cho Miền Tần số Phổ:
  1. CoordConv Cố định: Tự sinh 2 kênh tọa độ không gian chuẩn hóa x, y thuộc [-1, 1], ghép vào Tensor phổ thành
     `(B, 3, 224, 224)`. Giúp CNN khắc phục tính bất biến tịnh tiến và nhận biết chính xác vị trí tần số (tâm: thấp, rìa: cao).
  2. Stem Bảo toàn Độ phân giải: `Conv3x3 (3 -> 32, stride=1, padding=1) + BN + ReLU`, TUYỆT ĐỐI KHÔNG dùng MaxPool
     hay stride=2 ở lớp đầu để không làm nhòe các đỉnh xung gai AI có kích thước chỉ 1-2 pixel.
  3. Thân mạng 4 Residual Stages: Gồm 8 khối `SpectralBasicBlock` (Stage 1: 32->64; Stage 2: 64->128; Stage 3: 128->256;
     Stage 4: 256->256 với stride=2 có downsample shortcut). Chuẩn hóa chính xác 5,317,409 tham số.
  4. Zero-Init Residual: Khởi tạo Kaiming Normal toàn mạng kết hợp đặt duy nhất `bn2.weight = 0.0` của mỗi BasicBlock,
     giúp ổn định gradient tuyệt đối khi huấn luyện từ đầu (Train from scratch) 30 epochs qua `train_branch.py`.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from base import BranchBase  # noqa: E402


def _build_coord_grid(height: int, width: int) -> torch.Tensor:
    """
    Tạo ma trận 2 kênh tọa độ không gian chuẩn hóa x, y thuộc đoạn [-1.0, 1.0].
    - Kênh 0 (x): Trục ngang từ trái sang phải (-1.0 -> 1.0).
    - Kênh 1 (y): Trục dọc từ trên xuống dưới (-1.0 -> 1.0).

    Returns:
        Tensor shape `(1, 2, height, width)`, kiểu `float32`.
    """
    y_coords = torch.linspace(-1.0, 1.0, steps=height, dtype=torch.float32)
    x_coords = torch.linspace(-1.0, 1.0, steps=width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # Shape (1, 2, H, W)
    return grid


class SpectralBasicBlock(nn.Module):
    """
    Khối tích chập phần dư (Residual BasicBlock) chuyên dụng cho phổ tần số 2D.
    Bao gồm 2 lớp Conv 3x3, BatchNorm2d, ReLU và nhánh tắt (Shortcut).
    - Hỗ trợ Zero-Init Residual tại `self.bn2.weight = 0.0`.
    - Downsample tự động kích hoạt khi `stride != 1` HOẶC `in_channels != out_channels`
      (đặc biệt xử lý chính xác Stage 4: 256 -> 256 với stride=2).
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class SpectralBranch(BranchBase):
    """
    Nhánh Tần số Phổ (Spectral Stream) kiến trúc Residual Spectral CNN.

    Thông số kỹ thuật chuẩn:
    - Input shape: (B, 1, 224, 224) từ `batch["spectral"]`.
    - CoordConv: Tự động ghép 2 kênh tọa độ thành (B, 3, 224, 224).
    - Backbone: Stem (stride=1) + 4 Residual Stages (8 BasicBlocks) + Dual Pooling (GAP || GMP) + Projection.
    - Output feature: (B, 256) chuẩn hóa LayerNorm.
    - Classifier head: Dropout(0.3) -> Linear(256, 1) xuất logit (B,).
    - Tổng tham số chính xác: 5,317,409 tham số (Backbone: 5,317,152; Head: 257).
    """

    input_key: str = "spectral"
    feature_dim: int = 256
    always_frozen_backbone: bool = False

    def __init__(self, dropout: float = 0.3) -> None:
        """
        Khởi tạo nhánh Tần số Phổ SpectralBranch.

        Args:
            dropout: Tỷ lệ Dropout của tầng phân loại Classifier Head (mặc định 0.3 theo Bảng 1).
                     Tương thích với lời gọi `SpectralBranch(dropout=dropout)` trong `train_branch.py`.
        """
        super().__init__()

        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout phải nằm trong [0.0, 1.0), nhận được: {dropout}")

        # 1. Đăng ký buffer lưới tọa độ 224x224 cố định (persistent=False để không lưu vào state_dict checkpoint)
        initial_grid = _build_coord_grid(224, 224)
        self.register_buffer("_coord_grid", initial_grid, persistent=False)

        # 2. Tầng Stem bảo toàn độ phân giải (Resolution-Preserving Stem)
        # Nhận 3 kênh (1 kênh phổ + 2 kênh CoordConv), giữ nguyên 224x224 (stride=1, không MaxPool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 3. Khối 4 Tầng Phần dư (4 Residual Stages - 8 BasicBlocks)
        # Stage 1: (B, 32, 224, 224) -> (B, 64, 112, 112)
        self.stage1 = nn.Sequential(
            SpectralBasicBlock(in_channels=32, out_channels=64, stride=2),
            SpectralBasicBlock(in_channels=64, out_channels=64, stride=1),
        )

        # Stage 2: (B, 64, 112, 112) -> (B, 128, 56, 56)
        self.stage2 = nn.Sequential(
            SpectralBasicBlock(in_channels=64, out_channels=128, stride=2),
            SpectralBasicBlock(in_channels=128, out_channels=128, stride=1),
        )

        # Stage 3: (B, 128, 56, 56) -> (B, 256, 28, 28)
        self.stage3 = nn.Sequential(
            SpectralBasicBlock(in_channels=128, out_channels=256, stride=2),
            SpectralBasicBlock(in_channels=256, out_channels=256, stride=1),
        )

        # Stage 4: (B, 256, 28, 28) -> (B, 256, 14, 14) [Downsample shortcut kích hoạt vì stride=2]
        self.stage4 = nn.Sequential(
            SpectralBasicBlock(in_channels=256, out_channels=256, stride=2),
            SpectralBasicBlock(in_channels=256, out_channels=256, stride=1),
        )

        # 4. Gom Đặc Trưng Kép (Dual Spectral Pooling: GAP || GMP)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # 5. Tầng Chiếu Tuyến tính (Projection Layer: 512 -> 256)
        self.projection = nn.Sequential(
            nn.Linear(512, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )

        # 6. Đầu Phân loại Nhị phân Độc lập (Classifier Head: 256 -> 1 logit)
        self.classifier = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.feature_dim, 1),
        )

        # 7. Khởi tạo trọng số Kaiming Normal + Zero-Init Residual bn2.weight = 0.0
        self._init_weights()

    def _init_weights(self) -> None:############
        """
        Khởi tạo trọng số chuẩn SOTA cho mạng huấn luyện từ đầu (Train from scratch):
        - Conv2d: Kaiming Normal (He init) với mode='fan_out', nonlinearity='relu'.
        - BatchNorm2d: weight=1.0, bias=0.0.
        - Linear: Kaiming Normal với mode='fan_in', nonlinearity='linear' (bảo toàn phương sai chiều xuôi
          sigma_w = 1 / sqrt(fan_in), ngăn chặn bùng nổ logit tại tầng `classifier` `Linear(256, 1)`
          khi `fan_out = 1`), bias=0.0.
        - LayerNorm: weight=1.0, bias=0.0.
        - Zero-Init Residual: ĐẶT DUY NHẤT bn2.weight = 0.0 CỦA MỖI BASICBLOCK.
          Tất cả các BatchNorm2d khác (stem.bn, bn1, downsample.bn) giữ nguyên weight=1.0!
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # Bắt buộc dùng mode="fan_in", nonlinearity="linear" cho nn.Linear
                # để tránh chia cho fan_out=1 tại tầng classifier Linear(256, 1)
                ############
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Zero-Init Residual: Khởi tạo nhánh residual về 0
        for m in self.modules():
            if isinstance(m, SpectralBasicBlock):
                nn.init.constant_(m.bn2.weight, 0.0)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Điều khiển đóng/mở băng thân mạng phục vụ trích xuất đặc trưng Fusion:
        - `freeze=True`: Khóa 100% đạo hàm (`requires_grad=False`), dọn sạch `p.grad = None`
          để giải phóng bộ nhớ, và đưa toàn bộ module con của backbone về chế độ `.eval()`.
        - `freeze=False`: Mở khóa lại toàn bộ tham số thân mạng (`requires_grad=True`).
        """
        self._backbone_frozen = bool(freeze)
        classifier_params = set(self.classifier.parameters()) if self.classifier is not None else set()

        for p in self.parameters():
            if p not in classifier_params:
                p.requires_grad = not freeze
                if freeze:
                    p.grad = None

        if self.classifier is not None:
            for p in self.classifier.parameters():
                p.requires_grad = True

        if freeze:
            self._set_frozen_backbone_norms_eval()
        elif self.training:
            super().train(True)

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """Hàm công tắc tường minh mở/khóa thân mạng."""
        self.freeze_backbone(freeze=not trainable)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng phổ tần số 256 chiều từ Tensor phổ 2D FFT.

        Args:
            x: Tensor phổ 2D FFT 1 kênh, shape `(B, 1, H, W)` từ DataLoader.

        Returns:
            Tensor đặc trưng 2D shape `(B, 256)`, chuẩn hóa LayerNorm.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"SpectralBranch yêu cầu đầu vào 4D phổ 1 kênh kích thước (B, 1, H, W), "
                f"nhận được shape: {tuple(x.shape)}"
            )

        batch_size, _, height, width = x.shape

        # 1. Bổ sung 2 kênh CoordConv: xử lý đúng device, dtype và kích thước động
        ############
        if height == self._coord_grid.shape[2] and width == self._coord_grid.shape[3]:
            if self._coord_grid.device != x.device or self._coord_grid.dtype != x.dtype:
                self._coord_grid = self._coord_grid.to(device=x.device, dtype=x.dtype)
            coord = self._coord_grid.expand(batch_size, -1, -1, -1)
        else:
            coord = _build_coord_grid(height, width).to(device=x.device, dtype=x.dtype).expand(batch_size, -1, -1, -1)

        x_in = torch.cat([x, coord], dim=1)  # Shape (B, 3, H, W)

        # 2. Lan truyền qua Stem và 4 Residual Stages
        out = self.stem(x_in)       # (B, 32, 224, 224)
        out = self.stage1(out)      # (B, 64, 112, 112)
        out = self.stage2(out)      # (B, 128, 56, 56)
        out = self.stage3(out)      # (B, 256, 28, 28)
        out = self.stage4(out)      # (B, 256, 14, 14)

        # 3. Dual Spectral Pooling (GAP || GMP)
        gap = self.avgpool(out).flatten(start_dim=1)  # (B, 256)
        gmp = self.maxpool(out).flatten(start_dim=1)  # (B, 256)
        pooled = torch.cat([gap, gmp], dim=1)         # (B, 512)

        # 4. Tầng Chiếu Tuyến tính
        features = self.projection(pooled)            # (B, 256)

        return features


# ==============================================================================
# BỘ KIỂM THỬ ĐƠN VỊ ĐỘC LẬP TOÀN DIỆN (FFT.py Unit Tests)
# ==============================================================================
def run_unit_tests() -> None:
    """
    Bộ kiểm thử đơn vị độc lập cho nhánh SpectralBranch (FFT.py):
    1. Kiểm tra thuộc tính bắt buộc, hợp đồng giao diện và khớp chính xác 5,317,409 tham số.
    2. Kiểm tra forward_features() và forward() (B=4 và B=1, shape, dtype).
    3. Kiểm tra CoordConv xử lý đúng device, dtype và hỗ trợ kích thước tùy biến (Dynamic Grid).
    4. Kiểm tra Zero-Init Residual (bn2.weight == 0.0) và hành vi đạo hàm qua 2 bước tối ưu.
    5. Kiểm tra freeze_backbone(True) khóa cả đạo hàm, xóa p.grad = None và giữ BatchNorm ở .eval().
    6. Kiểm tra tương thích get_param_groups() với bộ tối ưu AdamW.
    """
    import time

    print("=" * 85)
    print("BẮT ĐẦU CHẠY KIỂM THỬ ĐỘC LẬP: code/system/branch/FFT.py (SpectralBranch)")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # TEST 1: Khởi tạo, Hợp đồng giao diện & Khớp chính xác 5,317,409 tham số
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 1] Khởi tạo mô hình, kiểm tra hợp đồng giao diện & tổng tham số:")
    t0 = time.time()
    model = SpectralBranch(dropout=0.3)
    init_time = (time.time() - t0) * 1000

    assert model.input_key == "spectral", f"input_key sai: {model.input_key}"
    assert model.feature_dim == 256, f"feature_dim sai: {model.feature_dim}"
    assert not model.always_frozen_backbone

    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    backbone_params = total_params - classifier_params

    print(f"-> Thời gian khởi tạo: {init_time:.2f} ms")
    print(f"-> Tổng tham số thân mạng (Backbone): {backbone_params:,} tham số (Kỳ vọng: 5,317,152)")
    print(f"-> Tổng tham số đầu phân loại (Head): {classifier_params:,} tham số (Kỳ vọng: 257)")
    print(f"-> TỔNG SỐ THAM SỐ TOÀN MÔ HÌNH: {total_params:,} tham số.")

    assert backbone_params == 5_317_152, f"Backbone params sai: {backbone_params} != 5,317,152"
    assert classifier_params == 257, f"Classifier params sai: {classifier_params} != 257"
    assert total_params == 5_317_409, f"Tổng tham số sai: {total_params} != 5,317,409"
    assert trainable_params == total_params, "Ban đầu toàn bộ tham số phải có requires_grad=True!"
    print("=> [PASS] Khớp chính xác 100% mốc 5,317,409 tham số thiết kế.")

    # -------------------------------------------------------------------------
    # TEST 2: Kiểm tra Forward Features, Forward Logits & Thang đo Khởi tạo Ban đầu
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 2] Kiểm tra forward_features(), forward() (B=4, B=1) & Thang đo Logit/Loss khởi tạo:")
    torch.manual_seed(42)
    model = SpectralBranch(dropout=0.3)
    model.eval()

    # Kiểm tra độ lệch chuẩn trọng số khởi tạo của Classifier Linear(256, 1) (kỳ vọng ~1/sqrt(256) = 0.0625)
    cls_w_std = float(model.classifier[1].weight.std().item())
    proj_w_std = float(model.projection[0].weight.std().item())
    print(f"-> Độ lệch chuẩn khởi tạo: Projection Linear(512,256) std={proj_w_std:.4f} (lý thuyết 0.0442) | "
          f"Classifier Linear(256,1) std={cls_w_std:.4f} (lý thuyết 0.0625)")
    assert 0.04 <= cls_w_std <= 0.09, f"Classifier weight std nằm ngoài dải an toàn: {cls_w_std:.4f} (bị lỗi fan_out=1?)"
    assert 0.035 <= proj_w_std <= 0.055, f"Projection weight std nằm ngoài dải an toàn: {proj_w_std:.4f}"

    dummy_4 = torch.randn(4, 1, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        feat_4 = model.forward_features(dummy_4)
        logits_4 = model(dummy_4)
        init_loss_4 = float(nn.BCEWithLogitsLoss()(logits_4, torch.tensor([0.0, 1.0, 0.0, 1.0])).item())
    assert feat_4.shape == (4, 256), f"Shape features sai: {feat_4.shape}, kỳ vọng (4, 256)"
    assert logits_4.shape == (4,), f"Shape logits sai: {logits_4.shape}, kỳ vọng (4,)"
    assert feat_4.dtype == torch.float32 and logits_4.dtype == torch.float32
    assert float(logits_4.abs().max().item()) < 2.0, f"Logit khởi tạo quá lớn: {logits_4}"
    assert 0.50 <= init_loss_4 <= 0.95, f"Initial BCE Loss lệch xa ln(2)=0.6931: {init_loss_4:.4f}"

    dummy_1 = torch.randn(1, 1, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        feat_1 = model.forward_features(dummy_1)
        logits_1 = model(dummy_1)
    assert feat_1.shape == (1, 256), f"Shape features khi B=1 sai: {feat_1.shape}"
    assert logits_1.shape == (1,), f"Shape logits khi B=1 bị co thành scalar: {logits_1.shape}, kỳ vọng (1,)"
    print(f"-> B=4: features={tuple(feat_4.shape)}, logits={tuple(logits_4.shape)}, max|logit|={logits_4.abs().max().item():.4f}, Initial BCE Loss={init_loss_4:.4f} (≈ ln(2)=0.6931)")
    print(f"-> B=1: features={tuple(feat_1.shape)}, logits={tuple(logits_1.shape)} (bảo toàn 1-D Tensor)")
    print("=> [PASS] Lan truyền xuôi trả về đúng chiều vector 256, logit 1-D và Initial Loss chuẩn xác.")

    # -------------------------------------------------------------------------
    # TEST 3: Kiểm tra CoordConv xử lý đúng device, dtype & kích thước động
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 3] Kiểm tra CoordConv (Device, Dtype & Kích thước động):")
    # Kiểm tra kích thước khác 224x224 (ví dụ 112x112)
    dummy_custom = torch.randn(2, 1, 112, 112, dtype=torch.float32)
    with torch.no_grad():
        feat_custom = model.forward_features(dummy_custom)
    assert feat_custom.shape == (2, 256), f"Lỗi với kích thước tùy biến: {feat_custom.shape}"
    print("-> Xử lý thành công kích thước ảnh tùy biến (112x112) qua Dynamic Grid Generation.")

    # Kiểm tra đồng bộ dtype bfloat16
    dummy_bf16 = torch.randn(2, 1, 224, 224, dtype=torch.bfloat16)
    model_bf16 = SpectralBranch(dropout=0.3).to(dtype=torch.bfloat16)
    with torch.no_grad():
        feat_bf16 = model_bf16.forward_features(dummy_bf16)
    assert feat_bf16.dtype == torch.bfloat16 and feat_bf16.shape == (2, 256)
    print("-> Xử lý thành công dữ liệu kiểu bfloat16 đồng bộ với buffer CoordConv.")
    print("=> [PASS] CoordConv vận hành chuẩn xác cả trên kích thước chuẩn lẫn tùy biến.")

    # -------------------------------------------------------------------------
    # TEST 4: Kiểm tra Zero-Init Residual & Hành vi Gradient 2 Bước
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 4] Kiểm tra Zero-Init Residual (bn2.weight == 0.0) & Đạo hàm 2 bước:")
    model.train()

    # 4.1. Xác nhận đúng các lớp bn2.weight bằng 0.0 và các BN khác bằng 1.0
    basic_blocks = [m for m in model.modules() if isinstance(m, SpectralBasicBlock)]
    assert len(basic_blocks) == 8, f"Số BasicBlocks sai: {len(basic_blocks)} != 8"

    for i, blk in enumerate(basic_blocks):
        assert torch.equal(blk.bn2.weight, torch.zeros_like(blk.bn2.weight)), f"Block {i} bn2.weight != 0.0"
        assert torch.equal(blk.bn1.weight, torch.ones_like(blk.bn1.weight)), f"Block {i} bn1.weight != 1.0"
        if blk.downsample is not None:
            ds_bn = blk.downsample[1]
            assert torch.equal(ds_bn.weight, torch.ones_like(ds_bn.weight)), f"Block {i} downsample bn != 1.0"
    assert torch.equal(model.stem[1].weight, torch.ones_like(model.stem[1].weight)), "Stem BN != 1.0"
    print("-> Xác nhận: 100% 8 BasicBlocks có bn2.weight == 0.0, toàn bộ BN khác giữ nguyên 1.0.")

    # 4.2. Kiểm tra đạo hàm Bước 1 (Zero-Init Step 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    inp_1 = torch.randn(4, 1, 224, 224, dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    logits_step1 = model(inp_1)
    loss_step1 = criterion(logits_step1, labels)
    loss_step1.backward()

    # Tại Step 1: Stem, Downsample, bn2 và Classifier nhận đạo hàm khác 0
    assert model.classifier[1].weight.grad is not None and model.classifier[1].weight.grad.abs().sum() > 0
    assert model.stem[0].weight.grad is not None and model.stem[0].weight.grad.abs().sum() > 0
    assert basic_blocks[0].bn2.weight.grad is not None and basic_blocks[0].bn2.weight.grad.abs().sum() > 0

    # Nhánh residual conv1 của block không có downsample tại Step 1 nhận đạo hàm = 0 (vì bn2.weight = 0)
    # Block 1 của Stage 1 (khối thứ hai, không có downsample):
    blk_identity = basic_blocks[1]
    assert blk_identity.conv1.weight.grad is not None
    assert torch.all(blk_identity.conv1.weight.grad == 0.0), "Step 1: conv1 của residual branch phải bằng 0.0!"
    print("-> Step 1: Đạo hàm truyền chuẩn xác qua shortcut & stem; bn2.weight nhận gradient khác 0.")

    # 4.3. Cập nhật trọng số & Kiểm tra đạo hàm Bước 2 (Step 2)
    optimizer.step()
    # Sau optimizer.step(), bn2.weight đã khác 0
    assert not torch.equal(basic_blocks[0].bn2.weight, torch.zeros_like(basic_blocks[0].bn2.weight))

    optimizer.zero_grad()
    inp_2 = torch.randn(4, 1, 224, 224, dtype=torch.float32)
    logits_step2 = model(inp_2)
    loss_step2 = criterion(logits_step2, labels)
    loss_step2.backward()

    # Tại Step 2: Toàn bộ conv1 và conv2 của cả 8 block đều nhận đạo hàm khác 0 bình thường!
    for i, blk in enumerate(basic_blocks):
        assert blk.conv1.weight.grad is not None and blk.conv1.weight.grad.abs().sum() > 0
        assert blk.conv2.weight.grad is not None and blk.conv2.weight.grad.abs().sum() > 0
    print("-> Step 2: Toàn bộ 8 BasicBlocks (conv1, conv2) nhận đạo hàm khác 0 mạnh mẽ.")
    print("=> [PASS] Cơ chế Zero-Init Residual và chu trình lan truyền đạo hàm đạt chuẩn tuyệt đối.")

    # -------------------------------------------------------------------------
    # TEST 5: Kiểm tra freeze_backbone(True) & Dọn sạch p.grad = None
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 5] Kiểm tra freeze_backbone(True) & Dọn sạch p.grad = None:")
    model.freeze_backbone(freeze=True)
    assert model.count_parameters(trainable_only=True) == 257, "Chỉ classifier được phép trainable khi freeze!"

    # Xác nhận mọi tham số backbone đều có requires_grad=False và p.grad = None
    for name, p in model.named_parameters():
        if "classifier" not in name:
            assert not p.requires_grad, f"Tham số {name} chưa được khóa!"
            assert p.grad is None, f"Tham số {name} còn sót grad rác!"

    # Kiểm tra khi gọi model.train(), BatchNorm của backbone vẫn ở chế độ .eval()
    model.train()
    assert not model.stem[1].training, "Stem BatchNorm bị bật sang train() khi đang freeze!"
    assert not basic_blocks[0].bn1.training, "BasicBlock BatchNorm bị bật sang train() khi đang freeze!"
    assert model.classifier.training, "Classifier phải ở chế độ train()!"

    # Mở khóa lại
    model.freeze_backbone(freeze=False)
    assert model.count_parameters(trainable_only=True) == 5_317_409
    assert model.stem[1].training, "Stem BatchNorm phải quay lại train() sau khi unfreeze!"
    print("=> [PASS] freeze_backbone(True/False) bảo vệ chặt chẽ BatchNorm và dọn sạch p.grad = None.")

    # -------------------------------------------------------------------------
    # TEST 6: Kiểm tra tương thích get_param_groups() với AdamW
    # -------------------------------------------------------------------------
    print("\n[KIỂM TRA 6] Kiểm tra get_param_groups() với AdamW:")
    groups = model.get_param_groups(backbone_lr=1e-3, head_lr=1e-3, weight_decay=1e-4)
    assert len(groups) == 2, f"Số groups sai: {len(groups)}, kỳ vọng 2"
    assert groups[0]["group_name"] == "backbone" and groups[0]["lr"] == 1e-3
    assert groups[1]["group_name"] == "head" and groups[1]["lr"] == 1e-3
    total_in_groups = sum(sum(p.numel() for p in g["params"]) for g in groups)
    assert total_in_groups == 5_317_409, f"Tổng tham số trong groups sai: {total_in_groups}"

    opt_test = torch.optim.AdamW(groups)
    assert len(opt_test.param_groups) == 2
    print("=> [PASS] get_param_groups() tương thích hoàn hảo với cấu hình huấn luyện của train_branch.py.")

    print("\n" + "=" * 85)
    print("TẤT CẢ 6 BÀI KIỂM THỬ ĐƠN VỊ CỦA FFT.py (SpectralBranch) ĐÃ VƯỢT QUA 100%!")
    print("=" * 85)


if __name__ == "__main__":
    run_unit_tests()

