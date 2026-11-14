# -*- coding: utf-8 -*-
"""
code/system/branch/EfficientNet.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Nhánh Không gian (Spatial Branch) — EfficientNet-B0 (`torchvision` IMAGENET1K_V1):
- Vai trò chuyên môn: Phân tích ô cắt gốc `view_native` (224x224, KHÔNG resize) để bắt các dấu vết
  vi mô mức pixel của bộ sinh AI (nhiễu giải mã VAE/upsampling, răng cưa biên, lỗi kết cấu bề mặt da/vải).
- Tuân thủ hợp đồng giao diện `BranchBase` (Mục 1, 2 & 7 của Chot_Cau_Hinh_Buoc_5B_3_Nhanh.md):
  + `input_key = "spatial"`         : Nhận `batch["spatial"]` shape `(B, 3, 224, 224)` đã chuẩn hóa ImageNet ở Bước 4A.
  + `feature_dim = 1280`            : Trích xuất đặc trưng qua `features(x) -> (B, 1280, 7, 7) -> AdaptiveAvgPool2d(1) -> (B, 1280)`.
  + `classifier`                    : `Dropout(0.2) -> Linear(1280, 1)` xuất 1 logit `(B,)` cho `BCEWithLogitsLoss`.
- Hỗ trợ Huấn luyện 2 Pha (Two-Phase Differential Fine-Tuning) & Ablation `trainable_stages`:
  + Pha A (Warmup Head - 1 epoch)   : `freeze_backbone(True)` -> khóa 100% đạo hàm & giữ mọi `BatchNorm2d` của
                                      backbone ở `.eval()`, chỉ học `classifier` (`1,281` tham số) với `lr = 1e-3`.
  + Pha B (Fine-tune - ~15 epochs)  : `freeze_backbone(False)` -> mở khóa theo cấu hình `trainable_stages`
                                      (mặc định `"all"`: toàn bộ 9 block `0..8`, ~4.01M tham số; hoặc `int`/danh sách
                                      stage cuối cho thí nghiệm ablation chống overfit sang generator Unseen),
                                      kết hợp Differential LR (`backbone_lr = 1e-4`, `head_lr = 1e-3`).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.ops import StochasticDepth

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from base import BranchBase  # noqa: E402


class SpatialBranch(BranchBase):
    """
    Nhánh Không gian (Spatial Stream) dựa trên EfficientNet-B0.

    Lưu ý quan trọng về tiền xử lý (Mục 2):
    - TUYỆT ĐỐI KHÔNG gọi `EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()` bên trong mô hình,
      bởi hàm đó của torchvision thực hiện `Resize(256) -> CenterCrop(224)` sẽ phá hủy hoàn toàn
      tương quan pixel gốc của `view_native` đã được cắt chuẩn ở Bước 3 & chuẩn hóa ở Bước 4A.
    """

    input_key: str = "spatial"
    feature_dim: int = 1280
    always_frozen_backbone: bool = False
    NUM_STAGES: int = 9  # efficientnet_b0.features gồm 9 block đánh số từ 0 đến 8

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
        trainable_stages: Optional[Union[str, int, Sequence[int]]] = "all",
    ) -> None:
        """
        Khởi tạo nhánh Không gian EfficientNet-B0.

        Args:
            pretrained: True để nạp trọng số `EfficientNet_B0_Weights.IMAGENET1K_V1`,
                        False để khởi tạo ngẫu nhiên (dùng khi nạp lại từ file checkpoint hoặc test offline).
            dropout: Tỷ lệ Dropout trước tầng phân loại tuyến tính (mặc định 0.2 theo Bảng 1).
            trainable_stages: Quy định các block trong `self.features` (0..8) sẽ được mở khóa ở Pha B:
                - `"all"` hoặc `None`: Tinh chỉnh toàn bộ 9 block `0..8` (cấu hình chính thức).
                - `"none"` hoặc `0`  : Đóng băng toàn bộ thân mạng, chỉ học `classifier`.
                - Số nguyên dương `k` (1..9): Chỉ mở khóa `k` block sâu nhất cuối mạng (ví dụ `k=2` -> `[7, 8]`).
                - Danh sách chỉ số `Sequence[int]`: Mở khóa chính xác các block được liệt kê (ví dụ `(6, 7, 8)`).
        """
        super().__init__()

        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout phải nằm trong [0.0, 1.0), nhận được: {dropout}")

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        # Chỉ giữ lại phần trích xuất đặc trưng không gian (9 stages: 0..8) và bộ gộp trung bình thích nghi.
        # Loại bỏ hoàn toàn `base_model.classifier` (1000 lớp ImageNet, 1.28M tham số thừa) để tiết kiệm VRAM
        # và tránh tham số mồ côi (grad=None) trong bộ tối ưu.
        self.features: nn.Sequential = base_model.features
        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))

        # Đầu phân loại nhị phân riêng cho nhánh Không gian (1280 -> 1 logit)
        self.classifier = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.feature_dim, 1),
        )

        # Lưu danh sách các stage được phép học khi mở khóa Pha B và áp dụng ngay trạng thái ban đầu
        self._active_stage_indices: Tuple[int, ...] = self._parse_trainable_stages(trainable_stages)
        self.freeze_backbone(freeze=(len(self._active_stage_indices) == 0))

    @classmethod
    def _parse_trainable_stages(
        cls, trainable_stages: Optional[Union[str, int, Sequence[int]]]
    ) -> Tuple[int, ...]:
        """Chuẩn hóa tham số `trainable_stages` thành tuple các chỉ số block hợp lệ thuộc `0..8`."""
        all_indices = tuple(range(cls.NUM_STAGES))

        if trainable_stages is None:
            return all_indices

        if isinstance(trainable_stages, str):
            mode = trainable_stages.strip().lower()
            if mode == "all":
                return all_indices
            if mode in ("none", "frozen", "head_only"):
                return ()
            raise ValueError(
                f"Chuỗi trainable_stages không hợp lệ: {trainable_stages!r}. "
                f"Hỗ trợ 'all' hoặc 'none'."
            )

        if isinstance(trainable_stages, bool):
            raise ValueError("trainable_stages không nhận kiểu bool; hãy truyền 'all', 'none', int hoặc list[int].")

        if isinstance(trainable_stages, int):
            if trainable_stages == 0:
                return ()
            if 1 <= trainable_stages <= cls.NUM_STAGES:
                return tuple(range(cls.NUM_STAGES - trainable_stages, cls.NUM_STAGES))
            raise ValueError(
                f"Số lượng stage mở khóa phải nằm trong [0, {cls.NUM_STAGES}], nhận được: {trainable_stages}"
            )

        if isinstance(trainable_stages, Sequence):
            parsed: List[int] = []
            for idx in trainable_stages:
                if isinstance(idx, bool) or not isinstance(idx, int) or not (0 <= idx < cls.NUM_STAGES):
                    raise ValueError(
                        f"Chỉ số stage phải là số nguyên trong [0, {cls.NUM_STAGES - 1}], nhận được: {idx!r}"
                    )
                if idx not in parsed:
                    parsed.append(idx)
            return tuple(sorted(parsed))

        raise TypeError(f"Kiểu dữ liệu trainable_stages không được hỗ trợ: {type(trainable_stages)}")

    @property
    def active_stage_indices(self) -> Tuple[int, ...]:
        """Trả về tuple các chỉ số stage (`0..8`) được cấu hình mở khóa trong Pha B."""
        return self._active_stage_indices

    def set_trainable_stages(
        self, trainable_stages: Optional[Union[str, int, Sequence[int]]]
    ) -> None:
        """
        Cập nhật động cấu hình `trainable_stages` (phục vụ ablation mở 1–2 stage cuối)
        và áp dụng ngay lập tức nếu mô hình đang không ở trạng thái khóa toàn cục.
        """
        self._active_stage_indices = self._parse_trainable_stages(trainable_stages)
        if len(self._active_stage_indices) == 0:
            self.freeze_backbone(True)
        elif not self._backbone_frozen:
            self.freeze_backbone(False)

    def _set_frozen_backbone_norms_eval(self) -> None:
        """
        Ghi đè hàm bảo vệ của `BranchBase.train(True)`:
        Khi `engine.py` gọi `model.train()` ở đầu mỗi epoch, ngoài các lớp `BatchNorm2d`,
        toàn bộ khối `stage` đang đóng băng (bao gồm cả các tầng `StochasticDepth` trong MBConv)
        bắt buộc phải được giữ ở chế độ `.eval()` để không cắt ngẫu nhiên đặc trưng của phần thân cố định.
        """
        if self._backbone_frozen or len(self._active_stage_indices) == 0:
            self.features.eval()
            return

        active_set = set(self._active_stage_indices)
        for stage_idx, stage_module in enumerate(self.features):
            if stage_idx not in active_set:
                stage_module.eval()

        super()._set_frozen_backbone_norms_eval()

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Điều khiển đóng/mở băng thân mạng EfficientNet phục vụ Huấn luyện 2 Pha & Fusion:
        - `freeze=True`  (Pha A Warmup Head hoặc Giai đoạn ghép Fusion):
          Khóa 100% toàn bộ 9 block `self.features` (`requires_grad=False`) và đưa TOÀN BỘ module con
          của thân mạng (cả `BatchNorm2d` lẫn `StochasticDepth`) về `.eval()`. Chỉ `self.classifier` được học.
        - `freeze=False` (Pha B Fine-tuning):
          Mở khóa chính xác các block nằm trong `self.active_stage_indices` (mặc định toàn bộ `0..8`).
          Nếu chỉ mở một phần (ablation), các block đầu mạng không nằm trong `active_stage_indices`
          vẫn tiếp tục bị khóa cả đạo hàm lẫn chế độ `.eval()` (`BatchNorm2d` & `StochasticDepth`).
        """
        # Đầu phân loại luôn mở khóa đạo hàm khi huấn luyện nhánh
        if self.classifier is not None:
            for p in self.classifier.parameters():
                p.requires_grad = True

        if freeze or len(self._active_stage_indices) == 0:
            self._backbone_frozen = True
            for p in self.features.parameters():
                p.requires_grad = False
                p.grad = None
            self._set_frozen_backbone_norms_eval()
            return

        self._backbone_frozen = False
        active_set = set(self._active_stage_indices)

        for stage_idx, stage_module in enumerate(self.features):
            is_trainable = stage_idx in active_set
            for p in stage_module.parameters():
                p.requires_grad = is_trainable
                if not is_trainable:
                    p.grad = None
            if self.training:
                stage_module.train(is_trainable)

        # Khóa chặt mọi module thuộc các stage bị đóng băng ở .eval()
        if self.training:
            self._set_frozen_backbone_norms_eval()

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """
        Hàm công tắc tường minh (alias trực quan của `freeze_backbone`):
        - `set_backbone_trainable(False)` -> Đóng băng thân mạng (Pha A khởi động head / Giai đoạn 4 Fusion).
        - `set_backbone_trainable(True)`  -> Mở khóa thân mạng theo `trainable_stages` (Pha B tinh chỉnh).
        """
        self.freeze_backbone(freeze=not trainable)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng không gian 1280 chiều từ Tensor `view_native`.

        Args:
            x: Tensor ảnh RGB đã chuẩn hóa ImageNet, bắt buộc kích thước `(B, 3, 224, 224)`.

        Returns:
            Tensor đặc trưng 2D shape `(B, 1280)`.
            (Không bọc `torch.no_grad()` để truyền đạo hàm bình thường khi fine-tune).
        """
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[-2:] != (224, 224):
            raise ValueError(
                f"SpatialBranch yêu cầu đầu vào 4D RGB kích thước cố định (B, 3, 224, 224) từ view_native, "
                f"nhận được: {tuple(x.shape)}"
            )

        feat_map = self.features(x)          # (B, 1280, 7, 7)
        pooled = self.avgpool(feat_map)      # (B, 1280, 1, 1)
        features = torch.flatten(pooled, 1)  # (B, 1280)
        return features

    def stage_summary(self) -> Dict[str, Any]:
        """Trả về thông tin tóm tắt trạng thái các block, BatchNorm & StochasticDepth phục vụ kiểm chứng."""
        bn_total = 0
        bn_eval = 0
        sd_total = 0
        sd_eval = 0
        for m in self.features.modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                bn_total += 1
                if not m.training:
                    bn_eval += 1
            elif isinstance(m, StochasticDepth):
                sd_total += 1
                if not m.training:
                    sd_eval += 1

        unlocked_now = [
            idx for idx, blk in enumerate(self.features)
            if any(p.requires_grad for p in blk.parameters())
        ]
        return {
            "total_params": self.count_parameters(trainable_only=False),
            "trainable_params": self.count_parameters(trainable_only=True),
            "configured_stages": list(self._active_stage_indices),
            "currently_unlocked_stages": unlocked_now,
            "backbone_bn_total": bn_total,
            "backbone_bn_in_eval": bn_eval,
            "backbone_sd_total": sd_total,
            "backbone_sd_in_eval": sd_eval,
        }


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP CHO NHÁNH SPATIAL (EfficientNet.py Unit Test)
# ==============================================================================
def run_unit_tests() -> None:
    print("=" * 80)
    print("KIỂM THỬ NHÁNH KHÔNG GIAN (SPATIAL BRANCH): code/system/branch/EfficientNet.py")
    print("=" * 80)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    print(f"  + Thiết bị kiểm thử : {device.type.upper()} (bf16={use_bf16})")

    # 1a. Kiểm tra kiến trúc thuần & hợp đồng tham số (pretrained=False)
    arch_model = SpatialBranch(pretrained=False, dropout=0.2, trainable_stages="all").to(device)
    assert arch_model.input_key == "spatial"
    assert arch_model.feature_dim == 1280
    total_params = arch_model.count_parameters(trainable_only=False)
    head_params = sum(p.numel() for p in arch_model.classifier.parameters())
    backbone_params = sum(p.numel() for p in arch_model.features.parameters())
    assert head_params == 1281, f"Head Linear(1280, 1) phải có đúng 1,281 tham số, nhận được: {head_params}"
    assert total_params == backbone_params + head_params == 4_008_829, f"Tổng tham số bất thường: {total_params}"

    # Chặn cả đầu vào sai số kênh (1 kênh) lẫn đầu vào sai kích thước (256x256)
    for bad_shape in ((2, 1, 224, 224), (2, 3, 256, 256), (2, 3, 128, 128)):
        try:
            arch_model(torch.randn(*bad_shape, device=device))
            raise AssertionError(f"Phải chặn đầu vào sai chuẩn (B, 3, 224, 224): {bad_shape}!")
        except ValueError:
            pass
    print(
        f"[PASS] 1. Kiểm thử kiến trúc & chốt chặn (B, 3, 224, 224): "
        f"Tổng={total_params:,} params (Backbone={backbone_params:,} + Head={head_params:,})."
    )

    # 1b. Kiểm tra tường minh việc nạp trọng số IMAGENET1K_V1 (không che giấu lỗi bằng fallback âm thầm)
    model = SpatialBranch(pretrained=True, dropout=0.2, trainable_stages="all").to(device)
    assert model.count_parameters(trainable_only=False) == 4_008_829
    print("[PASS] 2. Kiểm thử nạp trọng số pretrained EfficientNet_B0_Weights.IMAGENET1K_V1 thành công.")

    # 3. Kiểm tra shape forward_features() -> (B, 1280) và forward() -> (B,) với cả B=4, B=1 và AMP bf16
    model.eval()
    x_dummy = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        feats = model.forward_features(x_dummy)
        logits_b4 = model(x_dummy)
        logits_b1 = model(x_dummy[:1])
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_bf16 = model(x_dummy)
            assert torch.isfinite(logits_bf16).all() and logits_bf16.shape == (4,)

    assert feats.shape == (4, 1280) and torch.isfinite(feats).all()
    assert logits_b4.shape == (4,) and torch.isfinite(logits_b4).all()
    assert logits_b1.shape == (1,) and torch.isfinite(logits_b1).all()
    print(
        f"[PASS] 3. Chuẩn kích thước đầu ra: features={tuple(feats.shape)}, "
        f"logits(B=4)={tuple(logits_b4.shape)}, logits(B=1)={tuple(logits_b1.shape)}."
    )

    # 4. Pha A (Warmup Head): Khóa 100% đạo hàm, BatchNorm2d VÀ StochasticDepth ngay cả khi gọi model.train()
    model.freeze_backbone(True)
    model.train()  # Mô phỏng engine.train_one_epoch() gọi model.train() đầu epoch
    summary_phase_a = model.stage_summary()
    sd_modules = [m for m in model.features.modules() if isinstance(m, StochasticDepth)]
    assert len(sd_modules) > 0, "Không tìm thấy StochasticDepth trong EfficientNet-B0!"
    assert summary_phase_a["trainable_params"] == 1281
    assert summary_phase_a["currently_unlocked_stages"] == []
    assert summary_phase_a["backbone_bn_in_eval"] == summary_phase_a["backbone_bn_total"] > 0
    assert summary_phase_a["backbone_sd_in_eval"] == summary_phase_a["backbone_sd_total"] == len(sd_modules)
    assert all(not m.training for m in model.features.modules()), "Backbone Pha A còn module ở chế độ .train()!"
    assert model.classifier.training, "Classifier Pha A (Dropout) phải vẫn ở chế độ .train()!"

    # Chứng minh thực nghiệm: hai lần forward_features() liên tiếp ở Pha A khi model.train() phải cho vector giống hệt nhau
    with torch.no_grad():
        f_pass1 = model.forward_features(x_dummy)
        f_pass2 = model.forward_features(x_dummy)
    assert torch.equal(f_pass1, f_pass2), "Backbone Pha A bị StochasticDepth làm nhiễu ngẫu nhiên giữa 2 lần forward!"

    groups_a = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3, weight_decay=1e-4)
    assert len(groups_a) == 1 and groups_a[0]["group_name"] == "head" and groups_a[0]["lr"] == 1e-3

    first_bn = next(m for m in model.features.modules() if isinstance(m, nn.BatchNorm2d))
    bn_mean_snap = first_bn.running_mean.detach().clone()
    bn_var_snap = first_bn.running_var.detach().clone()
    bb_weight_before = model.features[0][0].weight.detach().clone()
    head_weight_before = model.classifier[-1].weight.detach().clone()

    opt_a = torch.optim.AdamW(groups_a)
    opt_a.zero_grad(set_to_none=True)
    y_dummy = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    loss_a = nn.BCEWithLogitsLoss()(model(x_dummy * 4.0 + 2.0), y_dummy)
    loss_a.backward()

    assert all(p.grad is None for p in model.features.parameters()), "Pha A rò rỉ gradient vào backbone!"
    assert all(p.grad is not None for p in model.classifier.parameters()), "Pha A thiếu gradient ở classifier!"
    opt_a.step()

    assert torch.equal(model.features[0][0].weight, bb_weight_before), "Pha A làm thay đổi trọng số backbone!"
    assert not torch.equal(model.classifier[-1].weight, head_weight_before), "Pha A không cập nhật trọng số head!"
    assert torch.equal(first_bn.running_mean, bn_mean_snap), "Pha A làm trôi running_mean của BatchNorm!"
    assert torch.equal(first_bn.running_var, bn_var_snap), "Pha A làm trôi running_var của BatchNorm!"
    print(
        f"[PASS] 4. Pha A (Warmup Head): Khóa 100% gradient, {summary_phase_a['backbone_bn_total']} BatchNorm2d "
        f"& {len(sd_modules)} StochasticDepth ở .eval(); trọng số backbone bất biến, chỉ head cập nhật."
    )

    # 5. Pha B (Full Fine-tuning): Chuyển pha tạo lại Optimizer với Differential LR -> chứng minh backbone THỰC SỰ cập nhật
    model.set_backbone_trainable(True)
    model.train()
    summary_phase_b = model.stage_summary()
    assert summary_phase_b["trainable_params"] == 4_008_829
    assert summary_phase_b["currently_unlocked_stages"] == list(range(9))
    assert summary_phase_b["backbone_bn_in_eval"] == 0
    assert summary_phase_b["backbone_sd_in_eval"] == 0

    groups_b = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3, weight_decay=1e-4)
    assert len(groups_b) == 2
    assert groups_b[0]["group_name"] == "backbone" and groups_b[0]["lr"] == 1e-4
    assert groups_b[1]["group_name"] == "head" and groups_b[1]["lr"] == 1e-3

    bb_stage0_before = model.features[0][0].weight.detach().clone()
    bb_stage8_before = model.features[8][0].weight.detach().clone()
    opt_b = torch.optim.AdamW(groups_b)
    opt_b.zero_grad(set_to_none=True)
    loss_b = nn.BCEWithLogitsLoss()(model(x_dummy * 4.0 + 2.0), y_dummy)
    loss_b.backward()
    opt_b.step()

    assert not torch.equal(model.features[0][0].weight, bb_stage0_before), "Pha B không cập nhật stage 0 sau optimizer.step()!"
    assert not torch.equal(model.features[8][0].weight, bb_stage8_before), "Pha B không cập nhật stage 8 sau optimizer.step()!"
    assert not torch.equal(first_bn.running_mean, bn_mean_snap), "Pha B BatchNorm phải cập nhật thống kê bình thường!"
    print(
        f"[PASS] 5. Pha A -> Pha B Transition & Optimizer Step: Mở {summary_phase_b['trainable_params']:,} params "
        f"(backbone=1e-4, head=1e-3); stage 0..8 đều nhận gradient & cập nhật trọng số thực tế."
    )

    # 6. Kiểm tra chế độ Ablation `trainable_stages=2` (chỉ mở [7, 8], khóa toàn bộ BN + StochasticDepth của [0..6])
    model.set_trainable_stages(2)
    model.train()  # Gọi lại model.train() để chứng minh stage 0..6 vẫn không bị hở StochasticDepth
    summary_ablation = model.stage_summary()
    assert summary_ablation["currently_unlocked_stages"] == [7, 8]
    assert all(not m.training for s in range(7) for m in model.features[s].modules()), "Stage 0..6 còn module ở .train()!"
    assert all(
        not m.training for s in range(7) for m in model.features[s].modules() if isinstance(m, StochasticDepth)
    ), "StochasticDepth ở stage 0..6 bị rò rỉ sang .train()!"
    assert all(
        m.training for s in (7, 8) for m in model.features[s].modules() if isinstance(m, StochasticDepth)
    ), "StochasticDepth ở stage mở khóa [7, 8] phải ở .train()!"

    frozen_stage0_before = model.features[0][0].weight.detach().clone()
    active_stage8_before = model.features[8][0].weight.detach().clone()
    opt_abl = torch.optim.AdamW(model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3, weight_decay=1e-4))
    opt_abl.zero_grad(set_to_none=True)
    loss_abl = nn.BCEWithLogitsLoss()(model(x_dummy), y_dummy)
    loss_abl.backward()
    assert all(p.grad is None for s in range(7) for p in model.features[s].parameters())
    assert any(p.grad is not None for p in model.features[7].parameters())
    assert any(p.grad is not None for p in model.features[8].parameters())
    opt_abl.step()
    assert torch.equal(model.features[0][0].weight, frozen_stage0_before), "Stage 0 đóng băng bị cập nhật!"
    assert not torch.equal(model.features[8][0].weight, active_stage8_before), "Stage 8 mở khóa không cập nhật!"
    print(
        f"[PASS] 6. Ablation `trainable_stages=2`: Chỉ mở [7, 8] ({summary_ablation['trainable_params']:,} params); "
        f"khóa hoàn toàn đạo hàm, BatchNorm2d & StochasticDepth của stage 0..6."
    )

    print("\n" + "-" * 80)
    print("UNIT TESTS KIẾN TRÚC & HUẤN LUYỆN 2 PHA (E1-E4): VƯỢT QUA 6/6 [PASS]!")
    print("-" * 80)

    # 7. Kiểm thử tích hợp tách biệt với DefactifyDataset thật (báo rõ PASS hoặc SKIP)
    prep_dir = CURRENT_DIR.parent / "prepare_data"
    meta_csv = CURRENT_DIR.parents[2] / "dataset" / "Defactify_dataset" / "metadata.csv"
    if prep_dir.exists() and meta_csv.exists():
        if str(prep_dir) not in sys.path:
            sys.path.insert(0, str(prep_dir))
        from dataset import build_dataloaders

        model.set_trainable_stages("all")
        model.eval()
        _, val_loader, _, _ = build_dataloaders(
            batch_size=8,
            num_workers=0,
            active_branches=("spatial",),
            persistent_workers=False,
        )
        real_batch = next(iter(val_loader))
        x_real = real_batch[model.input_key].to(device)
        with torch.no_grad():
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    f_real = model.forward_features(x_real)
                    l_real = model(x_real)
            else:
                f_real = model.forward_features(x_real)
                l_real = model(x_real)
        assert f_real.shape == (8, 1280) and l_real.shape == (8,)
        assert torch.isfinite(f_real).all() and torch.isfinite(l_real).all()
        print("[PASS] 7. Integration Test với DefactifyDataset thật: batch['spatial'] -> (8, 1280) & logits (8,).")
    else:
        print(
            "[SKIP] 7. Integration Test với DefactifyDataset: Không tìm thấy prepare_data hoặc metadata.csv."
        )

    print("\n" + "=" * 80)
    print("HOÀN TẤT TOÀN BỘ KIỂM THỬ CỦA EfficientNet.py!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()

