# -*- coding: utf-8 -*-
"""
code/system/branch/base.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Module lớp cơ sở trừu tượng (Abstract Base Class) cho toàn bộ các nhánh mô hình:
- SpatialBranch    (EfficientNet.py) : Trích xuất đặc trưng không gian (1280 chiều)
- SemanticBranch   (CLIP.py)         : Trích xuất đặc trưng ngữ nghĩa (512 chiều)
- SpectralBranch   (FFT.py)          : Trích xuất đặc trưng phổ tần số (256 chiều)

Quy định hợp đồng giao diện chung (Unified Interface Contract):
1. `input_key: str`        : Tên khóa trích xuất Tensor đầu vào từ batch của DefactifyDataset
                             ("spatial" | "semantic" | "spectral").
2. `feature_dim: int`      : Số chiều của vector đặc trưng đầu ra (1280 | 512 | 256).
3. `forward_features(x)`   : Trả về Tensor vector đặc trưng shape (B, feature_dim).
                             LƯU Ý: Không bọc `torch.no_grad()` ở lớp cơ sở để các nhánh trainable
                             (EfficientNet, FFT) có thể lan truyền đạo hàm bình thường.
4. `forward(x)`            : Trả về logit phân loại nhị phân shape (B,) (1 logit cho mỗi ảnh),
                             tương thích trực tiếp với BCEWithLogitsLoss và nhãn `batch["label_a"]`,
                             an toàn kể cả khi kích thước lô B = 1.
5. `count_parameters()`    : Tiện ích đếm tham số mô hình (toàn bộ hoặc chỉ trainable).
6. `freeze_backbone()`     : Tiện ích đóng/mở băng phần thân mạng phục vụ huấn luyện 2 pha.
7. `get_param_groups()`    : Tách nhóm tham số backbone / head cho Differential Learning Rate.
"""

import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class BranchBase(nn.Module, ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các nhánh nơ-ron đơn lẻ trong kiến trúc Defactify.

    Hỗ trợ an toàn trạng thái đóng băng:
    - `always_frozen_backbone`: Đặt `True` cho nhánh cố định 100% (như CLIP) để chặn mọi lệnh mở khóa nhầm.
    - Tự động giữ các lớp chuẩn hóa (`BatchNorm` / `_NormBase`) thuộc vùng tham số đang bị đóng băng
      ở chế độ `.eval()` ngay cả khi gọi `model.train()`, ngăn rò rỉ cập nhật `running_mean` / `running_var`.
    """

    VALID_INPUT_KEYS = ("spatial", "semantic", "spectral")

    input_key: str
    feature_dim: int
    always_frozen_backbone: bool = False

    def __init__(self) -> None:
        super().__init__()

        # Kiểm tra hợp đồng thuộc tính bắt buộc ngay khi khởi tạo lớp con
        if not hasattr(self, "input_key") or not isinstance(self.input_key, str) or self.input_key not in self.VALID_INPUT_KEYS:
            raise ValueError(
                f"Lớp {self.__class__.__name__} phải khai báo `input_key` thuộc {self.VALID_INPUT_KEYS}, "
                f"nhận được: {getattr(self, 'input_key', None)!r}"
            )
        if (
            not hasattr(self, "feature_dim")
            or isinstance(self.feature_dim, bool)
            or not isinstance(self.feature_dim, int)
            or self.feature_dim <= 0
        ):
            raise ValueError(
                f"Lớp {self.__class__.__name__} phải khai báo `feature_dim` là số nguyên dương (> 0), "
                f"nhận được: {getattr(self, 'feature_dim', None)!r}"
            )

        # Lớp phân loại độc lập cho từng nhánh: mapping feature_dim -> 1 logit
        # Các lớp con sẽ khởi tạo hoặc tùy biến classifier này (ví dụ: kèm Dropout)
        self.classifier: Optional[nn.Module] = None
        self._backbone_frozen: bool = bool(self.always_frozen_backbone)

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng biểu diễn mức cao từ Tensor đầu vào.

        Args:
            x: Tensor đầu vào tương ứng với nhánh, shape (B, C, H, W).
               - Spatial  : (B, 3, 224, 224) từ `view_native`
               - Semantic : (B, 3, 224, 224) từ `view_global`
               - Spectral : (B, 1, 224, 224) từ phổ tần số 2D FFT

        Returns:
            Tensor đặc trưng 2 chiều shape (B, feature_dim).
        """
        raise NotImplementedError("Mọi lớp nhánh con bắt buộc phải cài đặt phương thức `forward_features()`.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi phân loại nhị phân độc lập cho từng nhánh (phục vụ train lẻ).

        Args:
            x: Tensor đầu vào của nhánh, shape (B, C, H, W).

        Returns:
            Tensor logit 1 chiều shape (B,).
            Giá trị dương biểu thị nghiêng về AI, giá trị âm nghiêng về Real.
            Có thể tính trực tiếp `bce_loss(logits, batch["label_a"])` không lo lệch chiều,
            kể cả khi B = 1 (không bị co thành 0-d scalar).
        """
        if self.classifier is None:
            raise RuntimeError(
                f"Lớp {self.__class__.__name__} chưa khởi tạo `self.classifier`. "
                f"Hãy định nghĩa lớp phân loại trong `__init__` của nhánh con."
            )

        batch_size = int(x.shape[0])
        features = self.forward_features(x)  # Shape (B, feature_dim)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"{self.__class__.__name__}.forward_features(x) phải trả về shape (B, {self.feature_dim}), "
                f"nhận được: {tuple(features.shape)}"
            )
        if features.shape[0] != batch_size:
            raise ValueError(
                f"Batch dimension không khớp tại {self.__class__.__name__}.forward_features(x): "
                f"input={batch_size}, features={features.shape[0]}"
            )

        logits = self.classifier(features)

        # Chuẩn hóa an toàn về đúng 1 chiều (B,), bảo toàn shape (1,) khi B = 1
        if logits.ndim == 2 and logits.shape == (batch_size, 1):
            return logits.squeeze(1)
        if logits.ndim == 1 and logits.shape[0] == batch_size:
            return logits

        raise ValueError(
            f"{self.__class__.__name__}.classifier phải trả về shape ({batch_size}, 1) hoặc ({batch_size},), "
            f"nhận được: {tuple(logits.shape)}"
        )

    def _set_frozen_backbone_norms_eval(self) -> None:
        """
        Khóa trạng thái thống kê động (`running_mean`, `running_var`) cho các module thuộc vùng đóng băng:
        - Nếu `always_frozen_backbone=True`: đặt toàn bộ module ngoài `self.classifier` về `.eval()`.
        - Với nhánh có thể tinh chỉnh (như EfficientNet): đặt mọi lớp `_NormBase` ngoài `self.classifier`
          đang bị khóa đạo hàm (`_backbone_frozen=True` hoặc mọi tham số của lớp đó có `requires_grad=False`)
          về `.eval()` để không làm trôi thống kê BatchNorm khi `model.train()` được gọi mỗi epoch.
        """
        classifier_modules = set(self.classifier.modules()) if self.classifier is not None else set()

        if self.always_frozen_backbone:
            for m in self.children():
                if m not in classifier_modules:
                    m.eval()
            return

        for m in self.modules():
            if m is self or m in classifier_modules:
                continue
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m_params = list(m.parameters(recurse=False))
                is_norm_frozen = self._backbone_frozen or (
                    len(m_params) > 0 and all(not p.requires_grad for p in m_params)
                )
                if is_norm_frozen:
                    m.eval()

    def train(self, mode: bool = True) -> "BranchBase":
        """
        Ghi đè `nn.Module.train(mode)` để đảm bảo khi `engine.py` gọi `model.train()` ở đầu mỗi epoch,
        các lớp `BatchNorm` (hoặc backbone cố định như CLIP) đang đóng băng vẫn giữ nguyên chế độ `.eval()`.
        """
        super().train(mode)
        if mode:
            self._set_frozen_backbone_norms_eval()
        return self

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Đếm số lượng tham số của mô hình.

        Args:
            trainable_only: Nếu True, chỉ đếm các tham số có requires_grad=True.
                            Nếu False, đếm toàn bộ tham số của mô hình.

        Returns:
            Số lượng tham số (int).
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Tiện ích bật/tắt cập nhật trọng số VÀ thống kê BatchNorm cho phần thân (backbone), giữ nguyên classifier.
        - Nếu lớp con đặt `always_frozen_backbone = True` (ví dụ CLIP), gọi `freeze_backbone(False)` sẽ bị chặn cứng.

        Args:
            freeze: True để khóa đạo hàm & thống kê động của backbone, False để mở khóa.
        """
        if self.always_frozen_backbone and not freeze:
            raise RuntimeError(
                f"Nhánh {self.__class__.__name__} được cấu hình `always_frozen_backbone=True` "
                f"(khóa cứng 100%), không được phép gọi `freeze_backbone(False)`!"
            )

        self._backbone_frozen = bool(freeze)
        classifier_params = set(self.classifier.parameters()) if self.classifier is not None else set()
        for p in self.parameters():
            if p not in classifier_params:
                p.requires_grad = not freeze

        if freeze:
            self._set_frozen_backbone_norms_eval()
        elif self.training:
            # Khi mở khóa lại toàn bộ backbone và mô hình đang ở chế độ train, bật lại train() cho các lớp con
            super().train(True)

    def get_param_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """
        Tạo danh sách parameter groups tách biệt tốc độ học cho backbone và head (classifier),
        chỉ thu thập các tham số đang có `requires_grad=True`.
        Phục vụ trực tiếp huấn luyện 2 pha của EfficientNet (backbone_lr=1e-4, head_lr=1e-3).
        """
        classifier_params_set = set(self.classifier.parameters()) if self.classifier is not None else set()
        backbone_params = [
            p for p in self.parameters() if p.requires_grad and p not in classifier_params_set
        ]
        head_params = [
            p for p in self.parameters() if p.requires_grad and p in classifier_params_set
        ]

        param_groups: List[Dict[str, Any]] = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "group_name": "backbone",
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "group_name": "head",
            })
        return param_groups


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP CHO GIAO DIỆN CHUNG (BranchBase Unit Test)
# ==============================================================================
def run_unit_tests() -> None:
    """
    Kiểm tra tính đúng đắn của hợp đồng giao diện BranchBase thông qua mô hình giả lập.
    """
    print("=" * 80)
    print("KIỂM THỬ GIAO DIỆN CHUNG: code/system/branch/base.py")
    print("=" * 80)

    class DummyBranch(BranchBase):
        input_key = "spatial"
        feature_dim = 1280

        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(16)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                self.bn,
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, self.feature_dim),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.feature_dim, 1),
            )

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)

    model = DummyBranch()

    # 1. Kiểm tra thuộc tính bắt buộc & chặn khai báo sai hợp đồng
    assert model.input_key == "spatial", f"input_key sai: {model.input_key}"
    assert model.feature_dim == 1280, f"feature_dim sai: {model.feature_dim}"

    class BadKeyBranch(BranchBase):
        input_key = "invalid_key"
        feature_dim = 256

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            return x

    try:
        BadKeyBranch()
        raise AssertionError("Phải chặn input_key không hợp lệ!")
    except ValueError:
        pass
    print("[PASS] 1. Kiểm tra thuộc tính bắt buộc & chặn khai báo sai hợp đồng thành công.")

    # 2. Kiểm tra forward_features() & chặn sai lệch batch dimension
    dummy_input = torch.randn(4, 3, 224, 224)
    features = model.forward_features(dummy_input)
    assert features.shape == (4, 1280), f"Shape features sai: {features.shape}, kỳ vọng (4, 1280)"
    assert features.dtype == torch.float32

    class BadBatchDimBranch(BranchBase):
        input_key = "spatial"
        feature_dim = 1280

        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Linear(1280, 1)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # Cố tình trả về đúng feature_dim=1280 nhưng sai batch_size (B + 1)
            return torch.zeros(x.shape[0] + 1, self.feature_dim)

    try:
        BadBatchDimBranch()(dummy_input)
        raise AssertionError("Phải chặn lỗi lệch batch dimension!")
    except ValueError:
        pass
    print(f"[PASS] 2. forward_features(x) trả về đúng shape {tuple(features.shape)} & chặn lệch batch dimension.")

    # 3. Kiểm tra forward() trả về shape (B,) kể cả khi B = 4 và B = 1
    model.eval()
    logits = model(dummy_input)
    assert logits.shape == (4,), f"Shape logits sai: {logits.shape}, kỳ vọng (4,)"
    logits_b1 = model(dummy_input[:1])
    assert logits_b1.shape == (1,), f"Shape logits khi B=1 bị lỗi scalar: {logits_b1.shape}, kỳ vọng (1,)"
    model.train()
    logits_train = model(dummy_input)
    print(f"[PASS] 3. forward(x) trả về đúng 1D {tuple(logits.shape)} và bảo toàn (1,) khi B=1.")

    # 4. Kiểm tra tương thích BCEWithLogitsLoss (không bọc no_grad ở lớp cơ sở)
    dummy_labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits_train, dummy_labels)
    assert loss.ndim == 0 and torch.isfinite(loss), "Loss không hợp lệ!"
    loss.backward()
    assert any(p.grad is not None for p in model.backbone.parameters()), "Backbone không nhận được gradient!"
    print(f"[PASS] 4. BCEWithLogitsLoss và backward() qua backbone thành công, loss = {loss.item():.4f}.")

    # 5. Kiểm tra freeze_backbone(True) khóa cứng cả BatchNorm running_mean / running_var ngay cả khi model.train()
    total_params = model.count_parameters(trainable_only=False)
    trainable_before = model.count_parameters(trainable_only=True)
    assert total_params == trainable_before, "Ban đầu toàn bộ tham số phải trainable!"

    groups_full = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3)
    assert len(groups_full) == 2 and groups_full[0]["lr"] == 1e-4 and groups_full[1]["lr"] == 1e-3

    model.freeze_backbone(freeze=True)
    model.train()  # Mô phỏng engine gọi model.train() ở đầu epoch
    assert not model.bn.training, "BatchNorm thuộc vùng frozen vẫn ở chế độ train()!"
    assert model.classifier.training, "Classifier phải vẫn ở chế độ train()!"
    bn_mean_before = model.bn.running_mean.clone()
    bn_var_before = model.bn.running_var.clone()
    _ = model(torch.randn(4, 3, 224, 224) * 5.0 + 3.0)
    assert torch.equal(model.bn.running_mean, bn_mean_before), "running_mean của BatchNorm bị trôi khi backbone đang frozen!"
    assert torch.equal(model.bn.running_var, bn_var_before), "running_var của BatchNorm bị trôi khi backbone đang frozen!"

    trainable_after_freeze = model.count_parameters(trainable_only=True)
    expected_classifier_params = sum(p.numel() for p in model.classifier.parameters())
    assert trainable_after_freeze == expected_classifier_params
    groups_frozen = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3)
    assert len(groups_frozen) == 1 and groups_frozen[0]["group_name"] == "head"
    print(
        f"[PASS] 5. freeze_backbone(True) khóa cả đạo hàm lẫn BatchNorm running stats khi model.train(): "
        f"{trainable_before:,} -> {trainable_after_freeze:,} params."
    )

    # 6. Kiểm tra mở khóa lại & kiểm tra chốt chặn always_frozen_backbone=True (dành cho CLIP)
    model.freeze_backbone(freeze=False)
    model.train()
    assert model.bn.training, "BatchNorm phải quay lại train() sau khi unfreeze!"
    assert model.count_parameters(trainable_only=True) == total_params

    class LockedClipMock(BranchBase):
        input_key = "semantic"
        feature_dim = 512
        always_frozen_backbone = True

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(16, 512)
            self.classifier = nn.Linear(512, 1)
            self.freeze_backbone(True)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    locked_mock = LockedClipMock()
    try:
        locked_mock.freeze_backbone(False)
        raise AssertionError("Phải chặn lệnh unfreeze trên nhánh có always_frozen_backbone=True!")
    except RuntimeError:
        pass
    print("[PASS] 6. freeze_backbone(False) mở khóa chuẩn xác & khóa an toàn cho nhánh cố định (CLIP).")

    print("\n" + "=" * 80)
    print("TẤT CẢ UNIT TESTS CỦA base.py ĐÃ VƯỢT QUA 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
