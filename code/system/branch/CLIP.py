# -*- coding: utf-8 -*-
"""
code/system/branch/CLIP.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Nhánh 2 — Nhánh Ngữ nghĩa (Semantic Branch) sử dụng mô hình nền OpenAI CLIP ViT-B/16:
- Tiếp nhận tensor ảnh toàn cục từ `CLIPPreprocessor` (`batch["semantic"]`, shape `(B, 3, 224, 224)`).
- Sử dụng duy nhất phần mã hóa thị giác (`clip_model.visual`) của `ViT-B-16` (`pretrained="openai"`),
  loại bỏ hoàn toàn nhánh văn bản (Text Encoder) nhằm tiết kiệm VRAM và tránh dư thừa tính toán.
- Đóng băng 100% trọng số backbone (`always_frozen_backbone = True`) theo nguyên lý
  **Linear Probe / UnivFD** (Ojha et al., CVPR 2023), chuẩn hóa L2 vector đặc trưng 512 chiều
  và huấn luyện đầu phân loại tuyến tính duy nhất (513 tham số).
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = CURRENT_DIR.parent
DATA_DIR = SYSTEM_DIR / "prepare_data"
REPO_ROOT = CURRENT_DIR.parents[2]

for _path in (str(CURRENT_DIR), str(SYSTEM_DIR), str(DATA_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Đồng bộ danh tính lớp `BranchBase` trên mọi kiểu import (`from base`, `from branch.base`, `from code.system.branch.base`)
# để `issubclass(SemanticBranch, BranchBase)` và `isinstance(model, BranchBase)` luôn trả về True ở mọi script.
if "base" in sys.modules:
    from base import BranchBase  # noqa: E402
elif "branch.base" in sys.modules:
    from branch.base import BranchBase  # noqa: E402
    sys.modules["base"] = sys.modules["branch.base"]
elif __package__:
    from .base import BranchBase  # noqa: E402
    _base_mod = sys.modules[f"{__package__}.base"]
    sys.modules.setdefault("base", _base_mod)
    sys.modules.setdefault("branch.base", _base_mod)
else:
    from base import BranchBase  # noqa: E402

_loaded_base_mod = sys.modules.get("base") or sys.modules.get("branch.base")
if _loaded_base_mod is not None:
    sys.modules.setdefault("base", _loaded_base_mod)
    sys.modules.setdefault("branch.base", _loaded_base_mod)
    sys.modules.setdefault("code.system.branch.base", _loaded_base_mod)


CLIP_DEFAULT_MODEL: str = "ViT-B-16"
CLIP_DEFAULT_PRETRAINED: str = "openai"
CLIP_INPUT_SIZE: int = 224
CLIP_FEATURE_DIM: int = 512


class SemanticBranch(BranchBase):
    """
    Nhánh trích xuất đặc trưng ngữ nghĩa toàn cục từ OpenAI CLIP Vision Transformer (`ViT-B/16`).

    Kiến trúc cố định:
        1. Visual Backbone (`self.visual`): `VisionTransformer` (`ViT-B/16`, `QuickGELU`),
           đóng băng 100% tham số (`requires_grad=False`) và luôn duy trì ở chế độ `.eval()`.
        2. Chuẩn hóa đặc trưng (`l2_normalize=True`): Ép kiểu `float32` và chuẩn hóa L2
           về siêu cầu đơn vị (`||f||_2 = 1`) với `eps=1e-12` để ổn định cho Linear Probe và Fusion.
        3. Linear Probe Head (`self.classifier`): `Dropout(p) -> Linear(512, 1)` gồm 513 tham số.
    """

    input_key: str = "semantic"
    feature_dim: int = CLIP_FEATURE_DIM
    always_frozen_backbone: bool = True

    def __init__(
        self,
        model_name: str = CLIP_DEFAULT_MODEL,
        pretrained: Optional[Union[str, bool]] = CLIP_DEFAULT_PRETRAINED,
        dropout: float = 0.0,
        l2_normalize: bool = True,
        force_quick_gelu: Optional[bool] = None,
    ) -> None:
        """
        Khởi tạo nhánh CLIP SemanticBranch.

        Args:
            model_name: Tên kiến trúc CLIP trong `open_clip` (mặc định `"ViT-B-16"`).
            pretrained: Tag trọng số tiền huấn luyện (mặc định `"openai"`).
                Hỗ trợ cả kiểu `bool` để đồng bộ với `SpatialBranch`:
                - `True` hoặc `"openai"` -> nạp trọng số OpenAI `"openai"`.
                - `False` hoặc `None`    -> khởi tạo ngẫu nhiên không nạp trọng số.
            dropout: Tỷ lệ Dropout trước tầng tuyến tính Linear Probe (mặc định `0.0` theo UnivFD).
            l2_normalize: Có chuẩn hóa L2 vector đặc trưng 512 chiều sau backbone hay không.
            force_quick_gelu: Ép dùng hàm kích hoạt `QuickGELU`. Nếu `None`, tự động đọc từ
                cấu hình pretrained (với OpenAI `ViT-B-16` là `True` để khớp 100% trọng số gốc).
        """
        super().__init__()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"`model_name` phải là chuỗi khác rỗng, nhận được: {model_name!r}")

        resolved_pretrained = self._normalize_pretrained_arg(pretrained)

        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError(f"`dropout` phải là số thực trong [0.0, 1.0), nhận được: {dropout!r}")
        dropout_val = float(dropout)
        if not math.isfinite(dropout_val) or not (0.0 <= dropout_val < 1.0):
            raise ValueError(f"`dropout` phải nằm trong nửa khoảng [0.0, 1.0), nhận được: {dropout_val}")

        self.model_name = model_name.strip()
        self.pretrained = resolved_pretrained
        self.dropout_rate = dropout_val
        self.l2_normalize = bool(l2_normalize)
        self.use_quick_gelu = self._resolve_quick_gelu_flag(
            model_name=self.model_name,
            pretrained=self.pretrained,
            force_quick_gelu=force_quick_gelu,
        )

        self.visual = self._load_visual_backbone(
            model_name=self.model_name,
            pretrained=self.pretrained,
            use_quick_gelu=self.use_quick_gelu,
            expected_dim=self.feature_dim,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.feature_dim, 1, bias=True),
        )
        self._init_classifier_weights()

        # Đóng băng cứng 100% nhánh thị giác, xóa grad rác và khóa mọi module ngoài classifier ở `.eval()`.
        self.freeze_backbone(freeze=True)

    @staticmethod
    def _normalize_pretrained_arg(pretrained: Optional[Union[str, bool]]) -> Optional[str]:
        """Chuẩn hóa tham số `pretrained` (hỗ trợ cả `bool` lẫn `str` / `None`)."""
        if isinstance(pretrained, bool):
            return CLIP_DEFAULT_PRETRAINED if pretrained else None
        if pretrained is None:
            return None
        if isinstance(pretrained, str) and pretrained.strip():
            return pretrained.strip()
        raise ValueError(
            f"`pretrained` phải là `None`, `bool` hoặc chuỗi khác rỗng, nhận được: {pretrained!r}"
        )

    @staticmethod
    def _import_open_clip() -> Any:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Không tìm thấy thư viện `open_clip`. Hãy cài đặt bằng `pip install open-clip-torch`."
            ) from exc
        return open_clip

    @classmethod
    def _resolve_quick_gelu_flag(
        cls,
        model_name: str,
        pretrained: Optional[str],
        force_quick_gelu: Optional[bool],
    ) -> bool:
        """Xác định chính xác có dùng QuickGELU hay không để tránh lệch trọng số OpenAI CLIP."""
        if force_quick_gelu is not None:
            return bool(force_quick_gelu)
        if pretrained is not None:
            open_clip = cls._import_open_clip()
            pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
            return bool(pretrained_cfg.get("quick_gelu", False))
        # Khi khởi tạo ngẫu nhiên (`pretrained=None`), giữ cấu hình QuickGELU đồng nhất với OpenAI ViT-B-16.
        return "vit-b-16" in model_name.lower()

    @classmethod
    def _load_visual_backbone(
        cls,
        model_name: str,
        pretrained: Optional[str],
        use_quick_gelu: bool,
        expected_dim: int,
    ) -> nn.Module:
        """Tải mô hình CLIP từ `open_clip`, tách riêng `visual` và giải phóng nhánh văn bản."""
        open_clip = cls._import_open_clip()
        clip_model = open_clip.create_model(
            model_name=model_name,
            pretrained=pretrained,
            force_quick_gelu=use_quick_gelu,
        )
        if not hasattr(clip_model, "visual"):
            raise RuntimeError(f"Mô hình `open_clip` ({model_name!r}) không chứa thuộc tính `visual`.")

        visual = clip_model.visual
        # Gỡ liên kết và xóa clip_model ngay để GC thu hồi toàn bộ ~63.4M tham số Text Encoder.
        del clip_model

        visual_out_dim = getattr(visual, "output_dim", None)
        if visual_out_dim is not None and int(visual_out_dim) != expected_dim:
            raise ValueError(
                f"Chiều đặc trưng đầu ra của `visual` ({visual_out_dim}) "
                f"không khớp với `feature_dim` quy định ({expected_dim})."
            )
        return visual

    def _init_classifier_weights(self) -> None:
        """Khởi tạo tầng Linear Probe bằng Kaiming Normal (linear) và bias = 0.0."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="linear",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _set_frozen_backbone_norms_eval(self) -> None:
        """Giữ toàn bộ `self.visual` (LayerNorm, QuickGELU, Attention) ở `.eval()` khi `model.train()` chạy."""
        super()._set_frozen_backbone_norms_eval()
        if hasattr(self, "visual") and self.visual is not None:
            self.visual.eval()

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Khóa cứng 100% đạo hàm của `self.visual`, xóa sạch `p.grad = None`,
        giữ `self.classifier` mở khóa (`requires_grad=True`) và đặt `self.visual` ở `.eval()`.
        Nếu gọi `freeze_backbone(False)` sẽ bị chặn cứng bởi `RuntimeError`.
        """
        super().freeze_backbone(freeze=freeze)
        if hasattr(self, "visual") and self.visual is not None:
            for p in self.visual.parameters():
                p.requires_grad = False
                p.grad = None
            self.visual.eval()
        if self.classifier is not None:
            for p in self.classifier.parameters():
                p.requires_grad = True

    def set_backbone_trainable(self, trainable: bool = False) -> None:
        """Giao diện đồng bộ với nhánh Spatial/Spectral; luôn chặn mở khóa CLIP backbone."""
        self.freeze_backbone(freeze=not bool(trainable))

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Đầu vào của SemanticBranch phải là torch.Tensor, nhận được: {type(x).__name__}")
        if not torch.is_floating_point(x):
            raise TypeError(
                f"Đầu vào của SemanticBranch phải là tensor số thực đã chuẩn hóa CLIP, nhận dtype: {x.dtype}"
            )
        if x.ndim != 4:
            raise ValueError(
                f"Đầu vào của SemanticBranch phải có 4 chiều (B, 3, {CLIP_INPUT_SIZE}, {CLIP_INPUT_SIZE}), "
                f"nhận được shape: {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError(f"Batch size phải >= 1, nhận được shape: {tuple(x.shape)}")
        if tuple(x.shape[1:]) != (3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE):
            raise ValueError(
                f"SemanticBranch yêu cầu ảnh chuẩn CLIP shape (B, 3, {CLIP_INPUT_SIZE}, {CLIP_INPUT_SIZE}), "
                f"nhận được shape: {tuple(x.shape)}"
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng ngữ nghĩa 512 chiều từ ảnh toàn cục.

        Lưu ý tiền xử lý: Hàm này giả định `x` đã được chuẩn hóa sẵn bởi `CLIPPreprocessor`
        (Shortest-Edge Resize 224 Bicubic -> CenterCrop 224 -> CLIP Mean/Std), không resize lại.

        Args:
            x: Tensor đầu vào shape `(B, 3, 224, 224)`.

        Returns:
            Tensor đặc trưng shape `(B, 512)`, luôn ở kiểu `torch.float32`.
            Nếu `self.l2_normalize=True`, mỗi vector hàng có chuẩn Euclid `||f||_2 == 1`.
        """
        self._validate_input(x)

        if self.visual.training:
            self.visual.eval()

        # Tắt đồ thị gradient trên 86.2M tham số ViT-B/16 để tối ưu VRAM và tốc độ.
        with torch.no_grad():
            features = self.visual(x)

        # Ép kiểu về float32 trước chuẩn hóa L2 để tránh mất chính xác hoặc underflow eps dưới AMP fp16/bf16.
        features = features.float()

        if features.ndim != 2 or features.shape != (int(x.shape[0]), self.feature_dim):
            raise RuntimeError(
                f"Đầu ra bất thường từ CLIP visual backbone: kỳ vọng {(int(x.shape[0]), self.feature_dim)}, "
                f"nhận được {tuple(features.shape)}"
            )

        if self.l2_normalize:
            features = F.normalize(features, p=2.0, dim=-1, eps=1e-12)

        return features

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Phân loại trực tiếp từ vector đặc trưng 512 chiều đã trích xuất / lưu đệm (phục vụ Linear Probe trên cache).

        Args:
            features: Tensor đặc trưng shape `(B, 512)`.

        Returns:
            Tensor logit 1 chiều shape `(B,)`, bảo toàn `(1,)` khi `B = 1`.
        """
        if self.classifier is None:
            raise RuntimeError("SemanticBranch chưa khởi tạo `self.classifier`.")
        if not isinstance(features, torch.Tensor) or not torch.is_floating_point(features):
            raise TypeError("`features` đưa vào `forward_head` phải là Tensor số thực.")
        if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"`forward_head` yêu cầu `features` có shape (B, {self.feature_dim}) với B >= 1, "
                f"nhận được: {tuple(features.shape)}"
            )
        logits = self.classifier(features)
        return logits.squeeze(1) if logits.ndim == 2 and logits.shape[1] == 1 else logits.view(-1)

    def stage_summary(self) -> Dict[str, Any]:
        """Trả về thông tin tóm tắt cấu trúc, tham số và trạng thái chế độ của nhánh CLIP."""
        backbone_total = sum(p.numel() for p in self.visual.parameters())
        backbone_trainable = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.classifier.parameters())
        head_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)

        return {
            "branch_name": "SemanticBranch (CLIP)",
            "input_key": self.input_key,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "quick_gelu": self.use_quick_gelu,
            "feature_dim": self.feature_dim,
            "l2_normalize": self.l2_normalize,
            "dropout": self.dropout_rate,
            "always_frozen_backbone": self.always_frozen_backbone,
            "total_params": self.count_parameters(trainable_only=False),
            "trainable_params": self.count_parameters(trainable_only=True),
            "backbone_params": backbone_total,
            "backbone_trainable_params": backbone_trainable,
            "classifier_params": head_total,
            "classifier_trainable_params": head_trainable,
            "visual_training_mode": self.visual.training,
            "classifier_training_mode": self.classifier.training,
        }


CLIPBranch = SemanticBranch

__all__ = [
    "CLIP_DEFAULT_MODEL",
    "CLIP_DEFAULT_PRETRAINED",
    "CLIP_FEATURE_DIM",
    "CLIP_INPUT_SIZE",
    "CLIPBranch",
    "SemanticBranch",
    "run_unit_tests",
]


# ==============================================================================
# UNIT TEST SUITE (TỰ ĐỘNG KIỂM CHỨNG NHÁNH CLIP)
# ==============================================================================

def _test_contract_and_weights(model: SemanticBranch, captured_warnings: list[warnings.WarningMessage]) -> None:
    print("\n[Test 1] Kiểm tra hợp đồng BranchBase, trọng số OpenAI CLIP & QuickGELU...")
    from base import BranchBase as BaseDirect
    from branch.base import BranchBase as BaseFromPkg

    assert CLIPBranch is SemanticBranch
    assert issubclass(SemanticBranch, BranchBase)
    assert issubclass(SemanticBranch, BaseDirect)
    assert issubclass(SemanticBranch, BaseFromPkg)
    assert isinstance(model, BaseDirect) and isinstance(model, BaseFromPkg)
    assert model.input_key == "semantic"
    assert model.feature_dim == 512
    assert model.always_frozen_backbone is True
    assert not hasattr(model, "clip_model") and not hasattr(model, "transformer")

    assert SemanticBranch._normalize_pretrained_arg(True) == "openai"
    assert SemanticBranch._normalize_pretrained_arg(False) is None
    assert SemanticBranch._normalize_pretrained_arg(None) is None
    assert SemanticBranch._normalize_pretrained_arg("openai") == "openai"

    mismatch_warnings = [w for w in captured_warnings if "QuickGELU mismatch" in str(w.message)]
    assert len(mismatch_warnings) == 0, f"Phát hiện cảnh báo QuickGELU mismatch: {mismatch_warnings}"

    act_module = model.visual.transformer.resblocks[0].mlp[1]
    assert "QuickGELU" in type(act_module).__name__, f"Kỳ vọng QuickGELU, nhận được: {type(act_module)}"

    summary = model.stage_summary()
    assert summary["total_params"] == 86_193_153, f"Sai total_params: {summary['total_params']}"
    assert summary["backbone_params"] == 86_192_640, f"Sai backbone_params: {summary['backbone_params']}"
    assert summary["backbone_trainable_params"] == 0
    assert summary["classifier_params"] == 513
    assert summary["classifier_trainable_params"] == 513
    assert summary["trainable_params"] == 513
    print(f"  -> Summary: {summary}")
    print("  [PASS] Hợp đồng BranchBase (mọi kiểu import), QuickGELU và tham số (86,192,640 + 513) chính xác 100%.")


def _test_input_validation(model: SemanticBranch) -> None:
    print("\n[Test 2] Kiểm tra từ chối đầu vào sai kích thước / sai kiểu dữ liệu...")
    invalid_inputs = [
        torch.randn(3, 224, 224),                   # thiếu chiều batch
        torch.randn(2, 3, 256, 256),                # sai kích thước ảnh
        torch.randn(2, 1, 224, 224),                # sai số kênh
        torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8),  # chưa chuẩn hóa float
        torch.randn(0, 3, 224, 224),                # batch rỗng
    ]
    for bad_x in invalid_inputs:
        raised = False
        try:
            model.forward_features(bad_x)
        except (TypeError, ValueError):
            raised = True
        assert raised, f"Không bắt được lỗi với đầu vào shape={tuple(bad_x.shape)}, dtype={bad_x.dtype}"

    for bad_feat in (torch.randn(512), torch.randn(2, 256), torch.randn(0, 512)):
        raised_head = False
        try:
            model.forward_head(bad_feat)
        except (TypeError, ValueError):
            raised_head = True
        assert raised_head, f"forward_head không bắt được lỗi shape={tuple(bad_feat.shape)}"

    bad_dropout_raised = False
    try:
        SemanticBranch(pretrained=False, dropout=1.0)
    except ValueError:
        bad_dropout_raised = True
    assert bad_dropout_raised, "Không bắt được lỗi dropout=1.0"
    print("  [PASS] Bộ kiểm tra đầu vào từ chối đúng tất cả trường hợp bất hợp lệ.")


def _test_forward_shapes_and_l2_norm(model: SemanticBranch) -> None:
    print("\n[Test 3] Kiểm tra chiều forward_features, forward, forward_head (B=4, B=1) & chuẩn hóa L2...")
    model.eval()
    for batch_size in (4, 1):
        x = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
        feats = model.forward_features(x)
        logits = model.forward(x)
        logits_head = model.forward_head(feats)

        assert feats.shape == (batch_size, 512), f"Sai feature shape: {feats.shape}"
        assert feats.dtype == torch.float32
        assert torch.isfinite(feats).all()

        norms = torch.linalg.vector_norm(feats, ord=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"L2 norm lệch: {norms}"

        assert logits.shape == (batch_size,), f"Sai logit shape với B={batch_size}: {logits.shape}"
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert torch.equal(logits, logits_head), "forward(x) và forward_head(forward_features(x)) không khớp!"

    # Kiểm tra distinctness: 2 ảnh khác nhau phải cho vector đặc trưng khác nhau
    x_a = torch.zeros(1, 3, 224, 224)
    x_b = torch.randn(1, 3, 224, 224)
    cos_sim = F.cosine_similarity(model.forward_features(x_a), model.forward_features(x_b), dim=-1).item()
    assert cos_sim < 0.999, f"Đặc trưng CLIP bị collapse (cos_sim={cos_sim})"

    # Kiểm tra nhanh khi tắt l2_normalize=False (dùng pretrained=False không tải lại trọng số)
    model_raw = SemanticBranch(pretrained=False, l2_normalize=False)
    model_raw.eval()
    feats_raw = model_raw.forward_features(torch.randn(2, 3, 224, 224))
    raw_norms = torch.linalg.vector_norm(feats_raw, ord=2, dim=-1)
    assert not torch.allclose(raw_norms, torch.ones_like(raw_norms), atol=1e-2)
    print(f"  -> cos_sim(ảnh đen vs ảnh nhiễu) = {cos_sim:.4f}, raw L2 norm = {[round(v, 4) for v in raw_norms.tolist()]}")
    print("  [PASS] Đầu ra shape (B, 512) / (B,), forward_head, dtype float32 và L2 norm = 1.0 chuẩn xác.")


def _test_kaiming_init_and_initial_loss(model: SemanticBranch) -> None:
    print("\n[Test 4] Kiểm tra khởi tạo Kaiming Normal & giá trị loss khởi điểm...")
    linear_layer = model.classifier[1]
    assert isinstance(linear_layer, nn.Linear)
    weight_std = linear_layer.weight.data.std().item()
    expected_std = 1.0 / math.sqrt(512.0)  # ~0.04419
    assert 0.030 <= weight_std <= 0.060, f"Kaiming weight std bất thường: {weight_std:.5f} vs {expected_std:.5f}"
    assert torch.all(linear_layer.bias.data == 0.0), "Bias khởi tạo khác 0.0"

    model.eval()
    x = torch.randn(16, 3, 224, 224)
    targets = torch.cat([torch.zeros(8), torch.ones(8)])
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, targets).item()
    assert 0.60 <= loss <= 0.80, f"Loss khởi điểm bất thường (kỳ vọng ~0.6931): {loss:.4f}"
    print(f"  -> weight_std={weight_std:.5f} (lý thuyết={expected_std:.5f}), init BCE loss={loss:.4f}")
    print("  [PASS] Khởi tạo Kaiming Normal chuẩn xác, logit khởi điểm không bão hòa.")


def _test_frozen_mode_determinism_and_gradients(model: SemanticBranch) -> None:
    print("\n[Test 5] Kiểm tra khóa cứng 100% backbone, tính xác định & cách ly gradient...")
    for fn in (lambda: model.freeze_backbone(False), lambda: model.set_backbone_trainable(True)):
        blocked = False
        try:
            fn()
        except RuntimeError:
            blocked = True
        assert blocked, "Cho phép mở khóa backbone vi phạm always_frozen_backbone=True!"

    model.train(True)
    assert model.visual.training is False
    assert all(not m.training for m in model.visual.modules()), "Có submodule trong CLIP visual bị bật train()!"
    assert model.classifier.training is True

    x = torch.randn(3, 3, 224, 224)
    feat_1 = model.forward_features(x)
    feat_2 = model.forward_features(x)
    logit_1 = model.forward(x)
    logit_2 = model.forward(x)
    assert torch.equal(feat_1, feat_2), "forward_features không xác định trong chế độ train()!"
    assert torch.equal(logit_1, logit_2), "forward (dropout=0.0) không xác định trong chế độ train()!"

    # Kiểm tra nhanh với dropout > 0: visual vẫn xác định, riêng head có dropout ngẫu nhiên
    model_drop = SemanticBranch(pretrained=False, dropout=0.5)
    model_drop.train(True)
    assert torch.equal(model_drop.forward_features(x), model_drop.forward_features(x))
    diff_train = (model_drop.forward(x) - model_drop.forward(x)).abs().max().item()
    assert diff_train > 1e-6, "Dropout=0.5 trong head không hoạt động ở chế độ train()"
    model_drop.eval()
    assert torch.equal(model_drop.forward(x), model_drop.forward(x))

    # Kiểm tra backward & AdamW step
    model.zero_grad(set_to_none=True)
    param_groups = model.get_param_groups(backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-4)
    assert len(param_groups) == 1 and param_groups[0]["group_name"] == "head"
    assert sum(p.numel() for p in param_groups[0]["params"]) == 513

    visual_ref = model.visual.conv1.weight.detach().clone()
    head_ref = model.classifier[1].weight.detach().clone()

    optimizer = torch.optim.AdamW(param_groups)
    targets = torch.tensor([0.0, 1.0, 1.0])
    loss = F.binary_cross_entropy_with_logits(model(x), targets)
    loss.backward()

    for p in model.visual.parameters():
        assert p.requires_grad is False and p.grad is None, "Rò rỉ gradient vào CLIP visual!"
    for p in model.classifier.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()

    optimizer.step()
    assert torch.equal(model.visual.conv1.weight.detach(), visual_ref), "Trọng số CLIP visual bị thay đổi!"
    head_delta = (model.classifier[1].weight.detach() - head_ref).abs().max().item()
    assert head_delta > 0.0, "Trọng số Linear Probe head không cập nhật!"
    print(f"  -> Max thay đổi visual={0.0:.1f}, max thay đổi head={head_delta:.6f}")
    print("  [PASS] Backbone bất biến 100%, 0 rò rỉ gradient, chỉ cập nhật đúng 513 tham số head.")


def _test_amp_safety(model: SemanticBranch) -> None:
    print("\n[Test 6] Kiểm tra an toàn Mixed Precision (AMP fp16/bf16 -> float32 features)...")
    model.eval()
    x_cpu = torch.randn(2, 3, 224, 224)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        feats_cpu = model.forward_features(x_cpu)
        logits_cpu = model.forward(x_cpu)
    assert feats_cpu.dtype == torch.float32 and torch.isfinite(feats_cpu).all()
    assert torch.isfinite(logits_cpu).all()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        x_gpu = x_cpu.to(device)
        for amp_dt in (torch.float16, torch.bfloat16):
            if amp_dt == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dt):
                feats_gpu = model.forward_features(x_gpu)
                logits_gpu = model.forward(x_gpu)
            assert feats_gpu.dtype == torch.float32 and torch.isfinite(feats_gpu).all()
            assert torch.allclose(
                torch.linalg.vector_norm(feats_gpu, ord=2, dim=-1),
                torch.ones(2, device=device, dtype=torch.float32),
                atol=1e-4,
            )
            assert torch.isfinite(logits_gpu).all()
        model.to("cpu")
        torch.cuda.empty_cache()
        print("  -> Đã kiểm chứng thành công CPU bfloat16 và CUDA fp16/bf16 autocast.")
    else:
        print("  -> Không có CUDA, đã kiểm chứng CPU bfloat16 autocast.")
    print("  [PASS] Đầu ra chuẩn hóa L2 giữ độ chính xác float32 an toàn dưới AMP.")


def _test_dataset_and_extract_integration(model: SemanticBranch) -> None:
    print("\n[Test 7] Kiểm tra tích hợp DefactifyDataset (build_dataloaders & build_extract_loader)...")
    from dataset import DEFAULT_DATASET_ROOT, build_dataloaders, build_extract_loader

    if not Path(DEFAULT_DATASET_ROOT).exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset tại: {DEFAULT_DATASET_ROOT}")

    train_loader, _, _, _ = build_dataloaders(
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=4,
        num_workers=0,
        active_branches=("semantic",),
        persistent_workers=False,
    )
    batch = next(iter(train_loader))
    assert "semantic" in batch and "label_a" in batch

    x = batch["semantic"]
    y = batch["label_a"].float()
    assert x.shape == (4, 3, 224, 224) and x.dtype == torch.float32

    model.eval()
    feats = model.forward_features(x)
    logits = model.forward(x)
    loss = F.binary_cross_entropy_with_logits(logits, y)

    norms = torch.linalg.vector_norm(feats, ord=2, dim=-1)
    assert feats.shape == (4, 512) and torch.isfinite(feats).all()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert logits.shape == (4,) and torch.isfinite(logits).all()
    assert torch.isfinite(loss) and loss.item() > 0.0

    # Kiểm tra kết nối với build_extract_loader (chế độ lưu đệm gốc & lật ngang phục vụ extract_features.py)
    ext_orig = build_extract_loader("val", force_flip=False, active_branches=("semantic",), batch_size=4, num_workers=0)
    ext_flip = build_extract_loader("val", force_flip=True, active_branches=("semantic",), batch_size=4, num_workers=0)
    b_orig = next(iter(ext_orig))
    b_flip = next(iter(ext_flip))
    assert b_orig["filepath"] == b_flip["filepath"]
    assert b_orig["sample_idx"].tolist() == b_flip["sample_idx"].tolist()
    with torch.inference_mode():
        f_orig = model.forward_features(b_orig["semantic"])
        f_flip = model.forward_features(b_flip["semantic"])
    flip_diff = float((f_orig - f_flip).abs().max().item())
    assert flip_diff > 1e-4, "Đặc trưng ảnh lật ngang trùng hệt ảnh gốc!"

    print(f"  -> Train batch: input={tuple(x.shape)}, feats={tuple(feats.shape)}, logits={tuple(logits.shape)}, BCE={loss.item():.4f}")
    print(f"  -> Extract loader (orig vs flip): khớp sample_idx={b_orig['sample_idx'].tolist()}, max|f_orig - f_flip|={flip_diff:.4f}")
    print("  [PASS] Tích hợp DefactifyDataset (Train/Val + Extract Cache Loader) hoạt động trơn tru 100%.")


def run_unit_tests() -> None:
    """Chạy toàn bộ 7 bộ kiểm thử đơn vị và tích hợp cho nhánh CLIP (SemanticBranch)."""
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ ĐƠN VỊ NHÁNH CLIP (SemanticBranch - ViT-B/16)")
    print("=" * 80)

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(min(2, max(1, prev_threads)))
    try:
        torch.manual_seed(42)
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            model = SemanticBranch(
                model_name=CLIP_DEFAULT_MODEL,
                pretrained=CLIP_DEFAULT_PRETRAINED,
                dropout=0.0,
                l2_normalize=True,
            )

        _test_contract_and_weights(model, captured_warnings)
        _test_input_validation(model)
        _test_forward_shapes_and_l2_norm(model)
        _test_kaiming_init_and_initial_loss(model)
        _test_frozen_mode_determinism_and_gradients(model)
        _test_amp_safety(model)
        _test_dataset_and_extract_integration(model)

        print("\n" + "=" * 80)
        print("TẤT CẢ 7 NHÓM KIỂM THỬ NHÁNH CLIP ĐỀU PASS 100%!")
        print("=" * 80)
    finally:
        torch.set_num_threads(prev_threads)


if __name__ == "__main__":
    run_unit_tests()
