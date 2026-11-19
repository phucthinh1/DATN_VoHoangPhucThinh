# -*- coding: utf-8 -*-
"""
code/system/branch/Fusion_mlp.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Tầng Hợp nhất Đa nhánh (Multi-Branch Feature Fusion MLP — Stage 2):
- Vai trò: Tiếp nhận các vector đặc trưng đã chuẩn hóa/trích xuất từ tối đa 3 nhánh độc lập:
    1. Nhánh Spatial  (`EfficientNet-B0`)        : 1280 chiều
    2. Nhánh Semantic (`OpenAI CLIP ViT-B/16`)   :  512 chiều (đã chuẩn hóa L2, biên độ ~ ±0.04)
    3. Nhánh Spectral (`Residual Spectral CNN`)  :  256 chiều
- Kiến trúc chuẩn Baseline (Mục 5 & Mục 8.1 #12, Mục 8.2 #2 của Chot_Cau_Hinh_Buoc_5B_3_Nhanh.md):
    1. LayerNorm RIÊNG cho từng nhánh (`LayerNorm(1280)`, `LayerNorm(512)`, `LayerNorm(256)`)
       trước khi nối để cân bằng thang đo, tránh nhánh CLIP bị nhánh Spatial 1280-D lấn át.
    2. Branch Dropout tùy chọn (`branch_dropout_p`, MẶC ĐỊNH = 0.0 cho Baseline sạch; khi bật > 0
       chỉ tác dụng lúc `.train()` và luôn giữ tối thiểu `min_active_branches=2` nhánh).
    3. Ghép vector đặc trưng (`1280 + 512 + 256 = 2048` chiều khi dùng đủ 3 nhánh).
    4. Mạng MLP phân loại:
       `Linear(total_dim, 512) -> GELU -> Dropout(0.3) -> Linear(512, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)`
       Xuất ra đúng 1 logit `(B,)` phục vụ `BCEWithLogitsLoss`.
- Hỗ trợ chọn tập con nhánh (`active_branches`) để chạy công bằng toàn bộ 7 tổ hợp Ablation
  trong `code/system/train/train_fusion.py`.
"""

from __future__ import annotations

import itertools
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# HẰNG SỐ KIẾN TRÚC CHUẨN (ĐỒNG BỘ VỚI base.py & extract_features.py)
# ==============================================================================
BRANCH_ORDER: Tuple[str, ...] = ("spatial", "semantic", "spectral")

BRANCH_FEATURE_DIMS: Dict[str, int] = {
    "spatial": 1280,
    "semantic": 512,
    "spectral": 256,
}

# Tỷ lệ Dropout mặc định của Classifier Head riêng từng nhánh (Bảng 1 — Chot_Cau_Hinh_Buoc_5B_3_Nhanh.md)
BRANCH_DEFAULT_DROPOUTS: Dict[str, float] = {
    "spatial": 0.2,
    "semantic": 0.0,
    "spectral": 0.3,
}

# 7 tổ hợp nhánh không rỗng phục vụ nghiên cứu cắt tỉa (Ablation Study)
ALL_ABLATION_COMBINATIONS: Tuple[Tuple[str, ...], ...] = tuple(
    comb
    for r in range(1, len(BRANCH_ORDER) + 1)
    for comb in itertools.combinations(BRANCH_ORDER, r)
)


def normalize_branch_selection(
    branches: Optional[Union[str, Sequence[str]]],
    allowed_dims: Mapping[str, int] = BRANCH_FEATURE_DIMS,
) -> Tuple[str, ...]:
    """
    Chuẩn hóa danh sách nhánh được chọn theo thứ tự cố định `BRANCH_ORDER`
    (`spatial`, `semantic`, `spectral`), đồng thời chặn tên nhánh sai hoặc trùng lặp.
    """
    if branches is None:
        return BRANCH_ORDER

    if isinstance(branches, str):
        raw_items = [branches]
    elif isinstance(branches, Sequence):
        raw_items = list(branches)
    else:
        raise TypeError(
            f"`active_branches` phải là chuỗi hoặc Sequence[str], nhận được: {type(branches).__name__}"
        )

    if len(raw_items) == 0:
        raise ValueError("`active_branches` không được rỗng; phải chọn ít nhất 1 nhánh.")

    seen: List[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            raise TypeError(f"Tên nhánh phải là str, nhận được: {item!r}")
        key = item.strip().lower()
        if key not in allowed_dims:
            raise ValueError(
                f"Nhánh '{item}' không hợp lệ! Các nhánh hợp lệ: {BRANCH_ORDER}."
            )
        if key in seen:
            raise ValueError(f"Danh sách `active_branches` bị lặp nhánh '{key}': {raw_items}")
        seen.append(key)

    selected_set = set(seen)
    return tuple(br for br in BRANCH_ORDER if br in selected_set)


def _comb_exact_keep_prob(num_branches: int, min_active: int, drop_p: float) -> float:
    """
    Tính chính xác tuyệt đối xác suất biên `q = P(nhánh j được giữ)` khi áp dụng
    Branch Dropout Bernoulli(`1 - drop_p`) kèm ràng buộc giữ tối thiểu `min_active` nhánh.

    Nhờ chia lại phần giữ cho `q` lúc train (`x_j * mask_j / q`), kỳ vọng toán học
    `E[x_j'] == x_j` giữa `.train()` và `.eval()` được bảo toàn 100%, không lệch biên độ.
    """
    if drop_p <= 0.0 or num_branches <= min_active:
        return 1.0

    keep_p = 1.0 - drop_p
    expected_kept_count = 0.0
    for r in range(num_branches + 1):
        prob_r = math.comb(num_branches, r) * (keep_p ** r) * (drop_p ** (num_branches - r))
        effective_r = max(r, min_active)
        expected_kept_count += prob_r * float(effective_r)

    return expected_kept_count / float(num_branches)


class FusionMLP(nn.Module):
    """
    Mạng Hợp nhất Đặc trưng Đa nhánh (Multi-Branch Feature Fusion MLP).

    Luồng xử lý:
        1. Kiểm tra chặt shape, batch size và tính hữu hạn của các tensor đặc trưng đầu vào.
        2. Chuẩn hóa từng nhánh riêng biệt qua `self.branch_norms[branch]` (`nn.LayerNorm(dim)`).
        3. (Nếu `self.training` và `branch_dropout_p > 0` và `len(active_branches) > min_active_branches`):
           Ngẫu nhiên che bớt nhánh theo từng mẫu trong batch, luôn giữ >= `min_active_branches`
           nhánh hoạt động và nhân bù tỷ lệ nghịch `1 / q` để giữ nguyên kỳ vọng đầu vào.
        4. Nối vector các nhánh theo đúng thứ tự chuẩn `BRANCH_ORDER` thành `(B, total_feature_dim)`.
        5. Đẩy qua các khối ẩn `Linear -> GELU -> Dropout` (`512 -> 128`) để tạo vector hợp nhất
           `penultimate` 128-D (`forward_features`) và lớp cuối `Linear(128, 1)` trả về logit `(B,)`.
    """

    def __init__(
        self,
        active_branches: Optional[Union[str, Sequence[str]]] = BRANCH_ORDER,
        branch_dims: Optional[Mapping[str, int]] = None,
        hidden_dims: Sequence[int] = (512, 128),
        hidden_dropouts: Sequence[float] = (0.3, 0.2),
        branch_dropout_p: float = 0.0,
        min_active_branches: int = 2,
        layer_norm_eps: float = 1e-5,
        input_key: str = "fusion",
    ) -> None:
        """
        Khởi tạo FusionMLP.

        Args:
            active_branches: Tập các nhánh đầu vào (mặc định: cả 3 nhánh
                `("spatial", "semantic", "spectral")`). Hỗ trợ chọn tập con cho Ablation 7 tổ hợp.
            branch_dims: Bảng ánh xạ số chiều đặc trưng của từng nhánh
                (mặc định `{"spatial": 1280, "semantic": 512, "spectral": 256}`).
            hidden_dims: Kích thước các tầng ẩn MLP (mặc định `(512, 128)` -> `2048->512->128->1`).
            hidden_dropouts: Tỷ lệ Dropout sau mỗi tầng ẩn (mặc định `(0.3, 0.2)`).
            branch_dropout_p: Xác suất rơi nhánh khi train (mặc định `0.0` cho Baseline;
                biến thể ablation `0.15` giữ tối thiểu `min_active_branches` nhánh).
            min_active_branches: Số nhánh tối thiểu luôn giữ lại khi bật Branch Dropout (mặc định `2`).
            layer_norm_eps: Epsilon của `LayerNorm` từng nhánh (mặc định `1e-5`).
            input_key: Tên khóa trích xuất Tensor đầu vào từ batch khi kết nối trực tiếp
                với `engine.py` (`train_one_epoch`, `evaluate`, `fit`, `sanity_check_gate`),
                mặc định `"fusion"`.
        """
        super().__init__()

        if not isinstance(input_key, str) or not input_key.strip():
            raise ValueError(f"`input_key` phải là chuỗi khác rỗng, nhận được: {input_key!r}")

        resolved_dims: Dict[str, int] = dict(BRANCH_FEATURE_DIMS)
        if branch_dims is not None:
            for k, v in branch_dims.items():
                key = str(k).strip().lower()
                if key not in BRANCH_ORDER:
                    raise ValueError(f"Nhánh '{k}' trong `branch_dims` không thuộc {BRANCH_ORDER}.")
                if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                    raise ValueError(f"Chiều đặc trưng của nhánh '{key}' phải là số nguyên > 0, nhận được: {v!r}")
                resolved_dims[key] = int(v)

        self.active_branches: Tuple[str, ...] = normalize_branch_selection(
            active_branches, allowed_dims=resolved_dims
        )
        self.branch_dims: Dict[str, int] = {
            br: resolved_dims[br] for br in self.active_branches
        }
        self.total_feature_dim: int = sum(self.branch_dims.values())
        self.input_feature_dim: int = self.total_feature_dim
        self.input_key: str = input_key.strip()

        # Kiểm tra siêu tham số tầng ẩn & Dropout
        if not isinstance(hidden_dims, Sequence) or len(hidden_dims) == 0:
            raise ValueError("`hidden_dims` phải là danh sách khác rỗng các số nguyên dương, VD: (512, 128).")
        if isinstance(hidden_dropouts, Sequence) and len(hidden_dropouts) != len(hidden_dims):
            raise ValueError(
                f"Độ dài `hidden_dropouts` ({len(hidden_dropouts)}) phải bằng "
                f"`hidden_dims` ({len(hidden_dims)})."
            )
        if not isinstance(hidden_dropouts, Sequence):
            raise TypeError("`hidden_dropouts` phải là Sequence[float].")

        parsed_hidden: List[int] = []
        for d in hidden_dims:
            if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
                raise ValueError(f"Mỗi giá trị trong `hidden_dims` phải > 0, nhận được: {d!r}")
            parsed_hidden.append(int(d))

        parsed_drops: List[float] = []
        for dp in hidden_dropouts:
            if isinstance(dp, bool) or not isinstance(dp, (int, float)):
                raise TypeError(f"Dropout phải là số thực, nhận được: {dp!r}")
            dp_val = float(dp)
            if not math.isfinite(dp_val) or not (0.0 <= dp_val < 1.0):
                raise ValueError(f"Mỗi giá trị trong `hidden_dropouts` phải thuộc [0.0, 1.0), nhận được: {dp_val}")
            parsed_drops.append(dp_val)

        if isinstance(branch_dropout_p, bool) or not isinstance(branch_dropout_p, (int, float)):
            raise TypeError(f"`branch_dropout_p` phải là số thực, nhận được: {branch_dropout_p!r}")
        bdp_val = float(branch_dropout_p)
        if not math.isfinite(bdp_val) or not (0.0 <= bdp_val < 1.0):
            raise ValueError(f"`branch_dropout_p` phải thuộc [0.0, 1.0), nhận được: {bdp_val}")

        if isinstance(min_active_branches, bool) or not isinstance(min_active_branches, int) or min_active_branches < 1:
            raise ValueError(
                f"`min_active_branches` phải là số nguyên >= 1, nhận được: {min_active_branches!r}"
            )

        if (
            isinstance(layer_norm_eps, bool)
            or not isinstance(layer_norm_eps, (int, float))
            or not math.isfinite(float(layer_norm_eps))
            or float(layer_norm_eps) <= 0.0
        ):
            raise ValueError(f"`layer_norm_eps` phải là số thực dương (> 0), nhận được: {layer_norm_eps!r}")

        self.hidden_dims: Tuple[int, ...] = tuple(parsed_hidden)
        self.hidden_dropouts: Tuple[float, ...] = tuple(parsed_drops)
        self.branch_dropout_p: float = bdp_val
        self.min_active_branches: int = int(min_active_branches)
        self.layer_norm_eps: float = float(layer_norm_eps)

        # Xác suất giữ nhánh hiệu dụng cho scaling ngược khi bật Branch Dropout
        self._effective_min_active: int = min(self.min_active_branches, len(self.active_branches))
        if self.branch_dropout_p > 0.0 and len(self.active_branches) <= self._effective_min_active:
            warnings.warn(
                f"Branch Dropout (`branch_dropout_p={self.branch_dropout_p}`) sẽ KHÔNG rơi nhánh nào vì "
                f"số nhánh đang bật ({len(self.active_branches)}: {self.active_branches}) <= "
                f"min_active_branches ({self.min_active_branches}).",
                RuntimeWarning,
                stacklevel=2,
            )
        self._branch_keep_prob: float = _comb_exact_keep_prob(
            num_branches=len(self.active_branches),
            min_active=self._effective_min_active,
            drop_p=self.branch_dropout_p,
        )

        # 1. Tầng LayerNorm RIÊNG cho từng nhánh (tránh đặc trưng CLIP L2-norm bị chìm)
        self.branch_norms = nn.ModuleDict(
            {
                br: nn.LayerNorm(self.branch_dims[br], eps=self.layer_norm_eps)
                for br in self.active_branches
            }
        )

        # 2. Phần thân MLP (từ total_feature_dim -> hidden_dims[-1], VD: 2048 -> 512 -> 128)
        mlp_layers: List[nn.Module] = []
        in_dim = self.total_feature_dim
        for h_dim, drop_rate in zip(self.hidden_dims, self.hidden_dropouts):
            mlp_layers.append(nn.Linear(in_dim, h_dim, bias=True))
            mlp_layers.append(nn.GELU())
            if drop_rate > 0.0:
                mlp_layers.append(nn.Dropout(p=drop_rate))
            in_dim = h_dim
        self.feature_mlp: nn.Sequential = nn.Sequential(*mlp_layers)

        # 3. Đầu phân loại tuyến tính 1 logit (128 -> 1)
        self.penultimate_dim: int = in_dim
        self.output_feature_dim: int = in_dim
        # `feature_dim` thống nhất với hợp đồng BranchBase: chiều đầu ra của `forward_features()` (= 128)
        self.feature_dim: int = in_dim
        self.classifier: nn.Linear = nn.Linear(self.penultimate_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Khởi tạo trọng số chuẩn xác:
        - `LayerNorm`   : weight = 1.0, bias = 0.0.
        - Tầng ẩn MLP   : Kaiming Normal (`nonlinearity='relu'`, tương thích tốt với GELU), bias = 0.0.
        - `classifier`  : Kaiming Normal (`nonlinearity='linear'`, std = 1/sqrt(fan_in)), bias = 0.0
                          để BCE Loss khởi đầu ổn định quanh ln(2) ≈ 0.6931.
        """
        for norm in self.branch_norms.values():
            if isinstance(norm, nn.LayerNorm):
                if norm.weight is not None:
                    nn.init.constant_(norm.weight, 1.0)
                if norm.bias is not None:
                    nn.init.constant_(norm.bias, 0.0)

        for module in self.feature_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def _split_concatenated_tensor(self, concat_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Tách một Tensor 2D ghép sẵn `(B, total_feature_dim)` (theo đúng thứ tự `self.active_branches`)
        về lại từ điển đặc trưng từng nhánh `{br: Tensor(B, dim)}` để mỗi nhánh vẫn đi qua
        đúng `LayerNorm` riêng và `Branch Dropout` trước khi nối lại.
        """
        if not torch.is_floating_point(concat_tensor):
            raise TypeError(
                f"Tensor đầu vào của FusionMLP phải là kiểu số thực, nhận dtype: {concat_tensor.dtype}"
            )
        if concat_tensor.ndim != 2:
            raise ValueError(
                f"Tensor ghép đầu vào của FusionMLP phải là 2D (B, {self.total_feature_dim}), "
                f"nhận được shape: {tuple(concat_tensor.shape)}"
            )
        if int(concat_tensor.shape[0]) <= 0:
            raise ValueError(f"Batch size phải >= 1, nhận được shape: {tuple(concat_tensor.shape)}")
        if int(concat_tensor.shape[1]) != self.total_feature_dim:
            raise ValueError(
                f"Tổng chiều Tensor đầu vào ({concat_tensor.shape[1]}) không khớp với "
                f"`total_feature_dim` ({self.total_feature_dim}) của các nhánh {self.active_branches}."
            )
        if not torch.isfinite(concat_tensor).all():
            raise FloatingPointError("Phát hiện giá trị NaN/Inf trong Tensor đặc trưng đầu vào của FusionMLP!")

        split_sizes = [self.branch_dims[br] for br in self.active_branches]
        chunks = torch.split(concat_tensor, split_sizes, dim=1)
        return {br: chunk for br, chunk in zip(self.active_branches, chunks)}

    def _resolve_branch_inputs(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Hợp nhất đầu vào linh hoạt theo cả 3 kiểu:
          1. Dict các nhánh `features={"spatial": ..., "semantic": ..., "spectral": ...}` (hoặc cả `batch` dict).
          2. Keyword args `spatial=..., semantic=..., spectral=...`.
          3. Tensor 2D ghép sẵn `features=Tensor(B, total_feature_dim)` (tương thích trực tiếp với `engine.py`
             khi gọi `x = batch[model.input_key].to(device); logits = model(x)`).
        """
        if isinstance(features, torch.Tensor):
            if branch_kwargs:
                raise ValueError(
                    "Không được truyền đồng thời Tensor 2D ở `features` và tham số rời ở `branch_kwargs`!"
                )
            return self._split_concatenated_tensor(features)

        merged: Dict[str, torch.Tensor] = {}
        fallback_concat: Optional[torch.Tensor] = None
        if features is not None:
            if not isinstance(features, Mapping):
                raise TypeError(
                    f"`features` phải là Mapping (dict[str, Tensor]) hoặc Tensor 2D (B, {self.total_feature_dim}), "
                    f"nhận được: {type(features).__name__}. "
                    f"Nếu muốn truyền rời rạc, hãy dùng keyword args (spatial=..., semantic=..., spectral=...)."
                )
            for k, v in features.items():
                key = str(k).strip().lower()
                if key in BRANCH_ORDER and isinstance(v, torch.Tensor):
                    merged[key] = v
                elif key == self.input_key.lower() and isinstance(v, torch.Tensor):
                    fallback_concat = v

        for k, v in branch_kwargs.items():
            key = str(k).strip().lower()
            if key not in BRANCH_ORDER:
                raise TypeError(f"Tham số keyword không hợp lệ '{k}'. Chỉ hỗ trợ {BRANCH_ORDER}.")
            if key in merged:
                raise ValueError(f"Nhánh '{key}' bị truyền trùng ở cả `features` dict và keyword args!")
            merged[key] = v

        missing = [br for br in self.active_branches if br not in merged]
        if missing:
            if not merged and not branch_kwargs and fallback_concat is not None:
                return self._split_concatenated_tensor(fallback_concat)
            raise KeyError(
                f"FusionMLP thiếu đặc trưng của các nhánh bắt buộc: {missing}. "
                f"Nhánh đang kích hoạt: {self.active_branches}; nhận được: {list(merged.keys())}."
            )

        validated: Dict[str, torch.Tensor] = {}
        expected_batch: Optional[int] = None
        expected_device: Optional[torch.device] = None

        for br in self.active_branches:
            tensor = merged[br]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Đặc trưng nhánh '{br}' phải là torch.Tensor, nhận được: {type(tensor).__name__}")
            if not torch.is_floating_point(tensor):
                raise TypeError(f"Đặc trưng nhánh '{br}' phải là Tensor số thực, nhận dtype: {tensor.dtype}")
            if tensor.ndim != 2:
                raise ValueError(
                    f"Đặc trưng nhánh '{br}' phải là Tensor 2D (B, {self.branch_dims[br]}), "
                    f"nhận được shape: {tuple(tensor.shape)}"
                )
            if int(tensor.shape[0]) <= 0:
                raise ValueError(f"Batch size của nhánh '{br}' phải >= 1, nhận được: {tensor.shape[0]}")
            exp_dim = self.branch_dims[br]
            if int(tensor.shape[1]) != exp_dim:
                raise ValueError(
                    f"Chiều đặc trưng nhánh '{br}' sai lệch: kỳ vọng {exp_dim}, nhận được {tensor.shape[1]}."
                )
            if expected_batch is None:
                expected_batch = int(tensor.shape[0])
                expected_device = tensor.device
            else:
                if int(tensor.shape[0]) != expected_batch:
                    raise ValueError(
                        f"Batch size không khớp giữa các nhánh: '{self.active_branches[0]}'={expected_batch}, "
                        f"'{br}'={tensor.shape[0]}."
                    )
                if tensor.device != expected_device:
                    raise ValueError(
                        f"Device không khớp giữa các nhánh: '{self.active_branches[0]}'={expected_device}, "
                        f"'{br}'={tensor.device}."
                    )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Phát hiện giá trị NaN/Inf trong vector đặc trưng nhánh '{br}'!")

            validated[br] = tensor

        return validated

    def pack_branch_features(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Đóng gói từ điển đặc trưng các nhánh đang kích hoạt theo đúng thứ tự chuẩn `self.active_branches`
        thành một Tensor 2D ghép nối `(B, total_feature_dim)` (chưa qua LayerNorm).
        Phục vụ đóng gói `batch["fusion"]` trong `CachedFeatureDataset` để truyền thẳng qua `engine.py`.
        """
        branch_inputs = self._resolve_branch_inputs(features, **branch_kwargs)
        ordered = [branch_inputs[br] for br in self.active_branches]
        if len(ordered) == 1:
            return ordered[0]
        return torch.cat(ordered, dim=1)

    def _apply_branch_dropout(
        self,
        normed_tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Áp dụng Branch Dropout theo từng mẫu trong batch khi `self.training` và `branch_dropout_p > 0`.
        Luôn giữ tối thiểu `_effective_min_active` nhánh hoạt động cho mỗi mẫu và chia bù cho `_branch_keep_prob`.
        """
        num_branches = len(normed_tensors)
        if (
            not self.training
            or self.branch_dropout_p <= 0.0
            or num_branches <= self._effective_min_active
        ):
            return normed_tensors

        batch_size = int(normed_tensors[0].shape[0])
        device = normed_tensors[0].device

        # Mặt nạ Bernoulli độc lập (keep = True với xác suất 1 - branch_dropout_p)
        keep_mask = torch.rand((batch_size, num_branches), device=device) >= self.branch_dropout_p

        # Nếu mẫu nào bị rơi xuống dưới ngưỡng `min_active_branches`, phục hồi ngẫu nhiên đủ số nhánh
        row_counts = keep_mask.sum(dim=1)
        need_fix = row_counts < self._effective_min_active
        if bool(need_fix.any()):
            fix_rows = need_fix.nonzero(as_tuple=False).squeeze(1)
            scores = torch.rand((fix_rows.numel(), num_branches), device=device)
            # Đẩy nhánh đã chọn lên +2.0 để giữ nguyên chúng và chỉ bốc ngẫu nhiên thêm trong số nhánh bị rơi
            scores = scores + keep_mask[fix_rows].to(dtype=scores.dtype) * 2.0
            _, top_idx = torch.topk(scores, k=self._effective_min_active, dim=1)
            repaired = torch.zeros_like(keep_mask[fix_rows])
            repaired.scatter_(1, top_idx, True)
            keep_mask[fix_rows] = repaired

        inv_scale = 1.0 / self._branch_keep_prob
        out_tensors: List[torch.Tensor] = []
        for idx, t in enumerate(normed_tensors):
            mask_col = keep_mask[:, idx].to(dtype=t.dtype).unsqueeze(1) * inv_scale
            out_tensors.append(t * mask_col)
        return out_tensors

    def forward_normalized_branches(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Áp dụng LayerNorm riêng từng nhánh và trả về dict đặc trưng đã chuẩn hóa."""
        branch_inputs = self._resolve_branch_inputs(features, **branch_kwargs)
        return {
            br: self.branch_norms[br](branch_inputs[br])
            for br in self.active_branches
        }

    def forward_fused_embedding(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chuẩn hóa từng nhánh, áp dụng Branch Dropout (nếu đang train & p > 0)
        và ghép nối thành vector `(B, total_feature_dim)` (mặc định `(B, 2048)`).
        """
        branch_inputs = self._resolve_branch_inputs(features, **branch_kwargs)
        normed_list = [
            self.branch_norms[br](branch_inputs[br])
            for br in self.active_branches
        ]
        normed_list = self._apply_branch_dropout(normed_list)
        if len(normed_list) == 1:
            return normed_list[0]
        return torch.cat(normed_list, dim=1)

    def forward_features(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Trích xuất biểu diễn tầng ẩn cuối `(B, 128)` trước lớp `classifier` (hữu ích cho phân tích/PCA)."""
        fused = self.forward_fused_embedding(features, **branch_kwargs)
        return self.feature_mlp(fused)

    def forward(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lan truyền xuôi trả về logit nhị phân 1 chiều shape `(B,)`.
        An toàn tuyệt đối khi `B = 1` (trả về `(1,)`, không bị co thành 0-d scalar).
        """
        hidden = self.forward_features(features, **branch_kwargs)
        logits = self.classifier(hidden)
        return logits.squeeze(1)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Đếm tổng số tham số của mô hình FusionMLP."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> Dict[str, Any]:
        """Xuất từ điển cấu hình kiến trúc đầy đủ để lưu kèm checkpoint."""
        return {
            "model_type": "FusionMLP",
            "input_key": self.input_key,
            "active_branches": list(self.active_branches),
            "branch_dims": dict(self.branch_dims),
            "total_feature_dim": int(self.total_feature_dim),
            "input_feature_dim": int(self.input_feature_dim),
            "output_feature_dim": int(self.output_feature_dim),
            "feature_dim": int(self.feature_dim),
            "hidden_dims": list(self.hidden_dims),
            "hidden_dropouts": list(self.hidden_dropouts),
            "branch_dropout_p": float(self.branch_dropout_p),
            "min_active_branches": int(self.min_active_branches),
            "layer_norm_eps": float(self.layer_norm_eps),
            "penultimate_dim": int(self.penultimate_dim),
            "total_params": int(self.count_parameters(trainable_only=False)),
        }


class LinearProbeHead(nn.Module):
    """
    Đầu phân loại tuyến tính độc lập cho từng nhánh trên vector đặc trưng đã lưu đệm (Stage 2):
      - Nhánh `semantic` (CLIP Linear Probe, Mục 1 & Mục 3 & Mục 8.2 #4):
        `Dropout(0.0) -> Linear(512, 1)` gồm đúng **513 tham số**.
      - Nhánh `spatial`  (Head riêng EfficientNet-B0):
        `Dropout(0.2) -> Linear(1280, 1)` gồm đúng **1,281 tham số**.
      - Nhánh `spectral` (Head riêng Residual Spectral CNN):
        `Dropout(0.3) -> Linear(256, 1)` gồm đúng **257 tham số**.

    Đặc tính liên kết hệ thống:
      1. Cấu trúc `self.classifier = nn.Sequential(nn.Dropout(p), nn.Linear(feature_dim, 1))`
         có khóa `state_dict` (`classifier.1.weight`, `classifier.1.bias`) **trùng khớp 100%**
         với `SpatialBranch`, `SemanticBranch` và `SpectralBranch`.
      2. Nhờ đó, `LinearProbeHead` vừa dùng để huấn luyện `best_semantic.pth` trên cache `.pt`
         qua `engine.py`, vừa có thể nạp trực tiếp trọng số `classifier` từ `best_spatial.pth` /
         `best_spectral.pth` để đánh giá tức thì các nhánh đơn trên tập Test Clean & JPEG `90,70,50,30`
         từ cache `.pt` với sai số bằng `0.00e+00` so với chạy lại toàn bộ mạng CNN.
    """

    def __init__(
        self,
        branch: str = "semantic",
        feature_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        l2_normalize: bool = False,
        input_key: Optional[str] = None,
    ) -> None:
        super().__init__()

        if not isinstance(branch, str) or branch.strip().lower() not in BRANCH_FEATURE_DIMS:
            raise ValueError(
                f"`branch` của LinearProbeHead phải thuộc {BRANCH_ORDER}, nhận được: {branch!r}"
            )
        br = branch.strip().lower()

        if feature_dim is None:
            resolved_dim = BRANCH_FEATURE_DIMS[br]
        else:
            if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
                raise TypeError(f"`feature_dim` phải là số nguyên dương (int), nhận được: {feature_dim!r}")
            if feature_dim <= 0:
                raise ValueError(f"`feature_dim` phải là số nguyên > 0, nhận được: {feature_dim!r}")
            resolved_dim = int(feature_dim)

        if dropout is None:
            resolved_drop = BRANCH_DEFAULT_DROPOUTS[br]
        else:
            if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
                raise TypeError(f"`dropout` phải là số thực trong [0.0, 1.0), nhận được: {dropout!r}")
            resolved_drop = float(dropout)
            if not math.isfinite(resolved_drop) or not (0.0 <= resolved_drop < 1.0):
                raise ValueError(f"`dropout` phải thuộc [0.0, 1.0), nhận được: {dropout!r}")

        if not isinstance(l2_normalize, bool):
            raise TypeError(f"`l2_normalize` phải là bool (True/False), nhận được: {l2_normalize!r}")

        self.branch: str = br
        self.active_branches: Tuple[str, ...] = (br,)
        self.feature_dim: int = resolved_dim
        self.input_feature_dim: int = resolved_dim
        self.output_feature_dim: int = resolved_dim
        self.total_feature_dim: int = resolved_dim
        self.dropout_rate: float = resolved_drop
        self.l2_normalize: bool = l2_normalize
        self.input_key: str = br if (input_key is None or not str(input_key).strip()) else str(input_key).strip()

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.feature_dim, 1, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _extract_input_tensor(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        tensor: Optional[torch.Tensor] = None
        if isinstance(features, torch.Tensor):
            if branch_kwargs:
                raise ValueError(
                    "Không được truyền đồng thời Tensor ở `features` và tham số ở `branch_kwargs`!"
                )
            tensor = features
        elif isinstance(features, Mapping):
            if self.branch in features and isinstance(features[self.branch], torch.Tensor):
                tensor = features[self.branch]
            elif self.input_key in features and isinstance(features[self.input_key], torch.Tensor):
                tensor = features[self.input_key]

        if branch_kwargs:
            if self.branch in branch_kwargs:
                if tensor is not None:
                    raise ValueError(f"Nhánh '{self.branch}' bị truyền trùng ở cả `features` và `branch_kwargs`!")
                tensor = branch_kwargs[self.branch]
            else:
                raise KeyError(
                    f"LinearProbeHead('{self.branch}') chỉ nhận nhánh '{self.branch}', nhận được: {list(branch_kwargs.keys())}"
                )

        if tensor is None:
            raise KeyError(
                f"LinearProbeHead('{self.branch}') thiếu Tensor đặc trưng đầu vào cho nhánh '{self.branch}'."
            )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Đầu vào của LinearProbeHead phải là torch.Tensor, nhận: {type(tensor).__name__}")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"Đầu vào của LinearProbeHead phải là Tensor số thực, nhận dtype: {tensor.dtype}")
        if tensor.ndim != 2 or int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) != self.feature_dim:
            raise ValueError(
                f"LinearProbeHead('{self.branch}') yêu cầu Tensor 2D (B, {self.feature_dim}) với B >= 1, "
                f"nhận được shape: {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"Phát hiện giá trị NaN/Inf trong vector đặc trưng đầu vào của LinearProbeHead('{self.branch}')!"
            )
        return tensor

    def forward_features(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        x = self._extract_input_tensor(features, **branch_kwargs)
        if self.l2_normalize:
            x = F.normalize(x.float(), p=2.0, dim=-1, eps=1e-12).to(dtype=x.dtype)
        return x

    def forward(
        self,
        features: Optional[Union[Mapping[str, Any], torch.Tensor]] = None,
        **branch_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.forward_features(features, **branch_kwargs)
        logits = self.classifier(x)
        return logits.squeeze(1)

    def load_from_branch_checkpoint(
        self,
        filepath: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Nạp trọng số tầng phân loại (`classifier.1.weight`, `classifier.1.bias`) từ:
          - Checkpoint đầy đủ của nhánh Stage 1 (`best_spatial.pth`, `best_spectral.pth`) hoặc `SemanticBranch`
          - Hoặc checkpoint của chính `LinearProbeHead` (`best_semantic.pth`).
        """
        fp = Path(filepath)
        if not fp.is_file():
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {fp}")

        map_loc = torch.device(device) if device is not None else torch.device("cpu")
        raw_ckpt: Dict[str, Any] = torch.load(fp, map_location=map_loc, weights_only=False)
        state_dict = raw_ckpt.get("model_state_dict", raw_ckpt.get("state_dict", raw_ckpt))

        if "classifier.1.weight" in state_dict and "classifier.1.bias" in state_dict:
            head_state = {
                "classifier.1.weight": state_dict["classifier.1.weight"],
                "classifier.1.bias": state_dict["classifier.1.bias"],
            }
        elif "classifier.weight" in state_dict and "classifier.bias" in state_dict:
            head_state = {
                "classifier.1.weight": state_dict["classifier.weight"],
                "classifier.1.bias": state_dict["classifier.bias"],
            }
        elif "1.weight" in state_dict and "1.bias" in state_dict:
            head_state = {
                "classifier.1.weight": state_dict["1.weight"],
                "classifier.1.bias": state_dict["1.bias"],
            }
        else:
            raise KeyError(
                f"Checkpoint '{fp}' không chứa trọng số `classifier.1.weight` / `classifier.1.bias` hợp lệ!"
            )

        self.load_state_dict(head_state, strict=True)
        if device is not None:
            self.to(map_loc)
        return raw_ckpt

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "LinearProbeHead",
            "branch": self.branch,
            "input_key": self.input_key,
            "active_branches": [self.branch],
            "feature_dim": int(self.feature_dim),
            "total_feature_dim": int(self.total_feature_dim),
            "dropout": float(self.dropout_rate),
            "l2_normalize": bool(self.l2_normalize),
            "total_params": int(self.count_parameters(trainable_only=False)),
        }


def build_stage2_model(
    active_branches: Optional[Union[str, Sequence[str]]] = BRANCH_ORDER,
    head_type: str = "mlp",
    branch_dropout_p: float = 0.0,
    min_active_branches: int = 2,
    hidden_dims: Sequence[int] = (512, 128),
    hidden_dropouts: Sequence[float] = (0.3, 0.2),
    linear_dropout: Optional[float] = None,
) -> Union[FusionMLP, LinearProbeHead]:
    """
    Hàm khởi tạo thống nhất (Factory) cho `train_fusion.py` (Mục 8.1 #16 & Mục 8.2 #4):
      - `head_type="mlp"`   : Trả về `FusionMLP` (dùng cho Baseline 3 nhánh, biến thể Branch Dropout
                              và toàn bộ 7 tổ hợp Ablation khi muốn so sánh công bằng cùng loại head).
      - `head_type="linear"`: Trả về `LinearProbeHead` (chỉ áp dụng khi chọn đúng 1 nhánh, đặc biệt là
                              `--branches semantic --head linear` để huấn luyện Linear Probe 513 tham số cho CLIP).
    """
    norm_branches = normalize_branch_selection(active_branches)
    ht = str(head_type).strip().lower()
    if ht == "mlp":
        return FusionMLP(
            active_branches=norm_branches,
            hidden_dims=hidden_dims,
            hidden_dropouts=hidden_dropouts,
            branch_dropout_p=branch_dropout_p,
            min_active_branches=min_active_branches,
        )
    if ht == "linear":
        if len(norm_branches) != 1:
            raise ValueError(
                f"`head_type='linear'` (LinearProbeHead) chỉ hỗ trợ đúng 1 nhánh đơn, "
                f"nhưng nhận được {len(norm_branches)} nhánh: {norm_branches}."
            )
        return LinearProbeHead(
            branch=norm_branches[0],
            dropout=linear_dropout,
        )
    raise ValueError(f"`head_type` không hợp lệ: {head_type!r}. Chọn 'mlp' hoặc 'linear'.")


__all__ = [
    "ALL_ABLATION_COMBINATIONS",
    "BRANCH_DEFAULT_DROPOUTS",
    "BRANCH_FEATURE_DIMS",
    "BRANCH_ORDER",
    "FusionMLP",
    "LinearProbeHead",
    "build_stage2_model",
    "normalize_branch_selection",
    "run_unit_tests",
]


# ==============================================================================
# BỘ KIỂM THỬ ĐƠN VỊ ĐỘC LẬP (Fusion_mlp.py Unit Tests)
# ==============================================================================
def run_unit_tests() -> None:
    import io

    print("=" * 85)
    print("BẮT ĐẦU KIỂM THỬ ĐỘC LẬP: code/system/branch/Fusion_mlp.py (FusionMLP)")
    print("=" * 85)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # TEST 1: Kiến trúc Baseline chuẩn (3 nhánh, p=0.0) & khớp số tham số chính xác
    # -------------------------------------------------------------------------
    print("\n[Test 1] Kiểm tra kiến trúc Baseline (3 nhánh, p=0.0) & phân bổ tham số...")
    model = FusionMLP().to(device)
    assert model.active_branches == ("spatial", "semantic", "spectral")
    assert model.branch_dims == {"spatial": 1280, "semantic": 512, "spectral": 256}
    assert model.total_feature_dim == 2048
    assert model.input_feature_dim == 2048
    assert model.branch_dropout_p == 0.0
    assert model.penultimate_dim == 128
    assert model.output_feature_dim == model.feature_dim == 128

    norm_params = sum(p.numel() for p in model.branch_norms.parameters())
    mlp_params = sum(p.numel() for p in model.feature_mlp.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())
    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)

    # LayerNorm: (1280+512+256)*2 = 4,096
    # MLP: (2048*512+512) + (512*128+128) = 1,049,088 + 65,664 = 1,114,752
    # Classifier: 128*1+1 = 129
    # Tổng: 1,118,977
    assert norm_params == 4_096, f"Sai số tham số branch_norms: {norm_params}"
    assert mlp_params == 1_114_752, f"Sai số tham số feature_mlp: {mlp_params}"
    assert cls_params == 129, f"Sai số tham số classifier: {cls_params}"
    assert total_params == trainable_params == 1_118_977, f"Tổng tham số sai: {total_params}"
    cfg = model.get_config()
    assert cfg["input_feature_dim"] == cfg["total_feature_dim"] == 2048
    assert cfg["output_feature_dim"] == cfg["feature_dim"] == cfg["penultimate_dim"] == 128
    assert cfg["branch_dropout_p"] == 0.0
    print(
        f"  -> LayerNorm={norm_params:,} | MLP={mlp_params:,} | Head={cls_params:,} | "
        f"Tổng={total_params:,} tham số -> [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 2: Shape (B=32, B=4, B=1), keyword args, BCE Loss khởi tạo ~ ln(2)
    # -------------------------------------------------------------------------
    print("\n[Test 2] Kiểm tra shape đầu ra (B=32, B=4, B=1), keyword args & BCE Loss khởi tạo...")
    model.eval()
    for bs in (32, 4, 1):
        f_spa = torch.randn(bs, 1280, device=device)
        f_sem = torch.nn.functional.normalize(torch.randn(bs, 512, device=device), p=2, dim=-1)
        f_spe = torch.randn(bs, 256, device=device)

        logits_dict = model({"spatial": f_spa, "semantic": f_sem, "spectral": f_spe})
        logits_kw = model(spatial=f_spa, semantic=f_sem, spectral=f_spe)
        assert logits_dict.shape == (bs,) and logits_dict.ndim == 1
        assert torch.allclose(logits_dict, logits_kw, atol=1e-6)

        feats_128 = model.forward_features({"spatial": f_spa, "semantic": f_sem, "spectral": f_spe})
        assert feats_128.shape == (bs, 128)

    # Kiểm tra BCE loss khởi tạo quanh ln(2) = 0.6931
    y_bal = torch.cat([torch.zeros(16, device=device), torch.ones(16, device=device)])
    init_logits = model(
        {
            "spatial": torch.randn(32, 1280, device=device),
            "semantic": torch.nn.functional.normalize(torch.randn(32, 512, device=device), p=2, dim=-1),
            "spectral": torch.randn(32, 256, device=device),
        }
    )
    init_loss = float(nn.BCEWithLogitsLoss()(init_logits, y_bal).item())
    assert 0.3 < init_loss < 1.3, f"Initial BCE loss bất thường: {init_loss:.4f}"
    print(f"  -> B=1/4/32 đều trả shape (B,) chính xác | Initial BCE Loss={init_loss:.4f} -> [PASS]")

    # -------------------------------------------------------------------------
    # TEST 3: Kiểm chứng LayerNorm RIÊNG bảo vệ nhánh CLIP khỏi lệch thang đo
    # -------------------------------------------------------------------------
    print("\n[Test 3] Kiểm chứng LayerNorm riêng từng nhánh chống CLIP bị 'chìm'...")
    f_spa_raw = torch.randn(8, 1280, device=device) * 500.0 + 1000.0  # thang đo cực lớn
    f_sem_raw = torch.nn.functional.normalize(torch.randn(8, 512, device=device), p=2, dim=-1)  # ±0.04
    f_spe_raw = torch.randn(8, 256, device=device) * 0.5 - 2.0        # lệch âm rõ rệt

    normed = model.forward_normalized_branches(
        {"spatial": f_spa_raw, "semantic": f_sem_raw, "spectral": f_spe_raw}
    )
    for br_name, br_vec in normed.items():
        row_means = br_vec.mean(dim=1).abs().max().item()
        row_stds = br_vec.std(dim=1, unbiased=False)
        assert row_means < 1e-4, f"LayerNorm '{br_name}' không đưa mean về 0: {row_means}"
        assert float((row_stds - 1.0).abs().max().item()) < 1e-2, f"Std của '{br_name}' lệch khỏi 1.0"

    # Thay đổi mạnh thang đo spatial hoàn toàn không chạm tới normalized semantic/spectral
    normed_2 = model.forward_normalized_branches(
        {"spatial": torch.randn(8, 1280, device=device), "semantic": f_sem_raw, "spectral": f_spe_raw}
    )
    assert torch.allclose(normed["semantic"], normed_2["semantic"], atol=1e-7)
    assert torch.allclose(normed["spectral"], normed_2["spectral"], atol=1e-7)
    print("  -> Cả 3 nhánh đều đạt mean ≈ 0, std ≈ 1; CLIP miễn nhiễm 100% trước thang đo Spatial -> [PASS]")

    # -------------------------------------------------------------------------
    # TEST 4: Chặn nghiêm ngặt đầu vào sai / thiếu nhánh / lệch batch / NaN
    # -------------------------------------------------------------------------
    print("\n[Test 4] Kiểm tra bắt lỗi đầu vào thiếu nhánh, sai chiều, lệch batch & NaN...")
    base_batch = {
        "spatial": torch.randn(4, 1280, device=device),
        "semantic": torch.randn(4, 512, device=device),
        "spectral": torch.randn(4, 256, device=device),
    }

    bad_cases = [
        ("thiếu nhánh", {"spatial": base_batch["spatial"], "semantic": base_batch["semantic"]}, KeyError),
        ("sai feature_dim", {**base_batch, "semantic": torch.randn(4, 256, device=device)}, ValueError),
        ("tensor 1D", {**base_batch, "spectral": torch.randn(256, device=device)}, ValueError),
        ("lệch batch size", {**base_batch, "spectral": torch.randn(2, 256, device=device)}, ValueError),
        ("chứa NaN", {**base_batch, "spatial": torch.full((4, 1280), float("nan"), device=device)}, FloatingPointError),
    ]
    for case_name, bad_input, exc_type in bad_cases:
        try:
            _ = model(bad_input)
            raise AssertionError(f"Trường hợp '{case_name}' lẽ ra phải báo lỗi {exc_type.__name__}!")
        except exc_type:
            pass
    print(f"  -> Đã chặn chuẩn xác {len(bad_cases)}/{len(bad_cases)} ca lỗi đầu vào -> [PASS]")

    # -------------------------------------------------------------------------
    # TEST 5: Ablation đủ 7 tổ hợp nhánh + chuẩn hóa thứ tự nhánh
    # -------------------------------------------------------------------------
    print("\n[Test 5] Kiểm tra đủ 7 tổ hợp Ablation & chuẩn hóa thứ tự nhánh...")
    assert len(ALL_ABLATION_COMBINATIONS) == 7
    m_rev = FusionMLP(active_branches=("spectral", "spatial"))
    m_can = FusionMLP(active_branches=("spatial", "spectral"))
    assert m_rev.active_branches == m_can.active_branches == ("spatial", "spectral")
    assert m_rev.total_feature_dim == 1536

    for comb in ALL_ABLATION_COMBINATIONS:
        sub_model = FusionMLP(active_branches=comb).to(device)
        exp_dim = sum(BRANCH_FEATURE_DIMS[b] for b in comb)
        assert sub_model.total_feature_dim == exp_dim
        sub_inputs = {b: torch.randn(4, BRANCH_FEATURE_DIMS[b], device=device) for b in comb}
        sub_logits = sub_model(sub_inputs)
        assert sub_logits.shape == (4,)
        sub_logits.sum().backward()
    print(f"  -> Cả 7/7 tổ hợp Ablation đều forward/backward chuẩn xác -> [PASS]")

    # -------------------------------------------------------------------------
    # TEST 6: Branch Dropout (p=0.0 tắt hẳn vs p=0.15 giữ >= 2 nhánh & unbiased scaling)
    # -------------------------------------------------------------------------
    print("\n[Test 6] Kiểm tra Branch Dropout (Baseline p=0.0 vs Ablation p=0.15, min>=2)...")
    # p=0.0 phải bất biến khi bật/tắt Hidden Dropout = 0
    m_det = FusionMLP(hidden_dropouts=(0.0, 0.0), branch_dropout_p=0.0).to(device)
    m_det.train()
    fix_in = {
        "spatial": torch.randn(6, 1280, device=device),
        "semantic": torch.randn(6, 512, device=device),
        "spectral": torch.randn(6, 256, device=device),
    }
    out_tr1 = m_det(fix_in)
    m_det.eval()
    out_ev1 = m_det(fix_in)
    assert torch.allclose(out_tr1, out_ev1, atol=1e-6), "Baseline p=0.0 không được drop nhánh!"

    # p=0.15, min_active_branches=2 trên 2048 mẫu
    m_bdrop = FusionMLP(branch_dropout_p=0.15, min_active_branches=2).to(device)
    m_bdrop.train()
    # Truyền randn dịch tâm +1.0 để LayerNorm không triệt tiêu vector hằng thành 0
    big_rand = {
        "spatial": torch.randn(2048, 1280, device=device) + 1.0,
        "semantic": torch.randn(2048, 512, device=device) + 1.0,
        "spectral": torch.randn(2048, 256, device=device) + 1.0,
    }
    fused_tr = m_bdrop.forward_fused_embedding(big_rand)
    spa_alive = (fused_tr[:, :1280].abs().sum(dim=1) > 0)
    sem_alive = (fused_tr[:, 1280:1792].abs().sum(dim=1) > 0)
    spe_alive = (fused_tr[:, 1792:].abs().sum(dim=1) > 0)
    alive_counts = spa_alive.int() + sem_alive.int() + spe_alive.int()
    assert int(alive_counts.min().item()) >= 2, (
        f"Phát hiện mẫu còn < 2 nhánh hoạt động khi Branch Dropout! min={alive_counts.min().item()}"
    )
    dropped_some = bool((alive_counts == 2).any())
    assert dropped_some, "Branch Dropout p=0.15 phải rơi bớt 1 nhánh ở một số mẫu!"

    # Kiểm chứng xác suất sống thực nghiệm (cả trung bình lẫn biên riêng từng nhánh q_s, q_c, q_f)
    empirical_q = float((alive_counts.float() / 3.0).mean().item())
    q_spa = float(spa_alive.float().mean().item())
    q_sem = float(sem_alive.float().mean().item())
    q_spe = float(spe_alive.float().mean().item())
    assert abs(empirical_q - m_bdrop._branch_keep_prob) < 0.03, (
        f"Tỷ lệ giữ nhánh thực nghiệm ({empirical_q:.4f}) lệch kỳ vọng ({m_bdrop._branch_keep_prob:.4f})"
    )
    for br_q_name, br_q_val in (("spatial", q_spa), ("semantic", q_sem), ("spectral", q_spe)):
        assert abs(br_q_val - m_bdrop._branch_keep_prob) < 0.04, (
            f"Xác suất giữ nhánh '{br_q_name}' bị thiên lệch ({br_q_val:.4f} vs {m_bdrop._branch_keep_prob:.4f})"
        )

    # Kiểm chứng cảnh báo khi bật branch_dropout_p > 0 với <= 2 nhánh (min_active_branches=2)
    with warnings.catch_warnings(record=True) as caught_bdrop_warns:
        warnings.simplefilter("always")
        _ = FusionMLP(active_branches=("spatial", "semantic"), branch_dropout_p=0.15, min_active_branches=2)
        assert any("Branch Dropout" in str(w.message) for w in caught_bdrop_warns), (
            "Thiếu cảnh báo RuntimeWarning khi bật branch_dropout_p > 0 với <= 2 nhánh!"
        )

    m_bdrop.eval()
    fused_ev = m_bdrop.forward_fused_embedding(big_rand)
    assert bool((fused_ev.abs().sum(dim=1) > 0).all())
    print(
        f"  -> p=0.15 giữ tối thiểu {int(alive_counts.min().item())}/3 nhánh mọi mẫu | "
        f"q(spa/sem/spe)={q_spa:.3f}/{q_sem:.3f}/{q_spe:.3f} vs lý_thuyết={m_bdrop._branch_keep_prob:.4f} -> [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 7: Lưu/Nạp state_dict, tối ưu 1 bước AdamW & AMP bfloat16 (nếu có CUDA)
    # -------------------------------------------------------------------------
    print("\n[Test 7] Kiểm tra Save/Load, 1 bước tối ưu AdamW & AMP bfloat16...")
    m_train = FusionMLP().to(device)
    opt = torch.optim.AdamW(m_train.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    sample_x = {
        "spatial": torch.randn(16, 1280, device=device),
        "semantic": torch.nn.functional.normalize(torch.randn(16, 512, device=device), p=2, dim=-1),
        "spectral": torch.randn(16, 256, device=device),
    }
    sample_y = torch.cat([torch.zeros(8, device=device), torch.ones(8, device=device)])

    m_train.train()
    opt.zero_grad(set_to_none=True)
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        logits_step = m_train(sample_x)
        loss_step = crit(logits_step.float(), sample_y)
    loss_step.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(m_train.parameters(), max_norm=1.0)
    assert math.isfinite(float(grad_norm.item()))
    opt.step()

    # Kiểm tra Save/Load giữ logits giống nhau tuyệt đối ở eval()
    m_train.eval()
    with torch.no_grad():
        logits_before = m_train(sample_x)
    buf = io.BytesIO()
    torch.save({"config": m_train.get_config(), "state_dict": m_train.state_dict()}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, map_location=device, weights_only=True)
    m_reloaded = FusionMLP(
        active_branches=ckpt["config"]["active_branches"],
        branch_dims=ckpt["config"]["branch_dims"],
        hidden_dims=ckpt["config"]["hidden_dims"],
        hidden_dropouts=ckpt["config"]["hidden_dropouts"],
        branch_dropout_p=ckpt["config"]["branch_dropout_p"],
        min_active_branches=ckpt["config"]["min_active_branches"],
        layer_norm_eps=ckpt["config"]["layer_norm_eps"],
    ).to(device)
    m_reloaded.load_state_dict(ckpt["state_dict"], strict=True)
    m_reloaded.eval()
    with torch.no_grad():
        logits_after = m_reloaded(sample_x)
    max_diff = float((logits_before - logits_after).abs().max().item())
    assert max_diff == 0.0, f"Logit lệch sau Save/Load: max_diff={max_diff}"
    print(f"  -> AdamW step + Save/Load strict=True (max_diff={max_diff:.1e}, bf16={use_bf16}) -> [PASS]")

    # -------------------------------------------------------------------------
    # TEST 8: Đồng nhất 3 kiểu truyền đầu vào & Liên kết trực tiếp với engine.py
    # -------------------------------------------------------------------------
    print("\n[Test 8] Kiểm tra 3 kiểu truyền đầu vào (dict / kwargs / Tensor 2D) & tương thích engine.py...")
    m_reloaded.eval()
    packed_x = m_reloaded.pack_branch_features(sample_x)
    assert packed_x.shape == (16, 2048)
    with torch.no_grad():
        logits_from_dict = m_reloaded(sample_x)
        logits_from_kw = m_reloaded(
            spatial=sample_x["spatial"],
            semantic=sample_x["semantic"],
            spectral=sample_x["spectral"],
        )
        logits_from_tensor = m_reloaded(packed_x)
        logits_from_batch_fallback = m_reloaded({"fusion": packed_x, "label_a": sample_y})
    assert torch.equal(logits_from_dict, logits_from_kw)
    assert torch.equal(logits_from_dict, logits_from_tensor)
    assert torch.equal(logits_from_dict, logits_from_batch_fallback)

    # Kiểm chứng round-trip pack_branch_features -> _split_concatenated_tensor trên đủ 7 tổ hợp (dù dict đảo thứ tự)
    for comb in ALL_ABLATION_COMBINATIONS:
        m_comb = FusionMLP(active_branches=comb).to(device).eval()
        rev_dict = {br: sample_x[br] for br in reversed(comb)}
        packed_comb = m_comb.pack_branch_features(rev_dict)
        unpacked_comb = m_comb._split_concatenated_tensor(packed_comb)
        assert list(unpacked_comb.keys()) == list(m_comb.active_branches)
        for br in m_comb.active_branches:
            assert torch.equal(unpacked_comb[br], sample_x[br])
        with torch.no_grad():
            assert torch.equal(m_comb(rev_dict), m_comb(packed_comb))

    # Đối chiếu trực tiếp với engine.py (overfit_single_batch_check & checkpoint_consistency_check)
    train_dir = Path(__file__).resolve().parent.parent / "train"
    branch_dir = Path(__file__).resolve().parent
    for _p in (str(train_dir), str(branch_dir)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from engine import (
        _get_model_input_key,
        checkpoint_consistency_check,
        overfit_single_batch_check,
    )

    m_engine = FusionMLP().to(device)
    assert _get_model_input_key(m_engine) == "fusion"
    engine_batch = {
        "fusion": packed_x.detach().cpu(),
        "label_a": sample_y.detach().cpu(),
        "model_name": ["MS_COCO"] * 8 + ["Stable_Diffusion_XL"] * 8,
    }
    assert overfit_single_batch_check(m_engine, engine_batch, device=device, lr=3e-3, steps=100)
    assert checkpoint_consistency_check(m_engine, engine_batch, device=device)
    print("  -> Cả 3 kiểu truyền đầu vào khớp bit-for-bit; vượt qua Cổng 1 & Cổng 3 của engine.py -> [PASS]")

    # -------------------------------------------------------------------------
    # TEST 9: LinearProbeHead (validation + 513/1281/257 params) & Parity đủ 3 nhánh
    # -------------------------------------------------------------------------
    print("\n[Test 9] Kiểm tra LinearProbeHead & đối chứng khớp tuyệt đối đủ 3 nhánh (Spatial/Spectral/Semantic)...")
    probe_sem = build_stage2_model(active_branches=("semantic",), head_type="linear").to(device)
    probe_spa = LinearProbeHead(branch="spatial").to(device)
    probe_spe = LinearProbeHead(branch="spectral").to(device)
    assert isinstance(probe_sem, LinearProbeHead)
    assert probe_sem.count_parameters() == 513 and probe_sem.input_key == "semantic"
    assert probe_spa.count_parameters() == 1_281 and probe_spa.input_key == "spatial"
    assert probe_spe.count_parameters() == 257 and probe_spe.input_key == "spectral"

    # Kiểm tra chặn sai kiểu tham số ở LinearProbeHead
    for invalid_kwargs in (
        {"branch": "semantic", "feature_dim": 512.8},
        {"branch": "semantic", "l2_normalize": "false"},
        {"branch": "semantic", "dropout": "0.1"},
    ):
        try:
            _ = LinearProbeHead(**invalid_kwargs)
            raise AssertionError(f"LinearProbeHead lẽ ra phải chặn tham số sai kiểu: {invalid_kwargs}")
        except TypeError:
            pass

    from CLIP import SemanticBranch
    from EfficientNet import SpatialBranch
    from FFT import SpectralBranch

    ckpt_root = Path(__file__).resolve().parent.parent / "checkpoints"
    spa_ckpt_file = ckpt_root / "spatial_baseline" / "best_spatial.pth"
    spe_ckpt_file = ckpt_root / "spectral_baseline" / "best_spectral.pth"

    assert spa_ckpt_file.is_file(), f"Thiếu checkpoint bắt buộc để đối chứng: {spa_ckpt_file}"
    assert spe_ckpt_file.is_file(), f"Thiếu checkpoint bắt buộc để đối chứng: {spe_ckpt_file}"

    spa_full = SpatialBranch(pretrained=False, dropout=0.2).to(device).eval()
    spa_raw = torch.load(spa_ckpt_file, map_location=device, weights_only=False)
    spa_full.load_state_dict(spa_raw["model_state_dict"], strict=True)
    probe_spa.load_from_branch_checkpoint(spa_ckpt_file, device=device)
    probe_spa.eval()

    spe_full = SpectralBranch(dropout=0.3).to(device).eval()
    spe_raw = torch.load(spe_ckpt_file, map_location=device, weights_only=False)
    spe_full.load_state_dict(spe_raw["model_state_dict"], strict=True)
    probe_spe.load_from_branch_checkpoint(spe_ckpt_file, device=device)
    probe_spe.eval()

    # Đối chứng parity SemanticBranch (CLIP.py) <-> LinearProbeHead('semantic') qua checkpoint tạm
    # (do best_semantic.pth sẽ được train ở bước sau bởi train_fusion.py)
    sem_full = SemanticBranch(pretrained=False, dropout=0.0).to(device).eval()
    with tempfile.TemporaryDirectory() as tmp_sem_dir:
        tmp_sem_ckpt = Path(tmp_sem_dir) / "tmp_semantic.pth"
        torch.save({"model_state_dict": sem_full.state_dict()}, tmp_sem_ckpt)
        probe_sem.load_from_branch_checkpoint(tmp_sem_ckpt, device=device)
        probe_sem.eval()

    with torch.no_grad():
        dummy_img_spa = torch.randn(4, 3, 224, 224, device=device)
        feat_spa = spa_full.forward_features(dummy_img_spa)
        logit_spa_full = spa_full(dummy_img_spa)
        logit_spa_probe = probe_spa(feat_spa)
        spa_diff = float((logit_spa_full - logit_spa_probe).abs().max().item())
        assert spa_diff == 0.0, f"LinearProbeHead('spatial') lệch với SpatialBranch: {spa_diff}"

        dummy_img_spe = torch.randn(4, 1, 224, 224, device=device)
        feat_spe = spe_full.forward_features(dummy_img_spe)
        logit_spe_full = spe_full(dummy_img_spe)
        logit_spe_probe = probe_spe(feat_spe)
        spe_diff = float((logit_spe_full - logit_spe_probe).abs().max().item())
        assert spe_diff == 0.0, f"LinearProbeHead('spectral') lệch với SpectralBranch: {spe_diff}"

        dummy_img_sem = torch.randn(4, 3, 224, 224, device=device)
        feat_sem = sem_full.forward_features(dummy_img_sem)
        logit_sem_full = sem_full(dummy_img_sem)
        logit_sem_probe = probe_sem(feat_sem)
        sem_diff = float((logit_sem_full - logit_sem_probe).abs().max().item())
        assert sem_diff == 0.0, f"LinearProbeHead('semantic') lệch với SemanticBranch: {sem_diff}"

    print(
        f"  -> LinearProbeHead khớp 100% đủ 3 nhánh: spatial diff={spa_diff:.1e} | "
        f"spectral diff={spe_diff:.1e} | semantic diff={sem_diff:.1e} -> [PASS]"
    )

    # -------------------------------------------------------------------------
    # TEST 10: End-to-End Integration Smoke Test (extract_features -> FusionMLP/Probe -> metrics.py)
    # -------------------------------------------------------------------------
    print("\n[Test 10] End-to-End Integration Smoke Test (extract_features -> FusionMLP/Probe -> metrics.py)...")
    from extract_features import (
        DEFAULT_DATASET_ROOT,
        extract_split_features,
        load_branch_extractor,
    )
    from metrics import compute_binary_metrics, evaluate_by_generator

    ext_models = {}
    ext_ckpts = {}
    for br in BRANCH_ORDER:
        m_ext, c_info = load_branch_extractor(branch=br, device=device, allow_untrained=False)
        ext_models[br] = m_ext
        ext_ckpts[br] = c_info

    real_payloads = extract_split_features(
        models=ext_models,
        checkpoint_infos=ext_ckpts,
        split="val",
        dataset_root=DEFAULT_DATASET_ROOT,
        batch_size=8,
        num_workers=0,
        device=device,
        use_amp=use_bf16,
        amp_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        include_flip=False,
        max_batches=2,
        log_interval=0,
    )
    real_feats_dict = {
        br: real_payloads[br]["features"].to(device)
        for br in BRANCH_ORDER
    }
    real_labels = real_payloads["spatial"]["label_a"].numpy()
    real_models = real_payloads["spatial"]["model_names"]

    with torch.no_grad():
        fusion_real_logits = m_reloaded(real_feats_dict).float().cpu().numpy()
        sem_probe_logits = probe_sem(real_feats_dict["semantic"]).float().cpu().numpy()
        spa_probe_logits = probe_spa(real_feats_dict["spatial"]).float().cpu().numpy()

    m_summary = compute_binary_metrics(real_labels, spa_probe_logits, from_logits=True)
    gen_summary = evaluate_by_generator(real_labels, fusion_real_logits, real_models, from_logits=True)
    assert fusion_real_logits.shape == (16,) and sem_probe_logits.shape == (16,)
    assert "auc" in m_summary and "per_generator" in gen_summary
    print(
        f"  -> Đã nối thông suốt extract_features (16 mẫu Val thật) -> FusionMLP & LinearProbeHead -> "
        f"metrics.py (Spatial Val batch ACC={m_summary['acc']*100:.1f}%) -> [PASS]"
    )

    print("\n" + "=" * 85)
    print("TẤT CẢ 10 NHÓM KIỂM THỬ ĐƠN VỊ & LIÊN KẾT HỆ THỐNG Fusion_mlp.py ĐỀU PASS 100%!")
    print("=" * 85)


if __name__ == "__main__":
    run_unit_tests()
