# -*- coding: utf-8 -*-
"""
code/system/train/metrics.py
Đồ án tốt nghiệp: Phát hiện ảnh do AI sinh (Defactify Dataset)

Module tính toán toàn diện các thước đo đánh giá nhị phân (Real=0, AI=1):
1. `compute_binary_metrics()` : Tính toán trọn bộ chỉ số (ROC-AUC, Balanced Accuracy,
                               AP, ACC@0.5, F1, Precision, Recall, FPR) trên toàn tập dữ liệu.
2. `evaluate_by_generator()`  : Đánh giá phân rã theo từng generator (Pairwise 1:1 với 7,300 ảnh Real
                               MS_COCO) và tính trung bình Macro-Average cho nhóm Known, Unseen và All.

Chuẩn mực khoa học & thiết kế cốt lõi:
- Đánh giá trên toàn tập (Global aggregation): gom toàn bộ dự đoán trước khi tính,
  tuyệt đối không tính trung bình AUC/AP của từng batch.
- Chống bẫy lệch nhãn 1:3 ở tập Val: ưu tiên ROC-AUC (bất biến ngưỡng) và Balanced Accuracy
  (trọng số 50-50 cho 2 lớp) thay vì Raw Accuracy.
- An toàn kiểu dữ liệu AMP: hỗ trợ trực tiếp Tensor `bfloat16`/`float16`, chặn cứng `NaN` để
  không che giấu lỗi phân kỳ mô hình, và giữ nguyên độ chính xác `float64` (không làm tròn sớm).
- Ghi chú phương pháp luận (Mục 8.1 #15): Các cặp 1:1 dùng chung 7,300 ảnh Real MS_COCO làm mốc,
  do đó báo cáo theo từng generator và Macro-Average, không coi 5 cặp là 5 tập quan sát độc lập.
"""

import sys
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KNOWN_GENERATORS = [
    "Stable_Diffusion_2.1",
    "Stable_Diffusion_XL",
    "DALL-E_3",
]

UNSEEN_GENERATORS = [
    "Stable_Diffusion_3",
    "Midjourney_V6",
]

ALL_AI_GENERATORS = KNOWN_GENERATORS + UNSEEN_GENERATORS
REAL_MODEL_NAME = "MS_COCO"

METRIC_KEYS = ("auc", "balanced_acc", "ap", "acc", "f1", "precision", "recall", "fpr")


def _to_numpy(x: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> np.ndarray:
    """
    Chuyển đổi an toàn Tensor (kể cả bfloat16/float16 trên GPU) hoặc Sequence sang numpy.ndarray 1D float64.
    """
    if isinstance(x, torch.Tensor):
        # NumPy không hỗ trợ trực tiếp bfloat16 -> bắt buộc ép sang float32/float64 trước khi .numpy()
        x = x.detach().to(dtype=torch.float64).cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float64)
    return x.ravel()


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Hàm sigmoid ổn định số học tuyệt đối (không bao giờ gọi exp với số dương -> không tràn số hay cảnh báo).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr, dtype=np.float64)
    pos = x_arr >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def compute_binary_metrics(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[Union[int, float]]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[float]],
    from_logits: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """
    Tính toán toàn diện các chỉ số phân loại nhị phân (0 = Real, 1 = AI) trên toàn tập.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1), shape (N,).
        y_pred: Điểm dự đoán từ mô hình (logits hoặc probabilities), shape (N,).
        from_logits: Nếu True, y_pred là raw logits (BCE logit) và sẽ được biến đổi
                     sang xác suất bằng hàm Sigmoid. Nếu False, y_pred đã là xác suất [0, 1].
        threshold: Ngưỡng xác suất phân lớp để tính các chỉ số nhị phân cứng (mặc định 0.5).

    Returns:
        Dictionary chứa các chỉ số ở độ chính xác đầy đủ float64 (không làm tròn sớm):
        - 'auc'          : ROC-AUC (Diện tích dưới đường cong ROC)
        - 'balanced_acc' : Balanced Accuracy = (TPR + TNR) / 2
        - 'ap'           : Average Precision (Diện tích dưới đường PR)
        - 'acc'          : Standard Accuracy @ threshold
        - 'f1'           : F1-Score trên lớp AI
        - 'precision'    : Precision trên lớp AI
        - 'recall'       : Recall / True Positive Rate trên lớp AI
        - 'fpr'          : False Positive Rate = FP / (FP + TN) trên lớp Real
        - 'tp', 'fp', 'tn', 'fn': Các số đếm từ ma trận nhầm lẫn
    """
    if not np.isfinite(threshold) or not (0.0 < float(threshold) < 1.0):
        raise ValueError(f"Ngưỡng phân lớp threshold phải hữu hạn và nằm trong khoảng (0, 1), nhận được: {threshold}")

    y_t_raw = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)

    if len(y_t_raw) != len(y_p):
        raise ValueError(f"Kích thước y_true ({len(y_t_raw)}) và y_pred ({len(y_p)}) không khớp!")

    if len(y_t_raw) == 0:
        return {
            "auc": 0.0, "balanced_acc": 0.0, "ap": 0.0, "acc": 0.0,
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "num_real": 0, "num_ai": 0, "mean_prob_real": 0.0, "mean_prob_ai": 0.0,
        }

    # Chặn cứng cả NaN lẫn ±Inf để không bao giờ âm thầm chuyển ±Inf qua sigmoid thành 0/1 hay trả về AUC=0.5
    if not np.isfinite(y_t_raw).all():
        raise ValueError("y_true chứa giá trị NaN hoặc Inf!")
    if not np.isfinite(y_p).all():
        raise ValueError("y_pred chứa giá trị NaN hoặc Inf! Mô hình có dấu hiệu phân kỳ hoặc tràn số logits.")

    # Kiểm tra nhãn nhị phân hợp lệ {0, 1} (chặn lỗi truyền nhầm xác suất vào y_true)
    if not np.all((y_t_raw == 0.0) | (y_t_raw == 1.0)):
        bad_vals = np.unique(y_t_raw)[:5]
        raise ValueError(f"y_true chỉ được chứa nhãn nhị phân {{0, 1}}, phát hiện giá trị lạ: {bad_vals}")

    y_t = y_t_raw.astype(np.int64)

    # Chuyển đổi sang xác suất P(AI = 1)
    probs = sigmoid(y_p) if from_logits else np.clip(y_p, 0.0, 1.0)
    preds = (probs >= threshold).astype(np.int64)

    unique_labels = np.unique(y_t)
    has_both_classes = (len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels)

    if has_both_classes:
        auc = float(roc_auc_score(y_t, probs))
        ap = float(average_precision_score(y_t, probs))
        b_acc = float(balanced_accuracy_score(y_t, preds))
        acc = float(accuracy_score(y_t, preds))
        prec = float(precision_score(y_t, preds, zero_division=0))
        rec = float(recall_score(y_t, preds, zero_division=0))
        f1 = float(f1_score(y_t, preds, zero_division=0))
        cm = confusion_matrix(y_t, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        # Trường hợp biên tập con chỉ có đúng 1 lớp (ví dụ chạy debug 1 ảnh hoặc chỉ lọc 1 nguồn)
        auc = 0.5
        ap = float(unique_labels[0]) if len(unique_labels) == 1 else 0.0
        acc = float(np.mean(y_t == preds))
        b_acc = acc
        tp = int(((y_t == 1) & (preds == 1)).sum())
        tn = int(((y_t == 0) & (preds == 0)).sum())
        fp = int(((y_t == 0) & (preds == 1)).sum())
        fn = int(((y_t == 1) & (preds == 0)).sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    num_real = int((y_t == 0).sum())
    num_ai = int((y_t == 1).sum())
    mean_prob_real = float(probs[y_t == 0].mean()) if num_real > 0 else 0.0
    mean_prob_ai = float(probs[y_t == 1].mean()) if num_ai > 0 else 0.0

    return {
        "auc": auc,
        "balanced_acc": b_acc,
        "ap": ap,
        "acc": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "num_real": num_real,
        "num_ai": num_ai,
        "mean_prob_real": mean_prob_real,
        "mean_prob_ai": mean_prob_ai,
    }


def evaluate_by_generator(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[Union[int, float]]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[float]],
    model_names: Sequence[str],
    from_logits: bool = True,
    threshold: float = 0.5,
    require_balanced_pairs: bool = False,
    strict_known_generators: bool = False,
) -> Dict[str, Any]:
    """
    Đánh giá phân rã chi tiết hiệu năng theo từng bộ sinh AI (Generator-wise Pairwise 1:1 Evaluation):
    - Tách riêng toàn bộ ảnh Real (MS_COCO) làm neo chung để ghép cặp 1:1 với từng mô hình AI.
    - Trong tập Test: 7,300 Real vs 7,300 AI từng generator -> Tỉ lệ cân bằng 1:1 tuyệt đối
      (bật `require_balanced_pairs=True` và `strict_known_generators=True` khi chấm Test chính thức).
    - Tính Macro-Average (trên các cặp 1:1) cho nhóm Known, nhóm Unseen và toàn bộ 5 generators (macro_avg_all).
    - Tính thêm chỉ số gộp toàn tập (overall).

    Lưu ý phương pháp luận (Mục 8.1 #15 & phân biệt AP):
    - 5 cặp đánh giá dùng lại cùng 7,300 ảnh Real MS_COCO, nên báo cáo từng generator và Macro-Average,
      không coi 5 cặp là 5 tập quan sát độc lập thống kê.
    - Không so sánh trực tiếp `overall["ap"]` (tập Test gộp có tỉ lệ AI:Real = 5:1, baseline AP = 0.8333)
      với `macro_avg_all["ap"]` (trung bình 5 cặp 1:1, baseline AP = 0.5000). Luôn lấy `Macro-Avg Known`
      và `Macro-Avg Unseen` làm thước đo chính để đánh giá khả năng tổng quát hóa.
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    models = np.asarray([str(m) for m in model_names], dtype=object)

    if len(y_t) != len(models):
        raise ValueError(f"Chiều dài y_true ({len(y_t)}) và model_names ({len(models)}) không đồng nhất!")

    # 1. Đánh giá Overall gộp toàn bộ mẫu hiện có (đồng thời xác thực NaN/Inf & nhãn nhị phân {0, 1})
    overall_metrics = compute_binary_metrics(y_t, y_p, from_logits=from_logits, threshold=threshold)

    # 2. Tách mặt nạ toàn bộ ảnh Real (MS_COCO) và đối soát chéo 2 chiều giữa model_name <-> label_a
    real_mask = (models == REAL_MODEL_NAME)
    n_real = int(real_mask.sum())

    if n_real > 0 and not np.all(y_t[real_mask] == 0):
        raise ValueError(f"Lỗi toàn vẹn nhãn: '{REAL_MODEL_NAME}' bắt buộc có label_a = 0!")

    ai_mask = ~real_mask
    if int(ai_mask.sum()) > 0 and not np.all(y_t[ai_mask] == 1):
        bad_models = sorted(set(models[ai_mask & (y_t != 1)].tolist()))
        raise ValueError(f"Lỗi toàn vẹn nhãn: Các generator AI {bad_models} bắt buộc có label_a = 1!")

    # Giữ thứ tự chuẩn của 5 generators; tùy chọn chặn cứng nếu xuất hiện tên generator sai chính tả / lạ
    discovered_ai = [
        str(m) for m in dict.fromkeys(models.tolist())
        if m != REAL_MODEL_NAME and m not in ALL_AI_GENERATORS
    ]
    if strict_known_generators and discovered_ai:
        raise ValueError(
            f"Phát hiện tên generator ngoài danh mục chuẩn {ALL_AI_GENERATORS}: {discovered_ai}"
        )
    ordered_generators = ALL_AI_GENERATORS + discovered_ai

    if require_balanced_pairs and n_real == 0:
        raise ValueError("Bật require_balanced_pairs=True nhưng không có mẫu Real (MS_COCO) nào!")

    per_gen_results: Dict[str, Dict[str, Union[float, int]]] = {}

    for gen in ordered_generators:
        gen_mask = (models == gen)
        n_gen = int(gen_mask.sum())
        if n_gen == 0:
            continue

        if not np.all(y_t[gen_mask] == 1):
            raise ValueError(f"Lỗi toàn vẹn nhãn: '{gen}' bắt buộc có label_a = 1!")

        if require_balanced_pairs and n_real != n_gen:
            raise ValueError(
                f"Cặp 1:1 của generator '{gen}' không cân bằng: Real={n_real:,}, AI={n_gen:,}"
            )

        if n_real == 0:
            sub_y_t = y_t[gen_mask]
            sub_y_p = y_p[gen_mask]
        else:
            pair_mask = real_mask | gen_mask
            sub_y_t = y_t[pair_mask]
            sub_y_p = y_p[pair_mask]

        gen_metrics = compute_binary_metrics(sub_y_t, sub_y_p, from_logits=from_logits, threshold=threshold)
        if gen_metrics["num_real"] != n_real or gen_metrics["num_ai"] != n_gen:
            raise RuntimeError(
                f"Lệch số lượng mẫu nội bộ tại '{gen}': "
                f"label_counts=({gen_metrics['num_real']}, {gen_metrics['num_ai']}) vs mask_counts=({n_real}, {n_gen})"
            )
        per_gen_results[gen] = gen_metrics

    # 3. Tính Macro-Average từ các cặp 1:1 (không làm tròn kép)
    def _compute_macro_avg(gen_list: List[str]) -> Dict[str, Union[float, int]]:
        avail_gens = [g for g in gen_list if g in per_gen_results]
        if not avail_gens:
            empty_res: Dict[str, Union[float, int]] = {k: 0.0 for k in METRIC_KEYS}
            empty_res["num_generators"] = 0
            return empty_res

        avg_res: Dict[str, Union[float, int]] = {}
        for metric_k in METRIC_KEYS:
            vals = [float(per_gen_results[g][metric_k]) for g in avail_gens]
            avg_res[metric_k] = float(np.mean(vals))
        avg_res["num_generators"] = len(avail_gens)
        return avg_res

    macro_known = _compute_macro_avg(KNOWN_GENERATORS)
    macro_unseen = _compute_macro_avg(UNSEEN_GENERATORS)
    macro_all = _compute_macro_avg(list(per_gen_results.keys()))

    return {
        "per_generator": per_gen_results,
        "macro_avg_known": macro_known,
        "macro_avg_unseen": macro_unseen,
        "macro_avg_all": macro_all,
        "overall": overall_metrics,
        "shared_real_anchor_count": n_real,
    }


def format_metrics_summary(metrics: Dict[str, Any], title: str = "BẢNG CHỈ SỐ ĐÁNH GIÁ") -> str:
    """Tạo chuỗi tóm tắt kết quả đánh giá (chỉ làm tròn khi hiển thị)."""
    lines = [
        f"=== {title} ===",
        f"  + Số lượng mẫu : Real={metrics.get('num_real', 0):,} | AI={metrics.get('num_ai', 0):,}",
        f"  + P(AI) trung bình : Real={metrics.get('mean_prob_real', 0.0):.4f} | AI={metrics.get('mean_prob_ai', 0.0):.4f}",
        f"  + ROC-AUC      : {metrics.get('auc', 0.0):.4f}",
        f"  + Balanced ACC : {metrics.get('balanced_acc', 0.0) * 100:.2f}%",
        f"  + Avg Prec (AP): {metrics.get('ap', 0.0):.4f}",
        f"  + ACC@0.5      : {metrics.get('acc', 0.0) * 100:.2f}%",
        f"  + F1-Score     : {metrics.get('f1', 0.0):.4f}",
        f"  + Precision    : {metrics.get('precision', 0.0) * 100:.2f}%",
        f"  + Recall (TPR) : {metrics.get('recall', 0.0) * 100:.2f}%",
        f"  + FPR (Real)   : {metrics.get('fpr', 0.0) * 100:.2f}%",
        f"  + Ma trận      : TN={metrics.get('tn', 0):,}, FP={metrics.get('fp', 0):,}, "
        f"FN={metrics.get('fn', 0):,}, TP={metrics.get('tp', 0):,}",
    ]
    return "\n".join(lines)


def format_generator_report(gen_eval: Dict[str, Any], title: str = "BÁO CÁO CẶP 1:1 THEO GENERATOR") -> str:
    """Định dạng bảng kết quả chi tiết từng Generator và Macro-Average Known / Unseen cho Chương 4."""
    header = f"{'Generator / Nhóm':<24} | {'AUC':>8} | {'Bal-ACC':>8} | {'AP':>8} | {'ACC@0.5':>8} | {'F1':>8}"
    sep = "-" * len(header)
    lines = [f"=== {title} (Neo Real dùng chung: {gen_eval.get('shared_real_anchor_count', 0):,} ảnh) ===", header, sep]

    def _row(label: str, m: Dict[str, Any]) -> str:
        return (
            f"{label:<24} | {m.get('auc', 0.0):>8.4f} | {m.get('balanced_acc', 0.0) * 100:>7.2f}% | "
            f"{m.get('ap', 0.0):>8.4f} | {m.get('acc', 0.0) * 100:>7.2f}% | {m.get('f1', 0.0):>8.4f}"
        )

    per_gen = gen_eval.get("per_generator", {})
    for g, m in per_gen.items():
        tag = "[U] " if g in UNSEEN_GENERATORS else "[K] "
        lines.append(_row(tag + g, m))

    lines.append(sep)
    if gen_eval.get("macro_avg_known", {}).get("num_generators", 0) > 0:
        lines.append(_row("Macro-Avg Known", gen_eval["macro_avg_known"]))
    if gen_eval.get("macro_avg_unseen", {}).get("num_generators", 0) > 0:
        lines.append(_row("Macro-Avg Unseen", gen_eval["macro_avg_unseen"]))
    if gen_eval.get("macro_avg_all", {}).get("num_generators", 0) > 0:
        lines.append(_row("Macro-Avg All (1:1)", gen_eval["macro_avg_all"]))
    lines.append(_row("Overall (Pooled)", gen_eval.get("overall", {})))
    lines.append(
        "  * Lưu ý: Không so sánh trực tiếp AP của 'Overall (Pooled)' (tỉ lệ lớp dương lệch) với AP của các cặp 1:1; "
        "dùng 'Macro-Avg Known' & 'Macro-Avg Unseen' làm kết quả chính."
    )
    return "\n".join(lines)


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP CHO METRICS (metrics.py Unit Test)
# ==============================================================================
def run_unit_tests() -> None:
    print("=" * 80)
    print("KIỂM THỬ MODULE ĐÁNH GIÁ CHỈ SỐ: code/system/train/metrics.py")
    print("=" * 80)

    # 1. Kiểm tra phân tách hoàn hảo & bảo toàn độ chính xác float64
    y_true_perf = np.array([0, 0, 0, 1, 1, 1])
    y_pred_perf = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
    m_perf = compute_binary_metrics(y_true_perf, y_pred_perf)
    assert m_perf["auc"] == 1.0 and m_perf["balanced_acc"] == 1.0 and m_perf["ap"] == 1.0
    assert m_perf["acc"] == 1.0 and m_perf["fpr"] == 0.0
    print("[PASS] 1. Kiểm tra phân tách hoàn hảo (AUC = 1.0, Balanced ACC = 100%, AP = 1.0).")

    # 2. Kiểm tra bẫy lệch nhãn 1:3 (giống tập Val: 100 Real, 300 AI)
    y_true_skew = np.array([0] * 100 + [1] * 300)
    y_pred_dummy_ai = np.array([2.0] * 400)
    m_skew = compute_binary_metrics(y_true_skew, y_pred_dummy_ai)
    assert m_skew["acc"] == 0.75
    assert m_skew["balanced_acc"] == 0.5
    assert m_skew["auc"] == 0.5
    assert m_skew["fpr"] == 1.0
    print("[PASS] 2. Bẫy lệch nhãn 1:3: Raw ACC=75% nhưng Balanced ACC=50% & AUC=0.5 chuẩn xác.")

    # 3. Kiểm tra tương thích trực tiếp với Tensor torch.bfloat16 (chuẩn AMP RTX 3050) & cực trị ±1000
    bf16_logits = torch.tensor([-1000.0, -2.5, 2.5, 1000.0], dtype=torch.bfloat16)
    bf16_labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.bfloat16)
    m_bf16 = compute_binary_metrics(bf16_labels, bf16_logits)
    assert m_bf16["auc"] == 1.0 and m_bf16["balanced_acc"] == 1.0
    print("[PASS] 3. Tương thích Tensor bfloat16 & ổn định số học sigmoid tại logit ±1000 (không tràn số).")

    # 4. Kiểm tra chặn cứng NaN, +Inf, -Inf, ngưỡng sai và nhãn sai định dạng
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        try:
            compute_binary_metrics([0, 1], [0.5, bad_val])
            raise AssertionError(f"Phải báo lỗi ValueError khi y_pred chứa {bad_val}!")
        except ValueError:
            pass
    try:
        compute_binary_metrics([0.2, 0.8], [-1.0, 1.0])
        raise AssertionError("Phải báo lỗi ValueError khi y_true không phải nhị phân {0, 1}!")
    except ValueError:
        pass
    for bad_thresh in (0.0, 1.0, -0.2, 1.5, float("nan")):
        try:
            compute_binary_metrics([0, 1], [-1.0, 1.0], threshold=bad_thresh)
            raise AssertionError(f"Phải chặn threshold không hợp lệ: {bad_thresh}")
        except ValueError:
            pass
    print("[PASS] 4. Chặn cứng NaN, ±Inf, threshold ngoài (0,1) và nhãn không nhị phân thành công.")

    # 5. Kiểm tra trường hợp biên Single-class & Rỗng
    m_single = compute_binary_metrics(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0]))
    assert m_single["auc"] == 0.5 and m_single["acc"] == 1.0
    m_empty = compute_binary_metrics([], [])
    assert m_empty["tp"] == 0 and m_empty["auc"] == 0.0
    print("[PASS] 5. Trường hợp biên (Single class & Empty) an toàn, không cảnh báo rác.")

    # 6. Kiểm tra evaluate_by_generator(): ghép cặp 1:1, đối soát nhãn <-> model_name & kiểm tra cân bằng
    mock_y_true = np.array([0, 0, 1, 1, 1, 1])
    mock_y_pred = np.array([-2.0, -1.0, 2.0, 1.5, 3.0, -0.5])
    mock_models = [
        "MS_COCO", "MS_COCO",
        "Stable_Diffusion_2.1", "Stable_Diffusion_2.1",
        "Stable_Diffusion_3", "Stable_Diffusion_3",
    ]
    eval_gen = evaluate_by_generator(
        mock_y_true, mock_y_pred, mock_models,
        require_balanced_pairs=True, strict_known_generators=True,
    )
    assert eval_gen["shared_real_anchor_count"] == 2
    assert eval_gen["per_generator"]["Stable_Diffusion_2.1"]["auc"] == 1.0
    assert eval_gen["per_generator"]["Stable_Diffusion_3"]["auc"] == 1.0
    assert eval_gen["per_generator"]["Stable_Diffusion_3"]["balanced_acc"] == 0.75
    assert eval_gen["macro_avg_known"]["auc"] == 1.0
    assert eval_gen["macro_avg_unseen"]["balanced_acc"] == 0.75
    assert abs(eval_gen["macro_avg_all"]["balanced_acc"] - 0.875) < 1e-12

    # Phải báo lỗi nếu MS_COCO mang label_a=1 hoặc AI mang label_a=0
    try:
        evaluate_by_generator([0, 1, 1], [-1.0, 1.0, 2.0], ["MS_COCO", "MS_COCO", "DALL-E_3"])
        raise AssertionError("Phải chặn MS_COCO có label_a=1!")
    except ValueError:
        pass
    try:
        evaluate_by_generator([0, 0, 1], [-1.0, 1.0, 2.0], ["MS_COCO", "DALL-E_3", "DALL-E_3"])
        raise AssertionError("Phải chặn AI generator có label_a=0!")
    except ValueError:
        pass
    # Phải báo lỗi nếu bật require_balanced_pairs=True mà cặp lệch số lượng
    try:
        evaluate_by_generator(
            [0, 0, 1], [-1.0, -0.5, 1.5], ["MS_COCO", "MS_COCO", "DALL-E_3"],
            require_balanced_pairs=True,
        )
        raise AssertionError("Phải chặn cặp không cân bằng 1:1 khi require_balanced_pairs=True!")
    except ValueError:
        pass
    # Phải báo lỗi nếu bật strict_known_generators=True mà tên generator sai chính tả (ví dụ 'DALL-E 3')
    try:
        evaluate_by_generator(
            [0, 1], [-1.0, 1.5], ["MS_COCO", "DALL-E 3"],
            strict_known_generators=True,
        )
        raise AssertionError("Phải chặn tên generator sai chính tả khi strict_known_generators=True!")
    except ValueError:
        pass
    print("[PASS] 6. evaluate_by_generator() đối soát nhãn <-> model_name, cân bằng 1:1 & chặn sai tên generator chuẩn xác.")

    # 7. Kiểm thử cấu trúc metadata thật 43,800 dòng Test & đối chiếu tên generator (dùng logits mô phỏng Gaussian)
    from pathlib import Path
    resolved_self = Path(__file__).resolve()
    meta_csv = (
        resolved_self.parents[3] / "dataset" / "Defactify_dataset" / "metadata.csv"
        if len(resolved_self.parents) > 3
        else Path("dataset/Defactify_dataset/metadata.csv")
    )
    if meta_csv.exists():
        import pandas as pd
        df_all = pd.read_csv(meta_csv)
        unique_models_in_csv = sorted(df_all["model_name"].astype(str).unique().tolist())
        expected_models = sorted([REAL_MODEL_NAME] + ALL_AI_GENERATORS)
        assert unique_models_in_csv == expected_models, (
            f"Tên model_name trong metadata.csv không khớp hằng số chuẩn!\n"
            f"  Thực tế : {unique_models_in_csv}\n  Kỳ vọng : {expected_models}"
        )
        print(f"  + Đã đối chiếu khớp 100% tên model_name trong metadata.csv: {unique_models_in_csv}")

        test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
        rng = np.random.default_rng(42)
        y_test_real = test_df["label_a"].to_numpy(dtype=np.int64)
        m_test_real = test_df["model_name"].tolist()
        # LƯU Ý: Đây là logits mô phỏng Gaussian để kiểm định cấu trúc dữ liệu & hàm gộp cặp 1:1 (không phải inference mô hình)
        sim_logits = np.where(
            y_test_real == 0,
            rng.normal(-1.8, 1.0, size=len(test_df)),
            np.where(
                test_df["model_name"].isin(KNOWN_GENERATORS),
                rng.normal(2.2, 1.0, size=len(test_df)),
                rng.normal(1.2, 1.1, size=len(test_df)),
            ),
        )
        full_gen_eval = evaluate_by_generator(
            y_test_real, sim_logits, m_test_real,
            require_balanced_pairs=True,
            strict_known_generators=True,
        )
        assert full_gen_eval["shared_real_anchor_count"] == 7300
        assert len(full_gen_eval["per_generator"]) == 5
        for gen_name in ALL_AI_GENERATORS:
            gm = full_gen_eval["per_generator"][gen_name]
            assert gm["num_real"] == 7300 and gm["num_ai"] == 7300, f"Lệch số mẫu tại {gen_name}: {gm}"
        assert full_gen_eval["macro_avg_known"]["num_generators"] == 3
        assert full_gen_eval["macro_avg_unseen"]["num_generators"] == 2
        assert full_gen_eval["macro_avg_known"]["auc"] > full_gen_eval["macro_avg_unseen"]["auc"] > 0.85
        print(
            "[PASS] 7. Kiểm thử cấu trúc metadata & gộp cặp 1:1 trên 43,800 dòng Test thật "
            "(dùng logits mô phỏng Gaussian): khớp 100%!"
        )
        print("\n" + format_generator_report(full_gen_eval, "MINH HỌA GỘP CẶP 1:1 TRÊN 43,800 DÒNG METADATA TEST (LOGITS MÔ PHỎNG)"))
    else:
        print("\n" + format_generator_report(eval_gen, "VÍ DỤ BẢNG GENERATOR CẶP 1:1"))

    print("\n" + "=" * 80)
    print("TẤT CẢ UNIT TESTS CỦA metrics.py ĐÃ VƯỢT QUA 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
