# Kiến Trúc & Quy Trình Tiền Xử Lý – Tài Liệu Kỹ Thuật

---

## Phần 1: Đánh Giá Kiến Trúc ChatGPT Đề Xuất

**Kết luận ngắn: ChatGPT đúng. Đây là kiến trúc hợp lý nhất với ràng buộc tài nguyên và mục tiêu tổng quát hóa của đề tài.**

Tuy nhiên, có **3 điểm cần làm rõ hơn** để tránh sai sót khi triển khai:

### Điểm cần bổ sung #1 – Fine-tune EfficientNet cần rõ lộ trình

ChatGPT nói "fine-tune từng phần" nhưng chưa chỉ rõ thứ tự. Lộ trình đúng:

```
Giai đoạn 1 (Epoch 1–10):
  → Đóng băng TOÀN BỘ backbone EfficientNet
  → Chỉ train phần Classifier Head (MLP nhỏ cuối cùng)
  → Learning Rate = 1e-3
  → Mục tiêu: Học được quyết định phân loại từ feature sẵn có

Giai đoạn 2 (Epoch 11–30):
  → Mở băng 2-3 Block cuối của EfficientNet (ví dụ block 5, 6, 7)
  → Fine-tune với Learning Rate rất nhỏ = 1e-5
  → Mục tiêu: Thích nghi đặc trưng với forensic artifact của ảnh AI

Giai đoạn 3 (tùy chọn, nếu kết quả chưa tốt):
  → Mở băng thêm block 3, 4
  → LR = 5e-6 (cực nhỏ, giữ ổn định các feature đã học)
```

**Tại sao phải theo thứ tự này?** Nếu bạn mở băng toàn bộ ngay từ đầu, gradient từ classifier head mới (chưa ổn định) sẽ "phá vỡ" các trọng số ImageNet đã được huấn luyện tốt. Kết quả: model hội tụ chậm hơn và có thể overfit vào SD1.4/ADM.

### Điểm cần bổ sung #2 – CLIP Offline nhưng EfficientNet Online

ChatGPT đề xuất offline extraction cho cả EfficientNet sau khi fine-tune. Tôi **không khuyến nghị làm vậy trong giai đoạn fine-tuning**, vì:

- Khi EfficientNet đang fine-tune, trọng số thay đổi sau mỗi epoch → feature .npy sẽ lỗi thời ngay
- Chỉ extract EfficientNet feature offline **sau khi** fine-tune xong hoàn toàn, dùng cho fusion MLP sau này

### Điểm cần bổ sung #3 – SPAI là baseline độc lập, không phải nhánh fusion

Điều ChatGPT nói đúng hoàn toàn: SPAI chạy bên ngoài hệ thống 3 nhánh, chỉ để so sánh hiệu năng spectral. Không ghép SPAI vào MLP fusion.

---

## Phần 2: Nén JPEG Giả Lập (JPEG Augmentation) – Giải Thích Đầy Đủ

### Nén JPEG Giả Lập là gì?

Kỹ thuật **nén lại ảnh bằng thuật toán JPEG với chất lượng thấp hơn** trong quá trình huấn luyện, nhằm mô phỏng việc ảnh bị nén khi chia sẻ trên mạng xã hội (Facebook, Zalo, Telegram...).

```python
# Cách hoạt động về mặt kỹ thuật:
from PIL import Image
import io

def jpeg_compress(img_pil, quality=85):
    """
    Nén ảnh PIL sang buffer JPEG, rồi đọc lại.
    Ảnh bị mất thông tin do nén lossy → tạo blocking artifact.
    """
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()
```

### KHI NÀO áp dụng?

> **CHỈ trong giai đoạn HUẤN LUYỆN (train). KHÔNG dùng cho val và test.**

| Tập dữ liệu | Áp dụng Nén JPEG GS? |
|---|---|
| `data_train` – train split | ✅ CÓ (xác suất 50%) |
| `data_train` – val split | ❌ KHÔNG |
| `hide_data/VQDM` – test | ❌ KHÔNG |

### ÁP DỤNG CHO NHÃN NÀO?

> **Cho CẢ HAI: ảnh AI (nhãn=1) VÀ ảnh Real (nhãn=0)**

Đây là điều cực kỳ quan trọng. Tại sao phải áp dụng cho cả hai?

Vì audit của chúng ta đã phát hiện:
- Ảnh AI → toàn bộ PNG (không có JPEG artifact)
- Ảnh Real → toàn bộ JPEG (có JPEG artifact)

Nếu chỉ áp dụng nén JPEG lên ảnh AI thôi, bạn đang **vô tình tạo ra thiên lệch mới**: model sẽ học "ảnh bị nén mạnh thì là AI". Đây là confound mới, không phải giải pháp.

Giải pháp đúng: **Áp dụng ngẫu nhiên cho cả hai nhãn** với cùng xác suất → model buộc phải học đặc trưng thật sự thay vì shortcut JPEG/PNG.

### ÁP DỤNG CHO NHÁNH NÀO?

> **Áp dụng trên ẢNH GỐC trước khi đưa vào BẤT KỲ nhánh nào.**

Vì nén JPEG thay đổi giá trị pixel → nó ảnh hưởng đến cả 3 nhánh (Spatial, Semantic, Spectral) đồng thời. Nếu chỉ áp dụng cho 1 nhánh riêng, 3 nhánh sẽ nhận input mâu thuẫn nhau từ cùng một bức ảnh.

```
Ảnh gốc → [Nén JPEG GS (p=0.5, quality=75-95)] → Ảnh sau nén
                                                        │
                              ┌─────────────────────────┤
                              ↓         ↓               ↓
                        EfficientNet   CLIP            FFT
```

### Thông số khuyến nghị:

```python
# Xác suất áp dụng: 50%
# Quality range: 75-95 (mô phỏng nén nhẹ đến vừa, như trên mạng xã hội)
# KHÔNG dùng quality < 70 (sẽ tạo artifact quá rõ, không thực tế)
```

---

## Phần 3: Xử Lý File Metadata

### Cấu trúc metadata hiện tại

```
data_train/
├── ADM/
│   ├── metadata_train_ai.csv    ← cột: path, split, label, category, generator
│   ├── metadata_train_real.csv
│   └── metadata_val.csv
└── stable_diffusion_1.4/
    ├── metadata_train_ai.csv
    ├── metadata_train_real.csv
    └── metadata_val.csv

hide_data/
└── VQDM/
    ├── metadata_train_ai.csv    ← đây thực ra là test/ai
    ├── metadata_train_real.csv  ← đây thực ra là test/real
    └── metadata_val.csv
```

### Metadata được dùng để làm gì?

**Không sửa, không xóa file metadata gốc.** Chỉ đọc và tạo thêm file mới:

```
ĐỪNG làm:    metadata_train_ai.csv → sửa nội dung
NÊN làm:     metadata_train_ai.csv → đọc → tạo features/ bên cạnh
```

### Quy trình xử lý metadata:

**Bước 1:** Đọc và gộp metadata từ các generator thành 2 tập dữ liệu:

```python
# Tập TRAIN (ADM + SD1.4 gộp lại):
train_ai_rows   = đọc ADM/metadata_train_ai.csv + SD1.4/metadata_train_ai.csv
train_real_rows = đọc ADM/metadata_train_real.csv + SD1.4/metadata_train_real.csv

# Tập VAL (ADM + SD1.4 gộp lại):
val_rows = đọc ADM/metadata_val.csv + SD1.4/metadata_val.csv

# Tập TEST (VQDM – không gộp với train):
vqdm_ai_rows   = đọc VQDM/metadata_train_ai.csv
vqdm_real_rows = đọc VQDM/metadata_train_real.csv
```

**Bước 2:** Tạo file CSV gộp cho mỗi tập:

```
dataset/
├── processed/
│   ├── train_manifest.csv    ← path, label (0=Real, 1=AI), generator, split
│   ├── val_manifest.csv
│   └── vqdm_manifest.csv
```

**Bước 3:** Sau khi extract CLIP offline, tạo thêm:

```
dataset/
├── processed/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   ├── vqdm_manifest.csv
│   └── features/
│       ├── clip_train.npy      ← shape: [N_train, 768]
│       ├── clip_val.npy        ← shape: [N_val, 768]
│       ├── clip_vqdm.npy       ← shape: [N_vqdm, 768]
│       ├── labels_train.npy    ← shape: [N_train]
│       ├── labels_val.npy
│       └── labels_vqdm.npy
```

---

## Phần 4: Quy Trình Tiền Xử Lý Hoàn Chỉnh – Theo Giai Đoạn

### GIAI ĐOẠN 0 – Chuẩn Bị Metadata (chạy 1 lần)

```
MỤC TIÊU: Gộp metadata từ nhiều generator, tạo train/val/test manifest

INPUT:  data_train/ADM/*.csv, data_train/stable_diffusion_1.4/*.csv
        hide_data/VQDM/*.csv

OUTPUT: processed/train_manifest.csv
        processed/val_manifest.csv
        processed/vqdm_manifest.csv

THỜI GIAN: < 1 phút
```

Mỗi dòng trong manifest có dạng:

| path | label | generator | split |
|---|---|---|---|
| `data_train/ADM/train_ai/ai/262_adm_001.PNG` | 1 | ADM | train |
| `data_train/ADM/train_real/real/n01440764_12090.JPEG` | 0 | real | train |

---

### GIAI ĐOẠN 1 – Extract CLIP Feature (chạy 1 lần)

```
MỤC TIÊU: Trích xuất embedding từ CLIP frozen, lưu ra .npy

INPUT:  train_manifest.csv, val_manifest.csv, vqdm_manifest.csv
        + Ảnh gốc tương ứng

PREPROCESSING ÁP DỤNG:
  ✅ CLIP Processor (Resize + CenterCrop + CLIP Normalize)
  ❌ RandomHorizontalFlip
  ❌ Nén JPEG Giả Lập
  ❌ (KHÔNG augmentation vì extract 1 lần, lưu cố định)

OUTPUT: processed/features/clip_train.npy   [N_train × 768]
        processed/features/clip_val.npy     [N_val × 768]
        processed/features/clip_vqdm.npy    [N_vqdm × 768]

THỜI GIAN: 30-60 phút (trên GPU RTX 3050)
```

---

### GIAI ĐOẠN 2 – Fine-tune EfficientNet (Phase 1: Đóng băng)

```
MỤC TIÊU: Train Classifier Head với backbone đóng băng

INPUT:  Ảnh gốc từ train_manifest.csv

PREPROCESSING (áp dụng MỖI BATCH khi train):
  ✅ Resize (cạnh ngắn nhất → 256px)
  ✅ CenterCrop (224 × 224)
  ✅ ToTensor()
  ✅ Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ✅ RandomHorizontalFlip (p=0.5)
  ✅ Nén JPEG Giả Lập (p=0.5, quality=75-95, cho CẢ AI và Real)
  → Đây là 6 bước transform cho nhánh Spatial khi training

  [Val và VQDM]:
  ✅ Resize 256 → CenterCrop 224 → ToTensor → Normalize
  ❌ Không có flip, không có nén JPEG

THỜI GIAN: ~10 epoch × ~15 phút = 2-3 giờ
```

---

### GIAI ĐOẠN 3 – Fine-tune EfficientNet (Phase 2: Mở băng một phần)

```
MỤC TIÊU: Thích nghi đặc trưng EfficientNet với forensic artifact

INPUT:  Như Giai đoạn 2 (cùng augmentation pipeline)
THAY ĐỔI: Mở băng 2-3 Block cuối EfficientNet, LR = 1e-5

THỜI GIAN: ~20 epoch × ~20 phút = 6-8 giờ
```

---

### GIAI ĐOẠN 4 – Extract FFT Feature (chạy 1 lần sau khi ảnh đã ổn định)

```
MỤC TIÊU: Tạo biểu diễn phổ tần số cho mỗi bức ảnh

INPUT:  Ảnh gốc (KHÔNG augmentation)

PREPROCESSING:
  ✅ Resize 256 → CenterCrop 224
  ✅ Chuyển sang Grayscale (1 kênh)
  ✅ FFT 2D
  ✅ FFT Shift (đưa tần số thấp về tâm)
  ✅ Log Magnitude: log(1 + |FFT|)
  ✅ Normalize về [0, 1]

  [LƯU Ý QUAN TRỌNG]:
  ❌ KHÔNG Nén JPEG Giả Lập trước FFT
  ❌ KHÔNG RandomHorizontalFlip trước FFT
  → Vì augmentation thay đổi phổ tần số, làm sai lệch đặc trưng spectral

OUTPUT: processed/features/fft_train.npy   [N_train × 224 × 224]
        processed/features/fft_val.npy
        processed/features/fft_vqdm.npy
        (có thể tính on-the-fly nếu RAM cho phép)

THỜI GIAN: 10-20 phút
```

---

### GIAI ĐOẠN 5 – Extract EfficientNet Feature (offline, sau khi fine-tune xong)

```
MỤC TIÊU: Lưu feature cuối cùng của EfficientNet đã fine-tune

INPUT:  Ảnh gốc qua PREPROCESSING CỐ ĐỊNH (không augmentation)
        + Model EfficientNet đã fine-tune từ Giai đoạn 3

PREPROCESSING:
  ✅ Resize 256 → CenterCrop 224 → ToTensor → Normalize
  ❌ KHÔNG flip, KHÔNG nén JPEG (để feature deterministic)

OUTPUT: processed/features/efficientnet_train.npy  [N_train × D]
        processed/features/efficientnet_val.npy
        processed/features/efficientnet_vqdm.npy
        (D = số chiều feature, thường 1280 với EfficientNet-B0)
```

---

### GIAI ĐOẠN 6 – Train Fusion MLP

```
MỤC TIÊU: Kết hợp 3 luồng đặc trưng → Binary Classifier

INPUT:
  clip_train.npy          [N × 768]
  efficientnet_train.npy  [N × 1280]
  fft_train.npy           [N × 224×224 → flatten hoặc qua CNN nhỏ → D_fft]

KIẾN TRÚC:
  Concatenate → [768 + 1280 + D_fft] → MLP (Linear → ReLU → Dropout → Linear) → Sigmoid

THỜI GIAN: < 30 phút (chỉ train MLP nhỏ trên feature cố định)
```

---

## Tóm Tắt Toàn Bộ Pipeline

```
GĐ 0: Chuẩn bị manifest          (1 lần, < 1 phút)
GĐ 1: Extract CLIP offline        (1 lần, 30-60 phút)
GĐ 2: Fine-tune EfficientNet P1   (2-3 giờ)
GĐ 3: Fine-tune EfficientNet P2   (6-8 giờ)
GĐ 4: Extract FFT offline         (1 lần, 10-20 phút)
GĐ 5: Extract EfficientNet        (1 lần, 15-30 phút)
GĐ 6: Train Fusion MLP            (< 30 phút)
```

**Tổng thời gian ước tính: 10-12 giờ** (phần lớn là fine-tune EfficientNet).

---

## Bảng Kiểm Tra Nhanh

| Kỹ thuật | EfficientNet | CLIP | FFT | data_train | hide_data/VQDM |
|---|---|---|---|---|---|
| Resize 256 → CenterCrop 224 | ✅ | ✅ (CLIP proc.) | ✅ | ✅ | ✅ |
| Normalize ImageNet | ✅ | ❌ (CLIP norm.) | ❌ | ✅ | ✅ |
| RandomHorizontalFlip | ✅ train only | ❌ | ❌ | ✅ | ❌ |
| **Nén JPEG Giả Lập** | ✅ train only | ❌ (offline) | ❌ | ✅ | ❌ |
| FFT Transform | ❌ | ❌ | ✅ | ✅ | ✅ |
| Offline Feature Extraction | ✅ (sau GĐ 3) | ✅ (GĐ 1) | ✅ (GĐ 4) | ✅ | ✅ |
