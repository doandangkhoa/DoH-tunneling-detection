# Hướng dẫn sử dụng dữ liệu huấn luyện

Điều hướng: [README](../README.md) · [Công việc triển khai](../TODO.md).

Tài liệu chung cho A/B/C/D. Dữ liệu đã xử lý sẵn; các thành viên không cần đọc PCAP hoặc chạy lại pipeline để bắt đầu. Mã train/model hiện vẫn cần được triển khai.

## Đọc nhanh: chỉ cần mở phần của mình

Sau khi pull code và giải nén gói như mục 1, chọn đúng dòng của mình:

| Bạn là | Thư mục dùng | Ví dụ code cần đọc |
|---|---|---|
| **B** | `data/processed/xgboost/common/` | [B — Nạp dữ liệu XGBoost](#b--nạp-dữ-liệu-xgboost) |
| **C** | `data/processed/isolation_forest/common/` | [C — Nạp dữ liệu Isolation Forest](#c--nạp-dữ-liệu-isolation-forest) |
| **D** | `data/processed/cnn1d/` | [D — Nạp dữ liệu CNN](#d--nạp-dữ-liệu-cnn) |

**Bắt đầu bằng đúng ba file trong thư mục của bạn:**

```text
train.npz  → lấy X, y để học mô hình (IF chỉ có benign)
val.npz    → chọn cấu hình/checkpoint và threshold
test.npz   → dự đoán cuối sau khi khóa mô hình và threshold
```

NPZ đã chứa `sample_id`: giữ ID đi cùng từng dòng X để xuất prediction. D lấy thêm `mask` cho CNN. Không đưa ID hoặc nhãn vào X.

| Các file còn lại | Khi nào cần mở? |
|---|---|
| `preprocessing.json` | Lưu cùng thí nghiệm để biết đầu vào đã xử lý ra sao; không cần tự áp dụng lại lên X |
| `*_metadata.parquet` | Sau khi có prediction, phân tích lỗi theo browser/công cụ bằng cách join `sample_id` |
| `train.parquet`, `val.parquet`, `test.parquet` | Chỉ khi muốn xem bảng; là bản biểu diễn khác, không nối thêm vào dữ liệu NPZ |
| `*_ids.csv` | Chỉ khi muốn xem/đối chiếu danh sách mẫu; không cần đọc riêng để train |

Để nguyên các file phụ trong thư mục, chưa cần mở khi bắt đầu. X trong NPZ đã được xử lý; không chuẩn hóa lại, không tự chia lại train/val/test. Sau khi model hoạt động, đọc [mục 4 để xuất kết quả chung](#4-xuất-kết-quả-để-so-sánh-chung).

## Phạm vi huấn luyện đã thống nhất

**B/C/D dùng chung gói common hiện tại: 190.100 flow. Vòng đầu mỗi mô hình chạy một baseline với seed 42; CNN dùng 32 packet. Chưa cần train thêm bộ full, chạy nhiều seed hoặc grid search.**

Common bao phủ khoảng **72,74%** bộ bảng đầy đủ 261.355 flow. Đây là tập con ghép được PCAP–CSV đạt QA, không tương đương hoàn toàn bộ full: độ bao phủ và phân bố khác nhau theo nguồn. Nhiều flow hơn không tự chứng minh mô hình tổng quát hoặc ổn định hơn.

| Thứ tự | B/C/D thực hiện | Kết quả cần giữ |
|---|---|---|
| 1 | Nạp đúng thư mục common của mình, kiểm tra shape, ID và nhãn | Dữ liệu đọc được theo các ví dụ bên dưới |
| 2 | Train baseline từ YAML, seed 42; IF chỉ fit benign train | Model/checkpoint và config; CNN chọn checkpoint bằng validation |
| 3 | Dùng validation để chọn threshold tối đa F1; nếu cần sửa cấu hình thì chỉ dựa vào validation | Validation score, threshold cố định và log thí nghiệm |
| 4 | Khóa model/checkpoint và threshold rồi predict test | Prediction đủ 13.411 test IDs theo mục 4 |
| 5 | B dùng evaluator chung; C làm dashboard; A kiểm tra tích hợp | Bảng metric, PR curve, confusion matrix và demo từ kết quả thực |

Một lần train CNN gồm nhiều epoch theo cấu hình, không phải một epoch. Lưu checkpoint đã chọn để predict lại; không cần train lại chỉ để mở dashboard hoặc tính lại metric từ prediction.

**Giữ các kiểm tra bắt buộc:** không fit preprocessing trên val/test; không tự chia lại dữ liệu; ID/nhãn prediction phải khớp test. D kiểm thử mask/padding ở runtime. Khi phát hiện lỗi pipeline, shape, NaN hoặc kết quả bất thường thì sửa và chạy lại phần liên quan; không tiếp tục dựa trên output lỗi. Không chỉnh mô hình/ngưỡng dựa vào kết quả test rồi coi đó là cùng lần đánh giá cuối.

**Phần mở rộng để sau:** train XGBoost/IF trên bộ full, chạy nhiều seed, tuning lớn và thử sequence length khác. Nếu bổ sung, trình bày riêng phạm vi dữ liệu/thí nghiệm. Kết quả vòng đầu chỉ là baseline một seed trên common, chưa chứng minh độ ổn định nhiều lần chạy hoặc khả năng tổng quát ngoài dataset.

## 1. Nhận dữ liệu và bắt đầu

A gửi file **`common.tar.gz`** (dung lượng xem trên file A bàn giao). Clone/pull repository chỉ lấy code vì dữ liệu được Git ignore. Cả nhóm dùng cùng gói A bàn giao; không thay dữ liệu giữa một lần thí nghiệm.

Đặt gói tại thư mục gốc repository rồi giải nén:

```bash
# Chạy tại DoH-tunneling-detection/
tar -xzf common.tar.gz

# Đủ để chạy các ví dụ đọc NPZ bên dưới
python3 -m pip install numpy
# Chỉ cần khi đọc Parquet hoặc xuất prediction
python3 -m pip install pandas pyarrow
```

Cài thêm thư viện theo mô hình phụ trách: B dùng `xgboost`, C dùng `scikit-learn`, D dùng `torch`. Danh sách chung nằm trong `requirements.txt`.

Sau giải nén, dữ liệu nằm đúng cấu trúc sau:

```text
docs/
└── TRAINING_GUIDE.md                  ← Hướng dẫn này
data/
└── processed/
    ├── xgboost/common/                ← B dùng
    ├── isolation_forest/common/       ← C dùng
    └── cnn1d/                         ← D dùng
```

## 2. Mỗi thành viên dùng bộ nào?

| Người | Mô hình | Thư mục cần mở | Đầu vào chính |
|---|---|---|---|
| B | XGBoost | `data/processed/xgboost/common/` | `X`: bảng 29 đặc trưng mỗi flow |
| C | Isolation Forest | `data/processed/isolation_forest/common/` | `X`: bảng đã scale; train chỉ benign |
| D | 1D-CNN | `data/processed/cnn1d/` | `X`: chuỗi 32 packet × 3 kênh, kèm `mask` |

**Để so sánh ba mô hình, dùng đúng ba thư mục trên.** Trên máy A còn có bộ bảng đầy đủ tại `data/processed/xgboost/` và `data/processed/isolation_forest/`, gồm 261.355 flow trước khi lấy subset common. Bộ đầy đủ dành cho thí nghiệm bổ sung, không nằm trong gói bàn giao.

Common có 190.100 flow ghép được packet với flow. Val/test của cả ba mô hình có cùng ID, nhãn và thứ tự. Train của IF là phần benign của common train.

| Tập | XGBoost / CNN | Benign (0) | Malicious (1) | IF |
|---|---:|---:|---:|---:|
| train | 168.597 | 5.383 | 163.214 | 5.383 benign |
| val | 8.092 | 1.435 | 6.657 | 8.092, đủ hai lớp |
| test | 13.411 | 6.792 | 6.619 | 13.411, đủ hai lớp |

```text
train → học mô hình
val   → chọn cấu hình, early stopping và threshold
      → khóa mô hình + threshold
test  → đánh giá kết quả cuối của ba mô hình
```

Không tự chia lại dữ liệu, không fit preprocessing trên val/test, không cân bằng lại val/test. Nếu dùng trọng số lớp hoặc sampling, chỉ áp dụng train.

Cấu hình baseline tương ứng: [XGBoost](../configs/xgboost.yaml), [Isolation Forest](../configs/isolation_forest.yaml), [CNN](../configs/cnn1d.yaml). B/C/D cần nối YAML vào CLI train/predict; hiện các trainer vẫn là khung. YAML chỉ định đường dẫn common, array keys, tham số và output; `preprocessing.json` cạnh dữ liệu giữ tham số xử lý thực tế. Xem [cách dùng cấu hình](../README.md#dùng-cấu-hình).

## 3. Các file có ý nghĩa gì?

Mỗi thư mục có `train.npz`, `val.npz`, `test.npz`. Chỉ cần ba file này để nạp đầu vào mô hình.

| File | Dùng khi nào? |
|---|---|
| `*.npz` | Nạp bằng NumPy để train; chứa nhiều array trong một file |
| `*.parquet` | Xem dữ liệu dạng bảng bằng pandas; không phải file văn bản/CSV |
| `*_ids.csv` | Danh sách ID của mỗi tập; dùng đối chiếu, không chia lại |
| `*_metadata.parquet` | Nguồn, capture, nhóm và nhãn để phân tích kết quả; join bằng `sample_id` |
| `preprocessing.json` | Các tham số và quy tắc xử lý đã fit từ train; lưu cùng thí nghiệm |

Nhãn thống nhất: **`0 = benign DoH`, `1 = malicious/tunnel DoH`**. `sample_id` là mã định danh flow, dùng ghép prediction với nhãn và metadata. Không đưa ID, IP, tên nguồn, capture hoặc label vào feature matrix.

### B — Nạp dữ liệu XGBoost

```python
from pathlib import Path
import numpy as np

folder = Path("data/processed/xgboost/common")
with np.load(folder / "train.npz", allow_pickle=False) as data:
    X_train = data["X"]
    y_train = data["y"]
    feature_names = data["feature_names"]
with np.load(folder / "val.npz", allow_pickle=False) as data:
    X_val, y_val = data["X"], data["y"]
    val_ids = data["sample_id"]

print(X_train.shape, y_train.shape)  # (168597, 29) (168597,)
# model.fit(X_train, y_train, ...)
# Dùng X_val/y_val để chọn cấu hình và threshold.
```

`X` đã điền missing bằng tham số từ common train. `X_scaled` là cùng bảng sau robust scaling; XGBoost có thể dùng trực tiếp `X`. `X_if` là bản scale riêng cho IF, B không cần dùng. Giữ nguyên thứ tự `feature_names`. Nếu thử `scale_pos_weight`, tỷ lệ benign/malicious của common train là khoảng **0,032981**; lựa chọn cuối dựa vào validation.

Muốn xem bảng trực quan:

```python
import pandas as pd

frame = pd.read_parquet("data/processed/xgboost/common/train.parquet")
print(frame.head().to_string(index=False))
# Bảng có sample_id, các feature và label; không fit toàn bộ bảng này vào model.
```

### C — Nạp dữ liệu Isolation Forest

```python
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest

folder = Path("data/processed/isolation_forest/common")
with np.load(folder / "train.npz", allow_pickle=False) as data:
    X_train, y_train = data["X"], data["y"]
with np.load(folder / "val.npz", allow_pickle=False) as data:
    X_val, y_val = data["X"], data["y"]
    val_ids = data["sample_id"]

assert (y_train == 0).all()
print(X_train.shape)  # (5383, 29)
model = IsolationForest(random_state=42, n_jobs=1).fit(X_train)
val_scores = -model.score_samples(X_val)  # Điểm cao = nghi malicious hơn
# Chọn threshold từ val_scores/y_val, rồi giữ cố định cho test.
```

`X` ở thư mục IF đã được impute và robust-scale bằng tham số học **chỉ từ benign common train**. Không scale thêm lần nữa; không lấy benign của bộ full thay cho common train. IF score không phải xác suất.

### D — Nạp dữ liệu CNN

```python
from pathlib import Path
import numpy as np
import torch

folder = Path("data/processed/cnn1d")
with np.load(folder / "train.npz", allow_pickle=False) as data:
    X_train = data["X"]
    y_train = data["y"]
    mask = data["mask"]
    train_ids = data["sample_id"]

print(X_train.shape, mask.shape)  # (168597, 32, 3) (168597, 32)
inputs = torch.from_numpy(X_train.transpose(0, 2, 1).copy()).float()
valid_mask = torch.from_numpy(mask).bool()
targets = torch.from_numpy(y_train).float()  # BCEWithLogitsLoss cần nhãn float 0/1.
# inputs: (N, 3, 32), valid_mask: (N, 32), targets: (N,)
# Khi train: loss = criterion(logits.reshape(-1), targets)
# Thiết kế model để dùng valid_mask trong forward và pooling.
```

| Kênh | `X_raw` | `X` đưa vào CNN |
|---|---|---|
| 0 — length | IPv4 total length, byte | log1p rồi chuẩn hóa từ packet thật của train |
| 1 — direction | +1 hoặc -1 | Giữ +1/-1 |
| 2 — IAT | Khoảng cách thời gian giữa packet, giây; packet đầu bằng 0 | log1p rồi chuẩn hóa từ packet thật của train |

Mỗi flow có 4–32 packet thật. `mask=True` là packet thật; `False` là padding. Padding trong `X` bằng 0. Direction theo SYN initiator nếu quan sát được SYN đầu sequence, nếu thiếu thì theo endpoint đầu quan sát được. `X_raw` dành cho phân tích, không phải đầu vào đã chuẩn hóa.

Không chuẩn hóa lại `X`. Conv1d không tự sử dụng mask: D cần kiểm thử ở chế độ eval rằng thay giá trị padding nhưng giữ packet thật và mask thì prediction không đổi trong tolerance đã chọn. Muốn thử prefix 16 packet, chọn bằng validation; 64 packet cần A trích xuất thêm, không tự bịa packet.

## 4. Xuất kết quả để so sánh chung

Sau khi khóa mô hình và threshold bằng validation, nạp `test.npz` từ đúng thư mục của mình. Giữ `sample_id` và `y` đi cùng từng dòng `X` khi dự đoán.

Mỗi người xuất `outputs/<model>/predictions.parquet`, với `<model>` là `xgboost`, `isolation_forest` hoặc `cnn1d`:

| Cột | Nội dung |
|---|---|
| `sample_id` | ID từ file test, duy nhất mỗi dòng |
| `model` | Tên mô hình |
| `y_true` | `y` của file test, 0/1 |
| `y_score` | Score hữu hạn, càng cao càng nghi malicious |
| `y_pred` | `(y_score >= threshold)`, 0/1 |
| `threshold` | Ngưỡng cố định đã chọn từ validation |
| `split` | Chuỗi `test` |

Ví dụ sau khi đã có `test_scores` từ mô hình và `threshold` từ validation:

```python
from pathlib import Path
import numpy as np
import pandas as pd

model_name = "xgboost"  # C/D đổi sang tên mô hình của mình
folder = Path("data/processed/xgboost/common")  # C/D đổi đúng thư mục
with np.load(folder / "test.npz", allow_pickle=False) as data:
    test_ids, y_test = data["sample_id"], data["y"]

# test_scores phải được dự đoán theo đúng thứ tự X của test.npz.
result = pd.DataFrame({
    "sample_id": test_ids, "model": model_name, "y_true": y_test,
    "y_score": test_scores, "y_pred": (test_scores >= threshold).astype(int),
    "threshold": threshold, "split": "test",
})
output = Path("outputs") / model_name
output.mkdir(parents=True, exist_ok=True)
result.to_parquet(output / "predictions.parquet", index=False)
```

B kiểm tra ba file có đúng 13.411 ID test, không trùng/thiếu và nhãn khớp nhau. Lưu kèm model/checkpoint, config, seed, feature order, preprocessing, log train/validation, phiên bản thư viện và thời gian train/inference.

Báo cáo Precision, Recall, F1, **AP** (`average_precision_score`), FPR và TP/FN/FP/TN. Join `test_metadata.parquet` bằng `sample_id` để báo cáo Recall/support riêng dns2tcp, dnscat2, iodine và macro Recall theo công cụ; FPR kèm FP/TN theo Chrome/Firefox. Nguồn không có mẫu ghi N/A. Mỗi mô hình dùng một threshold đã khóa, không chỉnh theo nhãn test hay từng công cụ test.

## 5. Hiểu đúng dữ liệu và giới hạn

Nguồn là CIRA-CIC-DoHBrw-2020: browser Chrome/Firefox được gán benign; các capture tunnel dns2tcp/dnscat2/iodine được gán malicious. Chỉ giữ DoH theo bộ lọc resolver/TLS 443; loại baseline chưa rõ nhãn. Đây là nhãn theo kịch bản nguồn, không phải xác minh payload tunnel từng flow. Khi viết báo cáo, trích dẫn bài “Detection of DoH Tunnels using Time-series Classification of Encrypted Traffic”, IEEE CyberSciTech 2020 của MontazeriShatoori và cộng sự.

Pipeline loại vector đặc trưng trùng chính xác và giữ capture/nhóm/phiên kết nối trong cùng split. Chưa chứng minh loại hết near-duplicate. Common chỉ gồm flow ghép PCAP–CSV đạt QA, có selection bias; không đại diện toàn bộ dữ liệu gốc. Đã xét toàn bộ capture browser/tunnel có sẵn, giữ nguyên split và các ID cũ. Kiểm tra ngược cửa sổ thời gian/provenance và đối chiếu tensor với packet metadata nằm trong `scripts/audit_packet_mapping.py`; script cũng đọc lại mẫu packet từ PCAP. Các flow không ghép được vẫn ở bộ tabular đầy đủ, không được tạo sequence giả để bù.

| Nguồn | Train | Val | Test |
|---|---:|---:|---:|
| Chrome | 707 | 0 | 2.022 |
| Firefox | 4.676 | 1.435 | 4.770 |
| dns2tcp | 151.523 | 1.425 | 1.633 |
| dnscat2 | 4.452 | 3.145 | 2.348 |
| iodine | 7.239 | 2.087 | 2.638 |

Benign validation chỉ có Firefox, từ hai capture thuộc một nhóm cha. Không suy ra FPR thấp sẽ ổn định trên nhiều môi trường hoặc trên Chrome. Sau khi mở rộng PCAP, dns2tcp đã tăng số mẫu; vẫn phải báo cáo support và số nhóm độc lập khi diễn giải Recall. Không chuyển flow giữa các tập để làm đẹp phân bố.

CNN dùng prefix 32 packet, trong khi XGBoost/IF dùng thống kê toàn flow. Kết quả là so sánh các phương pháp với biểu diễn tương ứng; chưa tách riêng ảnh hưởng thuật toán. Khoảng 50,49% packet thật không có TCP payload; chúng khác padding. Dấu hiệu TLS chỉ dựa trên record header quan sát được, không reassembly/giải mã; outer type 23 có thể gồm handshake mã hóa. Chưa có kết quả model hoặc kiểm thử mask runtime để kết luận CNN học được hành vi tunnel.

## 6. Dành cho A: tái tạo và đóng gói

Các lệnh này cần raw và dữ liệu trung gian trên máy A; B/C/D không cần chạy:

```bash
python3 src/data/prepare_dataset.py
python3 src/data/prepare_packets.py
python3 scripts/audit_training_data.py
python3 scripts/audit_packet_mapping.py
python3 scripts/validate_prepared_data.py
python3 scripts/package_training_data.py
```

Script đóng gói tạo `data/exports/common.tar.gz`, chỉ chứa ba profile common và hướng dẫn này. QA chi tiết giữ tại `data/interim/qa/`, trạng thái kiểm tra tại `data/processed/validation.json`; không cần gửi các file đó cho người train. Master, flow features trước imputation, packet metadata, cache và snapshot cũ phục vụ A truy vết, không phải đầu vào train trực tiếp.
