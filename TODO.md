# TODO — Bảng giao việc và nghiệm thu

> Mọi công việc bên dưới là **chưa triển khai**; ô checkbox chỉ đánh dấu khi có PR, file và bằng chứng nghiệm thu. Nội dung tương ứng 34 Task ID trong kế hoạch nhóm 12 ngày.

- [ ] Điền tên: **A:** __________ · **B:** __________ · **C:** __________ · **D:** __________
- [ ] Ngày kickoff: ____/____/______ · Người điều phối: __________
- [ ] Tạo GitHub repo + 4 branch + board theo `docs/git_workflow.md`
- [ ] Chốt dataset, quyền sử dụng, contract và quy ước tên file

## Mốc nghiệm thu (Gates)

- [ ] **G1 — hết D2:** kiểm chứng TLS, nhãn và ghép thử packet ↔ flow ↔ nhãn.
- [ ] **G2 — hết D4:** master dataset v1 + flow/packet metadata + split manifest không chồng lấp.
- [ ] **G3 — hết D9:** ba mô hình xuất kết quả dự đoán chuẩn trên cùng test IDs.
- [ ] **G4 — hết D12:** evaluator, report, dashboard và dry-run hoàn chỉnh.

## Danh sách 34 nhiệm vụ

### A — Data Lead & Integration

> Owner: **A** · Reviewer chính: **D (Data) / B (Integration)**.

- [ ] **DATA-01 · D1 — Kiểm kê nguồn dữ liệu**
  - **Nghiệm thu:** Danh sách PCAP/CSV/nhãn, dung lượng, nguồn/phiên bản, giấy phép; báo cáo số file và rủi ro thiếu dữ liệu.
  - **Phụ thuộc:** Không; có thể bắt đầu ngay.
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DATA-02 · D2 — Xác thực TLS và nhãn**
  - **Nghiệm thu:** Có thống kê các lớp sau khi lọc, giải thích nhãn; chứng cứ PCAP có traffic TLS/DoH; nêu sai lệch giữa nhãn và traffic.
  - **Phụ thuộc:** DATA-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DATA-03 · D2 — Proof-of-concept ánh xạ flow ↔ packet**
  - **Nghiệm thu:** Ghép thử đúng nhiều mẫu ở cả hai lớp; định nghĩa quy tắc flow ID/5-tuple/thời gian; D xác nhận.
  - **Phụ thuộc:** DATA-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DATA-04 · D3 — Xây master dataset v1**
  - **Nghiệm thu:** Mỗi flow có `sample_id` duy nhất, label, capture_id, flow_id, thời gian, group; có kiểm tra schema.
  - **Phụ thuộc:** DATA-03
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DATA-05 · D4 — Tạo flow features + packet metadata**
  - **Nghiệm thu:** Xuất 2 file Parquet với `sample_id`; có báo cáo null/infinite và kiểm tra số lượng record.
  - **Phụ thuộc:** DATA-04
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DATA-06 · D4 — Chốt split manifests**
  - **Nghiệm thu:** Tạo train/val/test ID; giữ nhóm capture nếu khả thi; không overlap; có class distribution và split rationale.
  - **Phụ thuộc:** DATA-04
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DATA-07 · D5 — Data QA + leakage audit**
  - **Nghiệm thu:** Kiểm tra duplicate/gần duplicate; các cột IP, capture, label và nguồn không đi vào mô hình baseline.
  - **Phụ thuộc:** DATA-05, DATA-06
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **INT-01 · D10 — Kiểm tra nhất quán prediction**
  - **Nghiệm thu:** Set sample_id của 3 mô hình bằng nhau và khớp test; nhãn y_true khớp master; có script assert.
  - **Phụ thuộc:** CNN-07, EVAL-03, IF-04
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DOC-01 · D11 — Viết phần Dataset/Methodology**
  - **Nghiệm thu:** Nêu provenance, labeling, filtering, feature pipeline, split và rủi ro.
  - **Phụ thuộc:** DATA-07
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **QA-01 · D12 — Dry-run và bàn giao cuối**
  - **Nghiệm thu:** README, file demo, cấu hình, bảng kết quả đủ; chạy demo trên một môi trường mới hoặc sạch.
  - **Phụ thuộc:** INT-01, UI-03, DOC-01..04
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

### B — XGBoost & Evaluation

> Owner: **B** · Reviewer chính: **A (Model) / C (Evaluation)**.

- [ ] **EVAL-01 · D1 — Đề xuất Output Contract + metric contract**
  - **Nghiệm thu:** Nhóm thống nhất trường prediction, chiều score, AP/FPR/latency và file names; A/C/D duyệt.
  - **Phụ thuộc:** Không; có thể bắt đầu ngay.
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **XGB-01 · D3 — Skeleton preprocess/train trên mock**
  - **Nghiệm thu:** Có train/predict chạy được từ vài dòng dữ liệu giả đúng schema; không cần chờ A.
  - **Phụ thuộc:** EVAL-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **EVAL-02 · D4 — Viết evaluator + unit tests**
  - **Nghiệm thu:** Test vài trường hợp TP/TN/FP/FN, không có class dương, score trái chiều; metric chính xác.
  - **Phụ thuộc:** EVAL-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **XGB-02 · D5 — XGBoost baseline trên data v1**
  - **Nghiệm thu:** Lưu model, features config, validation predictions, thời gian huấn luyện.
  - **Phụ thuộc:** DATA-05, DATA-06, XGB-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **XGB-03 · D6 — Tuning + threshold trên validation**
  - **Nghiệm thu:** Có log cấu hình đã thử, lựa chọn theo validation; không dùng test.
  - **Phụ thuộc:** XGB-02
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **EVAL-03 · D7 — Vẽ biểu đồ chung**
  - **Nghiệm thu:** Confusion Matrix, PR curve, bảng metric và đồ thị so sánh từ file predictions.
  - **Phụ thuộc:** EVAL-02
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **EVAL-04 · D10 — Khóa giao thức đánh giá test**
  - **Nghiệm thu:** Duy nhất một evaluator chạy cả ba file; test không được dùng để tuning.
  - **Phụ thuộc:** INT-01, EVAL-02
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DOC-02 · D11 — Viết XGBoost + thảo luận so sánh**
  - **Nghiệm thu:** Có cách học, siêu tham số, feature importance, metric và diễn giải hạn chế.
  - **Phụ thuộc:** XGB-03
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

### C — Isolation Forest & Dashboard

> Owner: **C** · Reviewer chính: **B**.

- [ ] **IF-01 · D2 — Đặc tả IF benign-only**
  - **Nghiệm thu:** Nêu cách chọn benign train, score conversion, threshold validation và chỉ số đánh giá.
  - **Phụ thuộc:** EVAL-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **IF-02 · D3 — Skeleton IF trên mock**
  - **Nghiệm thu:** Fit/score/threshold/predict hoạt động độc lập với data thật.
  - **Phụ thuộc:** IF-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **UI-01 · D4 — Dashboard prototype bằng mock results**
  - **Nghiệm thu:** Có các trang/tabs metrics, chart, sample predictions; chưa bắt buộc có model thật.
  - **Phụ thuộc:** EVAL-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **IF-03 · D6 — Isolation Forest baseline**
  - **Nghiệm thu:** Lưu model, score trên validation, config và kiểm tra phân phối anomaly score.
  - **Phụ thuộc:** IF-02, DATA-05, DATA-06
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **IF-04 · D7 — Chọn threshold + phân tích FP**
  - **Nghiệm thu:** Ngưỡng chọn từ validation; lưu kết quả F1-opt và/hoặc FPR target, không dùng test.
  - **Phụ thuộc:** IF-03
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **UI-02 · D8 — Kết nối dashboard với contract**
  - **Nghiệm thu:** Dashboard đọc output XGB/IF/CNN theo chuẩn; lỗi file thiếu hiển thị thân thiện.
  - **Phụ thuộc:** UI-01, EVAL-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **UI-03 · D10 — Hoàn thiện dashboard so sánh**
  - **Nghiệm thu:** Bảng metric, PR curve, confusion matrix, latency; hiển thị đúng score/probability.
  - **Phụ thuộc:** UI-02, EVAL-03
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DOC-03 · D11 — Viết IF + demo guide**
  - **Nghiệm thu:** Nêu benign-only, ngưỡng, cảnh báo sai, cách chạy dashboard.
  - **Phụ thuộc:** IF-04, UI-03
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

### D — Packet Sequences & 1D-CNN

> Owner: **D** · Reviewer chính: **A**.

- [ ] **CNN-01 · D1 — Đặc tả packet sequence**
  - **Nghiệm thu:** Chốt cột packet_index/length/direction/IAT; convention hướng truyền; cách padding.
  - **Phụ thuộc:** DATA-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **CNN-02 · D2 — Đánh giá ghép flow ↔ packet**
  - **Nghiệm thu:** Review DATA-03; xuất thí dụ 5-10 flow; chứng minh thứ tự packet và label đúng.
  - **Phụ thuộc:** DATA-03, CNN-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **CNN-03 · D4 — Sequence builder + test**
  - **Nghiệm thu:** Tạo (N,C) cho từng sample_id; kiểm tra padding/truncation, IAT >=0, order, no overlap.
  - **Phụ thuộc:** CNN-02, DATA-05
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **CNN-04 · D5 — Kiến trúc CNN + train loop**
  - **Nghiệm thu:** Baseline Conv1D nhỏ, train/val, save checkpoint; chạy thử trên dữ liệu mẫu.
  - **Phụ thuộc:** CNN-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **CNN-05 · D7 — CNN baseline thật**
  - **Nghiệm thu:** Train/validation chạy được; lưu loss curve, config, seed, checkpoint; không dùng test.
  - **Phụ thuộc:** CNN-03, CNN-04, DATA-06
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **CNN-06 · D8 — Tuning kích thước sequence**
  - **Nghiệm thu:** So sánh N=16/32/64 nếu đủ thời gian; lựa chọn bằng validation.
  - **Phụ thuộc:** CNN-05
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **CNN-07 · D9 — Export predictions theo contract**
  - **Nghiệm thu:** Đúng tập test chung; đủ score, y_pred, sample_id; thời gian inference.
  - **Phụ thuộc:** CNN-05, EVAL-01
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

- [ ] **DOC-04 · D11 — Viết phần CNN + sequence**
  - **Nghiệm thu:** Có kiến trúc, input shape, padding/truncation, curves, giới hạn.
  - **Phụ thuộc:** CNN-05
  - **Bằng chứng:** PR/commit `__________` · file/log/test `__________`

## Quy tắc chung khi đánh dấu hoàn thành

- [ ] Không có cùng `sample_id` ở nhiều split; số liệu được xác nhận bằng test tự động.
- [ ] Không dùng test labels để tuning hoặc lựa chọn threshold.
- [ ] Người review xác nhận các schema, đường dẫn và file output đúng contract.
- [ ] Nếu thay đổi dữ liệu chuẩn, có version/manifest và thông báo tới toàn nhóm.
- [ ] Báo cáo rõ hạn chế của DoH dataset: không suy rộng thành mọi loại malware TLS.

## Mẫu cập nhật cuối ngày

| Thành viên | Task hoàn thành | Đang làm | Blocker | Cần ai hỗ trợ |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |
