# X-quang: AI-assisted Chest X-ray Multi-Label Diagnosis & Explainable AI (XAI)

Hệ thống Hỗ trợ Chẩn đoán và Quản lý Quy trình X-quang Ngực có Giải thích Kết quả (XAI) tích hợp DenseNet-121 và Asymmetric Loss (ASL).

## Cấu trúc Dự án
- `data/`: Dữ liệu và cấu hình metadata mẫu.
- `scripts/`: Toàn bộ mã nguồn huấn luyện, inference và file chạy tự động (`train_densenet_asl.py`, `RUN_TRAIN_50_EPOCHS.bat`).
- `reports/`: Báo cáo khoa học thực nghiệm và các biểu đồ benchmark (`MASTER_RESEARCH_REPORT_XAI_BENCHMARK.md`).
- `notebooks/`: Jupyter notebooks phục vụ phân tích dữ liệu và vẽ biểu đồ.
- `models/`: Cấu hình kiến trúc mô hình.
- `results/`: Kết quả đánh giá chỉ số AUROC, F1, mIoU, Pointing Game.
- `logs/`: Nhật ký thực nghiệm.
- `draft/` & `sandbox/`: Không gian thử nghiệm tính năng mới.
- `repos/`: Mã nguồn mở đối sánh (cheXlocalize, ragflow).

## Hướng dẫn Chạy Huấn luyện (50 Epochs)
Chạy trực tiếp file batch trên Windows:
```cmd
RUN_TRAIN_50_EPOCHS.bat
```
Hoặc dùng dòng lệnh:
```bash
python scripts/train_densenet_asl.py --epochs 50 --sample_size 50000 --batch_size 32
```
