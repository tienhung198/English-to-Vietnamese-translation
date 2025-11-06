# 🌍 Dự án Dịch máy tiếng Anh - Tiếng Việt sử dụng Mô hình Học sâu (Deep Learning)

## 1️⃣ Giới thiệu về dự án

Dự án này xây dựng một **hệ thống dịch tự động tiếng Anh sang tiếng Việt** dựa trên **mô hình học sâu (Deep Learning)**.  
Hệ thống gồm 2 phần chính:

- **File `traning_model.py`**: Xây dựng và huấn luyện mô hình dịch sử dụng kiến trúc **Transformer (Seq2Seq)**.
- **File `app.py`**: Tạo giao diện web bằng **Flask**, cho phép người dùng nhập câu tiếng Anh và nhận kết quả dịch tiếng Việt.

Mục tiêu của dự án là giúp người dùng dễ dàng trải nghiệm khả năng dịch tự động của mô hình học sâu thông qua một giao diện web đơn giản, trực quan.

---

## 2️⃣ Giới thiệu ngắn gọn về mô hình sử dụng

Mô hình được sử dụng là **Seq2Seq Transformer**, được định nghĩa trong file `hih.py`.  
Cấu trúc mô hình bao gồm các thành phần chính:

- **Embedding Layer** – Biểu diễn từ dưới dạng vector số học.  
- **Positional Encoding** – Giúp mô hình nhận biết vị trí của từng từ trong câu.  
- **Encoder Layer** – Mã hóa ngữ nghĩa của câu nguồn (tiếng Anh).  
- **Decoder Layer** – Giải mã và sinh câu đích (tiếng Việt).  
- **Multi-Head Attention** – Cơ chế giúp mô hình tập trung vào các phần quan trọng của câu nguồn khi dịch.  
- **Generator Layer** – Biến đầu ra của mô hình thành xác suất phân phối trên từ vựng tiếng Việt.

Mô hình được huấn luyện bằng **PyTorch** và **torchtext**, sử dụng hàm mất mát **CrossEntropyLoss** và bộ tối ưu **Adam**.  
Nó học cách ánh xạ ngữ nghĩa giữa các cặp câu tiếng Anh – tiếng Việt trong tập dữ liệu song ngữ.

---

## 3️⃣ Cách mô hình hoạt động và kết quả nhận được

### ⚙️ Quy trình hoạt động

1. **Người dùng nhập** một câu tiếng Anh vào giao diện web.  
2. **Flask** (trong `app.py`) nhận dữ liệu và truyền cho hàm `translate()`.  
3. Câu đầu vào được **tokenize** và **chuyển đổi sang chỉ số** thông qua lớp `Field` của `torchtext`.  
4. Các chỉ số được đưa vào mô hình **Seq2Seq Transformer** để tạo ra chuỗi đầu ra tiếng Việt.  
5. Kết quả được **giải mã** về dạng câu tiếng Việt hoàn chỉnh và **hiển thị trên trình duyệt**.

### 🧩 Sơ đồ tổng quát

```
English sentence → Tokenize → Encode → Transformer Model → Decode → Vietnamese sentence
```

### 🧠 Kết quả nhận được

- Mô hình có khả năng dịch tương đối chính xác các câu thông dụng tiếng Anh → tiếng Việt.  
- Câu dịch thể hiện được ngữ pháp và ngữ nghĩa tương đối tự nhiên.  
- Thời gian xử lý trung bình: < 1 giây / câu.  
- Hệ thống giao diện web Flask hoạt động ổn định, dễ sử dụng.

---

## ⚙️ Hướng dẫn chạy dự án

### 1. Cài đặt môi trường và thư viện cần thiết
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
torchtext==0.12.0 flask spacy dill numpy==1.26.4 --extra-index-url https://download.pytorch.org/whl/cu113
python -m spacy download en_core_web_sm
```

### 2. Chạy ứng dụng web
```bash
python app.py
```

### 3. Truy cập trình duyệt tại:
```
http://127.0.0.1:5000
```

Nhập câu tiếng Anh để xem bản dịch tiếng Việt do mô hình Transformer sinh ra.

---

## 🧭 Định hướng phát triển

- Tối ưu tốc độ xử lý khi dịch các câu dài.  
- Nâng cấp mô hình với kiến trúc **Transformer-Big** hoặc **mBART**.  
- Bổ sung giao diện chọn mô hình hoặc tải mô hình huấn luyện khác.  
- Triển khai hệ thống lên **Hugging Face Hub** hoặc **Web API**.

---

## 👤 Tác giả

**Trần Việt Tiến Hưng**  
Ngành: Trí tuệ Nhân tạo – Đại học Nguyễn Tất Thành  
📧 Email: hungtvt218@gmail.com

---

> *Dự án được thực hiện nhằm mục đích nghiên cứu và học tập trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP) và Dịch máy (Machine Translation) sử dụng học sâu (Deep Learning).*

