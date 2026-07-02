# mo_hinh_5c
# 📊 Ứng dụng Phân loại Rủi ro Khách hàng

## 📋 Mô tả ứng dụng

Ứng dụng web **Streamlit** sử dụng mô hình **Logistic Regression** để phân loại rủi ro của khách hàng dựa trên các chỉ số đánh giá:

- **TC** (Tư cách): 5 chỉ số (TC1-TC5)
- **NL** (Năng lực): 4 chỉ số (NL1-NL4)
- **DK** (Độc lập): 5 chỉ số (DK1-DK5)
- **V** (Vị trí): 6 chỉ số (V1-V6)
- **TS** (Tính số): 4 chỉ số (TS1-TS4)

**Biến mục tiêu:** `PD` (Phân loại Rủi do)
- `0` = Không có rủi ro
- `1` = Có rủi ro

## 🚀 Cài đặt

### Yêu cầu
- Python 3.8 hoặc cao hơn
- pip (quản lý gói Python)

### Các bước cài đặt

1. **Clone hoặc tải về ứng dụng:**
   ```bash
   # Đặt tệp vào một thư mục
   cd /đường/dẫn/ứng/dụng
   ```

2. **Tạo môi trường ảo (tùy chọn nhưng được khuyến nghị):**
   ```bash
   python -m venv venv
   
   # Trên Windows:
   venv\Scripts\activate
   
   # Trên macOS/Linux:
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt mặc định tại `http://localhost:8501`

## 📁 Cấu trúc file dữ liệu đầu vào

File dữ liệu phải là **CSV** (định dạng: UTF-8) với các cột bắt buộc:

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `Dấu thời gian` | String | Dấu thời gian (không bắt buộc) |
| `TC1-TC5` | Số nguyên | Chỉ số Tư cách (1-5) |
| `NL1-NL4` | Số nguyên | Chỉ số Năng lực (1-5) |
| `DK1-DK5` | Số nguyên | Chỉ số Độc lập (1-5) |
| `V1-V6` | Số nguyên | Chỉ số Vị trí (1-5) |
| `TS1-TS4` | Số nguyên | Chỉ số Tính số (1-5) |
| `NN` | Số nguyên | (Tùy chọn, không dùng) |
| `PD` | Số nguyên | **Biến mục tiêu** (0 hoặc 1) |

**Ví dụ dòng dữ liệu:**
```csv
Dấu thời gian,TC1,TC2,TC3,TC4,TC5,NL1,NL2,NL3,NL4,DK1,DK2,DK3,DK4,DK5,V1,V2,V3,V4,V5,V6,TS1,TS2,TS3,TS4,NN,PD
9/19/2023 8:55,5,5,4,5,4,5,4,5,4,4,4,4,5,4,4,5,4,4,5,4,5,4,5,4,1,0
```

## 📊 Cấu trúc ứng dụng (6 thành phần)

### 1️⃣ **Sidebar - Cấu hình & Tải dữ liệu**
- 📁 Tải file CSV
- 🤖 Cấu hình tham số Logistic Regression:
  - `max_iter`: Số lần lặp tối đa (mặc định: 1000)
  - `random_state`: Seed tái tạo kết quả (mặc định: 23)
  - `test_size`: Tỷ lệ kiểm định (mặc định: 20%)
- 🎯 Nút "Huấn luyện mô hình" để bắt đầu training
- 🔄 Nút "Đặt lại" để xóa kết quả

### 2️⃣ **Tab: Tổng quan dữ liệu**
- 📊 Số lượng dòng, cột, dung lượng file
- 📄 Hiển thị 10 dòng dữ liệu đầu
- 📊 Bảng thống kê mô tả (describe) của các biến
- 🎯 Phân phối biến mục tiêu (PD)

### 3️⃣ **Tab: Trực quan hóa dữ liệu**
- 📈 Histogram cho các biến đầu vào
- Có thể chọn tối đa 4 biến để hiển thị
- 🎯 Biểu đồ bar phân phối biến mục tiêu
- Dùng Plotly để tương tác động

### 4️⃣ **Tab: Kết quả huấn luyện & Kiểm định**
- 📊 Kích thước tập huấn luyện & kiểm định
- 🏆 Chỉ tiêu hiệu suất:
  - **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
  - **Precision** = TP / (TP + FP)
  - **Recall** = TP / (TP + FN)
  - **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
  - **ROC-AUC** = Diện tích dưới đường cong ROC
- 🔲 Ma trận nhầm lẫn (Confusion Matrix) dạng heatmap
- 📋 Báo cáo chi tiết (classification report)

### 5️⃣ **Tab: Sử dụng mô hình - Chế độ 1: Nhập trực tiếp**
- 📝 Form nhập thông tin khách hàng
- 24 slider tương ứng với 24 biến đầu vào
- 🔮 Nút "Dự báo" để dự báo rủi ro
- 📈 Hiển thị kết quả:
  - Nhãn dự báo (Không rủi ro / Có rủi ro)
  - Xác suất chi tiết cho từng lớp
  - Biểu đồ xác suất dạng bar

### 6️⃣ **Tab: Sử dụng mô hình - Chế độ 2: Tải file dự báo**
- 📊 Tải file CSV có cùng cấu trúc biến đầu vào
- 📋 Bảng kết quả dự báo hàng loạt
- 📊 Thống kê số lượng khách hàng có/không rủi ro
- 📥 Nút tải về kết quả (CSV, UTF-8)

## 🤖 Thuật toán & Tham số

### Loại mô hình
**Logistic Regression** - Mô hình phân loại nhị phân (Binary Classification)

### Tham số mặc định (từ notebook gốc)
| Tham số | Giá trị mặc định | Khoảng khả dụng |
|--------|-----------------|-----------------|
| `max_iter` | 1000 | 100 - 2000 |
| `random_state` | 23 | 0 - 1000 |
| `test_size` | 0.2 (20%) | 0.1 - 0.5 (10% - 50%) |

### Công thức dự báo
```
P(y=1 | X) = 1 / (1 + exp(-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)))

Nếu P(y=1 | X) ≥ 0.5 → Dự báo: Có rủi ro (1)
Nếu P(y=1 | X) < 0.5  → Dự báo: Không rủi ro (0)
```

## 📊 Ví dụ dữ liệu mẫu

Tệp `5c.csv` (có sẵn) chứa:
- **150 dòng** dữ liệu khách hàng
- **27 cột** (24 biến đầu vào + 1 biến mục tiêu + 2 cột khác)
- Kích thước: ~12 KB

## ⚙️ Yêu cầu phiên bản Streamlit

**Phiên bản được khuyến nghị:** `>=1.28.0`

Các tính năng sử dụng:
- `st.set_page_config()` - Cấu hình trang (layout wide)
- `st.tabs()` - Tạo tab giao diện
- `st.file_uploader()` - Tải file
- `st.session_state` - Lưu trữ trạng thái session
- `st.form()` - Form nhập liệu
- `@st.cache_data` - Cache dữ liệu

## 🐛 Khắc phục sự cố

### Lỗi: "ModuleNotFoundError: No module named 'streamlit'"
**Giải pháp:** Cài đặt lại các gói
```bash
pip install -r requirements.txt
```

### Lỗi: "Cột thiếu trong dữ liệu"
**Giải pháp:** Kiểm tra file CSV có đủ các cột sau:
```
TC1, TC2, TC3, TC4, TC5, NL1, NL2, NL3, NL4, DK1, DK2, DK3, DK4, DK5, 
V1, V2, V3, V4, V5, V6, TS1, TS2, TS3, TS4, PD
```

### Lỗi: "Dữ liệu rỗng"
**Giải pháp:** Đảm bảo file CSV:
- Không rỗng (>0 dòng dữ liệu)
- Có dòng header hợp lệ
- Định dạng UTF-8

### Port 8501 đã được sử dụng
**Giải pháp:** Chạy trên port khác
```bash
streamlit run app.py --server.port 8502
```

## 📝 Ghi chú phiên bản

### v1.0 (Initial Release)
- ✅ Cấu hình sidebar đầy đủ
- ✅ 6 thành phần giao diện
- ✅ Huấn luyện Logistic Regression
- ✅ Dự báo đơn và hàng loạt
- ✅ Xuất kết quả CSV
- ✅ Tương tác động với Plotly
- ✅ Xử lý lỗi file input

## 💡 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu
Tạo file CSV có cấu trúc như mô tả (hoặc dùng `5c.csv` mẫu)

### Bước 2: Tải file
Bấm nút "Chọn file CSV" ở sidebar để tải dữ liệu

### Bước 3: Xem thông tin dữ liệu
Chuyển sang tab "Tổng quan dữ liệu" để kiểm tra dữ liệu đã tải đúng

### Bước 4: Khám phá dữ liệu
Xem các biểu đồ ở tab "Trực quan hóa dữ liệu"

### Bước 5: Huấn luyện mô hình
Điều chỉnh tham số (nếu muốn) rồi bấm "🎯 Huấn luyện mô hình" ở sidebar

### Bước 6: Xem kết quả
Chuyển sang tab "Kết quả huấn luyện" để xem chỉ tiêu hiệu suất

### Bước 7: Dự báo
Dùng tab "Sử dụng mô hình" để dự báo rủi ro khách hàng mới

## 📧 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra lỗi trong terminal
2. Xem phần "Khắc phục sự cố" trên
3. Đảm bảo file dữ liệu có cấu trúc đúng

## 📄 Giấy phép

Ứng dụng này được tạo cho mục đích học tập và kinh doanh.

---

**Thông tin liên hệ:** Dùng chế độ nhập trực tiếp hoặc tải file để bắt đầu dự báo! 🚀
