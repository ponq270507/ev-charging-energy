# Gói tái lập Group 25 — bản dựng mới 06/11/2026

## Cách chạy trên Google Colab

1. Giải nén `ADY_Group25_Colab_Package_New.zip` trên máy.
2. Mở Google Colab, chọn File → Upload notebook và tải file
   `EV_Charging_Arrival_Prediction_Group25_Revised.ipynb` trong thư mục ADY_Group25.
3. Chọn môi trường Python 3, CPU. Chọn Runtime → Run all.
4. Khi notebook yêu cầu gói dự án, tải lên chính `ADY_Group25_Colab_Package_New.zip`.
5. Nếu ô cài đặt yêu cầu khởi động lại runtime sau khi cập nhật thư viện, chọn
   Runtime → Restart session, rồi Run all lại. Đây là bước cần thiết khi Colab đã nạp phiên bản khác.
6. Giữ cả năm biến RUN_* ở False cho lần đầu. Notebook sẽ kiểm tra dữ liệu,
   dựng đặc trưng, tính lại chỉ số từ dự đoán đã lưu, kiểm tra mô hình và xuất hình.
7. Kết quả nằm trong `analysis_outputs/`; hình PDF độc lập nằm trong
   `report_source/figures/figure1.pdf` đến `figure4.pdf`.
   File `Group25_analysis_outputs.zip` được tạo khi chạy để tải kết quả về.

## Gói mới gồm những gì?

- Workbook và notebook nền lấy từ hai file bạn cung cấp trong Desktop/1.
- CSV gốc được giữ nguyên biểu diễn để tương thích các kiểm tra hash của thực nghiệm.
  `compare_uploaded_sources.py` so sánh từng cột với workbook, chuẩn hóa ngày/giờ Excel.
- Module Python, mô hình đã huấn luyện và dự đoán/thực nghiệm đã lưu kế thừa từ gói cũ.
- Bổ sung các ô balanced search, nested temporal evaluation và các phụ thuộc còn thiếu.
- Bộ hình xuất trực tiếp bằng mã; Hình 3 dùng Matplotlib với kiểm tra nhãn không chạm cột.
- `PACKAGE_INFO.json` ghi nguồn, hash và phần kế thừa; `SHA256SUMS.txt` kiểm tra toàn bộ file.
- Không kèm bản PDF/LaTeX lịch sử để tránh nhầm với bài Overleaf bạn đang sửa.

## Hiểu đúng chế độ chạy

Chế độ mặc định kiểm tra lại kết quả của mô hình đã lưu; không phải huấn luyện mới.
Các cờ RUN_* = True yêu cầu chạy lại thực nghiệm và có thể mất nhiều thời gian.
Xem REPRODUCE.md trước khi bật. Kết quả từng mô hình có thể khác nhẹ khi huấn luyện
trên hệ điều hành/phần cứng khác. Các kiểm tra khắt khe sẽ báo khác biệt thay vì sửa số.

Kiểm tra tại máy dựng được ghi ở `verification/notebook_execution.json` và
`verification/local_checks.json`. Đây là kiểm tra Python cục bộ, không phải chứng nhận
đã thao tác giao diện Google Colab. Dữ liệu test đã từng được xem; kết luận mang tính khám phá.

Khi đưa lên GitHub, tải các file/thư mục đã giải nén để người đọc xem được mã trực tiếp.
Có thể thêm ZIP vào một Release sau. Dữ liệu có giấy phép riêng, xem DATA_SOURCES.md.
