1. Mô tả chương trình

- Chương trình Chess AI sử dụng thuật toán Alpha-Beta Pruning và Iterative Deepening

- Có 11 cấp độ khó khác nhau từ dễ đến khó

- AI tự động chơi cả 2 bên (Trắng và Đen)


2. Yêu cầu hệ thống

- Cần cài đặt Python 3.7 hoặc mới hơn

- Các thư viện cần thiết:

	+ python-chess

	+ pillow

	+ tkinter
	+ pandas 
	+ matplotlib

3. Hướng dẫn cài đặt

- Tải toàn bộ thư mục chương trình

- Tạo thư mục 'assets' trong cùng thư mục với file chess_ai.py

- Đặt các file ảnh quân cờ vào thư mục assets

- Các file ảnh cần có tên theo quy định:

	+ PW.png (Tốt Trắng)

	+ r.png (Xe Đen)

(xem thêm trong file code phần PIECE_MAPPING)

4. Cấu trúc thư mục
chess-ai/
├── chess_ai.py          # Mã nguồn chính
├── test_ai.py           # File test AI 
├── assets/              # Thư mục chứa ảnh quân cờ
│   ├── PW.png           # Tốt Trắng
│   ├── r.png            # Xe Đen
│   ├── ...              # Các ảnh khác (xem PIECE_MAPPING)
└── README.md            # Hướng dẫn này

5. Cách chạy chương trình

- Mở terminal/command prompt

- Di chuyển đến thư mục chứa chương trình

- Chạy lệnh: python chess_ai.py để chạy thử chương trình chess

- Chạy lệnh: python test_ai.py để chạy thử thử nghiệm


6. Chức năng chính

- Bàn cờ hiển thị bằng giao diện đồ họa

- Có thanh trượt điều chỉnh tốc độ AI

- Hiển thị lịch sử các nước đi

- Tự động kết thúc khi ván cờ kết thúc


7. Lưu ý quan trọng

- Nếu không có thư mục assets, chương trình vẫn chạy nhưng hiển thị bằng ký tự

- File chess_ai.log sẽ ghi lại các lỗi nếu có

- Có thể chỉnh sửa code để thay đổi cách AI đánh giá thế cờ
