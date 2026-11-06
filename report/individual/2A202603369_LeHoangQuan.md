Dựa trên hình ảnh bạn cung cấp, đây là nội dung mẫu để bạn điền vào file Markdown (`MSSV_HoTen.md`) cho báo cáo cá nhân của mình. Bạn hãy thay thế các phần thông tin cá nhân và số liệu cho phù hợp nhé.

---

# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

**MSSV:** 2A202603369
**Họ và tên:** Lê Hoàng Quân
**SEED** (in ở ô đầu notebook): 1101

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](https://www.google.com/search?q=day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](https://www.google.com/search?q=lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](https://www.google.com/search?q=hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính

| Phương pháp | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.890 | 0.919 | 0.071 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.934 | 0.974 | 0.210 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.911 | 0.950 | 50.297 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.963 | 1.000 | 31.794 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.963 | 1.000 | 31.888 | 0.000 |
| 4a. LLM nhỏ zero-shot | 0.000 | 0.000 | 1918.707 | 0.005 |
| 4b. LLM lớn zero-shot | 0.000 | 0.000 | 1185.367 | 0.047 |
| 4c. LLM nhỏ few-shot | 0.000 | 0.000 | 1411.607 | 0.016 |

## 3. Dự đoán và thực tế

* **Dự đoán 1:** `predict("dl_curve", "...")`
* **Dự đoán:** Validation Loss sẽ giảm ở các epoch đầu rồi tăng lại (dấu hiệu overfit).
* **Thực tế:** Đúng như dự đoán, Validation Loss đạt điểm thấp nhất tại epoch 6 rồi bắt đầu tăng, trong khi Train Loss tiếp tục giảm.
* **Vì sao lệch (nếu có):** Không lệch, kết quả phản ánh rõ hiện tượng overfitting.


* **Dự đoán 2:** `predict("llm_models", "...")`
* **Dự đoán:** LLM sẽ có F1 tương đương hoặc nhỉnh hơn Embedding nhưng thua xa về tốc độ.
* **Thực tế:** LLM zero-shot cho kết quả F1 = 0 trên tập test nhỏ, thấp hơn rất nhiều so với Embedding, và tốc độ cực kỳ chậm (p95 > 1000ms).
* **Vì sao lệch:** Do đánh giá quá cao khả năng phân loại zero-shot của LLM trong khi không tinh chỉnh kỹ prompt và các thông số (như format đầu ra không chuẩn dẫn đến invalid cao).


* **Dự đoán 3:** `predict("hard_eval", "...")`
* **Dự đoán:** Accuracy của BoW sẽ rớt mạnh nhất ở nhóm "Né từ khóa", LLM dễ dính "Prompt injection".
* **Thực tế:** BoW đúng 17% ở nhóm "Né từ khóa", LLM nhỏ zero-shot đúng 0% ở nhóm "Prompt injection".
* **Vì sao lệch:** Phù hợp với dự đoán. Các mô hình truyền thống thất bại khi gặp nhiễu dữ liệu, còn LLM thất bại khi bị dẫn dắt bởi chỉ thị độc hại.



## 4. Phân tích 50 case khó

* **Model tụt mạnh nhất:** Nhóm các mô hình BoW (BoW + LogReg, BoW + MLP).
* **Ở nhóm case:** "Thao túng / lừa đảo", "Spam giọng tự nhiên", "Né từ khoá".
* **2 case cụ thể và cơ chế khiến model sai:**
* **Case 29 ("Né từ khoá"):** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd.` Mô hình BoW sai vì các ký tự chữ bị thay bằng số, tạo ra các token không có trong từ điển (vocab) đã học, dẫn đến mô hình không nhận diện được bất kỳ từ khóa spam nào.
* **Case 11 ("Spam giọng tự nhiên"):** `Hey it's Jess from the gym...` Mô hình BoW sai vì không có các từ khóa spam điển hình (như "FREE", "WIN"), mà sử dụng ngôn ngữ tự nhiên, giao tiếp hàng ngày, khiến mô hình nhầm lẫn với tin nhắn hợp lệ (ham).



## 5. Thử thách cá nhân

* **Đề bài:** ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?
* **Cách làm:** Viết vòng lặp để lấy giá trị xác suất từ `predict_proba(X_te, w, b)`, sau đó gán nhãn dựa trên các ngưỡng [0.2, 0.5, 0.8] và tính toán các chỉ số metrics tương ứng.
* **Kết quả:**
* Ngưỡng 0.2 -> Precision: 0.865 | Recall: 0.938 | F1: 0.900
* Ngưỡng 0.5 -> Precision: 1.000 | Recall: 0.802 | F1: 0.890
* Ngưỡng 0.8 -> Precision: 1.000 | Recall: 0.656 | F1: 0.792


* **Giải thích:** Ngân hàng nên chọn ngưỡng 0.8. Để tránh chặn nhầm tin nhắn OTP quan trọng (ham thành spam), mô hình cần có Precision cao nhất (giảm thiểu False Positives). Ngưỡng 0.8 đảm bảo Precision đạt 1.000, dù phải chấp nhận bỏ sót một số tin spam (Recall giảm).

## 6. Phương án cho sản phẩm thật

* **Chọn phương án:** 3b. Encoder (MiniLM) + LogReg.
* **Dẫn số liệu:** Mô hình đạt độ chính xác F1 rất cao (0.963 trên tập test, 1.000 trên tập SUB). Tốc độ xử lý đạt yêu cầu với độ trễ p95 là 31.794 ms (thoả mãn SLA < 50 ms/tin). Chi phí vận hành rẻ (không tốn chi phí API/USD như LLM).
* **Chấp nhận đánh đổi:** Phải xử lý tiền xử lý (encode) văn bản qua mô hình ngôn ngữ (MiniLM) nên tốc độ không thể cực nhanh (như BoW + LogReg chỉ tốn 0.071 ms), đồng thời yêu cầu phần cứng mạnh hơn một chút để xử lý embedding.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1. ML | "Giải thích cách hoạt động của dòng `w = w - lr * (X.T @ error) / n` khi đoán sai tin spam có chứa chữ 'free'." | AI giải thích việc tính toán gradient và hướng cập nhật trọng số. | Tôi đã kiểm chứng lại với công thức Gradient Descent và viết lại bằng câu văn của mình. |
| 2. DL | "Dự đoán sự thay đổi Validation Loss khi huấn luyện." | AI dự đoán sẽ có hiện tượng overfitting sau vài epoch. | Đã đối chiếu với biểu đồ thực tế khi chạy thử nghiệm và xác nhận epoch có validation loss thấp nhất. |
| 3. kNN | "Viết hàm `knn_predict` để lấy k tin nhắn gần nhất và quyết định nhãn." | AI cung cấp đoạn code lấy index gần nhất và tính tổng vote. | Đã rà soát lại để đảm bảo không dùng thư viện ngoài và tối ưu logic phân loại. |

* **Điều tôi hiểu ra hôm nay mà trước đó chưa hiểu:**
* Sự khác biệt rõ ràng giữa việc học đặc trưng (Feature Learning) của Deep Learning/Embedding so với việc chọn đặc trưng thủ công của Machine Learning (BoW).
* LLM không phải lúc nào cũng là giải pháp tốt nhất. Cho các bài toán phân loại nhanh, độ trễ thấp và quy mô lớn, các mô hình nhỏ (như Embedding + ML) lại hiệu quả và tiết kiệm hơn rất nhiều so với việc gọi API LLM tốn kém.