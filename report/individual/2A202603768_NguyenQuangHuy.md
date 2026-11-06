# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603768
- **Họ và tên:** Nguyễn Quang Huy
- **SEED** (in ở ô đầu notebook): 2800

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

|Mô hình|	F1 test|	F1 SUB 40|	p95 (ms/tin)|	USD/1k tin|
|1. BoW + LogReg (tự code)|	0.9239|	0.9474|	0.0828|	0.0|
2. BoW + MLP (Deep Learning)	0.9662	0.9474	0.3748	0.0
3a. Encoder (MiniLM) + kNN	0.9541	1.0000	67.8690	0.0
3b. Encoder (MiniLM) + LogReg	0.9854	1.0000	38.7296	0.0
3c. Encoder (MiniLM) + MLP head	0.9806	1.0000	63.2966	0.0
4a. LLM nhỏ zero-shot	0.9048	0.9048	3440.4777	0.0
4b. LLM lớn zero-shot	0.9474	0.9474	2419.0088	0.0
4c. LLM nhỏ few-shot	0.9000	0.9000	3497.1713	0.0

## 3. Dự đoán và thực tế

(a) ml_lr, learning rate.

Dự đoán: lr=50 làm loss dao động mạnh và không hội tụ.
Thực tế: loss vọt lên khoảng 2,5 ở vài epoch đầu rồi tụt xuống và hội tụ về 0,006.
Vì sao lệch: gradient của logistic regression bị chặn vì |p − y| ≤ 1, nên trọng số không nổ tung. Khi sigmoid bão hòa thì các tin đã đoán đúng gần như không còn tạo gradient. Loss train nhỏ chưa chắc mô hình tốt, phải so F1 trên tập val.

(b) llm_fewshot, few-shot.

Dự đoán: F1 tăng lên 0,92 đến 0,95.
Thực tế: F1 = 0,900, thấp hơn zero-shot (0,905), vẫn sai 4/40 tin.
Vì sao lệch: 4 ví dụ lấy theo thứ tự đầu của tập train, không chọn lọc, nên không dạy thêm ranh giới nào. Prompt dài gấp khoảng 2 lần nên chỉ tốn thêm token input.

(c) hard_eval, 50 case khó.

Dự đoán: BoW chặn nhầm tin ham có từ khoá, LLM yếu nhất ở prompt injection, LLM nhỏ đạt 0,75 đến 0,85.
Thực tế: BoW đúng 100% ở hai nhóm ham (“tưởng spam” và “nói về spam”), LLM đúng 6/6 ở nhóm injection, LLM nhỏ đạt accuracy 0,92.
Vì sao lệch: BoW đúng ở nhóm ham chỉ vì gần như đoán ham cho mọi tin (F1 chỉ 0,286), không phải vì hiểu ngữ cảnh. LLM đọc tin như dữ liệu nên không bị lệnh “ignore all previous instructions” đổi nhãn. Nhóm này chỉ có 6 tin nên chưa đủ kết luận prompt phòng thủ có tác dụng.

## 4. Phân tích 50 case khó

Tụt nhiều nhất: BoW + LogReg, F1 từ 0,947 (SUB) xuống 0,286 trên 50 case khó (0,263 khi thêm 5 case của em). Nhóm tệ nhất là spam giọng tự nhiên (đúng 0%), thao túng (12%) và né từ khoá (17%). Hai bộ có phân phối khác nhau (SUB cân bằng 20/20, bộ khó 29 spam/21 ham) nên em so theo xu hướng, không so số lẻ.

Case #11 (giả làm người quen ở phòng gym, “3 months for the price of 1, just reply JOIN”): BoW đoán ham. Tin không có “free”, “win”, “prize” hay số điện thoại. Các từ còn lại (“hey”, “this week”, “reply”, “sort it”) phổ thông, trọng số gần 0 hoặc âm, nên tổng w·x + b không vượt 0,5.

Case #14 (việc làm tại nhà, “making about £300 a day, no experience needed, just text me back”): BoW cũng đoán ham. Dữ liệu train toàn spam quảng cáo kiểu cũ (prize, txt, claim, call), không có kiểu lừa việc làm. BoW chỉ cộng điểm từng từ nên không nhận ra mẫu “kiếm tiền dễ dàng, nhắn lại để biết cách”.

Embedding cũng yếu ở hai nhóm này (3b chỉ đúng 10% ở spam giọng tự nhiên) vì ranh giới chỉ học từ nhãn train. LLM đúng 100% ở các nhóm spam vì không phụ thuộc phân phối train, nhưng bị từ khoá kéo ở tin ham (#2 “FREE pizza”, #8, #9, #47).

## 5. Thử thách cá nhân

Đề #5: Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.

Cách làm: dùng lại các biến vocab, w, b, emb_model, mu, sd, w_e, b_e, tính xác suất p(spam) của từng tin bằng cả hai mô hình.

Tin	Thật	BoW (p)	Embedding (p)
reward... respond with your postcode	spam	ham ✗ (0,31)	ham ✗ (0,44)
picked for a gift, tap the link	spam	ham ✗ (0,29)	spam ✓ (0,72)
Are you free to grab coffee	ham	ham ✓ (0,24)	ham ✓ (0,01)
hope you win the match	ham	ham ✓ (0,05)	ham ✓ (0,00)

BoW đúng 2/4, Embedding đúng 3/4.

Giải thích:

Tin 1: không có từ khoá quen thuộc. BoW chỉ cộng điểm các từ riêng lẻ nên không đủ. Embedding đã nghiêng về spam (0,44) nhưng chưa qua ngưỡng, vì kiểu “trả lời kèm postcode” hiếm trong train.
Tin 2: embedding đọc cả câu “được chọn, bấm link nhận quà” nên gần cụm spam trúng thưởng. BoW không có từ mạnh nào.
Tin 3 và 4: cả hai đúng. Embedding chắc hơn (p = 0,01 và 0,00 so với 0,24 và 0,05) vì hiểu “free” nghĩa là rảnh và “win” là lời chúc.

## 6. Phương án cho sản phẩm thật

Chọn: BoW + MLP làm tầng chính, kèm router lên tầng sau.

Số liệu độ trễ: p95 = 0,34 ms, khoảng 29.000 tin/s, trong khi nhu cầu trung bình chỉ khoảng 116 tin/s. Embedding có p95 57 đến 65 ms nên vượt SLA, LLM có p95 khoảng 2,4 giây nên không đạt.
Số liệu chất lượng: F1 test = 0,966.
Chi phí LLM nếu dùng cho toàn bộ: gpt-6-luna cần 100 × 0,10/1M + 1 × 0,50/1M = 0,0000105 USD mỗi tin, tức khoảng 105 USD/ngày, khoảng 3.150 USD/tháng.

Router: BoW + MLP xử lý mọi tin, tin chắc chắn (p < 0,1 hoặc p > 0,9) quyết định luôn. Tin lưỡng lự đi qua Embedding + LogReg, tin còn không chắc hoặc hai tầng bất đồng mới lên LLM. Em dùng dải rộng vì BoW cho spam thật chỉ p = 0,29 đến 0,31, nằm ngoài dải [0,3; 0,7]. Nếu khoảng 10% tin lên LLM thì chi phí còn khoảng 300 USD/tháng.

Đánh đổi chấp nhận:

BoW + MLP chỉ đạt F1 khoảng 0,45 trên 50 case khó, nên sẽ bỏ sót spam kiểu mới (giọng quen biết, mạo danh). Router giảm bớt nhưng không loại hết.
Nếu hơn 5% tin đi qua LLM thì p95 của cả hệ thống vượt SLA, nên nhánh LLM phải chạy bất đồng bộ hoặc giữ tỉ lệ dưới 5%.
Số đo độ trễ lấy trên CPU Colab, môi trường thật có thể khác.

## 7. AI log và điều đã học

Mỗi khi sử dụng AI, em thường đặt câu hỏi về những vấn đề cần giải quyết trước mắt mà bản thân chưa hiểu rõ hoặc chưa biết cách thực hiện. Sau khi nhận được câu trả lời, em chủ động kiểm tra, đối chiếu và đánh giá độ chính xác cũng như tính đáng tin cậy của thông tin. Thay vì sao chép nguyên văn nội dung từ AI, em sẽ diễn đạt lại bằng ngôn ngữ và cách hiểu của bản thân để đảm bảo nắm vững kiến thức. Qua buổi học hôm nay, em nhận ra rằng độ chính xác (Accuracy) cao không đồng nghĩa với khả năng phát hiện spam tốt, đồng thời mô hình phức tạp hơn chưa chắc mang lại hiệu quả vượt trội. Vì vậy, việc lựa chọn mô hình cần dựa trên sự cân bằng giữa chất lượng, tốc độ xử lý và chi phí triển khai.
