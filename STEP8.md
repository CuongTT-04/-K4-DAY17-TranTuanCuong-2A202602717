# Báo Cáo Phân Tích Hiệu Năng Hệ Thống Memory (Day 17 - Track 3)

Tài liệu này trình bày kết quả benchmark thực tế, phân tích định lượng dựa trên số liệu đo lường, và luận giải các cơ chế mở rộng (bonus) cho hệ thống **Memory Systems for AI Agent**.

---

## 1. Kết quả Benchmark Thực Nghiệm

Dữ liệu dưới đây được đo lường thực tế bằng lệnh `python src/benchmark.py` trên môi trường sạch:

### Bảng 1: Standard Benchmark (`data/conversations.json`)
*10 hội thoại (~10 lượt/hội thoại) của người dùng `dungct`, đánh giá khả năng nhớ qua nhiều phiên ngắn bình thường.*

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1,098 | 12,024 | 0.00 | 0.30 | 0 | 0 |
| **Advanced** | 1,828 | 20,406 | **0.96** | **0.97** | 258 | 0 |

### Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json`)
*1 hội thoại 16 lượt dày đặc của `dungct_stress`, kiểm tra khả năng nén ngữ cảnh khi lịch sử hội thoại rất dài.*

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 222 | 21,433 | 0.00 | 0.30 | 0 | 0 |
| **Advanced** | 497 | **10,867** | **1.00** | **1.00** | 207 | **3** |

---

## 2. Phân Tích Định Lượng & Chuỗi Logic Cốt Lõi

### Câu hỏi 1: Vì sao Advanced Agent có Recall vượt trội so với Baseline Agent?
* **Số liệu thực tế:** 
  * Ở bảng Standard: Baseline đạt `0.00` recall, trong khi Advanced đạt **`0.96`** (Response quality đạt **`0.97`**).
  * Ở bảng Stress: Baseline tiếp tục đạt `0.00`, trong khi Advanced đạt tuyệt đối **`1.00`** (Response quality **`1.00`**).
* **Cơ chế kỹ thuật:**
  * **Baseline Agent** cô lập `SessionState` chặt chẽ theo `thread_id`. Khi câu hỏi recall được đặt trong một thread mới (`{conv_id}-recall-{idx}`), Baseline hoàn toàn không có lịch sử hội thoại trước đó và không có bộ nhớ ngoài, do đó không thể trả lời các sự kiện từ phiên cũ (Recall = 0).
  * **Advanced Agent** sở hữu tầng **Persistent Memory** với `UserProfileStore`:
    1. Hàm `extract_profile_updates()` phân tách các fact ổn định (tên, nơi ở, nghề nghiệp, sở thích) ra khỏi dòng hội thoại thô.
    2. Các fact này được ghi bền vững xuống đĩa dưới file `state/profiles/{user_id}/User.md` (thể hiện qua `Memory growth = 258 bytes`).
    3. Khi sang thread mới, hàm `_offline_response()` nạp trực tiếp hồ sơ từ `User.md` để trả lời chính xác thông tin cá nhân của người dùng bất kể phiên làm việc đã đóng.

---

### Câu hỏi 2: Vì sao Advanced Agent lại tốn nhiều token hơn ở hội thoại ngắn?
* **Số liệu thực tế (Bảng Standard):**
  * `Agent tokens only`: Advanced tốn $1,828$ tokens so với $1,098$ tokens của Baseline (tăng $\approx 66.5\%$).
  * `Prompt tokens processed`: Advanced tiêu thụ $20,406$ tokens so với $12,024$ tokens của Baseline (tăng $\approx 69.7\%$).
  * `Compactions = 0` ở cả hai agent.
* **Cơ chế kỹ thuật:**
  * Ở các cuộc hội thoại ngắn (~10 lượt, mỗi lượt vài câu ngắn), tổng lượng token trong thread chưa bao giờ vượt qua ngưỡng `compact_threshold_tokens = 1000`. Do đó, cơ chế compact hoàn toàn không được kích hoạt (`Compactions = 0`).
  * Trong mỗi lượt của Advanced Agent, prompt ngữ cảnh không chỉ chứa các tin nhắn gần nhất mà còn phải **nhúng toàn bộ nội dung của `User.md`** để phục vụ việc cá nhân hóa:
    $$\text{Prompt Tokens} = \text{Tokens}(User.md) + \text{Tokens}(Summary) + \text{Tokens}(Recent\ Messages)$$
  * Việc nhúng `User.md` tạo ra một khoản chi phí cố định (overhead) trên từng lượt. Khi hội thoại ngắn, khoản tiết kiệm từ việc nén là **bằng 0**, khiến Advanced Agent tốn kém hơn Baseline cả về prompt context lẫn output generation.

---

### Câu hỏi 3: Vì sao Compact Memory lại tạo ra bước ngoặt ở hội thoại dài?
* **Số liệu thực tế (Bảng Stress):**
  * `Prompt tokens processed`: Baseline tiêu tốn tới **$21,433$ tokens**, trong khi Advanced chỉ tiêu tốn **$10,867$ tokens** — giúp tiết kiệm **$49.3\%$ chi phí ngữ cảnh**.
  * `Compactions`: Advanced kích hoạt nén **$3$ lần**, trong khi Baseline là $0$.
  * `Agent tokens only`: Advanced ($497$) vẫn cao hơn Baseline ($222$) do sinh câu trả lời đầy đủ và cấu trúc 3-bullet.
* **Cơ chế kỹ thuật:**
  * **Tại sao chỉ tối ưu `Prompt tokens processed` mà không tối ưu `Agent tokens only`?**
    * `Agent tokens only` đo độ dài câu trả lời do model tự sinh. Advanced trả lời có chiều sâu, đủ 3 ý bullet và ví dụ thực chiến nên token output tự nhiên sẽ dài hơn câu trả lời ngắn/từ chối của Baseline.
    * `Prompt tokens processed` đo tổng khối lượng ngữ cảnh mà agent phải "gánh" tích lũy qua từng lượt:
      * Với Baseline: Ngữ cảnh phình to tuyến tính theo số lượt $N$, tổng prompt xử lý tăng theo cấp số nhân $O(N^2)$ vì lượt thứ 16 phải đọc lại nguyên văn toàn bộ 15 lượt trước.
      * Với Advanced: Khi tích lũy vượt ngưỡng token, `CompactMemoryManager` cắt các tin nhắn cũ và chuyển thành bản tóm tắt ngắn gọn có giới hạn (bounded summary), chỉ giữ nguyên văn `compact_keep_messages = 4` tin nhắn gần nhất. Nhờ đó, chi phí ngữ cảnh được "chặn trần" và tăng trưởng gần như tuyến tính $O(N)$ thay vì $O(N^2)$.

---

### Câu hỏi 4: Tốc độ tăng trưởng của file Memory và rủi ro đi kèm trong thực tế?
* **Quan sát số liệu:**
  * `Memory growth (bytes)` tăng từ $0$ lên $258$ bytes ở Standard Benchmark và $207$ bytes ở Stress Benchmark.
  * Tốc độ tăng trưởng chậm và ổn định do bộ trích xuất fact chỉ lưu trữ các thực thể cốt lõi dạng key-value Markdown thay vì nối toàn bộ văn bản.
* **Phân tích rủi ro trong môi trường Production:**
  1. **Rủi ro Memory Bloat (File phình vô hạn):** Nếu người dùng trò chuyện hàng tháng/hàng năm mà mọi chi tiết vụn vặt đều bị nhét vào `User.md`, file profile sẽ vượt quá context window của model hoặc làm tăng vọt chi phí inference ở mọi lượt chat đầu vào.
  2. **Rủi ro Fact Poisoning / Hallucination Tích Lũy:** Nếu hệ thống vô tình ghi nhận một câu nói đùa, câu giả định hoặc thông tin sai vào `User.md`, thông tin sai lệch này sẽ trở thành "chân lý vĩnh viễn" (persistent truth), đầu độc câu trả lời của agent ở mọi phiên sau.
  3. **Chi phí I/O và Đỗ trễ (Latency):** Việc đọc/ghi đĩa liên tục trên từng lượt chat có thể gây bottleneck khi scale lên hàng nghìn người dùng đồng thời.

---

## 3. Phần Mở Rộng Kỹ Thuật (Bonus - Hướng tới mốc 90-100 điểm)

Hệ thống đã triển khai hai cơ chế mở rộng nâng cao trong [src/memory_store.py](file:///d:/VinAI/day17-cohort4-MemorySystems4Agent/src/memory_store.py) để giải quyết các rủi ro trên:

### Bonus 1: Cơ chế Xử lý Xung đột và Đính chính (Conflict Handling & Correction Logic)
1. **Vấn đề giải quyết:**
   * Trong thực tế, thông tin người dùng không cố định mà thay đổi theo thời gian (ví dụ: `dungct` đổi nơi ở từ Đà Nẵng sang Huế; `dungct_stress` chuyển từ Huế sang làm việc ở Đà Nẵng; nghề nghiệp đổi từ Backend sang MLOps).
   * Ngoài ra, người dùng thường đưa ra thông tin gây nhiễu: *"Hà Nội chỉ là nơi đi họp 2 ngày"* hay *"product manager chỉ là câu đùa"*.
2. **Cách cải thiện Recall và Token:**
   * Thay vì nối thêm (append) văn bản mù quáng làm file profile phình to và chứa 2 đáp án mâu thuẫn, hàm `upsert_fact()` và regex trong `extract_profile_updates()` nhận diện các mẫu câu đính chính (*"không còn ở..."*, *"đã cập nhật từ..."*).
   * Giá trị cũ bị xóa hoàn toàn và thay thế bằng giá trị mới duy nhất. Nhờ vậy:
     * File memory giữ kích thước siêu gọn (~$200-250$ bytes).
     * Tránh được hiện tượng agent trả lời nước đôi hoặc nhớ nhầm thông tin cũ, đưa Recall câu hỏi khó tại Stress Test lên **$1.00$ tuyệt đối**.
3. **Rủi ro kỹ thuật phát sinh:**
   * **Rủi ro False Overwrite (Ghi đè nhầm):** Nếu regex hoặc LLM hiểu sai ngữ cảnh (ví dụ: người dùng nói *"Mình nhớ hồi ở Đà Nẵng..."* nhưng agent lại tưởng người dùng chuyển về Đà Nẵng), fact hiện tại sẽ bị xóa mất mà không thể khôi phục nếu không có hệ thống versioning / event-sourcing.

### Bonus 2: Bộ lọc Câu hỏi & Ngưỡng tự tin (Question Filtering & Confidence Guardrail)
1. **Vấn đề giải quyết:**
   * Ngăn chặn việc người dùng đặt câu hỏi kiểm tra (*"Mình tên gì?"*, *"Bạn có nhớ style mình thích không?"*) bị hệ thống bóc tách nhầm thành fact mới (ví dụ nhầm từ *"gì"* thành tên người dùng).
2. **Cách cải thiện Recall và Token:**
   * Hệ thống kiểm tra mẫu câu hỏi (`is_pure_question`): bỏ qua hoàn toàn các câu kết thúc bằng dấu `?` hoặc chứa từ khóa truy vấn (`nhắc lại`, `hỏi lại`, `cho mình biết`).
   * Chỉ ghi nhận khi người dùng chủ động tuyên bố dữ kiện. Nhờ đó, file `User.md` không bị nhiễm rác, dữ liệu luôn sạch và chất lượng phản hồi duy trì ở mức **$0.97 - 1.00$**.
3. **Rủi ro kỹ thuật phát sinh:**
   * **Rủi ro False Negative (Bỏ sót thông tin):** Trong trường hợp người dùng đặt câu hỏi phức hợp có lồng ghép thông tin mới (ví dụ: *"Mình vừa chuyển sang làm MLOps, bạn có gợi ý lộ trình học nào không?"*), hệ thống có thể phân loại nhầm đây là câu hỏi và bỏ qua fact nghề nghiệp mới.
