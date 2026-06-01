// ================================================================
// RAG Configuration — Thông số điều chỉnh hệ thống RAG
// Chỉnh sửa các thông số ở đây để tối ưu hóa kết quả Chatbot
// ================================================================

export const RAG_CONFIG = {
  // Số lượng tài liệu liên quan tối đa đưa vào Prompt cho Gemini
  // Tăng → Câu trả lời có nhiều thông tin hơn nhưng tốn nhiều Token hơn
  TOP_K: 3,

  // Ngưỡng điểm Cosine tối thiểu để xem tài liệu là "liên quan"
  // Khoảng 0.0 - 1.0. Tăng → Chặt hơn, giảm → Lỏng hơn
  SIMILARITY_THRESHOLD: 0.60,

  // Điểm thưởng cho sự kiện mà người dùng ĐANG XEM trong không gian 3D
  // (Context-Aware Boost). Điểm này cộng thêm vào Cosine Score để ưu tiên
  // sự kiện hiện tại lên đầu danh sách kết quả truy vấn.
  CONTEXT_BOOST_SCORE: 0.25,

  // Model Embedding để vector hóa văn bản (768 chiều)
  EMBEDDING_MODEL: "text-embedding-004",

  // Model LLM để sinh câu trả lời cuối cùng (Khuyến nghị: gemini-1.5-flash ổn định hơn và ít bị 503)
  GENERATION_MODEL: "gemini-2.5-pro",

  // Số token tối đa cho câu trả lời (Tăng lên 1024 để tránh bị ngắt nửa câu)
  MAX_OUTPUT_TOKENS: 4000,

  // Độ sáng tạo của AI (0 = chính xác, 1 = sáng tạo)
  TEMPERATURE: 0.4,
};
