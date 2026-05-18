// Vercel Serverless Function — Proxy an toàn tới Gemini API
// API key được giữ bí mật trong Vercel Environment Variables
export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { message } = req.body || {};
  if (!message?.trim()) return res.status(400).json({ error: "Thiếu nội dung tin nhắn" });

  const GEMINI_KEY   = process.env.GEMINI_API_KEY;
  const FB_KEY       = process.env.FIREBASE_API_KEY;
  const FB_PROJECT   = process.env.FIREBASE_PROJECT_ID || "trienlam3d-84c03";

  try {
    // 1. Lấy dữ liệu hiện vật từ Firestore để làm context cho AI
    let context = "Đây là bảo tàng ảo 3D UTC. Chưa có dữ liệu hiện vật.";
    try {
      const fsRes = await fetch(
        `https://firestore.googleapis.com/v1/projects/${FB_PROJECT}/databases/(default)/documents/museum/data?key=${FB_KEY}`
      );
      if (fsRes.ok) {
        const fsData = await fsRes.json();
        const db = JSON.parse(fsData.fields?.jsonData?.stringValue || "{}");
        const exhibits = (db.exhibits || []).filter(e => e.isPublished !== false);
        context = `Bảo tàng: ${db.museumName || "Triển Lãm 3D UTC"}\nTổng số hiện vật: ${exhibits.length}\n\nDanh sách hiện vật:\n` +
          exhibits.map(e => {
            const yr = e.yearRange?.from
              ? (e.yearRange.from === e.yearRange.to
                  ? `Năm: ${e.yearRange.from}`
                  : `Giai đoạn: ${e.yearRange.from}–${e.yearRange.to}`)
              : "";
            const tags = (e.tags || []).length > 0 ? `Tags: [${e.tags.join(", ")}]` : "";
            return `- ${e.name} (ID: ${e.id}, Thể loại: ${e.category || "Chung"}${yr ? ", " + yr : ""}${tags ? ", " + tags : ""}): ${e.description || "Chưa có mô tả"}`;
          }).join("\n");
      }
    } catch (_) {}

    // 2. Gọi Gemini API
    const prompt = `Bạn là hướng dẫn viên ảo tại bảo tàng 3D UTC. Trả lời ngắn gọn, thân thiện bằng tiếng Việt (tối đa 4 câu).

Thông tin bảo tàng:
${context}

Câu hỏi: ${message}`;

    const geminiRes = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { temperature: 0.7, maxOutputTokens: 400 }
        })
      }
    );

    const geminiData = await geminiRes.json();
    const reply = geminiData.candidates?.[0]?.content?.parts?.[0]?.text
      || "Xin lỗi, tôi không thể trả lời lúc này. Vui lòng thử lại!";

    return res.status(200).json({ reply });

  } catch (err) {
    return res.status(500).json({ error: "Lỗi server", message: err.message });
  }
}
