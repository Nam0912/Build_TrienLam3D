import { RAG_CONFIG } from "./rag-config.js";


let chunkCache = null;          
let cacheExhibitCount = 0;
let cacheTimestamp = 0;
const CACHE_MAX_AGE = 30 * 60 * 1000;

const GREETING_PATTERNS = [
  /^(xin\s+)?chào/i,
  /^hello/i, /^hi\b/i, /^hey\b/i,
  /^chào\s+(bạn|anh|chị|em|mọi\s+người)/i,
];

const FAREWELL_PATTERNS = [
  /^(tạm\s+biệt|bye|goodbye)/i,
  /^cảm\s+ơn/i, /^cám\s+ơn/i,
  /^thank/i,
];

const GREETING_RESPONSES = [
  "Xin chào! Tôi là Tư vấn viên ảo của triển lãm. Bạn có thể hỏi tôi về bất kỳ tác phẩm nào đang trưng bày nhé!",
  "Chào bạn! Hãy hỏi tôi bất kỳ điều gì về các tác phẩm trong triển lãm! 🎨",
];

const FAREWELL_RESPONSES = [
  "Cảm ơn bạn đã tham quan! Chúc bạn có trải nghiệm tuyệt vời tại triển lãm. 🎨",
  "Rất vui được hỗ trợ bạn! Nếu cần thêm thông tin, đừng ngại hỏi nhé!",
];

function classifyIntent(message) {
  const trimmed = message.trim();
  for (const p of GREETING_PATTERNS) {
    if (p.test(trimmed)) return "GREETING";
  }
  for (const p of FAREWELL_PATTERNS) {
    if (p.test(trimmed)) return "FAREWELL";
  }
  return "SPECIFIC";
}

const BLOCKED_PATTERNS = [
  /ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)/i,
  /bỏ\s+qua\s+(tất\s+cả\s+)?(quy\s+tắc|hướng\s+dẫn|lệnh)/i,
  /pretend\s+(you\s+are|to\s+be)/i,
  /giả\s+(vờ|bộ|làm)\s/i,
  /write\s+(me\s+)?(a\s+)?(code|script|program)/i,
  /viết\s+(cho\s+(tôi|mình)\s+)?(code|mã|kịch\s+bản|chương\s+trình)/i,
  /system\s*prompt/i,
  /jailbreak/i,
  /hãy\s+(quên|bỏ|xóa)\s+(hết|tất\s+cả|mọi)/i,
];

function checkSafety(message) {
  for (const pattern of BLOCKED_PATTERNS) {
    if (pattern.test(message)) {
      return { safe: false, reason: "blocked_pattern" };
    }
  }
  return { safe: true };
}

function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;

  return dotProduct / denominator;
}

async function getEmbedding(text, apiKey) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${RAG_CONFIG.EMBEDDING_MODEL}:embedContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: `models/${RAG_CONFIG.EMBEDDING_MODEL}`,
      content: { parts: [{ text: text }] },
      taskType: "RETRIEVAL_QUERY",
    }),
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(`Embedding API lỗi: ${err.error?.message || response.status}`);
  }

  const data = await response.json();
  return data.embedding?.values ?? null;
}

// ================================================================
// 5. CHUNKING — Chia nhỏ tài liệu thành các đoạn (chunk)
// Mỗi hiện vật tạo ra 1 chunk chính + N chunk phụ (mỗi trang page)
// Giúp tìm kiếm ngữ nghĩa chính xác hơn ở cấp độ đoạn văn
// ================================================================
function createChunks(exhibits) {
  const chunks = [];

  for (const exhibit of exhibits) {
    // Chunk chính: Tên + Mô tả tổng quan + Thể loại + Phòng ban
    const mainText = [
      exhibit.name ?? "",
      exhibit.description ?? "",
      exhibit.category ?? "",
      exhibit.department ?? "",
    ].filter(Boolean).join(". ");

    if (mainText.trim()) {
      chunks.push({
        chunkId: `${exhibit.id}__main`,
        exhibitId: exhibit.id,
        text: mainText,
        vector: null,
      });
    }

    // Chunk phụ: Mỗi trang (page) thuyết minh là một chunk riêng biệt
    (exhibit.pages ?? []).forEach((page, idx) => {
      const pageText = [
        exhibit.name ?? "",        // Kèm tên hiện vật để giữ ngữ cảnh
        page.title ?? "",
        page.description ?? "",
      ].filter(Boolean).join(". ");

      if (pageText.trim() && pageText.length > 20) {
        chunks.push({
          chunkId: `${exhibit.id}__page_${idx}`,
          exhibitId: exhibit.id,
          text: pageText,
          vector: null,
        });
      }
    });
  }

  return chunks;
}

// ================================================================
// 6. EMBEDDING CACHE (Chunked) — Bộ nhớ đệm vector theo chunk
// ================================================================
async function ensureEmbeddingCache(exhibits, apiKey) {
  const now = Date.now();
  const cacheExpired = (now - cacheTimestamp) > CACHE_MAX_AGE;
  const countChanged = cacheExhibitCount !== exhibits.length;

  if (chunkCache && !cacheExpired && !countChanged) {
    console.log(`[RAG Cache] Sử dụng cache có sẵn (${chunkCache.length} chunks, tuổi: ${Math.round((now - cacheTimestamp) / 1000)}s)`);
    return;
  }

  console.log("[RAG Cache] Đang tạo chunks và tính embedding...");
  const chunks = createChunks(exhibits);

  for (const chunk of chunks) {
    try {
      chunk.vector = await getEmbedding(chunk.text, apiKey);
    } catch (e) {
      console.warn(`[RAG Cache] Lỗi embedding chunk ${chunk.chunkId}: ${e.message}`);
    }
  }

  chunkCache = chunks;
  cacheExhibitCount = exhibits.length;
  cacheTimestamp = now;
  console.log(`[RAG Cache] Hoàn tất! ${chunkCache.length} chunks từ ${exhibits.length} hiện vật.`);
}

// ================================================================
// 7. KEYWORD SEARCH — Tìm kiếm từ khóa (Fallback bậc 2)
// Được sử dụng khi Embedding API gặp lỗi
// ================================================================
function keywordSearch(question, exhibits, currentExhibitId) {
  const words = question.toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .split(/\s+/)
    .filter(w => w.length > 1);

  if (words.length === 0) return [];

  const scored = exhibits.map(exhibit => {
    const searchText = [
      exhibit.name ?? "",
      exhibit.description ?? "",
      exhibit.category ?? "",
      ...(exhibit.pages ?? []).map(p => `${p.title ?? ""} ${p.description ?? ""}`),
    ].join(" ").toLowerCase();

    let hits = 0;
    for (const word of words) {
      if (searchText.includes(word)) hits++;
    }

    // Context boost cho hiện vật đang xem
    if (currentExhibitId && exhibit.id === currentExhibitId) {
      hits += 2;
    }

    return { exhibit, hits };
  });

  return scored
    .filter(item => item.hits > 0)
    .sort((a, b) => b.hits - a.hits)
    .slice(0, RAG_CONFIG.TOP_K)
    .map(item => item.exhibit);
}

// ================================================================
// 8. RETRIEVE TOP-K (Chunked + Multi-level Fallback)
// Điểm tương đồng được tính ở cấp chunk, sau đó gom theo hiện vật
// ================================================================
async function retrieveTopK(question, exhibits, currentExhibitId, apiKey) {
  await ensureEmbeddingCache(exhibits, apiKey);

  const questionVector = await getEmbedding(question, apiKey);
  if (!questionVector) throw new Error("Không thể vector hóa câu hỏi.");

  // Tính điểm cho từng chunk
  const chunkScores = chunkCache
    .filter(chunk => chunk.vector !== null)
    .map(chunk => ({
      ...chunk,
      similarity: cosineSimilarity(questionVector, chunk.vector),
    }));

  // Gom điểm theo exhibitId: lấy điểm CAO NHẤT trong các chunk của cùng hiện vật
  const exhibitScoreMap = new Map();
  for (const cs of chunkScores) {
    const current = exhibitScoreMap.get(cs.exhibitId) || 0;
    if (cs.similarity > current) {
      exhibitScoreMap.set(cs.exhibitId, cs.similarity);
    }
  }

  // Áp dụng Context Boost và xây dựng danh sách điểm
  const scoredExhibits = exhibits.map(exhibit => {
    let similarity = exhibitScoreMap.get(exhibit.id) || 0;
    if (currentExhibitId && exhibit.id === currentExhibitId) {
      similarity += RAG_CONFIG.CONTEXT_BOOST_SCORE;
    }
    return { exhibit, similarity };
  });

  const filtered = scoredExhibits
    .filter(item => item.similarity >= RAG_CONFIG.SIMILARITY_THRESHOLD)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, RAG_CONFIG.TOP_K);

  console.log("[RAG] Kết quả truy vấn (chunked):");
  filtered.forEach(item =>
    console.log(`  - [${item.exhibit.id}] ${item.exhibit.name}: ${item.similarity.toFixed(4)}`)
  );

  // Fallback bậc 1: Nếu không có kết quả nhưng có currentExhibitId
  if (filtered.length === 0 && currentExhibitId) {
    const current = exhibits.find(e => e.id === currentExhibitId);
    if (current) {
      console.log("[RAG] Fallback bậc 1: Sử dụng hiện vật đang xem.");
      return [current];
    }
  }

  return filtered.map(item => item.exhibit);
}

// ================================================================
// 9. PROMPT BUILDER — Xây dựng Prompt (có Multi-turn History)
// ================================================================
function buildRAGPrompt(question, retrievedExhibits, companyName, allExhibits, history) {
  const contextDocs = retrievedExhibits.map((e, i) => {
    const pages = (e.pages ?? [])
      .map(p => `  [Trang: ${p.title ?? ""}] ${p.description ?? ""}`)
      .join("\n");
    return `--- Tài liệu ${i + 1}: ${e.name} (ID: ${e.id}, Phòng ban: ${e.department ?? "N/A"}) ---
Mô tả: ${e.description ?? "Chưa có mô tả"}
${pages}`;
  }).join("\n\n");

  let globalContext = "";
  if (allExhibits && allExhibits.length > 0) {
    const list = allExhibits.map((e, i) => {
      const tagsArr = [];
      if (e.category) tagsArr.push(e.category);
      if (e.department) tagsArr.push(e.department);
      if (e.tags) {
        if (Array.isArray(e.tags)) tagsArr.push(...e.tags);
        else tagsArr.push(e.tags);
      }

      const tagsString = tagsArr.filter(Boolean).join(", ");
      const tagInfo = tagsString ? ` [Thể loại/Tag: ${tagsString}]` : "";
      return `${i + 1}. ${e.name}${tagInfo}`;
    }).join("\n");

    globalContext = `\nTHÔNG TIN TỔNG QUAN VỀ TRIỂN LÃM:\nTriển lãm hiện có tổng cộng ${allExhibits.length} hiện vật/tác phẩm. Danh sách sơ lược:\n${list}\n`;
  }

  // Lịch sử hội thoại (Multi-turn)
  let historySection = "";
  if (history && history.length > 0) {
    const historyText = history.map(h =>
      `${h.role === "user" ? "Người dùng" : "Tư vấn viên"}: ${h.text}`
    ).join("\n");
    historySection = `\nLỊCH SỬ HỘI THOẠI GẦN ĐÂY (dùng để hiểu ngữ cảnh đại từ như "nó", "bức tranh đó", "ông ấy"):\n${historyText}\n`;
  }

  return `Bạn là Tư vấn viên ảo chuyên nghiệp của ${companyName ?? "Triển lãm"}.
Nhiệm vụ: TRẢ LỜI TRỰC TIẾP câu hỏi dựa HOÀN TOÀN vào TÀI LIỆU bên dưới.

QUY TẮC:
- KHÔNG chào hỏi, KHÔNG tự giới thiệu
- Trả lời NGẮN GỌN (tối đa 3-4 câu), bám sát tài liệu
- Nếu tài liệu không có thông tin, nói: "Tôi chưa có thông tin cụ thể về vấn đề này."
- Dùng tiếng Việt, giọng văn chuyên nghiệp và thân thiện
- TUYỆT ĐỐI KHÔNG trả lời câu hỏi ngoài phạm vi triển lãm (không viết code, không làm thơ, không giải toán)
- Nếu lịch sử hội thoại có sẵn, hãy sử dụng để hiểu ngữ cảnh đại từ (ví dụ: "ông ấy" = người được nhắc trước đó)
${historySection}${globalContext}
TÀI LIỆU CHI TIẾT (được truy xuất tự động bởi hệ thống RAG):
${contextDocs || "Không tìm thấy tài liệu chi tiết liên quan."}

CÂU HỎI: ${question}

TRẢ LỜI:`;
}

// ================================================================
// 10. MAIN HANDLER — Hàm xử lý chính
// ================================================================
export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { message, currentExhibitId, history } = req.body || {};
  if (!message?.trim()) return res.status(400).json({ error: "Thiếu nội dung tin nhắn" });

  const GEMINI_KEY = process.env.GEMINI_API_KEY;
  const FB_KEY     = process.env.FIREBASE_API_KEY;
  const FB_PROJECT = process.env.FIREBASE_PROJECT_ID || "trienlam3d-84c03";

  if (!GEMINI_KEY) {
    return res.status(500).json({ reply: "Lỗi cấu hình: GEMINI_API_KEY chưa được cài đặt." });
  }

  // ── Bước 1: Safety Filter — Chặn Prompt Injection & ngoài phạm vi ──
  const safety = checkSafety(message);
  if (!safety.safe) {
    console.log(`[Safety] Chặn tin nhắn: "${message}" (lý do: ${safety.reason})`);
    return res.status(200).json({
      reply: RAG_CONFIG.OUT_OF_SCOPE_RESPONSE,
      rag: { used: false, blocked: true },
    });
  }

  // ── Bước 2: Intent Classification — Phân loại ý định ──
  const intent = classifyIntent(message);

  if (intent === "GREETING") {
    const resp = GREETING_RESPONSES[Math.floor(Math.random() * GREETING_RESPONSES.length)];
    console.log(`[Intent] GREETING → Trả lời mẫu`);
    return res.status(200).json({ reply: resp, rag: { used: false, intent } });
  }

  if (intent === "FAREWELL") {
    const resp = FAREWELL_RESPONSES[Math.floor(Math.random() * FAREWELL_RESPONSES.length)];
    console.log(`[Intent] FAREWELL → Trả lời mẫu`);
    return res.status(200).json({ reply: resp, rag: { used: false, intent } });
  }

  try {
    // ── Bước 3: Tải dữ liệu từ Firestore ──
    let exhibits = [];
    let companyName = "Triển lãm Doanh nghiệp";

    try {
      const fsRes = await fetch(
        `https://firestore.googleapis.com/v1/projects/${FB_PROJECT}/databases/(default)/documents/museum/data?key=${FB_KEY}`
      );
      if (fsRes.ok) {
        const fsData = await fsRes.json();
        const db = JSON.parse(fsData.fields?.jsonData?.stringValue || "{}");
        companyName = db.museumName || db.companyName || companyName;
        exhibits = (db.exhibits || []).filter(e => e.isPublished !== false);
        console.log(`[RAG] Đã tải ${exhibits.length} hiện vật từ Firestore.`);
      }
    } catch (e) {
      console.warn("[RAG] Không thể tải Firestore:", e.message);
    }

    // ── Bước 4: RAG Retrieval (với Multi-level Fallback) ──
    let retrievedExhibits = [];
    let ragUsed = false;
    let fallbackLevel = 0;

    if (exhibits.length > 0) {
      try {
        // Bậc 0: RAG đầy đủ (Chunked Embedding + Cosine Similarity)
        retrievedExhibits = await retrieveTopK(message, exhibits, currentExhibitId, GEMINI_KEY);
        ragUsed = true;
        console.log(`[RAG] Truy xuất được ${retrievedExhibits.length} tài liệu liên quan.`);
      } catch (e) {
        // Bậc 2: Keyword Search (khi Embedding API lỗi)
        console.warn("[RAG] Embedding thất bại, fallback bậc 2: Keyword Search.", e.message);
        retrievedExhibits = keywordSearch(message, exhibits, currentExhibitId);
        fallbackLevel = 2;

        if (retrievedExhibits.length > 0) {
          ragUsed = true;
          console.log(`[RAG Fallback 2] Keyword Search tìm được ${retrievedExhibits.length} kết quả.`);
        } else {
          // Bậc 3: Sử dụng hiện vật đang xem hoặc Global Context
          fallbackLevel = 3;
          if (currentExhibitId) {
            const current = exhibits.find(ex => ex.id === currentExhibitId);
            if (current) {
              retrievedExhibits = [current];
              console.log("[RAG Fallback 3] Sử dụng hiện vật đang xem.");
            }
          }
          // Nếu vẫn không có → buildRAGPrompt sẽ dùng Global Context (danh sách sơ lược)
        }
      }
    }

    // ── Bước 5: Xây dựng Prompt (với lịch sử hội thoại Multi-turn) ──
    const trimmedHistory = (history || []).slice(-(RAG_CONFIG.MAX_HISTORY_TURNS * 2));
    const prompt = buildRAGPrompt(message, retrievedExhibits, companyName, exhibits, trimmedHistory);

    // ── Bước 6: Gọi Gemini API (với vòng lặp thử lại khi lỗi 503) ──
    let geminiData = null;
    const maxRetries = 3;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      const geminiRes = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${RAG_CONFIG.GENERATION_MODEL}:generateContent?key=${GEMINI_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: {
              temperature: RAG_CONFIG.TEMPERATURE,
              maxOutputTokens: RAG_CONFIG.MAX_OUTPUT_TOKENS,
            },
          }),
        }
      );

      geminiData = await geminiRes.json();

      if (!geminiData.error || geminiData.error.code !== 503) {
        break;
      }

      if (attempt < maxRetries) {
        const waitTime = attempt * 1200;
        console.warn(`[RAG] Lỗi 503 (High Demand). Đang thử lại lần ${attempt}/${maxRetries - 1} sau ${waitTime}ms...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }

    // ── Bước 7: Xử lý lỗi Gemini ──
    if (geminiData.error) {
      console.error("[RAG] Gemini API error:", JSON.stringify(geminiData.error));

      if (geminiData.error.code === 503) {
        return res.status(200).json({
          reply: "Hệ thống AI hiện đang bị quá tải nhẹ. Vui lòng đợi vài giây và hỏi lại nhé!",
        });
      }

      if (geminiData.error.code === 429) {
        return res.status(200).json({
          reply: "Bạn đang hỏi quá nhanh! Hạn mức miễn phí chỉ cho phép 15-20 tin nhắn mỗi phút. Vui lòng đợi khoảng 1 phút rồi thử lại nhé.",
        });
      }

      return res.status(200).json({
        reply: `Lỗi AI (${geminiData.error.code}): ${geminiData.error.message}`,
      });
    }

    // ── Bước 8: Trả về kết quả ──
    const reply = geminiData.candidates?.[0]?.content?.parts?.[0]?.text
      || "Xin lỗi, tôi không thể trả lời lúc này. Vui lòng thử lại!";

    return res.status(200).json({
      reply,
      rag: {
        used: ragUsed,
        retrievedCount: retrievedExhibits.length,
        retrievedIds: retrievedExhibits.map(e => e.id),
        modelUsed: RAG_CONFIG.GENERATION_MODEL,
        intent,
        fallbackLevel,
      },
    });

  } catch (err) {
    console.error("[RAG] Lỗi server:", err.message);
    return res.status(500).json({ error: "Lỗi server", message: err.message });
  }
}
