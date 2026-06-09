

import { RAG_CONFIG } from "./rag-config.js";

let exhibitEmbeddingCache = null;  
let cacheExhibitCount = 0;       
let cacheTimestamp = 0;            
const CACHE_MAX_AGE = 30 * 60 * 1000; 

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

async function ensureEmbeddingCache(exhibits, apiKey) {
  const now = Date.now();
  const cacheExpired = (now - cacheTimestamp) > CACHE_MAX_AGE;
  const countChanged = cacheExhibitCount !== exhibits.length;

  if (exhibitEmbeddingCache && !cacheExpired && !countChanged) {
    console.log(`[RAG Cache] Sử dụng cache có sẵn (${exhibitEmbeddingCache.size} hiện vật, tuổi: ${Math.round((now - cacheTimestamp) / 1000)}s)`);
    return;
  }

  console.log("[RAG Cache] Đang tính embedding cho tất cả hiện vật (lần đầu hoặc cache hết hạn)...");
  exhibitEmbeddingCache = new Map();

  for (const exhibit of exhibits) {
    const exhibitText = [
      exhibit.name ?? "",
      exhibit.description ?? "",
      exhibit.category ?? "",
      exhibit.department ?? "",
      ...(exhibit.pages ?? []).map(p => `${p.title ?? ""} ${p.description ?? ""}`),
    ].filter(Boolean).join(". ");

    try {
      const vec = await getEmbedding(exhibitText, apiKey);
      exhibitEmbeddingCache.set(exhibit.id, vec);
    } catch (e) {
      console.warn(`[RAG Cache] Lỗi embedding ${exhibit.id}: ${e.message}`);
      exhibitEmbeddingCache.set(exhibit.id, null);
    }
  }

  cacheExhibitCount = exhibits.length;
  cacheTimestamp = now;
  console.log(`[RAG Cache] Hoàn tất! Đã cache ${exhibitEmbeddingCache.size} hiện vật.`);
}

async function retrieveTopK(question, exhibits, currentExhibitId, apiKey) {
  await ensureEmbeddingCache(exhibits, apiKey);

  const questionVector = await getEmbedding(question, apiKey);
  if (!questionVector) throw new Error("Không thể vector hóa câu hỏi.");

  const scoredExhibits = exhibits.map((exhibit) => {
    const exhibitVector = exhibitEmbeddingCache.get(exhibit.id);
    let similarity = exhibitVector
      ? cosineSimilarity(questionVector, exhibitVector)
      : 0;

    if (currentExhibitId && exhibit.id === currentExhibitId) {
      similarity += RAG_CONFIG.CONTEXT_BOOST_SCORE;
    }

    return { exhibit, similarity };
  });

  const filtered = scoredExhibits
    .filter(item => item.similarity >= RAG_CONFIG.SIMILARITY_THRESHOLD)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, RAG_CONFIG.TOP_K);

  console.log("[RAG] Kết quả truy vấn:");
  filtered.forEach(item =>
    console.log(`  - [${item.exhibit.id}] ${item.exhibit.name}: ${item.similarity.toFixed(4)}`)
  );

  if (filtered.length === 0 && currentExhibitId) {
    const current = exhibits.find(e => e.id === currentExhibitId);
    if (current) {
      console.log("[RAG] Không có kết quả ngưỡng cao, fallback về sự kiện hiện tại.");
      return [current];
    }
  }

  return filtered.map(item => item.exhibit);
}

function buildRAGPrompt(question, retrievedExhibits, companyName, allExhibits) {
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

  return `Bạn là Tư vấn viên ảo chuyên nghiệp của ${companyName ?? "Triển lãm"}.
Nhiệm vụ: TRẢ LỜI TRỰC TIẾP câu hỏi dựa HOÀN TOÀN vào TÀI LIỆU bên dưới.

QUY TẮC:
- KHÔNG chào hỏi, KHÔNG tự giới thiệu
- Trả lời NGẮN GỌN (tối đa 3-4 câu), bám sát tài liệu
- Nếu tài liệu không có thông tin, nói: "Tôi chưa có thông tin cụ thể về vấn đề này."
- Dùng tiếng Việt, giọng văn chuyên nghiệp và thân thiện
${globalContext}
TÀI LIỆU CHI TIẾT (được truy xuất tự động bởi hệ thống RAG):
${contextDocs || "Không tìm thấy tài liệu chi tiết liên quan."}

CÂU HỎI: ${question}

TRẢ LỜI:`;
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { message, currentExhibitId } = req.body || {};
  if (!message?.trim()) return res.status(400).json({ error: "Thiếu nội dung tin nhắn" });

  const GEMINI_KEY = process.env.GEMINI_API_KEY;
  const FB_KEY     = process.env.FIREBASE_API_KEY;
  const FB_PROJECT = process.env.FIREBASE_PROJECT_ID || "trienlam3d-84c03";

  if (!GEMINI_KEY) {
    return res.status(500).json({ reply: "Lỗi cấu hình: GEMINI_API_KEY chưa được cài đặt." });
  }

  try {
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
        console.log(`[RAG] Đã tải ${exhibits.length} sự kiện từ Firestore.`);
      }
    } catch (e) {
      console.warn("[RAG] Không thể tải Firestore:", e.message);
    }

    let retrievedExhibits = [];
    let ragUsed = false;

    if (exhibits.length > 0) {
      try {
        retrievedExhibits = await retrieveTopK(message, exhibits, currentExhibitId, GEMINI_KEY);
        ragUsed = true;
        console.log(`[RAG] Truy xuất được ${retrievedExhibits.length} tài liệu liên quan.`);
      } catch (e) {
        console.warn("[RAG] Embedding thất bại, fallback về context đơn:", e.message);
        if (currentExhibitId) {
          const current = exhibits.find(ex => ex.id === currentExhibitId);
          if (current) retrievedExhibits = [current];
        }
      }
    }
    
    const prompt = buildRAGPrompt(message, retrievedExhibits, companyName, exhibits);

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

    const reply = geminiData.candidates?.[0]?.content?.parts?.[0]?.text
      || "Xin lỗi, tôi không thể trả lời lúc này. Vui lòng thử lại!";

    return res.status(200).json({
      reply,
      rag: {
        used: ragUsed,
        retrievedCount: retrievedExhibits.length,
        retrievedIds: retrievedExhibits.map(e => e.id),
        modelUsed: RAG_CONFIG.GENERATION_MODEL,
      },
    });

  } catch (err) {
    console.error("[RAG] Lỗi server:", err.message);
    return res.status(500).json({ error: "Lỗi server", message: err.message });
  }
}
