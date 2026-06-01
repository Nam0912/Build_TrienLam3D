// ================================================================
// Vercel Serverless Function — RAG-Enhanced Chatbot Proxy
// Kiến trúc: Lightweight In-Request RAG
//
// Luồng xử lý:
//   1. Nhận câu hỏi + currentExhibitId từ Unity WebGL
//   2. Tải dữ liệu sự kiện từ Firebase Firestore
//   3. Vector hóa câu hỏi bằng Gemini Embedding API (text-embedding-004)
//   4. Tính Cosine Similarity giữa câu hỏi và từng sự kiện
//   5. Chọn Top-K sự kiện liên quan nhất (+ Boost sự kiện đang xem)
//   6. Gửi Top-K + câu hỏi cho Gemini để sinh câu trả lời
// ================================================================

import { RAG_CONFIG } from "./rag-config.js";

// ----------------------------------------------------------------
// 1. HÀM TOÁN HỌC: Cosine Similarity
// Tính độ tương đồng giữa 2 vector đa chiều.
// Công thức: cos(θ) = (A · B) / (||A|| × ||B||)
// Kết quả trong khoảng [0, 1]: 1 = hoàn toàn giống nhau
// ----------------------------------------------------------------
function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;

    let dotProduct = 0;    // Tích vô hướng A · B
    let normA = 0;         // ||A||²
    let normB = 0;         // ||B||²

    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;

    return dotProduct / denominator; // cos(θ)
}

// ----------------------------------------------------------------
// 2. HÀM EMBEDDING: Chuyển văn bản → Vector 768 chiều
// Sử dụng Google text-embedding-004 qua Gemini REST API
// ----------------------------------------------------------------
async function getEmbedding(text, apiKey) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${RAG_CONFIG.EMBEDDING_MODEL}:embedContent?key=${apiKey}`;

    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            model: `models/${RAG_CONFIG.EMBEDDING_MODEL}`,
            content: { parts: [{ text: text }] },
            taskType: "RETRIEVAL_QUERY", // Tối ưu cho tìm kiếm ngữ nghĩa
        }),
    });

    if (!response.ok) {
        const err = await response.json();
        throw new Error(`Embedding API lỗi: ${err.error?.message || response.status}`);
    }

    const data = await response.json();
    return data.embedding?.values ?? null; // Mảng float 768 chiều
}

// ----------------------------------------------------------------
// 3. HÀM RAG: Truy xuất Top-K tài liệu liên quan nhất
// Input:  câu hỏi, danh sách sự kiện, currentExhibitId (có thể null)
// Output: Mảng Top-K sự kiện có điểm tương đồng cao nhất
// ----------------------------------------------------------------
async function retrieveTopK(question, exhibits, currentExhibitId, apiKey) {
    // 3.1. Vector hóa câu hỏi
    const questionVector = await getEmbedding(question, apiKey);
    if (!questionVector) throw new Error("Không thể vector hóa câu hỏi.");

    // 3.2. Tính Cosine Similarity cho từng sự kiện song song (Promise.all)
    const scoredExhibits = await Promise.all(
        exhibits.map(async (exhibit) => {
            // Gộp các trường văn bản quan trọng thành 1 chuỗi để embedding
            const exhibitText = [
                exhibit.name ?? "",
                exhibit.description ?? "",
                exhibit.category ?? "",
                exhibit.department ?? "",
                // Nối thêm text của các trang phụ nếu có
                ...(exhibit.pages ?? []).map(p => `${p.title ?? ""} ${p.description ?? ""}`),
            ].filter(Boolean).join(". ");

            let similarity = 0;
            try {
                const exhibitVector = await getEmbedding(exhibitText, apiKey);
                similarity = cosineSimilarity(questionVector, exhibitVector);
            } catch (_) {
                // Nếu lỗi embedding 1 sự kiện, bỏ qua và gán điểm 0
                similarity = 0;
            }

            // 3.3. Context-Aware Boost: Cộng thêm điểm cho sự kiện đang xem
            // Đây là cơ chế nhận thức ngữ cảnh không gian 3D của hệ thống
            if (currentExhibitId && exhibit.id === currentExhibitId) {
                similarity += RAG_CONFIG.CONTEXT_BOOST_SCORE;
            }

            return { exhibit, similarity };
        })
    );

    // 3.4. Lọc theo ngưỡng và sắp xếp giảm dần theo điểm
    const filtered = scoredExhibits
        .filter(item => item.similarity >= RAG_CONFIG.SIMILARITY_THRESHOLD)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, RAG_CONFIG.TOP_K);

    // 3.5. Log điểm để kiểm tra (hiển thị trong Vercel Console)
    console.log("[RAG] Kết quả truy vấn:");
    filtered.forEach(item =>
        console.log(`  - [${item.exhibit.id}] ${item.exhibit.name}: ${item.similarity.toFixed(4)}`)
    );

    // Nếu không có kết quả nào vượt ngưỡng, trả về sự kiện đang xem (fallback)
    if (filtered.length === 0 && currentExhibitId) {
        const current = exhibits.find(e => e.id === currentExhibitId);
        if (current) {
            console.log("[RAG] Không có kết quả ngưỡng cao, fallback về sự kiện hiện tại.");
            return [current];
        }
    }

    return filtered.map(item => item.exhibit);
}

// ----------------------------------------------------------------
// 4. XÂY DỰNG PROMPT TỪ KẾT QUẢ RAG
// ----------------------------------------------------------------
function buildRAGPrompt(question, retrievedExhibits, companyName, allExhibits) {
    const contextDocs = retrievedExhibits.map((e, i) => {
        const pages = (e.pages ?? [])
            .map(p => `  [Trang: ${p.title ?? ""}] ${p.description ?? ""}`)
            .join("\n");
        return `--- Tài liệu ${i + 1}: ${e.name} (ID: ${e.id}, Phòng ban: ${e.department ?? "N/A"}) ---
Mô tả: ${e.description ?? "Chưa có mô tả"}
${pages}`;
    }).join("\n\n");

    // Tạo danh sách tổng quan tất cả hiện vật (để bot biết số lượng và tên các tác phẩm khác)
    let globalContext = "";      
    if (allExhibits && allExhibits.length > 0) {
        const list = allExhibits.map((e, i) => {
            // Gom các trường phân loại lại với nhau
            const tagsArr = [];
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

// ----------------------------------------------------------------
// 5. HANDLER CHÍNH của Vercel Serverless Function
// ----------------------------------------------------------------
export default async function handler(req, res) {
    // CORS headers
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === "OPTIONS") return res.status(200).end();
    if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

    // Parse request body
    const { message, currentExhibitId } = req.body || {};
    if (!message?.trim()) return res.status(400).json({ error: "Thiếu nội dung tin nhắn" });

    const GEMINI_KEY = process.env.GEMINI_API_KEY;
    const FB_KEY     = process.env.FIREBASE_API_KEY;
    const FB_PROJECT = process.env.FIREBASE_PROJECT_ID || "trienlam3d-84c03";

    if (!GEMINI_KEY) {
        return res.status(500).json({ reply: "Lỗi cấu hình: GEMINI_API_KEY chưa được cài đặt." });
    }

    try {
        // ----------------------------------------------------------
        // BƯỚC 1: Tải dữ liệu từ Firestore
        // ----------------------------------------------------------
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

        // ----------------------------------------------------------
        // BƯỚC 2: RAG — Truy xuất tài liệu liên quan
        // ----------------------------------------------------------
        let retrievedExhibits = [];
        let ragUsed = false;

        if (exhibits.length > 0) {
            try {
                retrievedExhibits = await retrieveTopK(message, exhibits, currentExhibitId, GEMINI_KEY);
                ragUsed = true;
                console.log(`[RAG] Truy xuất được ${retrievedExhibits.length} tài liệu liên quan.`);
            } catch (e) {
                // Nếu RAG lỗi (ví dụ: Embedding API limit), fallback về cách cũ
                console.warn("[RAG] Embedding thất bại, fallback về context đơn:", e.message);
                if (currentExhibitId) {
                    const current = exhibits.find(ex => ex.id === currentExhibitId);
                    if (current) retrievedExhibits = [current];
                }
            }
        }

        // ----------------------------------------------------------
        // BƯỚC 3: Xây dựng RAG Prompt và gọi Gemini
        // ----------------------------------------------------------
        const prompt = buildRAGPrompt(message, retrievedExhibits, companyName, exhibits);

        let geminiData = null;
        const maxRetries = 3; // Thử lại tối đa 3 lần nếu gặp 503

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

            // Nếu thành công hoặc là lỗi KHÁC 503 thì thoát vòng lặp
            if (!geminiData.error || geminiData.error.code !== 503) {
                break;
            }

            // Nếu gặp lỗi 503 và còn số lần thử, chờ 1 khoảng thời gian rồi thử lại (Exponential backoff)
            if (attempt < maxRetries) {
                const waitTime = attempt * 1200; // 1.2s, 2.4s
                console.warn(`[RAG] Lỗi 503 (High Demand). Đang thử lại lần ${attempt}/${maxRetries - 1} sau ${waitTime}ms...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }

        if (geminiData.error) {
            console.error("[RAG] Gemini API error:", JSON.stringify(geminiData.error));

            // Nếu sau các lần thử vẫn 503, trả về câu báo thân thiện hơn
            if (geminiData.error.code === 503) {
                return res.status(200).json({
                    reply: "Hệ thống AI hiện đang bị quá tải nhẹ. Vui lòng đợi vài giây và hỏi lại nhé!",
                });
            }

            // Xử lý lỗi 429 (Hết hạn mức - Quota Exceeded)
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

        // Trả về kết quả + metadata RAG (để Unity có thể log/debug)
        return res.status(200).json({
            reply,
            rag: {
                used: ragUsed,
                retrievedCount: retrievedExhibits.length,
                retrievedIds: retrievedExhibits.map(e => e.id),
                modelUsed: RAG_CONFIG.GENERATION_MODEL, // Thêm dòng này để dễ theo dõi
            },
        });

    } catch (err) {
        console.error("[RAG] Lỗi server:", err.message);
        return res.status(500).json({ error: "Lỗi server", message: err.message });
    }
}
