# ============================================
# qa_system.py
# Question Answering - Llama-3 Chat as PRIMARY
# (extractive QA models fail on slide-format text)
# ============================================

import os
import re
import logging

logger = logging.getLogger(__name__)

# Load HF token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    for env_file in ["key.env", ".env"]:
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        HF_TOKEN = line.strip().split("=", 1)[1]
                        break

CHAT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def answer_question(question: str, context: str, **kwargs) -> dict:
    """Answer a question using Llama-3 chat (best accuracy for lecture content)."""
    if not question or not question.strip():
        return {"answer": "Please provide a question.", "confidence": 0.0,
                "context_snippet": "", "all_answers": []}
    if not context or not context.strip():
        return {"answer": "No document loaded.", "confidence": 0.0,
                "context_snippet": "", "all_answers": []}

    # Try Llama-3 chat FIRST (much better for lecture Q&A)
    if HF_TOKEN:
        try:
            return _qa_llama(question, context)
        except Exception as e:
            logger.warning(f"Llama QA failed: {e}")
            # Try extractive as fallback
            try:
                return _qa_api_extractive(question, context)
            except Exception as e2:
                logger.warning(f"Extractive QA also failed: {e2}")

    return _qa_local(question, context)


def _qa_llama(question: str, context: str) -> dict:
    """Use Llama-3 chat for QA — understands lecture content much better."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    # Use full context (up to 5000 chars for Llama)
    ctx = context[:5000] if len(context) > 5000 else context

    prompt = f"""You are a study assistant. Answer the student's question based ONLY on the lecture content below.

Rules:
- Answer ONLY from the lecture content. Do not use external knowledge.
- If the answer is clearly stated in the text, quote the relevant part.
- If the answer is not in the text, say "This information is not found in the lecture."
- Be concise but complete. Give the actual definition or explanation.
- Do NOT mention doctors, professors, or lecturers unless specifically asked.

LECTURE CONTENT:
{ctx}

STUDENT QUESTION: {question}

ANSWER:"""

    response = client.chat_completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.1
    )
    answer = response.choices[0].message.content.strip()

    # Remove any "ANSWER:" prefix the model might add
    answer = re.sub(r'^(?:ANSWER|Answer|A):\s*', '', answer).strip()

    # Find relevant context snippet
    q_words = set(question.lower().split())
    sentences = re.split(r'(?<=[.!?])\s+', context)
    best_snippet = ""
    best_score = 0
    for sent in sentences:
        score = sum(1 for w in q_words if w in sent.lower())
        if score > best_score:
            best_score = score
            best_snippet = sent

    return {
        "answer": answer,
        "confidence": 0.92,
        "context_snippet": best_snippet[:300] if best_snippet else ctx[:300],
        "all_answers": []
    }


def _qa_api_extractive(question: str, context: str) -> dict:
    """Fallback: HF extractive QA API."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    # Split into chunks for extractive QA
    chunks = _chunk_text(context, max_chars=1500)
    candidates = []

    for chunk in chunks:
        try:
            result = client.question_answering(
                question=question, context=chunk,
                model="deepset/roberta-base-squad2"
            )
            answer = result.answer if hasattr(result, 'answer') else ''
            score = result.score if hasattr(result, 'score') else 0
            if answer and score > 0.1:
                candidates.append({
                    "answer": answer, "confidence": round(score, 4),
                    "context_snippet": chunk[:300]
                })
        except Exception:
            continue

    if not candidates:
        return {"answer": "Could not find an answer.", "confidence": 0.0,
                "context_snippet": "", "all_answers": []}

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    best = candidates[0]
    return {"answer": best["answer"], "confidence": best["confidence"],
            "context_snippet": best["context_snippet"], "all_answers": candidates[:3]}


def _qa_local(question: str, context: str) -> dict:
    """Last fallback: local DistilBERT."""
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from utils.chunking import smart_chunk_text

    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    chunks = smart_chunk_text(context, max_tokens=350, overlap=80)
    candidates = []
    for chunk in chunks:
        inputs = tokenizer(question, chunk, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits, dim=1).item()
        end_idx = torch.argmax(outputs.end_logits, dim=1).item()
        if end_idx < start_idx:
            end_idx = start_idx
        tokens = inputs["input_ids"][0][start_idx:end_idx+1]
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        score = (outputs.start_logits[0][start_idx] + outputs.end_logits[0][end_idx]).item()
        confidence = torch.sigmoid(torch.tensor(score)).item()
        if answer and answer not in ("[CLS]", "[SEP]", "[PAD]"):
            candidates.append({"answer": answer, "confidence": round(confidence, 4),
                                "context_snippet": chunk[:250]})

    if not candidates:
        return {"answer": "Could not find an answer.", "confidence": 0.0,
                "context_snippet": "", "all_answers": []}
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    best = candidates[0]
    return {"answer": best["answer"], "confidence": best["confidence"],
            "context_snippet": best["context_snippet"], "all_answers": candidates[:3]}


def _chunk_text(text: str, max_chars: int = 1500) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return round(2 * precision * recall / (precision + recall), 4)
