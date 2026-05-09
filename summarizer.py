# ============================================
# summarizer.py
# Comprehensive lecture summarization using
# Llama-3 chat model via HF Inference API
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


def _preprocess_slides(text: str) -> str:
    """Convert slide-format bullet points into clean text."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        line = re.sub(r'^[\-\*\u2022\u25CF\u25CB]+\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip()
        if line and line[-1] not in '.!?':
            line += '.'
        if len(line) > 3:
            cleaned.append(line)
    return ' '.join(cleaned)


def _chunk_text(text: str, max_chars: int = 2000) -> list:
    """Split text into chunks by sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


def summarize_text(text: str, **kwargs) -> dict:
    """
    Generate a COMPREHENSIVE lecture summary using Llama-3 chat model.
    Each chunk gets a detailed summary, all are combined.
    """
    if not text or not text.strip():
        return {"summary": "No text provided.", "chunk_summaries": [],
                "num_chunks": 0, "model_used": "none"}

    processed = _preprocess_slides(text)
    if len(processed.split()) < 15:
        processed = text

    # Try HF API (Llama-3 chat) first
    if HF_TOKEN:
        try:
            return _summarize_with_llama(processed)
        except Exception as e:
            logger.warning(f"Llama API failed: {e}, trying BART API...")
            try:
                return _summarize_with_bart(processed)
            except Exception as e2:
                logger.warning(f"BART API failed: {e2}, falling back to local")

    return _summarize_local(processed)


def _summarize_with_llama(text: str) -> dict:
    """Use Llama-3 chat model for comprehensive summarization."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    chunks = _chunk_text(text, max_chars=2000)
    print(f"[Summarizer] Using {CHAT_MODEL} for {len(chunks)} chunk(s)...")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[Summarizer] Processing chunk {i+1}/{len(chunks)}...")

        prompt = f"""You are a study assistant. Write a COMPREHENSIVE and DETAILED summary of this lecture content. 

Requirements:
- Cover ALL key concepts, definitions, and important points
- Explain each concept clearly so a student can study from this summary
- Include examples when mentioned
- Organize the summary with clear paragraphs
- The summary should be detailed enough to replace reading the original lecture
- Do NOT add information that is not in the original content

LECTURE CONTENT:
{chunk}

COMPREHENSIVE SUMMARY:"""

        response = client.chat_completion(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        chunk_summaries.append(summary)

    # Combine all chunk summaries
    if len(chunk_summaries) == 1:
        final = chunk_summaries[0]
    else:
        # For multiple chunks, combine and optionally create an overview
        final = "\n\n---\n\n".join(chunk_summaries)

    return {
        "summary": final,
        "chunk_summaries": chunk_summaries,
        "num_chunks": len(chunks),
        "model_used": CHAT_MODEL + " (API)"
    }


def _summarize_with_bart(text: str) -> dict:
    """Fallback: use BART summarization API."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    chunks = _chunk_text(text, max_chars=2500)
    print(f"[Summarizer] Using BART API for {len(chunks)} chunk(s)...")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        result = client.summarization(chunk, model="facebook/bart-large-cnn")
        summary = result.summary_text if hasattr(result, 'summary_text') else str(result)
        chunk_summaries.append(summary)

    final = '\n\n'.join(chunk_summaries)
    return {
        "summary": final,
        "chunk_summaries": chunk_summaries,
        "num_chunks": len(chunks),
        "model_used": "facebook/bart-large-cnn (API)"
    }


def _summarize_local(text: str) -> dict:
    """Last resort: use local DistilBART model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from utils.chunking import smart_chunk_text

    model_name = "sshleifer/distilbart-cnn-6-6"
    print(f"[Summarizer Local] Using {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    chunks = smart_chunk_text(text, max_tokens=600, overlap=50)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            ids = model.generate(inputs["input_ids"], max_new_tokens=200,
                                 min_length=50, num_beams=1, no_repeat_ngram_size=3)
        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    final = '\n\n'.join(summaries)
    return {"summary": final, "chunk_summaries": summaries,
            "num_chunks": len(chunks), "model_used": model_name + " (local)"}
