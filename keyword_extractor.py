# ============================================
# keyword_extractor.py
# Extracts key concepts, definitions, and terms
# Uses Llama-3 for intelligent extraction
# ============================================

import os
import re
import logging
from collections import Counter

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


def extract_keywords(text: str, method: str = "auto", max_keywords: int = 15, **kwargs) -> list:
    """Extract key concepts, definitions, and terms from text."""
    if not text or not text.strip():
        return []

    # Try Llama-3 for intelligent extraction
    if HF_TOKEN:
        try:
            results = _extract_with_llama(text)
            if results and len(results) >= 3:
                return results
        except Exception as e:
            logger.warning(f"Llama extraction failed: {e}")

    # Fallback: pattern-based extraction
    return _extract_fallback(text, max_keywords)


def _extract_with_llama(text: str) -> list:
    """Use Llama-3 to intelligently extract key concepts."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    ctx = text[:4000] if len(text) > 4000 else text

    prompt = f"""Analyze this lecture and extract the key information. Be specific and detailed.

For DEFINITIONS: Write the full definition as it appears in or can be derived from the lecture.
For KEY POINTS: Write the most important concepts and ideas.
For TERMS: List the technical vocabulary.

Format EXACTLY like this (each on its own line):
DEF: Term Name - The complete definition or explanation
DEF: Another Term - Its complete definition or explanation
KEY: An important concept or point from the lecture
KEY: Another important concept
TERM: technical_term_1
TERM: technical_term_2

LECTURE CONTENT:
{ctx}

Extract the key information:"""

    response = client.chat_completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.2
    )
    output = response.choices[0].message.content.strip()
    print(f"[Keywords] Llama-3 extraction complete.")

    return _parse_structured_output(output)


def _parse_structured_output(output: str) -> list:
    """Parse the DEF:/KEY:/TERM: formatted output."""
    results = []

    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Remove bullet/number prefixes
        line = re.sub(r'^[\-\*\d]+[.)]\s*', '', line).strip()

        # DEF: Term - Definition
        def_match = re.match(r'^DEF:\s*(.+?)\s*[-–—:]\s*(.+)', line, re.IGNORECASE)
        if def_match:
            term = def_match.group(1).strip()
            definition = def_match.group(2).strip()
            if term and definition and len(definition) > 5:
                results.append({
                    "text": f"{term}: {definition}",
                    "term": term,
                    "type": "definition",
                    "importance": "high"
                })
            continue

        # DEF: without separator (just definition text)
        def_match2 = re.match(r'^DEF:\s*(.+)', line, re.IGNORECASE)
        if def_match2:
            text = def_match2.group(1).strip()
            if text and len(text) > 10:
                # Try to split on first : or -
                parts = re.split(r'\s*[-–—:]\s*', text, 1)
                if len(parts) == 2 and len(parts[1]) > 5:
                    results.append({
                        "text": f"{parts[0].strip()}: {parts[1].strip()}",
                        "term": parts[0].strip(),
                        "type": "definition",
                        "importance": "high"
                    })
                else:
                    results.append({
                        "text": text,
                        "term": text.split()[0] if text.split() else "",
                        "type": "definition",
                        "importance": "high"
                    })
            continue

        # KEY: Important point
        key_match = re.match(r'^KEY:\s*(.+)', line, re.IGNORECASE)
        if key_match:
            text = key_match.group(1).strip()
            if text and len(text) > 5:
                results.append({
                    "text": text,
                    "type": "key_point",
                    "importance": "high" if len(text) > 30 else "medium"
                })
            continue

        # TERM: keyword
        term_match = re.match(r'^TERM:\s*(.+)', line, re.IGNORECASE)
        if term_match:
            term = term_match.group(1).strip()
            if term and len(term) > 2:
                # May be comma-separated
                for kw in term.split(','):
                    kw = kw.strip()
                    if kw and len(kw) > 2:
                        results.append({
                            "keyword": kw,
                            "type": "keyword",
                            "importance": "medium"
                        })
            continue

    return results


def _extract_fallback(text: str, max_keywords: int = 15) -> list:
    """Pattern-based extraction when API is unavailable."""
    results = []

    # Extract definitions
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 500:
            continue
        if re.search(r'\b(?:is|are|refers?\s+to|means?|defined\s+as)\b', sent, re.IGNORECASE):
            match = re.match(r'^([A-Z][^.]{2,40}?)\s+(?:is|are|refers)', sent, re.IGNORECASE)
            term = match.group(1).strip() if match else ""
            results.append({
                "text": sent, "term": term,
                "type": "definition", "importance": "high"
            })
            if len(results) >= 5:
                break

    # Extract keywords by frequency
    stop_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for',
        'not', 'on', 'with', 'as', 'do', 'at', 'this', 'but', 'by', 'from',
        'or', 'an', 'will', 'all', 'there', 'what', 'so', 'up', 'out',
        'if', 'about', 'who', 'get', 'which', 'go', 'when', 'can', 'like',
        'also', 'than', 'then', 'now', 'only', 'its', 'over',
        'use', 'how', 'even', 'new', 'any', 'these', 'such',
        'was', 'were', 'been', 'has', 'had', 'are', 'is', 'very', 'more',
        'each', 'may', 'should', 'being', 'used', 'based', 'data'
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    counter = Counter(filtered)

    for w, c in counter.most_common(max_keywords):
        results.append({
            "keyword": w, "frequency": c,
            "type": "keyword",
            "importance": "high" if c > 5 else "medium"
        })

    return results


def highlight_keywords_in_text(text: str, keywords: list) -> str:
    """Highlight keywords in text with **bold** markers."""
    highlighted = text
    kw_strings = sorted(
        [kw.get("keyword", kw.get("term", "")) for kw in keywords if kw.get("keyword") or kw.get("term")],
        key=len, reverse=True
    )
    for kw in kw_strings:
        if kw:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted = pattern.sub(f"**{kw}**", highlighted)
    return highlighted
