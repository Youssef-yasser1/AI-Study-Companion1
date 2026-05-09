# ============================================
# flashcard_generator.py
# Generates study flashcards using Llama-3
# ============================================

import os
import re
import logging

logger = logging.getLogger(__name__)

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


def generate_flashcards(text: str, max_cards: int = 15) -> list:
    """Generate study flashcards from text using Llama-3."""
    if not text or not text.strip():
        return []

    if HF_TOKEN:
        try:
            return _generate_with_llama(text, max_cards)
        except Exception as e:
            logger.warning(f"Llama flashcard generation failed: {e}")

    return _generate_fallback(text, max_cards)


def _generate_with_llama(text: str, max_cards: int) -> list:
    """Use Llama-3 to generate flashcards."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    ctx = text[:4000] if len(text) > 4000 else text

    prompt = f"""Create {max_cards} study flashcards from this lecture content.
Each flashcard should have a TERM (front of card) and DEFINITION (back of card).

Rules:
- Cover the most important concepts, definitions, and facts
- Definitions should be clear and concise (1-2 sentences)
- Include both terminology and conceptual understanding
- Make them useful for exam preparation

Format EXACTLY like this (one per line):
CARD: Term or Question | Clear definition or answer

LECTURE CONTENT:
{ctx}

Generate {max_cards} flashcards:"""

    response = client.chat_completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.3
    )
    output = response.choices[0].message.content.strip()
    print(f"[Flashcards] Generated from Llama-3.")
    return _parse_flashcards(output)


def _parse_flashcards(output: str) -> list:
    """Parse CARD: term | definition format."""
    cards = []
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Remove numbering and CARD: prefix
        line = re.sub(r'^\d+[.)]\s*', '', line)
        line = re.sub(r'^CARD:\s*', '', line, flags=re.IGNORECASE)
        line = re.sub(r'^[-*]\s*', '', line)

        if '|' in line:
            parts = line.split('|', 1)
            term = parts[0].strip()
            definition = parts[1].strip()
            if term and definition and len(definition) > 5:
                cards.append({
                    "term": term,
                    "definition": definition,
                    "category": "general"
                })
        elif ':' in line and len(line) > 15:
            parts = line.split(':', 1)
            term = parts[0].strip()
            definition = parts[1].strip()
            if term and definition and len(definition) > 5 and len(term) < 80:
                cards.append({
                    "term": term,
                    "definition": definition,
                    "category": "general"
                })

    return cards


def _generate_fallback(text: str, max_cards: int) -> list:
    """Simple extraction when API unavailable."""
    cards = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        match = re.match(r'^(.+?)\s+(?:is|are|refers?\s+to|means?)\s+(.+)', sent, re.IGNORECASE)
        if match and len(match.group(2)) > 10:
            cards.append({
                "term": match.group(1).strip(),
                "definition": match.group(2).strip().rstrip('.'),
                "category": "general"
            })
        if len(cards) >= max_cards:
            break
    return cards
