# ============================================
# quiz_generator.py
# MCQ generation using Llama-3 Chat API
# Generates proper questions with difficulty levels
# ============================================

import os
import re
import json
import random
import logging
from datetime import datetime

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


def generate_mcqs(text: str, num_questions: int = 5,
                  difficulty: str = "medium", model_name: str = None) -> list:
    """
    Generate MCQs using Llama-3 chat API.
    Falls back to local T5 model if API unavailable.
    """
    if not text or not text.strip():
        return []

    # Try Llama-3 API first
    if HF_TOKEN:
        try:
            return _generate_with_llama(text, num_questions, difficulty)
        except Exception as e:
            logger.warning(f"Llama MCQ generation failed: {e}")

    # Fallback to local T5
    return _generate_with_t5(text, num_questions, difficulty)


def _generate_with_llama(text: str, num_questions: int, difficulty: str) -> list:
    """Use Llama-3 chat to generate proper MCQ questions."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)

    # Truncate text to fit in prompt
    ctx = text[:4000] if len(text) > 4000 else text

    difficulty_instructions = {
        "easy": "Make questions about basic definitions and simple facts. Options should be clearly distinguishable.",
        "medium": "Make questions that test understanding of concepts. Include some tricky but fair distractors.",
        "hard": "Make challenging questions that require deep understanding. Distractors should be plausible and related to the topic."
    }

    diff_inst = difficulty_instructions.get(difficulty, difficulty_instructions["medium"])

    prompt = f"""You are a professor creating a multiple choice exam. Generate EXACTLY {num_questions} MCQ questions from this lecture content.

RULES:
- Each question MUST be directly based on the lecture content below
- Each question must have EXACTLY 4 options (A, B, C, D)
- Only ONE option should be correct
- Distractors (wrong answers) must be plausible and related to the topic
- {diff_inst}
- Questions should cover DIFFERENT topics from the lecture
- Do NOT repeat questions

FORMAT (follow this EXACTLY for each question):
Q: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
ANSWER: [correct letter]

LECTURE CONTENT:
{ctx}

Generate {num_questions} questions now:"""

    print(f"[Quiz] Generating {num_questions} MCQs ({difficulty}) with Llama-3...")

    response = client.chat_completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.4
    )
    output = response.choices[0].message.content.strip()

    # Parse the output
    mcqs = _parse_mcq_output(output, difficulty)
    print(f"[Quiz] Parsed {len(mcqs)} questions from Llama-3 output.")

    # If we didn't get enough, try again with remaining
    if len(mcqs) < num_questions:
        remaining = num_questions - len(mcqs)
        try:
            existing_qs = [q["question"] for q in mcqs]
            more = _generate_more_questions(client, ctx, remaining, difficulty, existing_qs)
            mcqs.extend(more)
        except Exception:
            pass

    # Number questions and limit to requested count
    for i, q in enumerate(mcqs[:num_questions]):
        q["id"] = i + 1
    return mcqs[:num_questions]


def _generate_more_questions(client, ctx, count, difficulty, existing_questions):
    """Generate additional questions avoiding duplicates."""
    existing_str = "\n".join(f"- {q}" for q in existing_questions)

    prompt = f"""Generate {count} MORE MCQ questions from this lecture. Do NOT repeat these existing questions:
{existing_str}

FORMAT:
Q: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
ANSWER: [correct letter]

LECTURE CONTENT:
{ctx[:3000]}"""

    response = client.chat_completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.5
    )
    return _parse_mcq_output(response.choices[0].message.content.strip(), difficulty)


def _parse_mcq_output(output: str, difficulty: str) -> list:
    """Parse Llama-3 MCQ output into structured format."""
    mcqs = []
    lines = output.split('\n')

    current_question = None
    current_options = {}
    correct_answer = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match question line: "Q: ...", "Q1: ...", "Q1. ...", "1. ..." etc.
        q_match = re.match(r'^(?:Q\d*[:.]\s*|(?:\d+)[:.]\s*(?:Q[:.]\s*)?)(.*)', line)
        if q_match and len(q_match.group(1)) > 10:
            # Save previous question if complete
            if current_question and len(current_options) == 4 and correct_answer:
                mcqs.append(_build_mcq(current_question, current_options, correct_answer, difficulty))
            current_question = q_match.group(1).strip()
            current_options = {}
            correct_answer = None
            continue

        # Match option lines: "A) ...", "A. ...", "a) ..."
        opt_match = re.match(r'^([A-Da-d])[).]\s*(.*)', line)
        if opt_match and current_question:
            letter = opt_match.group(1).upper()
            option_text = opt_match.group(2).strip()
            if option_text:
                current_options[letter] = option_text
            continue

        # Match answer line: "ANSWER: A", "Answer: B", "Correct: C"
        ans_match = re.match(r'^(?:ANSWER|Answer|Correct|correct)[:\s]+([A-Da-d])', line)
        if ans_match:
            correct_answer = ans_match.group(1).upper()
            continue

    # Don't forget the last question
    if current_question and len(current_options) == 4 and correct_answer:
        mcqs.append(_build_mcq(current_question, current_options, correct_answer, difficulty))

    return mcqs


def _build_mcq(question, options, correct_answer, difficulty):
    """Build a standardized MCQ dict."""
    return {
        "id": 0,  # Will be set later
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "correct_text": options.get(correct_answer, ""),
        "difficulty": difficulty
    }


def _generate_with_t5(text: str, num_questions: int, difficulty: str) -> list:
    """Fallback: Local T5 model for MCQ generation."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from utils.chunking import smart_chunk_text

    model_name = "valhalla/t5-small-qg-hl"
    print(f"[Quiz Local] Using {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    chunks = smart_chunk_text(text, max_tokens=300, overlap=30)
    mcqs = []
    used_answers = set()

    for chunk in chunks:
        if len(mcqs) >= num_questions:
            break
        # Extract answer candidates
        words = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', chunk)
        candidates = list(set(words))
        random.shuffle(candidates)

        for answer in candidates[:3]:
            if len(mcqs) >= num_questions:
                break
            if answer.lower() in used_answers:
                continue

            highlighted = chunk.replace(answer, f"<hl> {answer} <hl>", 1)
            input_text = f"generate question: {highlighted}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                output = model.generate(inputs["input_ids"], max_new_tokens=64, num_beams=2)
            question = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            question = re.sub(r'^question:\s*', '', question, flags=re.IGNORECASE)

            if len(question) > 10:
                distractors = [c for c in candidates if c.lower() != answer.lower()][:3]
                while len(distractors) < 3:
                    distractors.append("None of the above")
                options = distractors[:3] + [answer]
                random.shuffle(options)
                correct_idx = options.index(answer)

                mcqs.append({
                    "id": len(mcqs) + 1,
                    "question": question,
                    "options": {chr(65+i): opt for i, opt in enumerate(options)},
                    "correct_answer": chr(65 + correct_idx),
                    "correct_text": answer,
                    "difficulty": difficulty
                })
                used_answers.add(answer.lower())

    return mcqs[:num_questions]


# ============================================
# Scoring System
# ============================================
class QuizScorer:
    def __init__(self):
        self.results = []

    def grade_quiz(self, mcqs: list, user_answers: dict) -> dict:
        correct_count = 0
        details = []
        for mcq in mcqs:
            q_id = mcq["id"]
            user_ans = user_answers.get(q_id, user_answers.get(str(q_id), ""))
            is_correct = user_ans.upper() == mcq["correct_answer"].upper()
            if is_correct:
                correct_count += 1
            details.append({
                "question_id": q_id, "question": mcq["question"],
                "user_answer": user_ans.upper(), "correct_answer": mcq["correct_answer"],
                "correct_text": mcq["correct_text"], "is_correct": is_correct
            })
        total = len(mcqs)
        pct = round((correct_count / total * 100), 1) if total > 0 else 0
        grade = ("A+" if pct >= 90 else "A" if pct >= 80 else "B" if pct >= 70
                 else "C" if pct >= 60 else "D" if pct >= 50 else "F")
        result = {"score": correct_count, "total": total, "percentage": pct,
                  "grade": grade, "details": details, "timestamp": datetime.now().isoformat()}
        self.results.append(result)
        return result

    def save_results(self, filepath: str = "data/quiz_results.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def load_results(self, filepath: str = "data/quiz_results.json") -> list:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            return self.results
        except FileNotFoundError:
            return []

    def get_statistics(self) -> dict:
        if not self.results:
            return {"message": "No quiz results available."}
        pcts = [r["percentage"] for r in self.results]
        return {"total_attempts": len(self.results), "average_score": round(sum(pcts)/len(pcts), 1),
                "best_score": max(pcts), "worst_score": min(pcts),
                "total_questions_answered": sum(r["total"] for r in self.results)}
