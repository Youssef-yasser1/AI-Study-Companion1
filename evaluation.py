# ============================================
# evaluation.py
# Evaluation metrics for QA and Summarization
# Includes F1 Score, ROUGE Score, and test cases
# ============================================

import re
import logging
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Try importing rouge_score
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


# ============================================
# F1 Score for Question Answering
# ============================================

def compute_f1(prediction: str, ground_truth: str) -> dict:
    """
    Compute token-level F1, Precision, and Recall between
    prediction and ground truth strings.
    
    This is the standard QA evaluation metric used in SQuAD.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Expected correct answer
    
    Returns:
        Dictionary with 'f1', 'precision', 'recall' scores
    """
    # Normalize text
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(ground_truth).split()
    
    if not pred_tokens and not truth_tokens:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    
    if not pred_tokens or not truth_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    # Count common tokens
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute Exact Match (EM) score.
    Returns 1.0 if normalized prediction matches ground truth exactly.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Expected correct answer
    
    Returns:
        1.0 if match, 0.0 otherwise
    """
    return 1.0 if _normalize_text(prediction) == _normalize_text(ground_truth) else 0.0


# ============================================
# ROUGE Score for Summarization
# ============================================

def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Compute ROUGE scores for summarization evaluation.
    
    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap
    ROUGE-L: Longest common subsequence
    
    Args:
        prediction: Generated summary
        reference: Reference/gold summary
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    if HAS_ROUGE:
        return _compute_rouge_library(prediction, reference)
    else:
        logger.warning("rouge_score not installed. Using simple ROUGE implementation.")
        return _compute_rouge_simple(prediction, reference)


def _compute_rouge_library(prediction: str, reference: str) -> dict:
    """Compute ROUGE using the rouge_score library."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    scores = scorer.score(reference, prediction)
    
    return {
        "rouge1": {
            "precision": round(scores['rouge1'].precision, 4),
            "recall": round(scores['rouge1'].recall, 4),
            "f1": round(scores['rouge1'].fmeasure, 4)
        },
        "rouge2": {
            "precision": round(scores['rouge2'].precision, 4),
            "recall": round(scores['rouge2'].recall, 4),
            "f1": round(scores['rouge2'].fmeasure, 4)
        },
        "rougeL": {
            "precision": round(scores['rougeL'].precision, 4),
            "recall": round(scores['rougeL'].recall, 4),
            "f1": round(scores['rougeL'].fmeasure, 4)
        }
    }


def _compute_rouge_simple(prediction: str, reference: str) -> dict:
    """Simple ROUGE implementation (fallback without library)."""
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return {
            "rouge1": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "rouge2": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "rougeL": {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        }
    
    # ROUGE-1 (unigram)
    r1_common = Counter(pred_tokens) & Counter(ref_tokens)
    r1_num = sum(r1_common.values())
    r1_p = r1_num / len(pred_tokens) if pred_tokens else 0
    r1_r = r1_num / len(ref_tokens) if ref_tokens else 0
    r1_f1 = 2 * r1_p * r1_r / (r1_p + r1_r) if (r1_p + r1_r) > 0 else 0
    
    # ROUGE-2 (bigram)
    pred_bigrams = [f"{pred_tokens[i]} {pred_tokens[i+1]}" for i in range(len(pred_tokens)-1)]
    ref_bigrams = [f"{ref_tokens[i]} {ref_tokens[i+1]}" for i in range(len(ref_tokens)-1)]
    r2_common = Counter(pred_bigrams) & Counter(ref_bigrams)
    r2_num = sum(r2_common.values())
    r2_p = r2_num / len(pred_bigrams) if pred_bigrams else 0
    r2_r = r2_num / len(ref_bigrams) if ref_bigrams else 0
    r2_f1 = 2 * r2_p * r2_r / (r2_p + r2_r) if (r2_p + r2_r) > 0 else 0
    
    # ROUGE-L (LCS)
    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    rl_p = lcs_len / len(pred_tokens) if pred_tokens else 0
    rl_r = lcs_len / len(ref_tokens) if ref_tokens else 0
    rl_f1 = 2 * rl_p * rl_r / (rl_p + rl_r) if (rl_p + rl_r) > 0 else 0
    
    return {
        "rouge1": {"f1": round(r1_f1, 4), "precision": round(r1_p, 4), "recall": round(r1_r, 4)},
        "rouge2": {"f1": round(r2_f1, 4), "precision": round(r2_p, 4), "recall": round(r2_r, 4)},
        "rougeL": {"f1": round(rl_f1, 4), "precision": round(rl_p, 4), "recall": round(rl_r, 4)}
    }


def _lcs_length(a: list, b: list) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _normalize_text(text: str) -> str:
    """Normalize text for evaluation: lowercase, remove articles/punctuation."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================
# Example Test Cases
# ============================================

def run_example_tests():
    """
    Run example test cases to validate evaluation metrics.
    """
    print("=" * 60)
    print("EVALUATION METRICS - TEST CASES")
    print("=" * 60)
    
    # --- F1 Score Tests ---
    print("\n--- F1 Score Tests (QA) ---\n")
    
    test_cases_f1 = [
        {
            "prediction": "machine learning is a subset of artificial intelligence",
            "ground_truth": "machine learning is a branch of artificial intelligence",
            "description": "Partial match"
        },
        {
            "prediction": "Python",
            "ground_truth": "Python",
            "description": "Exact match"
        },
        {
            "prediction": "deep learning uses neural networks",
            "ground_truth": "natural language processing uses text data",
            "description": "No overlap"
        },
        {
            "prediction": "the transformer architecture was introduced in 2017",
            "ground_truth": "transformer architecture introduced 2017",
            "description": "Close match with extra words"
        }
    ]
    
    for tc in test_cases_f1:
        result = compute_f1(tc["prediction"], tc["ground_truth"])
        em = compute_exact_match(tc["prediction"], tc["ground_truth"])
        print(f"  Test: {tc['description']}")
        print(f"    Prediction:   '{tc['prediction']}'")
        print(f"    Ground Truth: '{tc['ground_truth']}'")
        print(f"    F1={result['f1']}, Precision={result['precision']}, "
              f"Recall={result['recall']}, EM={em}")
        print()
    
    # --- ROUGE Score Tests ---
    print("\n--- ROUGE Score Tests (Summarization) ---\n")
    
    test_cases_rouge = [
        {
            "prediction": "Machine learning enables computers to learn from data and make predictions.",
            "reference": "Machine learning allows computers to learn from data without being explicitly programmed to make predictions.",
            "description": "Good summary"
        },
        {
            "prediction": "The weather is nice today.",
            "reference": "Deep learning models use multiple layers of neural networks for feature extraction.",
            "description": "Completely different"
        }
    ]
    
    for tc in test_cases_rouge:
        result = compute_rouge(tc["prediction"], tc["reference"])
        print(f"  Test: {tc['description']}")
        print(f"    Generated: '{tc['prediction'][:60]}...'")
        print(f"    Reference: '{tc['reference'][:60]}...'")
        print(f"    ROUGE-1 F1: {result['rouge1']['f1']}")
        print(f"    ROUGE-2 F1: {result['rouge2']['f1']}")
        print(f"    ROUGE-L F1: {result['rougeL']['f1']}")
        print()
    
    print("=" * 60)
    print("All evaluation tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    run_example_tests()
