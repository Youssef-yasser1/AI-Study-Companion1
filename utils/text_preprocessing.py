# ============================================
# utils/text_preprocessing.py
# Text cleaning and preprocessing utilities
# ============================================

import re
import string


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text:
    - Replace multiple spaces with single space
    - Replace multiple newlines with double newline (paragraph break)
    - Strip leading/trailing whitespace
    """
    # Replace multiple newlines with paragraph separator
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Replace single newlines with space (within paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    return text.strip()


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters from text while preserving readability.
    
    Args:
        text: Input text to clean
        keep_punctuation: If True, keep basic punctuation marks
    
    Returns:
        Cleaned text string
    """
    if keep_punctuation:
        # Keep letters, numbers, basic punctuation, and Arabic characters
        pattern = r'[^\w\s\u0600-\u06FF.,;:?!\'"()\-/]'
    else:
        # Keep only letters, numbers, spaces, and Arabic characters
        pattern = r'[^\w\s\u0600-\u06FF]'
    
    text = re.sub(pattern, '', text)
    return text


def remove_headers_footers(text: str) -> str:
    """
    Remove common PDF headers/footers like page numbers, 
    repeated headers, etc.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            cleaned_lines.append(line)
            continue
        # Skip lines that are just page numbers
        if re.match(r'^\d{1,4}$', stripped):
            continue
        # Skip lines like "Page X of Y"
        if re.match(r'^[Pp]age\s+\d+\s+(of|/)\s+\d+$', stripped):
            continue
        # Skip very short lines that look like headers (all caps, < 5 words)
        # but keep them if they might be section titles
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_text(text: str) -> str:
    """
    Main text cleaning pipeline. Applies all preprocessing steps
    in the correct order.
    
    Args:
        text: Raw text extracted from PDF
    
    Returns:
        Cleaned and preprocessed text
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Remove headers/footers
    text = remove_headers_footers(text)
    
    # Step 2: Remove special characters
    text = remove_special_chars(text, keep_punctuation=True)
    
    # Step 3: Normalize whitespace
    text = normalize_whitespace(text)
    
    # Step 4: Final strip
    text = text.strip()
    
    return text


def get_sentences(text: str) -> list:
    """
    Split text into sentences using regex-based sentence boundary detection.
    Handles both English and Arabic sentence endings.
    
    Args:
        text: Input text
    
    Returns:
        List of sentence strings
    """
    # Split on sentence-ending punctuation followed by space or end of string
    sentences = re.split(r'(?<=[.!?؟])\s+', text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
