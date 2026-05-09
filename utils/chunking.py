# ============================================
# utils/chunking.py
# Text chunking strategies for handling long documents
# ============================================

import re


def chunk_text(text: str, max_words: int = 500, overlap_words: int = 50) -> list:
    """
    Simple word-based chunking with overlap.
    Splits text into chunks of approximately max_words words,
    with overlap_words words of overlap between consecutive chunks.
    
    Args:
        text: Input text to chunk
        max_words: Maximum number of words per chunk
        overlap_words: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    words = text.split()
    
    # If text is short enough, return as single chunk
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = end - overlap_words
        
        # Prevent infinite loop
        if start >= len(words) - overlap_words:
            break
    
    return chunks


def smart_chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> list:
    """
    Smart chunking that respects sentence boundaries.
    Instead of cutting mid-sentence, this function tries to break
    at sentence endings to preserve context and readability.
    
    Args:
        text: Input text to chunk
        max_tokens: Approximate max words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks with sentence-boundary awareness
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?؟])\s+', text)
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        word_count = len(sentence.split())
        
        # If adding this sentence exceeds the limit and we have content
        if current_length + word_count > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Calculate overlap: keep last N words for context
            overlap_text = ' '.join(current_chunk).split()
            if len(overlap_text) > overlap:
                overlap_sentences = overlap_text[-overlap:]
                current_chunk = [' '.join(overlap_sentences)]
                current_length = overlap
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += word_count
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [text]


def get_chunk_count(text: str, max_tokens: int = 500) -> int:
    """
    Estimate the number of chunks a text will be split into.
    Useful for progress tracking.
    
    Args:
        text: Input text
        max_tokens: Max words per chunk
    
    Returns:
        Estimated number of chunks
    """
    word_count = len(text.split())
    if word_count <= max_tokens:
        return 1
    return (word_count // max_tokens) + 1
