# utils/__init__.py
# Utility package for AI Study Companion
from utils.text_preprocessing import clean_text, normalize_whitespace, remove_special_chars
from utils.chunking import chunk_text, smart_chunk_text

__all__ = [
    "clean_text",
    "normalize_whitespace", 
    "remove_special_chars",
    "chunk_text",
    "smart_chunk_text",
]
