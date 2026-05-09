# ============================================
# pdf_loader.py
# PDF text extraction and processing module
# ============================================

import io
import os
import re
import logging
from typing import Optional

# Try PyPDF2 first, fall back to pdfplumber
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from utils.text_preprocessing import clean_text

# Configure logger
logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Handles PDF file loading and text extraction.
    Supports both PyPDF2 and pdfplumber as extraction backends.
    Includes error handling for corrupt, empty, or invalid PDFs.
    """
    
    def __init__(self, backend: str = "auto"):
        """
        Initialize PDFLoader.
        
        Args:
            backend: 'pypdf2', 'pdfplumber', or 'auto' (tries both)
        """
        self.backend = backend
        self._validate_backend()
    
    def _validate_backend(self):
        """Ensure at least one PDF library is available."""
        if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
            raise ImportError(
                "No PDF library found. Install PyPDF2 or pdfplumber:\n"
                "  pip install PyPDF2 pdfplumber"
            )
    
    def load_from_path(self, file_path: str) -> str:
        """
        Extract text from a PDF file on disk.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            Extracted and cleaned text
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF or is empty
        """
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File '{file_path}' is not a PDF file.")
        
        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading: {file_path}")
        
        return self.load_from_bytes(pdf_bytes)
    
    def load_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes (e.g., from file upload).
        
        Args:
            pdf_bytes: Raw bytes of the PDF file
        
        Returns:
            Extracted and cleaned text
        
        Raises:
            ValueError: If PDF is empty or unreadable
        """
        if not pdf_bytes:
            raise ValueError("Empty PDF data provided.")
        
        raw_text = self._extract_text(pdf_bytes)
        
        if not raw_text or not raw_text.strip():
            raise ValueError(
                "No text could be extracted from the PDF. "
                "The file may be scanned/image-based or empty."
            )
        
        # Clean and preprocess
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text:
            raise ValueError("PDF text is empty after preprocessing.")
        
        return cleaned_text
    
    def _extract_text(self, pdf_bytes: bytes) -> str:
        """
        Extract raw text from PDF bytes using the configured backend.
        
        Args:
            pdf_bytes: Raw PDF bytes
        
        Returns:
            Raw extracted text
        """
        if self.backend == "pypdf2" and HAS_PYPDF2:
            return self._extract_with_pypdf2(pdf_bytes)
        elif self.backend == "pdfplumber" and HAS_PDFPLUMBER:
            return self._extract_with_pdfplumber(pdf_bytes)
        else:
            # Auto mode: try PyPDF2 first, then pdfplumber
            text = ""
            if HAS_PYPDF2:
                try:
                    text = self._extract_with_pypdf2(pdf_bytes)
                except Exception as e:
                    logger.warning(f"PyPDF2 extraction failed: {e}")
            
            if not text.strip() and HAS_PDFPLUMBER:
                try:
                    text = self._extract_with_pdfplumber(pdf_bytes)
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed: {e}")
            
            return text
    
    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> str:
        """Extract text using PyPDF2."""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        
        if len(reader.pages) == 0:
            raise ValueError("PDF has no pages.")
        
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())
            except Exception as e:
                logger.warning(f"Failed to extract page {i+1}: {e}")
                continue
        
        return '\n\n'.join(pages_text)
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text using pdfplumber."""
        pages_text = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("PDF has no pages.")
            
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text.strip())
                except Exception as e:
                    logger.warning(f"Failed to extract page {i+1}: {e}")
                    continue
        
        return '\n\n'.join(pages_text)
    
    def get_page_count(self, pdf_bytes: bytes) -> int:
        """
        Get the number of pages in a PDF.
        
        Args:
            pdf_bytes: Raw PDF bytes
        
        Returns:
            Number of pages
        """
        if HAS_PYPDF2:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            return len(reader.pages)
        elif HAS_PDFPLUMBER:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                return len(pdf.pages)
        return 0
    
    def get_metadata(self, pdf_bytes: bytes) -> dict:
        """
        Extract PDF metadata (title, author, etc.).
        
        Args:
            pdf_bytes: Raw PDF bytes
        
        Returns:
            Dictionary of metadata fields
        """
        metadata = {}
        if HAS_PYPDF2:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                meta = reader.metadata
                if meta:
                    metadata = {
                        "title": meta.get("/Title", ""),
                        "author": meta.get("/Author", ""),
                        "subject": meta.get("/Subject", ""),
                        "pages": len(reader.pages)
                    }
            except Exception:
                pass
        return metadata


# ============================================
# Convenience functions
# ============================================

def load_pdf(source, backend: str = "auto") -> str:
    """
    Convenience function to load and extract text from a PDF.
    
    Args:
        source: File path (str) or file bytes (bytes)
        backend: 'pypdf2', 'pdfplumber', or 'auto'
    
    Returns:
        Cleaned extracted text
    """
    loader = PDFLoader(backend=backend)
    
    if isinstance(source, str):
        return loader.load_from_path(source)
    elif isinstance(source, bytes):
        return loader.load_from_bytes(source)
    else:
        raise TypeError(f"Expected str or bytes, got {type(source)}")


# ============================================
# Image Text Extraction (multi-method OCR)
# ============================================

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

# Load HF token
_HF_TOKEN = os.environ.get("HF_TOKEN")
if not _HF_TOKEN:
    for _env_file in ["key.env", ".env"]:
        if os.path.exists(_env_file):
            with open(_env_file, 'r') as _f:
                for _line in _f:
                    if _line.startswith("HF_TOKEN="):
                        _HF_TOKEN = _line.strip().split("=", 1)[1]
                        break


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using multiple methods.
    Tries: HF Document QA API -> pytesseract -> easyocr
    """
    errors = []

    # Method 1: HF Document Question Answering (understands layout, tables)
    if _HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=_HF_TOKEN)

            # Ask multiple questions to extract different parts of the image
            questions = [
                "What is all the text content in this document?",
                "What are the headings and titles?",
                "What are the bullet points or listed items?",
                "What is in the tables?",
                "What are the key terms and definitions?"
            ]

            all_text = []
            for q in questions:
                try:
                    results = client.document_question_answering(
                        image=image_bytes,
                        question=q,
                        model="impira/layoutlm-document-qa"
                    )
                    for r in results:
                        answer = r.answer if hasattr(r, 'answer') else str(r)
                        if answer and answer.strip() and answer.strip() not in all_text:
                            all_text.append(answer.strip())
                except Exception:
                    continue

            if all_text:
                combined = '. '.join(all_text)
                if len(combined) > 20:
                    print(f"[Image OCR] Extracted {len(combined)} chars via Document QA API")
                    return clean_text(combined)
        except Exception as e:
            errors.append(f"HF DocQA: {e}")
            logger.warning(f"HF Document QA failed: {e}")

    # Method 2: pytesseract (if Tesseract binary installed)
    try:
        from PIL import Image
        import pytesseract
        # Auto-detect Tesseract path on Windows
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe'),
        ]
        for tp in tesseract_paths:
            if os.path.exists(tp):
                pytesseract.pytesseract.tesseract_cmd = tp
                break
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        if text and text.strip() and len(text.strip()) > 10:
            print(f"[Image OCR] Extracted {len(text)} chars via pytesseract")
            return clean_text(text)
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"pytesseract: {e}")
        logger.warning(f"pytesseract failed: {e}")

    # Method 3: EasyOCR (if installed)
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(image_bytes, detail=0, paragraph=True)
        text = '\n'.join(results)
        if text and text.strip():
            print(f"[Image OCR] Extracted {len(text)} chars via EasyOCR")
            return clean_text(text)
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"easyocr: {e}")

    # Method 4: HF image-to-text (limited but works as last resort)
    if _HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=_HF_TOKEN)
            result = client.image_to_text(
                image_bytes,
                model="microsoft/trocr-base-printed"
            )
            text = result.generated_text if hasattr(result, 'generated_text') else str(result)
            if text and text.strip() and len(text.strip()) > 5:
                return clean_text(text)
        except Exception as e:
            errors.append(f"trocr: {e}")

    error_details = "; ".join(errors) if errors else "No OCR methods available"
    raise ValueError(
        f"Could not extract text from image. {error_details}. "
        f"Try installing Tesseract-OCR: https://github.com/tesseract-ocr/tesseract"
    )


def load_file(source_bytes: bytes, filename: str) -> str:
    """Load text from any supported file (PDF or image)."""
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if ext == '.pdf':
        return load_pdf(source_bytes)
    elif ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(source_bytes)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP"
        )

