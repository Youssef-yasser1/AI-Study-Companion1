# ============================================
# study_mode.py
# Combined Study Mode - integrates all features
# Summary + Q&A + Quiz in one session
# ============================================

import json
import os
import logging
from datetime import datetime
from typing import Optional

from pdf_loader import PDFLoader, load_pdf, load_file, IMAGE_EXTENSIONS
from summarizer import summarize_text
from qa_system import answer_question
from quiz_generator import generate_mcqs, QuizScorer
from keyword_extractor import extract_keywords, highlight_keywords_in_text

logger = logging.getLogger(__name__)


class StudySession:
    """
    Complete study session that combines all features:
    - PDF/Image loading and text extraction
    - Text summarization
    - Question answering
    - MCQ quiz generation with scoring
    - Keyword highlighting
    """
    
    def __init__(self):
        """Initialize a new study session."""
        self.pdf_loader = PDFLoader(backend="auto")
        self.scorer = QuizScorer()
        
        # Session state
        self.document_text = None
        self.document_name = None
        self.summary = None
        self.keywords = None
        self.current_quiz = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # History
        self.qa_history = []
        self.quiz_history = []
        
        # Cached results for PDF export
        self.last_summary = ''
        self.last_keywords = []
    
    def load_document(self, source, filename: str = None) -> dict:
        """
        Load a PDF or image document and extract text.
        
        Args:
            source: File path (str) or file bytes (bytes)
            filename: Original filename (for format detection)
        
        Returns:
            Dict with document info
        """
        try:
            # Reset cached results on new document
            self.summary = None
            self.keywords = None
            self.current_quiz = None
            
            if isinstance(source, str):
                self.document_name = os.path.basename(source)
                ext = os.path.splitext(source)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    with open(source, 'rb') as f:
                        self.document_text = load_file(f.read(), source)
                else:
                    self.document_text = self.pdf_loader.load_from_path(source)
            elif isinstance(source, bytes):
                fn = filename or "uploaded_document.pdf"
                self.document_name = fn
                ext = '.' + fn.rsplit('.', 1)[-1].lower() if '.' in fn else '.pdf'
                if ext in IMAGE_EXTENSIONS:
                    self.document_text = load_file(source, fn)
                else:
                    self.document_text = self.pdf_loader.load_from_bytes(source)
            else:
                raise TypeError(f"Expected str or bytes, got {type(source)}")
            
            word_count = len(self.document_text.split())
            
            return {
                "status": "success",
                "document_name": self.document_name,
                "word_count": word_count,
                "char_count": len(self.document_text),
                "preview": self.document_text[:500] + "..." if len(self.document_text) > 500 else self.document_text
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_summary(self) -> dict:
        """
        Generate a summary of the loaded document.
        Caches the result for subsequent calls.
        
        Returns:
            Summary result dictionary
        """
        if not self.document_text:
            return {"status": "error", "error": "No document loaded. Use load_document() first."}
        
        if self.summary:
            return {"status": "success", "summary": self.summary, "cached": True}
        
        try:
            result = summarize_text(self.document_text)
            self.summary = result["summary"]
            self.last_summary = result["summary"]
            return {
                "status": "success",
                "summary": result["summary"],
                "chunk_summaries": result["chunk_summaries"],
                "num_chunks": result["num_chunks"],
                "cached": False
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question about the loaded document.
        
        Args:
            question: The user's question
        
        Returns:
            Answer result dictionary
        """
        if not self.document_text:
            return {"status": "error", "error": "No document loaded. Use load_document() first."}
        
        try:
            result = answer_question(question, self.document_text)
            
            # Save to history
            self.qa_history.append({
                "question": question,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "success",
                "question": question,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "context_snippet": result["context_snippet"]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_quiz(
        self,
        num_questions: int = 5,
        difficulty: str = "medium"
    ) -> dict:
        """
        Generate a quiz from the loaded document.
        
        Args:
            num_questions: Number of MCQs to generate
            difficulty: 'easy', 'medium', or 'hard'
        
        Returns:
            Quiz data dictionary
        """
        if not self.document_text:
            return {"status": "error", "error": "No document loaded. Use load_document() first."}
        
        if difficulty not in ("easy", "medium", "hard"):
            return {"status": "error", "error": "Difficulty must be 'easy', 'medium', or 'hard'."}
        
        try:
            mcqs = generate_mcqs(
                self.document_text,
                num_questions=num_questions,
                difficulty=difficulty
            )
            
            self.current_quiz = mcqs
            self.last_quiz = mcqs  # Save for PDF export
            
            return {
                "status": "success",
                "num_questions": len(mcqs),
                "difficulty": difficulty,
                "questions": mcqs
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def submit_quiz_answers(self, user_answers: dict) -> dict:
        """
        Submit answers for the current quiz and get scored results.
        
        Args:
            user_answers: Dict mapping question ID to answer letter
                         e.g., {1: "A", 2: "C", 3: "B"}
        
        Returns:
            Graded quiz result dictionary
        """
        if not self.current_quiz:
            return {"status": "error", "error": "No quiz active. Generate a quiz first."}
        
        try:
            result = self.scorer.grade_quiz(self.current_quiz, user_answers)
            self.quiz_history.append(result)
            self.last_quiz_result = result  # Save for PDF export
            
            return {
                "status": "success",
                "score": result["score"],
                "total": result["total"],
                "percentage": result["percentage"],
                "grade": result["grade"],
                "details": result["details"]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_keywords(self, max_keywords: int = 15) -> dict:
        """
        Extract important keywords from the document.
        
        Args:
            max_keywords: Maximum keywords to return
        
        Returns:
            Keywords result dictionary
        """
        if not self.document_text:
            return {"status": "error", "error": "No document loaded."}
        
        if self.keywords:
            return {"status": "success", "keywords": self.keywords, "cached": True}
        
        try:
            keywords = extract_keywords(
                self.document_text,
                method="auto",
                max_keywords=max_keywords
            )
            self.keywords = keywords
            self.last_keywords = keywords
            
            return {
                "status": "success",
                "keywords": keywords,
                "cached": False
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_highlighted_text(self) -> dict:
        """
        Get the document text with important keywords highlighted.
        
        Returns:
            Dictionary with highlighted text
        """
        if not self.document_text:
            return {"status": "error", "error": "No document loaded."}
        
        # Extract keywords first if not done yet
        if not self.keywords:
            kw_result = self.get_keywords()
            if kw_result["status"] == "error":
                return kw_result
        
        highlighted = highlight_keywords_in_text(self.document_text, self.keywords)
        
        return {
            "status": "success",
            "highlighted_text": highlighted,
            "keyword_count": len(self.keywords)
        }
    
    def get_session_summary(self) -> dict:
        """
        Get a summary of the entire study session.
        
        Returns:
            Session statistics dictionary
        """
        quiz_stats = self.scorer.get_statistics()
        
        return {
            "session_id": self.session_id,
            "document_name": self.document_name,
            "has_summary": self.summary is not None,
            "questions_asked": len(self.qa_history),
            "quizzes_taken": len(self.quiz_history),
            "quiz_statistics": quiz_stats,
            "qa_history": self.qa_history
        }
    
    def save_session(self, directory: str = "data"):
        """
        Save the entire session data to disk.
        
        Args:
            directory: Directory to save session files
        """
        os.makedirs(directory, exist_ok=True)
        
        session_data = {
            "session_id": self.session_id,
            "document_name": self.document_name,
            "summary": self.summary,
            "keywords": self.keywords,
            "qa_history": self.qa_history,
            "quiz_history": self.quiz_history,
            "scorer_results": self.scorer.results
        }
        
        filepath = os.path.join(directory, f"session_{self.session_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Also save quiz results separately
        self.scorer.save_results(os.path.join(directory, "quiz_results.json"))
        
        print(f"[Study Session] Session saved to {filepath}")
        return filepath
