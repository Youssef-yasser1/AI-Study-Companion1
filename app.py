# ============================================
# app.py
# Flask Web Application for AI Study Companion
# Features: PDF upload, Summary, Q&A, Quiz
# ============================================

import os
import json
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS

from study_mode import StudySession
from evaluation import compute_f1, compute_rouge
import database as db
from flashcard_generator import generate_flashcards
from pdf_exporter import export_study_guide

# ============================================
# App Configuration
# ============================================
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = 'data/uploads'

CORS(app)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Global study session storage
# In production, use a proper session store / database
# ============================================
study_sessions = {}


def get_or_create_session(session_id: str) -> StudySession:
    """Get existing study session or create a new one."""
    if session_id not in study_sessions:
        study_sessions[session_id] = StudySession()
    return study_sessions[session_id]


# ============================================
# Routes
# ============================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """
    Upload and process a PDF or image file.
    Extracts text and prepares it for analysis.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"status": "error", "error": "No file selected"}), 400
        
        # Supported formats
        allowed_ext = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in allowed_ext:
            return jsonify({"status": "error", "error": f"Unsupported format. Accepted: PDF, PNG, JPG, BMP, TIFF, WEBP"}), 400
        
        # Read file bytes
        file_bytes = file.read()
        
        if not file_bytes:
            return jsonify({"status": "error", "error": "Empty file uploaded"}), 400
        
        # Create or get session
        sid = request.form.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        study_session = get_or_create_session(sid)
        
        # Load document (PDF or image)
        result = study_session.load_document(file_bytes, filename=file.filename)
        result['session_id'] = sid
        result['filename'] = file.filename
        study_session.document_name = file.filename
        
        # Save session to database
        if result["status"] == "success":
            try:
                db.save_session(sid, file.filename, study_session.document_text or '')
            except Exception:
                pass
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/summarize', methods=['POST'])
def summarize():
    """Generate a summary of the uploaded document."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        
        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No document loaded. Upload a PDF first."}), 400
        
        study_session = study_sessions[sid]
        result = study_session.get_summary()
        
        if result["status"] == "success":
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer a question based on the document."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        question = data.get('question', '').strip()
        
        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No document loaded. Upload a PDF first."}), 400
        
        if not question:
            return jsonify({"status": "error", "error": "Please provide a question."}), 400
        
        study_session = study_sessions[sid]
        result = study_session.ask_question(question)
        
        if result["status"] == "success":
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        logger.error(f"QA error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/quiz/generate', methods=['POST'])
def generate_quiz():
    """Generate MCQ quiz from the document."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        num_questions = data.get('num_questions', 5)
        difficulty = data.get('difficulty', 'medium')
        
        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No document loaded. Upload a PDF first."}), 400
        
        if difficulty not in ('easy', 'medium', 'hard'):
            difficulty = 'medium'
        
        study_session = study_sessions[sid]
        result = study_session.generate_quiz(
            num_questions=min(num_questions, 20),
            difficulty=difficulty
        )
        
        if result["status"] == "success":
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz():
    """Submit quiz answers and get score."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        answers = data.get('answers', {})
        
        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No active session."}), 400
        
        if not answers:
            return jsonify({"status": "error", "error": "No answers provided."}), 400
        
        study_session = study_sessions[sid]
        result = study_session.submit_quiz_answers(answers)
        
        # Save quiz result to database
        if result["status"] == "success":
            try:
                db.save_quiz_result(
                    sid, result.get("score", 0), result.get("total", 0),
                    result.get("percentage", 0), result.get("grade", ""),
                    result.get("difficulty", "medium"), result.get("details", [])
                )
            except Exception:
                pass
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        logger.error(f"Quiz submission error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/keywords', methods=['POST'])
def get_keywords():
    """Extract keywords from the document."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        max_keywords = data.get('max_keywords', 15)
        
        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No document loaded."}), 400
        
        study_session = study_sessions[sid]
        result = study_session.get_keywords(max_keywords=max_keywords)
        
        if result["status"] == "success":
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        logger.error(f"Keyword extraction error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/session/stats', methods=['POST'])
def session_stats():
    """Get session statistics."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        
        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No active session."}), 400
        
        study_session = study_sessions[sid]
        stats = study_session.get_session_summary()
        
        return jsonify({"status": "success", **stats}), 200
    
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/session/save', methods=['POST'])
def save_session():
    """Save session data to disk."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        
        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No active session."}), 400
        
        study_session = study_sessions[sid]
        filepath = study_session.save_session()
        
        return jsonify({
            "status": "success",
            "filepath": filepath,
            "message": "Session saved successfully."
        }), 200
    
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================
# Flashcard Routes
# ============================================

@app.route('/api/flashcards/generate', methods=['POST'])
def gen_flashcards():
    """Generate flashcards from the document."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        max_cards = data.get('max_cards', 10)

        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No document loaded."}), 400

        study_session = study_sessions[sid]
        text = study_session.document_text or ''
        cards = generate_flashcards(text, max_cards=min(max_cards, 20))

        if cards:
            try:
                db.save_flashcards(sid, cards)
            except Exception:
                pass
            return jsonify({"status": "success", "flashcards": cards, "count": len(cards)}), 200
        else:
            return jsonify({"status": "error", "error": "Could not generate flashcards."}), 400

    except Exception as e:
        logger.error(f"Flashcard error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/flashcards/review', methods=['POST'])
def review_flashcard():
    """Update flashcard mastery after review."""
    try:
        data = request.get_json()
        fc_id = data.get('flashcard_id')
        correct = data.get('correct', False)

        if not fc_id:
            return jsonify({"status": "error", "error": "Missing flashcard_id."}), 400

        db.update_flashcard_mastery(fc_id, correct)
        return jsonify({"status": "success", "message": "Flashcard updated."}), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/flashcards/due', methods=['POST'])
def due_flashcards():
    """Get flashcards due for review (spaced repetition)."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        cards = db.get_due_flashcards(sid)
        return jsonify({"status": "success", "flashcards": cards, "count": len(cards)}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================
# Dashboard & Export Routes
# ============================================

@app.route('/api/dashboard', methods=['POST'])
def dashboard():
    """Get study dashboard statistics."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')
        stats = db.get_study_stats(sid if sid else None)
        return jsonify({"status": "success", **stats}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    """Export study materials to PDF."""
    try:
        data = request.get_json()
        sid = data.get('session_id', '')

        if not sid or sid not in study_sessions:
            return jsonify({"status": "error", "error": "No active session."}), 400

        study_session = study_sessions[sid]

        # Gather all session data
        session_data = {
            "document_name": getattr(study_session, 'document_name', 'Document'),
            "summary": getattr(study_session, 'last_summary', ''),
            "keywords": getattr(study_session, 'last_keywords', []),
            "flashcards": db.get_flashcards(sid),
            "quiz_history": db.get_quiz_history(sid),
            "last_quiz": getattr(study_session, 'last_quiz', []),
            "last_quiz_result": getattr(study_session, 'last_quiz_result', None)
        }

        filepath = export_study_guide(session_data)
        return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))

    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/upload/multi', methods=['POST'])
def upload_multi():
    """Upload multiple files to one session."""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"status": "error", "error": "No files uploaded."}), 400

        sid = request.form.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        study_session = get_or_create_session(sid)

        allowed_ext = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        loaded_files = []
        combined_text = study_session.document_text or ''

        for f in files:
            ext = '.' + f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
            if ext not in allowed_ext:
                continue
            file_bytes = f.read()
            if not file_bytes:
                continue
            try:
                from pdf_loader import load_file
                text = load_file(file_bytes, f.filename)
                combined_text += '\n\n' + text
                loaded_files.append(f.filename)
            except Exception as e:
                logger.warning(f"Failed to load {f.filename}: {e}")

        if not loaded_files:
            return jsonify({"status": "error", "error": "No valid files could be processed."}), 400

        study_session.document_text = combined_text
        study_session.document_name = ', '.join(loaded_files)

        try:
            db.save_session(sid, study_session.document_name, combined_text)
        except Exception:
            pass

        return jsonify({
            "status": "success", "session_id": sid,
            "files_loaded": loaded_files, "count": len(loaded_files),
            "word_count": len(combined_text.split()),
            "message": f"Successfully loaded {len(loaded_files)} file(s)."
        }), 200

    except Exception as e:
        logger.error(f"Multi-upload error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all saved sessions."""
    try:
        sessions = db.get_all_sessions()
        return jsonify({"status": "success", "sessions": sessions}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({"status": "error", "error": "File too large. Maximum size is 50MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "error": "Internal server error."}), 500


# ============================================
# Run App
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  AI Study Companion - Web Server")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )