# ============================================
# pdf_exporter.py
# Export study materials to PDF
# ============================================

import os
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def safe_text(text: str) -> str:
    """Sanitize text to only include characters supported by FPDF's default latin-1 Helvetica font."""
    if not text:
        return ""
    # Replace common unicode quotes and dashes with ascii equivalents
    text = text.replace('”', '"').replace('“', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('–', '-').replace('—', '-').replace('•', '-')
    # Force encode to latin-1 and replace anything else with '?'
    return str(text).encode('latin-1', 'replace').decode('latin-1')

def export_study_guide(session_data: dict, output_dir: str = "data/exports") -> str:
    """Export a complete study guide to PDF."""
    from fpdf import FPDF

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_name = re.sub(r'[^\w\s\-.]', '', session_data.get("document_name", "study_guide"))
    doc_name = doc_name.replace(' ', '_')[:30]
    filename = f"{doc_name}_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- Title Page ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 40, "", ln=True)
    pdf.cell(0, 15, safe_text("AI Study Companion"), ln=True, align="C")
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 12, safe_text("Study Guide"), ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, safe_text(f"Document: {session_data.get('document_name', 'N/A')}"), ln=True, align="C")
    pdf.cell(0, 8, safe_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align="C")

    # ---- Summary Section ----
    summary = session_data.get("summary", "")
    if summary:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(50, 100, 200)
        pdf.cell(0, 12, safe_text("Summary"), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        _write_formatted_text(pdf, summary)

    # ---- Keywords / Key Concepts ----
    keywords = session_data.get("keywords", [])
    if keywords:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(50, 100, 200)
        pdf.cell(0, 12, safe_text("Key Concepts & Definitions"), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

        definitions = [k for k in keywords if k.get("type") == "definition"]
        key_points = [k for k in keywords if k.get("type") == "key_point"]
        terms = [k for k in keywords if k.get("type") == "keyword"]

        if definitions:
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(100, 50, 200)
            pdf.cell(0, 10, safe_text("Definitions"), ln=True)
            pdf.set_text_color(0, 0, 0)
            for d in definitions:
                pdf.set_font("Helvetica", "B", 11)
                term = d.get("term", "")
                if term:
                    pdf.set_x(22)
                    pdf.cell(0, 7, safe_text(term), ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_x(25)
                try:
                    pdf.multi_cell(0, 6, safe_text(d.get('text', '')))
                except Exception as e:
                    logger.warning("Skipped a definition line due to FPDF layout error.")
                pdf.ln(2)

        if key_points:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(100, 50, 200)
            pdf.cell(0, 10, safe_text("Key Points"), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            for kp in key_points:
                try:
                    pdf.multi_cell(0, 6, safe_text(f"  * {kp.get('text', '')}"))
                except Exception:
                    logger.warning("Skipped a key point line due to FPDF layout error.")
                pdf.ln(1)

        if terms:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(100, 50, 200)
            pdf.cell(0, 10, safe_text("Important Terms"), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            term_list = ", ".join(t.get("keyword", "") for t in terms if t.get("keyword"))
            try:
                pdf.multi_cell(0, 6, safe_text(f"  {term_list}"))
            except Exception:
                logger.warning("Skipped important terms line due to FPDF layout error.")

    # ---- Flashcards ----
    flashcards = session_data.get("flashcards", [])
    if flashcards:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(50, 100, 200)
        pdf.cell(0, 12, safe_text("Flashcards"), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        for i, fc in enumerate(flashcards, 1):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_x(22)
            pdf.cell(0, 7, safe_text(f"{i}. {fc.get('term', '')}"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(27)
            try:
                pdf.multi_cell(0, 6, safe_text(fc.get('definition', '')))
            except Exception:
                logger.warning("Skipped a flashcard definition due to FPDF layout error.")
            pdf.ln(3)

    # ---- Last Quiz Questions & Answers ----
    last_quiz = session_data.get("last_quiz", [])
    last_quiz_result = session_data.get("last_quiz_result")
    
    if last_quiz:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(50, 100, 200)
        pdf.cell(0, 12, safe_text("Multiple Choice Questions"), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        
        user_answers = {}
        if last_quiz_result and "details" in last_quiz_result:
            for d in last_quiz_result["details"]:
                user_answers[d["question_id"]] = {
                    "answer": d["user_answer"],
                    "is_correct": d["is_correct"]
                }
                
        for i, q in enumerate(last_quiz, 1):
            pdf.set_font("Helvetica", "B", 12)
            try:
                pdf.multi_cell(0, 6, safe_text(f"Q{i}: {q.get('question', '')}"))
            except Exception:
                logger.warning("Skipped a quiz question due to FPDF layout error.")
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)
            
            options = q.get('options', {})
            for letter, opt_text in options.items():
                pdf.set_x(25)
                try:
                    pdf.multi_cell(0, 6, safe_text(f"{letter}) {opt_text}"))
                except Exception:
                    logger.warning("Skipped a quiz option due to FPDF layout error.")
                
            pdf.ln(2)
            
            correct_ans = safe_text(q.get('correct_answer', ''))
            
            if q.get('id') in user_answers:
                u_ans = user_answers[q['id']]
                u_ans_text = safe_text(u_ans['answer'])
                
                if u_ans['is_correct']:
                    pdf.set_text_color(0, 150, 0)
                    pdf.cell(0, 6, f"    Your Answer: {u_ans_text} (Correct)", ln=True)
                else:
                    pdf.set_text_color(200, 0, 0)
                    pdf.cell(0, 6, f"    Your Answer: {u_ans_text} (Incorrect)", ln=True)
                    pdf.set_text_color(0, 150, 0)
                    pdf.cell(0, 6, f"    Correct Answer: {correct_ans}", ln=True)
            else:
                pdf.set_text_color(0, 150, 0)
                pdf.cell(0, 6, f"    Answer: {correct_ans}", ln=True)
                
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)

    # ---- Quiz History ----
    quiz_history = session_data.get("quiz_history", [])
    if quiz_history:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(50, 100, 200)
        pdf.cell(0, 12, safe_text("Quiz History"), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        for i, qr in enumerate(quiz_history, 1):
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, safe_text(f"  Quiz {i}: {qr.get('score', 0)}/{qr.get('total', 0)} ({qr.get('percentage', 0)}%) - Grade: {qr.get('grade', '')}"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, safe_text(f"    Difficulty: {qr.get('difficulty', 'N/A')}"), ln=True)
            pdf.ln(2)

    pdf.output(filepath)
    print(f"[PDF Export] Saved to {filepath}")
    return filepath

def _write_formatted_text(pdf, text):
    """Write text to PDF, handling markdown-like formatting."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^#{1,3}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---$', '', text, flags=re.MULTILINE)

    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith(('- ', '* ', '• ')):
                    pdf.multi_cell(0, 6, safe_text(f"  {line}"))
                else:
                    pdf.multi_cell(0, 6, safe_text(line))
            except Exception:
                logger.warning("Skipped a summary line due to FPDF layout error.")
        pdf.ln(3)
