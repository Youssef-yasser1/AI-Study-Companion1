# ============================================
# main.py
# CLI Interface for AI Study Companion
# Provides interactive command-line study mode
# ============================================

import os
import sys
import json

from study_mode import StudySession
from evaluation import run_example_tests


def print_banner():
    """Print the application banner."""
    print()
    print("=" * 60)
    print("   📚  AI STUDY COMPANION  📚")
    print("   Powered by Transformer Models")
    print("=" * 60)
    print()


def print_menu():
    """Print the main menu options."""
    print("\n" + "-" * 40)
    print("  STUDY MENU")
    print("-" * 40)
    print("  1. 📄 Load PDF Document")
    print("  2. 📋 Generate Summary")
    print("  3. ❓ Ask a Question (Q&A)")
    print("  4. 📝 Generate Quiz (MCQs)")
    print("  5. 🎯 Take Quiz")
    print("  6. 🔑 Extract Keywords")
    print("  7. 📊 View Session Stats")
    print("  8. 💾 Save Session")
    print("  9. 📈 Run Evaluation Tests")
    print("  0. 🚪 Exit")
    print("-" * 40)


def load_pdf_interactive(session: StudySession):
    """Interactive PDF loading."""
    print("\n📄 LOAD PDF DOCUMENT")
    print("-" * 30)
    
    path = input("Enter PDF file path: ").strip()
    
    if not path:
        print("❌ No path provided.")
        return
    
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    
    print("⏳ Loading PDF...")
    result = session.load_document(path)
    
    if result["status"] == "success":
        print(f"✅ Document loaded: {result['document_name']}")
        print(f"   Word count: {result['word_count']}")
        print(f"   Preview: {result['preview'][:200]}...")
    else:
        print(f"❌ Error: {result['error']}")


def generate_summary_interactive(session: StudySession):
    """Interactive summary generation."""
    print("\n📋 GENERATE SUMMARY")
    print("-" * 30)
    
    if not session.document_text:
        print("❌ No document loaded. Load a PDF first (option 1).")
        return
    
    print("⏳ Generating summary (this may take a moment)...")
    result = session.get_summary()
    
    if result["status"] == "success":
        print(f"\n{'=' * 50}")
        print("📋 DOCUMENT SUMMARY")
        print(f"{'=' * 50}")
        if result.get("cached"):
            print("(Cached result)")
        print(f"\n{result['summary']}")
        print(f"\n{'=' * 50}")
        if "num_chunks" in result:
            print(f"Processed in {result['num_chunks']} chunk(s)")
    else:
        print(f"❌ Error: {result['error']}")


def ask_question_interactive(session: StudySession):
    """Interactive question answering."""
    print("\n❓ ASK A QUESTION")
    print("-" * 30)
    
    if not session.document_text:
        print("❌ No document loaded. Load a PDF first (option 1).")
        return
    
    print("Type 'back' to return to menu.\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ('back', 'quit', 'exit', ''):
            break
        
        print("⏳ Searching for answer...")
        result = session.ask_question(question)
        
        if result["status"] == "success":
            print(f"\n📖 Answer: {result['answer']}")
            print(f"🎯 Confidence: {result['confidence']:.1%}")
            if result.get("context_snippet"):
                print(f"📄 Context: ...{result['context_snippet'][:150]}...")
            print()
        else:
            print(f"❌ Error: {result['error']}")


def generate_quiz_interactive(session: StudySession):
    """Interactive quiz generation."""
    print("\n📝 GENERATE QUIZ")
    print("-" * 30)
    
    if not session.document_text:
        print("❌ No document loaded. Load a PDF first (option 1).")
        return
    
    # Get number of questions
    try:
        num = input("Number of questions (default 5): ").strip()
        num_questions = int(num) if num else 5
    except ValueError:
        num_questions = 5
    
    # Get difficulty
    print("Difficulty levels: easy, medium, hard")
    difficulty = input("Difficulty (default medium): ").strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    
    print(f"⏳ Generating {num_questions} {difficulty} MCQs...")
    result = session.generate_quiz(num_questions=num_questions, difficulty=difficulty)
    
    if result["status"] == "success":
        print(f"\n✅ Generated {result['num_questions']} questions!\n")
        display_quiz(result["questions"])
    else:
        print(f"❌ Error: {result['error']}")


def display_quiz(questions: list):
    """Display quiz questions."""
    print("=" * 50)
    print("📝 QUIZ")
    print("=" * 50)
    
    for q in questions:
        print(f"\nQ{q['id']}. {q['question']}")
        print(f"   [{q['difficulty'].upper()}]")
        for letter, option in q['options'].items():
            print(f"     {letter}) {option}")
    
    print("\n" + "=" * 50)


def take_quiz_interactive(session: StudySession):
    """Interactive quiz taking with scoring."""
    print("\n🎯 TAKE QUIZ")
    print("-" * 30)
    
    if not session.current_quiz:
        print("❌ No quiz available. Generate a quiz first (option 4).")
        return
    
    questions = session.current_quiz
    user_answers = {}
    
    print(f"\nAnswer {len(questions)} questions. Enter A, B, C, or D.\n")
    
    for q in questions:
        print(f"\nQ{q['id']}. {q['question']}")
        print(f"   [{q['difficulty'].upper()}]")
        for letter, option in q['options'].items():
            print(f"     {letter}) {option}")
        
        while True:
            answer = input("   Your answer: ").strip().upper()
            if answer in ('A', 'B', 'C', 'D'):
                user_answers[q['id']] = answer
                break
            else:
                print("   ⚠️ Please enter A, B, C, or D.")
    
    # Grade the quiz
    print("\n⏳ Grading quiz...")
    result = session.submit_quiz_answers(user_answers)
    
    if result["status"] == "success":
        print(f"\n{'=' * 50}")
        print("📊 QUIZ RESULTS")
        print(f"{'=' * 50}")
        print(f"\n  Score: {result['score']}/{result['total']}")
        print(f"  Percentage: {result['percentage']}%")
        print(f"  Grade: {result['grade']}")
        
        print(f"\n  Detailed Results:")
        for detail in result["details"]:
            status = "✅" if detail["is_correct"] else "❌"
            print(f"    {status} Q{detail['question_id']}: "
                  f"You answered {detail['user_answer']}, "
                  f"Correct: {detail['correct_answer']} "
                  f"({detail['correct_text']})")
        
        print(f"\n{'=' * 50}")
    else:
        print(f"❌ Error: {result['error']}")


def extract_keywords_interactive(session: StudySession):
    """Interactive keyword extraction."""
    print("\n🔑 EXTRACT KEYWORDS")
    print("-" * 30)
    
    if not session.document_text:
        print("❌ No document loaded. Load a PDF first (option 1).")
        return
    
    print("⏳ Extracting keywords...")
    result = session.get_keywords(max_keywords=15)
    
    if result["status"] == "success":
        print(f"\n{'=' * 50}")
        print("🔑 IMPORTANT KEYWORDS")
        print(f"{'=' * 50}")
        
        for i, kw in enumerate(result["keywords"], 1):
            importance_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                kw.get("importance", "low"), "⚪"
            )
            print(f"  {i:2d}. {importance_icon} {kw['keyword']} "
                  f"(importance: {kw.get('importance', 'unknown')})")
        
        print(f"\n{'=' * 50}")
    else:
        print(f"❌ Error: {result['error']}")


def view_stats_interactive(session: StudySession):
    """View session statistics."""
    print("\n📊 SESSION STATISTICS")
    print("-" * 30)
    
    stats = session.get_session_summary()
    
    print(f"  Session ID: {stats['session_id']}")
    print(f"  Document: {stats['document_name'] or 'None loaded'}")
    print(f"  Summary generated: {'Yes' if stats['has_summary'] else 'No'}")
    print(f"  Questions asked: {stats['questions_asked']}")
    print(f"  Quizzes taken: {stats['quizzes_taken']}")
    
    if stats['quiz_statistics'] and 'total_attempts' in stats['quiz_statistics']:
        qs = stats['quiz_statistics']
        print(f"\n  Quiz Statistics:")
        print(f"    Total attempts: {qs['total_attempts']}")
        print(f"    Average score: {qs['average_score']}%")
        print(f"    Best score: {qs['best_score']}%")
        print(f"    Total questions answered: {qs['total_questions_answered']}")
    
    if stats['qa_history']:
        print(f"\n  Recent Q&A:")
        for qa in stats['qa_history'][-5:]:
            print(f"    Q: {qa['question'][:50]}...")
            print(f"    A: {qa['answer'][:50]}... (conf: {qa['confidence']:.1%})")


def main():
    """Main CLI entry point."""
    print_banner()
    
    session = StudySession()
    
    while True:
        print_menu()
        
        choice = input("\nSelect option (0-9): ").strip()
        
        if choice == "1":
            load_pdf_interactive(session)
        elif choice == "2":
            generate_summary_interactive(session)
        elif choice == "3":
            ask_question_interactive(session)
        elif choice == "4":
            generate_quiz_interactive(session)
        elif choice == "5":
            take_quiz_interactive(session)
        elif choice == "6":
            extract_keywords_interactive(session)
        elif choice == "7":
            view_stats_interactive(session)
        elif choice == "8":
            filepath = session.save_session()
            print(f"✅ Session saved to: {filepath}")
        elif choice == "9":
            run_example_tests()
        elif choice == "0":
            print("\n👋 Goodbye! Happy studying!")
            break
        else:
            print("⚠️ Invalid option. Please enter 0-9.")


if __name__ == "__main__":
    main()
