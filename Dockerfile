FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pytesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/exports

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "2"]
