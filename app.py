from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
from docx import Document
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'templates/static/uploads'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"
    return text

def extract_text_from_file(file_path):
    """Extract text based on file type"""
    file_ext = file_path.rsplit('.', 1)[1].lower()
    
    if file_ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == 'docx':
        return extract_text_from_docx(file_path)
    elif file_ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

def extract_keywords(text):
    """Extract important keywords from text"""
    # Simple keyword extraction - can be improved with NLTK
    words = re.findall(r'\b[a-z]+\b', text.lower())
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    return list(set(keywords))

def calculate_match_score(resume_text, job_description):
    """Calculate match score using TF-IDF and cosine similarity"""
    if not resume_text or not job_description:
        return 0, [], []
    
    # Vectorize texts
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        score_percentage = round(similarity_score * 100, 2)
    except:
        score_percentage = 0
    
    # Extract keywords
    resume_keywords = set(extract_keywords(resume_text))
    job_keywords = set(extract_keywords(job_description))
    matching_keywords = list(resume_keywords.intersection(job_keywords))
    missing_keywords = list(job_keywords - resume_keywords)
    
    return score_percentage, matching_keywords, missing_keywords

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        resume_text = ""
        
        # Check if file is uploaded
        if 'resume_file' in request.files:
            file = request.files['resume_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                resume_text = extract_text_from_file(filepath)
                # Clean up
                os.remove(filepath)
        
        # Check if text is pasted
        if 'resume_text' in request.form and request.form['resume_text']:
            resume_text = request.form['resume_text']
        
        job_description = request.form.get('job_description', '')
        
        if not resume_text.strip():
            return jsonify({'error': 'Please upload a resume file or paste resume text'}), 400
        
        if not job_description.strip():
            return jsonify({'error': 'Please enter a job description'}), 400
        
        # Calculate match score
        score, matching_keywords, missing_keywords = calculate_match_score(resume_text, job_description)
        
        # Determine match level
        if score >= 80:
            match_level = "Excellent"
            color = "green"
        elif score >= 60:
            match_level = "Good"
            color = "blue"
        elif score >= 40:
            match_level = "Average"
            color = "orange"
        else:
            match_level = "Poor"
            color = "red"
        
        # Resume summary
        resume_length = len(resume_text.split())
        resume_preview = resume_text[:300] + "..." if len(resume_text) > 300 else resume_text
        
        return jsonify({
            'success': True,
            'match_score': score,
            'match_level': match_level,
            'color': color,
            'matching_keywords': matching_keywords[:10],
            'missing_keywords': missing_keywords[:10],
            'matching_count': len(matching_keywords),
            'missing_count': len(missing_keywords),
            'resume_length': resume_length
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)