"""
ResumeIQ — app.py
Flask backend for AI-powered resume screening and ATS analysis.
"""
 
import os
import re
import logging
from pathlib import Path
import json
from werkzeug.security import generate_password_hash, check_password_hash
 
import PyPDF2
from docx import Document
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
 
# ─────────────────────────────────────────────
# App configuration
# ─────────────────────────────────────────────
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
app = Flask(__name__)
 
UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_CONTENT_MB = 16
 
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
 
UPLOAD_FOLDER.mkdir(exist_ok=True)
 
 
# ─────────────────────────────────────────────
# NLP stop-words (extended)
# ─────────────────────────────────────────────
 
STOP_WORDS: frozenset = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "have", "will",
    "your", "are", "you", "was", "were", "been", "into", "their", "they",
    "them", "about", "there", "which", "these", "also", "more", "other",
    "such", "than", "but", "not", "can", "all", "any", "our", "its",
    "has", "had", "may", "must", "use", "per", "get", "set", "new",
    "one", "two", "how", "who", "why", "each", "both", "some", "out",
    "her", "his", "him", "she", "via", "etc", "e.g.", "i.e.",
})
 
 
# ─────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────
 
def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allow-list."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )
 
 
def extract_text_from_pdf(filepath: str) -> str:
    """Extract plain text from a PDF file."""
    text_parts: list[str] = []
    try:
        with open(filepath, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", filepath, exc)
    return "\n".join(text_parts)
 
 
def extract_text_from_docx(filepath: str) -> str:
    """Extract plain text from a .docx file."""
    try:
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        logger.warning("DOCX extraction failed for %s: %s", filepath, exc)
        return ""
 
 
def extract_text_from_txt(filepath: str) -> str:
    """Read plain text from a .txt file."""
    try:
        with open(filepath, encoding="utf-8") as fh:
            return fh.read()
    except Exception as exc:
        logger.warning("TXT read failed for %s: %s", filepath, exc)
        return ""
 
 
def extract_text(filepath: str) -> str:
    """Dispatch text extraction based on file extension."""
    ext = filepath.rsplit(".", 1)[-1].lower()
    extractors = {
        "pdf":  extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "txt":  extract_text_from_txt,
    }
    extractor = extractors.get(ext)
    return extractor(filepath) if extractor else ""
 
 
# ─────────────────────────────────────────────
# NLP helpers
# ─────────────────────────────────────────────
 
def extract_keywords(text: str) -> set[str]:
    """Return a set of meaningful words from text (length ≥ 3, not stop-words)."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in tokens if w not in STOP_WORDS}
 
 
def calculate_match_score(resume: str, job_desc: str) -> tuple[float, list[str], list[str]]:
    """
    Compute TF-IDF cosine similarity and keyword overlap.
 
    Returns:
        score         – float 0-100, cosine similarity
        matching      – list of shared keywords (capped at 20)
        missing       – list of job-only keywords (capped at 20)
    """
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([resume, job_desc])
        raw_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        score = round(raw_similarity * 100, 1)
    except Exception as exc:
        logger.error("Similarity calculation failed: %s", exc)
        score = 0.0
 
    resume_kw = extract_keywords(resume)
    job_kw    = extract_keywords(job_desc)
 
    matching = sorted(resume_kw & job_kw)[:20]
    missing  = sorted(job_kw - resume_kw)[:20]
 
    return score, matching, missing
 
 
def calculate_ats_score(
    base_score: float,
    resume_text: str,
    missing_keywords: list[str],
) -> float:
    """
    Adjust the base cosine score with heuristic ATS penalties.
 
    Penalties:
        -10  if > 10 keywords are missing
        -10  if resume is too short (< 150 words)
        -5   if resume is suspiciously long (> 800 words)
    """
    ats = base_score
 
    if len(missing_keywords) > 10:
        ats -= 10
 
    word_count = len(resume_text.split())
    if word_count < 150:
        ats -= 10
    elif word_count > 800:
        ats -= 5
 
    return max(0.0, min(100.0, round(ats, 1)))
 
 
def classify_match(score: float) -> str:
    """Return a classification level for a given score."""
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Average"
    return "Low"
 
 USER_DB = "users.json"

def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=4)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
 
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    users = load_users()

    email = data.get("email")
    password = data.get("password")

    if email in users:
        return jsonify({"error": "User already exists"}), 400

    users[email] = generate_password_hash(password)
    save_users(users)

    return jsonify({"message": "Signup successful"})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    users = load_users()

    email = data.get("email")
    password = data.get("password")

    if email not in users:
        return jsonify({"error": "User not found"}), 404

    if not check_password_hash(users[email], password):
        return jsonify({"error": "Incorrect password"}), 401

    return jsonify({"message": "Login successful"})

@app.route("/")
def home():
    """Serve the main page."""
    return render_template("index.html")
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Accepts multipart form data:
        resume_file     – uploaded resume (PDF / DOCX / TXT)
        job_description – plain-text job description
    Returns JSON with scores, keywords, and stats.
    """
    resume_text = ""
 
    # ── Extract resume text ──
    uploaded = request.files.get("resume_file")
    if uploaded and uploaded.filename and allowed_file(uploaded.filename):
        filename = secure_filename(uploaded.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            uploaded.save(filepath)
            resume_text = extract_text(filepath)
            logger.info("Extracted %d words from '%s'", len(resume_text.split()), filename)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
 
    # Fallback: raw text field
    if not resume_text.strip():
        resume_text = request.form.get("resume_text", "")
 
    job_desc = request.form.get("job_description", "")
 
    # ── Validation ──
    if not resume_text.strip():
        return jsonify({"error": "Please upload a valid resume (PDF, DOCX, or TXT)."}), 400
    if not job_desc.strip():
        return jsonify({"error": "Please provide a job description."}), 400
 
    # ── Analysis ──
    match_score, matching, missing = calculate_match_score(resume_text, job_desc)
    ats_score   = calculate_ats_score(match_score, resume_text, missing)
    level = classify_match(match_score)
 
    return jsonify({
        "success":          True,
        "match_score":      match_score,
        "ats_score":        ats_score,
        "match_level":      level,
        "matching_keywords": matching[:10],
        "missing_keywords":  missing[:10],
        "matching_count":   len(matching),
        "missing_count":    len(missing),
        "resume_length":    len(resume_text.split()),
    })
 
 
# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    app.run(debug=True)