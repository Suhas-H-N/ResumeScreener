"""
ResumeIQ — app.py
Clean, production-style Flask backend for AI Resume Screener
"""

import os
import re
import json
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
USER_DB = "users.json"


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=4)


# ─────────────────────────────────────────────
# File Processing
# ─────────────────────────────────────────────

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.warning(f"PDF error: {e}")
    return text


def extract_text_from_docx(filepath):
    try:
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning(f"DOCX error: {e}")
        return ""


def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def extract_text(filepath):
    ext = filepath.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(filepath)
    elif ext == "docx":
        return extract_text_from_docx(filepath)
    elif ext == "txt":
        return extract_text_from_txt(filepath)
    return ""

SKILL_DB = {
    "python", "java", "c++", "javascript", "html", "css",
    "react", "node", "flask", "django",
    "sql", "mongodb",
    "machine learning", "deep learning", "nlp", "data analysis",
    "pandas", "numpy", "scikit-learn",
    "aws", "docker", "kubernetes", "git"
}

# ─────────────────────────────────────────────
# NLP Logic
# ─────────────────────────────────────────────

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "will"
}


def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def extract_skills(text):
    words = re.findall(r"\b[a-zA-Z+\-#]{2,}\b", text.lower())
    skills = set()
    for word in words:
        if word in SKILL_DB:
            skills.add(word)
        # Check for compound skills
        for skill in SKILL_DB:
            if skill.replace(" ", "").replace("-", "").replace("+", "") in word:
                skills.add(skill)
    return list(skills)


def calculate_match_score(resume, job_desc):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([resume, job_desc])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        score = round(similarity * 100, 1)
    except:
        score = 0

    resume_kw = extract_keywords(resume)
    job_kw = extract_keywords(job_desc)

    matching = list(resume_kw & job_kw)
    missing = list(job_kw - resume_kw)

    return score, matching, missing


def calculate_ats_score(score, resume_text, missing_keywords):
    ats = score

    if len(missing_keywords) > 10:
        ats -= 10

    word_count = len(resume_text.split())

    if word_count < 150:
        ats -= 10
    elif word_count > 800:
        ats -= 5

    return max(0, min(100, round(ats, 1)))


def classify_match(score):
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    return "Low"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login-page")
def login_page():
    return render_template("login.html")


@app.route("/signup-page")
def signup_page():
    return render_template("signup.html")


# ─────────────────────────────────────────────
# Auth APIs
# ─────────────────────────────────────────────

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    users = load_users()

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not name or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    if email in users:
        return jsonify({"error": "User already exists"}), 400

    users[email] = {
        "name": name,
        "password": generate_password_hash(password)
    }

    save_users(users)

    return jsonify({"message": "Signup successful"})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    users = load_users()

    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if email not in users:
        return jsonify({"error": "User not found"}), 404

    if not check_password_hash(users[email]["password"], password):
        return jsonify({"error": "Incorrect password"}), 401

    return jsonify({
        "message": "Login successful",
        "user": users[email]["name"]
    })


# ─────────────────────────────────────────────
# Resume Analysis
# ─────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():

    resume_text = ""

    file = request.files.get("resume_file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(path)
        resume_text = extract_text(path)

        if os.path.exists(path):
            os.remove(path)

    if not resume_text:
        resume_text = request.form.get("resume_text", "")

    job_desc = request.form.get("job_description", "")

    if not resume_text.strip():
        return jsonify({"error": "Please provide resume"}), 400

    if not job_desc.strip():
        return jsonify({"error": "Please provide job description"}), 400

    score, matching, missing = calculate_match_score(resume_text, job_desc)
    ats_score = calculate_ats_score(score, resume_text, missing)
    level = classify_match(score)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)
    missing_skills = list(set(job_skills) - set(resume_skills))

    return jsonify({
        "success": True,
        "match_score": score,
        "ats_score": ats_score,
        "match_level": level,
        "matching_keywords": matching[:10],
        "missing_keywords": missing[:10],
        "matching_count": len(matching),
        "missing_count": len(missing),
        "resume_length": len(resume_text.split()),
        "resume_skills": resume_skills,
        "missing_skills": missing_skills[:10]
    })


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)