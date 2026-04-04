import os
import re
import json
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# ------------------ CONFIG ------------------
app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
USER_DB = "users.json"


# ------------------ UTILS ------------------
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


# ------------------ TEXT EXTRACTION ------------------
def extract_text(filepath):
    ext = filepath.split(".")[-1].lower()

    try:
        if ext == "pdf":
            text = ""
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text

        elif ext == "docx":
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)

        elif ext == "txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    except:
        return ""

    return ""


# ------------------ NLP ------------------
STOP_WORDS = {"the", "and", "for", "with", "this", "that"}


def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def calculate_match_score(resume, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, job_desc])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

    resume_kw = extract_keywords(resume)
    job_kw = extract_keywords(job_desc)

    return round(score, 1), list(resume_kw & job_kw), list(job_kw - resume_kw)


def classify(score):
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    return "Low"


# ------------------ NEW FEATURES ------------------

def generate_recommendations(missing_keywords):
    if not missing_keywords:
        return ["Your resume is well optimized 🎉"]

    return [
        f"Add keywords like: {', '.join(missing_keywords[:5])}",
        "Improve formatting and ATS compatibility",
        "Use action verbs and measurable results"
    ]


def generate_pdf(data):
    file_path = "report.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"Match Score: {data['match_score']}%", styles["Normal"]),
        Paragraph(f"Level: {data['level']}", styles["Normal"]),
        Paragraph(f"Missing Keywords: {', '.join(data['missing_keywords'])}", styles["Normal"]),
        Paragraph(f"Recommendations: {', '.join(data['recommendations'])}", styles["Normal"])
    ]

    doc.build(content)
    return file_path


# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    users = load_users()

    if data["email"] in users:
        return jsonify({"error": "User exists"}), 400

    users[data["email"]] = {
        "name": data["name"],
        "password": generate_password_hash(data["password"])
    }

    save_users(users)
    return jsonify({"message": "Signup successful"})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    users = load_users()

    if data["email"] not in users:
        return jsonify({"error": "User not found"}), 404

    if not check_password_hash(users[data["email"]]["password"], data["password"]):
        return jsonify({"error": "Wrong password"}), 401

    return jsonify({"message": "Login successful"})


@app.route("/analyze", methods=["POST"])
def analyze():
    resume_text = ""
    file = request.files.get("resume_file")

    if file and allowed_file(file.filename):
        path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(path)
        resume_text = extract_text(path)
        os.remove(path)

    if not resume_text:
        resume_text = request.form.get("resume_text", "")

    job_desc = request.form.get("job_description", "")

    if not resume_text or not job_desc:
        return jsonify({"error": "Missing input"}), 400

    score, matching, missing = calculate_match_score(resume_text, job_desc)
    level = classify(score)

    recommendations = generate_recommendations(missing)

    return jsonify({
        "match_score": score,
        "level": level,
        "matching_keywords": matching[:10],
        "missing_keywords": missing[:10],
        "recommendations": recommendations
    })


@app.route("/download-report", methods=["POST"])
def download_report():
    data = request.json
    file_path = generate_pdf(data)
    return send_file(file_path, as_attachment=True)


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)