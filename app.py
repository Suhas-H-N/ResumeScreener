"""
ResumeIQ v2.0 — AI Resume Screener
Author: Suhas H N
"""

import os, re, json, uuid, logging
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from collections import Counter

from flask import (Flask, jsonify, render_template, request,
                   send_file, session, redirect, url_for)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import PyPDF2
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "resumeiq_dev_secret_2026")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
USER_DB            = "users.json"
HISTORY_DB         = "history.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# SKILL TAXONOMY
# ──────────────────────────────────────────────
TECH_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go",
    "rust", "kotlin", "swift", "ruby", "php", "scala", "r",
    # Web
    "react", "angular", "vue", "nextjs", "nodejs", "express", "django",
    "flask", "fastapi", "spring", "html", "css", "tailwind",
    # Data / ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "opencv", "nlp", "machine learning",
    "deep learning", "data science", "sql", "mongodb", "postgresql",
    "mysql", "redis", "elasticsearch",
    # DevOps / Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
    "jenkins", "github actions", "ci/cd", "linux", "bash",
    # Tools
    "git", "jira", "figma", "postman", "graphql", "rest", "api",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving",
    "critical thinking", "adaptability", "creativity", "time management",
    "collaboration", "attention to detail", "analytical", "organized",
    "motivated", "proactive", "presentation", "negotiation", "mentoring",
}

ALL_SKILLS = TECH_SKILLS | SOFT_SKILLS

# ──────────────────────────────────────────────
# USER / HISTORY DB
# ──────────────────────────────────────────────
def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_users():    return load_json(USER_DB, {})
def save_users(u):   save_json(USER_DB, u)
def load_history():  return load_json(HISTORY_DB, {})
def save_history(h): save_json(HISTORY_DB, h)

# ──────────────────────────────────────────────
# AUTH DECORATOR
# ──────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated

# ──────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ──────────────────────────────────────────────
# TEXT EXTRACTION
# ──────────────────────────────────────────────
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
            doc = DocxDocument(filepath)
            return "\n".join(p.text for p in doc.paragraphs)
        elif ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
    return ""

# ──────────────────────────────────────────────
# NLP & SCORING
# ──────────────────────────────────────────────
STOP_WORDS = {
    "the","and","for","with","this","that","are","was","were","have",
    "has","had","will","would","can","could","should","may","might",
    "shall","been","being","from","into","onto","upon","your","their",
    "our","which","when","where","how","what","who","why","also","than",
    "then","just","but","not","all","any","both","each","few","more",
    "most","other","some","such","only","own","same","too","very",
}

def clean_text(text):
    text = re.sub(r"[^\w\s\+\#]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def extract_keywords(text):
    words = re.findall(r"\b[a-z][a-z0-9\+\#]{2,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}

def extract_skills_from_text(text):
    text_lower = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill)
    return found

def calculate_match_score(resume, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        vectors = vectorizer.fit_transform([resume, job_desc])
        score   = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    except Exception:
        score = 0.0

    resume_kw = extract_keywords(clean_text(resume))
    job_kw    = extract_keywords(clean_text(job_desc))

    matching = list(resume_kw & job_kw)
    missing  = list(job_kw - resume_kw)

    return round(score, 1), matching, missing

def classify(score):
    if score >= 80: return "Excellent"
    if score >= 60: return "Good"
    if score >= 40: return "Average"
    return "Low"

def score_color(score):
    if score >= 80: return "#10b981"
    if score >= 60: return "#f59e0b"
    if score >= 40: return "#f97316"
    return "#ef4444"

# ──────────────────────────────────────────────
# ATS SCORING (NEW)
# ──────────────────────────────────────────────
def calculate_ats_score(resume_text, job_desc):
    """
    Holistic ATS scoring across 5 dimensions:
    keyword density, skill match, length, format signals, experience signals.
    """
    scores = {}

    # 1. Keyword density
    job_kw    = extract_keywords(clean_text(job_desc))
    resume_kw = extract_keywords(clean_text(resume_text))
    if job_kw:
        overlap = len(resume_kw & job_kw) / len(job_kw)
        scores["keyword_density"] = round(min(overlap * 100, 100))
    else:
        scores["keyword_density"] = 0

    # 2. Skill match
    job_skills    = extract_skills_from_text(job_desc)
    resume_skills = extract_skills_from_text(resume_text)
    if job_skills:
        matched = len(resume_skills & job_skills)
        scores["skill_match"] = round(min(matched / len(job_skills) * 100, 100))
    else:
        scores["skill_match"] = 50

    # 3. Length quality (300-1200 words optimal)
    word_count = len(resume_text.split())
    if 300 <= word_count <= 1200:
        scores["length"] = 100
    elif word_count < 150:
        scores["length"] = 30
    elif word_count < 300:
        scores["length"] = 60
    else:
        scores["length"] = 75

    # 4. Section presence signals
    section_keywords = ["experience", "education", "skills", "projects",
                        "summary", "objective", "certifications", "achievements"]
    text_lower = resume_text.lower()
    found_sections = sum(1 for s in section_keywords if s in text_lower)
    scores["sections"] = round((found_sections / len(section_keywords)) * 100)

    # 5. Action verb / impact signals
    action_verbs = ["developed", "built", "designed", "led", "managed",
                    "improved", "achieved", "implemented", "created",
                    "optimized", "delivered", "increased", "reduced"]
    found_verbs = sum(1 for v in action_verbs if v in text_lower)
    scores["action_verbs"] = round(min(found_verbs / 5 * 100, 100))

    # Weighted overall
    weights = {"keyword_density": 0.30, "skill_match": 0.30,
               "length": 0.15, "sections": 0.15, "action_verbs": 0.10}
    overall = sum(scores[k] * weights[k] for k in scores)
    scores["overall"] = round(overall, 1)

    return scores

# ──────────────────────────────────────────────
# RESUME STATS (NEW)
# ──────────────────────────────────────────────
def resume_stats(text):
    words     = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    lines     = [l for l in text.splitlines() if l.strip()]

    # Readability: Flesch-Kincaid approximation
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    syllables  = sum(count_syllables(w) for w in words)
    fk_score   = max(0, round(
        206.835 - 1.015 * (word_count / sent_count)
               - 84.6  * (syllables / max(word_count, 1))
    ))

    return {
        "word_count":     word_count,
        "char_count":     len(text),
        "sentence_count": sent_count,
        "line_count":     len(lines),
        "avg_word_len":   round(sum(len(w) for w in words) / max(word_count, 1), 1),
        "readability":    fk_score,
    }

def count_syllables(word):
    word = word.lower()
    count = len(re.findall(r"[aeiou]+", word))
    return max(count, 1)

# ──────────────────────────────────────────────
# RECOMMENDATIONS ENGINE (UPGRADED)
# ──────────────────────────────────────────────
def generate_recommendations(missing_keywords, missing_skills, ats_scores, stats):
    tips = []

    if missing_skills:
        top = list(missing_skills)[:5]
        tips.append(f"Add these in-demand skills: {', '.join(top)}")

    if missing_keywords:
        top_kw = missing_keywords[:4]
        tips.append(f"Include job-specific keywords: {', '.join(top_kw)}")

    if ats_scores.get("sections", 100) < 70:
        tips.append("Add clear section headers: Experience, Education, Skills, Projects")

    if ats_scores.get("action_verbs", 100) < 60:
        tips.append("Use strong action verbs: Led, Developed, Optimized, Delivered, Achieved")

    if ats_scores.get("length", 100) < 60:
        if stats.get("word_count", 0) < 300:
            tips.append("Resume is too short — expand your experience and project descriptions")
        else:
            tips.append("Resume may be too long — aim for 400–900 words for better ATS parsing")

    if ats_scores.get("keyword_density", 100) < 50:
        tips.append("Keyword density is low — mirror the language used in the job description")

    tips.append("Quantify achievements with numbers (e.g. 'Reduced load time by 40%')")
    tips.append("Keep formatting clean — avoid tables, columns, and images for ATS compatibility")

    return tips[:6]  # cap at 6

# ──────────────────────────────────────────────
# PDF REPORT GENERATOR (UPGRADED)
# ──────────────────────────────────────────────
def generate_pdf_report(data):
    filename  = f"ResumeIQ_Report_{uuid.uuid4().hex[:8]}.pdf"
    file_path = filename

    doc    = SimpleDocTemplate(file_path, pagesize=letter,
                               leftMargin=50, rightMargin=50,
                               topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()

    accent  = colors.HexColor("#6ee7b7")
    accent2 = colors.HexColor("#6366f1")
    dark    = colors.HexColor("#0d0f14")
    gray    = colors.HexColor("#64748b")

    title_style = ParagraphStyle("Title2", parent=styles["Heading1"],
                                  fontSize=22, textColor=colors.HexColor("#1e293b"),
                                  spaceAfter=4)
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"],
                                  fontSize=13, textColor=colors.HexColor("#1e293b"),
                                  spaceBefore=14, spaceAfter=4)
    body_style  = ParagraphStyle("Body2", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#334155"),
                                  spaceAfter=4, leading=14)
    tip_style   = ParagraphStyle("Tip", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#334155"),
                                  spaceAfter=3, leftIndent=12, leading=14)

    score       = data.get("match_score", 0)
    ats         = data.get("ats_scores", {})
    story       = []

    # ── Header ──
    story.append(Paragraph("ResumeIQ — AI Resume Analysis Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=gray)
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=accent, spaceAfter=12))

    # ── Scores table ──
    story.append(Paragraph("Match Overview", h2_style))
    score_color_hex = score_color(score)
    score_data = [
        ["Match Score", "ATS Score", "Level", "Words"],
        [
            f"{score}%",
            f"{ats.get('overall', 0)}%",
            data.get("level", "—"),
            str(data.get("stats", {}).get("word_count", "—")),
        ]
    ]
    score_table = Table(score_data, colWidths=[120, 120, 120, 120])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.HexColor("#1e293b")),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  10),
        ("FONTNAME",    (0, 1), (-1, 1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 1), (-1, 1),  16),
        ("TEXTCOLOR",   (0, 1), (0, 1),   colors.HexColor(score_color_hex.lstrip("#") and score_color_hex)),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, 1), [colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 14))

    # ── ATS Breakdown ──
    if ats:
        story.append(Paragraph("ATS Score Breakdown", h2_style))
        ats_labels = {
            "keyword_density": "Keyword Density",
            "skill_match":     "Skill Match",
            "sections":        "Section Structure",
            "action_verbs":    "Action Verbs",
            "length":          "Resume Length",
        }
        ats_rows = [["Dimension", "Score", "Status"]]
        for key, label in ats_labels.items():
            val = ats.get(key, 0)
            status = "✓ Good" if val >= 70 else ("△ Fair" if val >= 40 else "✗ Needs work")
            ats_rows.append([label, f"{val}%", status])
        ats_table = Table(ats_rows, colWidths=[200, 80, 200])
        ats_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#f1f5f9")),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("ALIGN",       (1, 0), (1, -1),  "CENTER"),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING",  (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(ats_table)
        story.append(Spacer(1, 14))

    # ── Skills ──
    resume_skills  = data.get("resume_skills", [])
    missing_skills = data.get("missing_skills", [])

    if resume_skills:
        story.append(Paragraph("Detected Skills", h2_style))
        story.append(Paragraph(", ".join(resume_skills), body_style))
        story.append(Spacer(1, 8))

    if missing_skills:
        story.append(Paragraph("Missing Skills (from Job Description)", h2_style))
        story.append(Paragraph(", ".join(missing_skills), body_style))
        story.append(Spacer(1, 8))

    # ── Keywords ──
    matching_kw = data.get("matching_keywords", [])
    missing_kw  = data.get("missing_keywords", [])

    if matching_kw:
        story.append(Paragraph("Matching Keywords", h2_style))
        story.append(Paragraph(", ".join(matching_kw), body_style))

    if missing_kw:
        story.append(Paragraph("Missing Keywords", h2_style))
        story.append(Paragraph(", ".join(missing_kw), body_style))

    # ── Recommendations ──
    recommendations = data.get("recommendations", [])
    if recommendations:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Recommendations", h2_style))
        for i, tip in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {tip}", tip_style))

    # ── Footer ──
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph("Generated by ResumeIQ v2.0 — Suhas H N",
                            ParagraphStyle("Footer", parent=styles["Normal"],
                                           fontSize=8, textColor=gray,
                                           alignment=1, spaceBefore=6)))

    doc.build(story)
    return file_path

# ──────────────────────────────────────────────
# AUTH ROUTES
# ──────────────────────────────────────────────
@app.route("/signup", methods=["POST"])
def signup():
    data  = request.json or {}
    name  = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    pwd   = data.get("password", "")

    if not name or not email or not pwd:
        return jsonify({"error": "All fields required"}), 400
    if len(pwd) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    users = load_users()
    if email in users:
        return jsonify({"error": "Email already registered"}), 409

    users[email] = {
        "name":     name,
        "password": generate_password_hash(pwd),
        "joined":   datetime.utcnow().isoformat(),
    }
    save_users(users)
    return jsonify({"message": "Account created successfully"}), 201


@app.route("/login", methods=["POST"])
def login():
    data  = request.json or {}
    email = data.get("email", "").strip().lower()
    pwd   = data.get("password", "")

    users = load_users()
    if email not in users:
        return jsonify({"error": "No account found with this email"}), 404
    if not check_password_hash(users[email]["password"], pwd):
        return jsonify({"error": "Incorrect password"}), 401

    session["user_email"] = email
    session["user_name"]  = users[email]["name"]
    return jsonify({"message": "Login successful", "user": users[email]["name"]})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


@app.route("/me")
def me():
    if "user_email" not in session:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "name":  session["user_name"],
        "email": session["user_email"],
    })

# ──────────────────────────────────────────────
# CORE ANALYZE ROUTE
# ──────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    resume_text = ""
    file        = request.files.get("resume_file")

    if file and file.filename and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        path  = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        file.save(path)
        resume_text = extract_text(path)
        try:
            os.remove(path)
        except Exception:
            pass

    if not resume_text:
        resume_text = request.form.get("resume_text", "").strip()

    job_desc = request.form.get("job_description", "").strip()

    if not resume_text:
        return jsonify({"error": "Please upload a resume or paste resume text"}), 400
    if not job_desc:
        return jsonify({"error": "Please enter a job description"}), 400
    if len(resume_text) < 50:
        return jsonify({"error": "Resume text is too short to analyze"}), 400

    # ── Scoring ──
    score, matching_kw, missing_kw = calculate_match_score(resume_text, job_desc)
    level       = classify(score)
    ats_scores  = calculate_ats_score(resume_text, job_desc)
    stats       = resume_stats(resume_text)

    # ── Skills ──
    resume_skills  = sorted(extract_skills_from_text(resume_text))
    job_skills     = sorted(extract_skills_from_text(job_desc))
    missing_skills = sorted(set(job_skills) - set(resume_skills))

    # ── Recommendations ──
    recommendations = generate_recommendations(
        missing_kw, missing_skills, ats_scores, stats
    )

    result = {
        "match_score":        score,
        "level":              level,
        "ats_scores":         ats_scores,
        "matching_keywords":  matching_kw[:15],
        "missing_keywords":   missing_kw[:15],
        "resume_skills":      resume_skills[:20],
        "job_skills":         job_skills[:20],
        "missing_skills":     missing_skills[:15],
        "recommendations":    recommendations,
        "stats":              stats,
    }

    # ── Save to history if logged in ──
    if "user_email" in session:
        history = load_history()
        email   = session["user_email"]
        if email not in history:
            history[email] = []
        history[email].insert(0, {
            "id":          uuid.uuid4().hex[:8],
            "date":        datetime.utcnow().isoformat(),
            "match_score": score,
            "ats_score":   ats_scores["overall"],
            "level":       level,
            "job_snippet": job_desc[:80] + ("…" if len(job_desc) > 80 else ""),
        })
        history[email] = history[email][:20]   # keep last 20
        save_history(history)

    return jsonify(result)

# ──────────────────────────────────────────────
# HISTORY ROUTE (NEW)
# ──────────────────────────────────────────────
@app.route("/history")
@login_required
def get_history():
    history = load_history()
    email   = session["user_email"]
    return jsonify(history.get(email, []))


@app.route("/history/<entry_id>", methods=["DELETE"])
@login_required
def delete_history_entry(entry_id):
    history = load_history()
    email   = session["user_email"]
    if email in history:
        history[email] = [e for e in history[email] if e["id"] != entry_id]
        save_history(history)
    return jsonify({"success": True})

# ──────────────────────────────────────────────
# COMPARE ROUTE (NEW) — compare two resumes
# ──────────────────────────────────────────────
@app.route("/compare", methods=["POST"])
def compare():
    job_desc = request.form.get("job_description", "").strip()
    texts    = []

    for key in ["resume_a", "resume_b"]:
        file = request.files.get(f"{key}_file")
        text = ""
        if file and file.filename and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            path  = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(path)
            text = extract_text(path)
            try: os.remove(path)
            except: pass
        if not text:
            text = request.form.get(key, "").strip()
        texts.append(text)

    if not texts[0] or not texts[1] or not job_desc:
        return jsonify({"error": "Provide both resumes and a job description"}), 400

    results = []
    for text in texts:
        score, matching, missing = calculate_match_score(text, job_desc)
        ats_scores = calculate_ats_score(text, job_desc)
        results.append({
            "match_score": score,
            "ats_score":   ats_scores["overall"],
            "level":       classify(score),
            "skills":      sorted(extract_skills_from_text(text))[:15],
        })

    winner = "A" if results[0]["match_score"] >= results[1]["match_score"] else "B"
    return jsonify({"resume_a": results[0], "resume_b": results[1], "winner": winner})

# ──────────────────────────────────────────────
# PDF DOWNLOAD (UPGRADED)
# ──────────────────────────────────────────────
@app.route("/download-report", methods=["POST"])
def download_report():
    data      = request.json or {}
    file_path = generate_pdf_report(data)
    return send_file(
        file_path,
        as_attachment=True,
        download_name="ResumeIQ_Report.pdf",
        mimetype="application/pdf",
    )

# ──────────────────────────────────────────────
# PAGE ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login-page")
def login_page():
    return render_template("login.html")

@app.route("/signup-page")
def signup_page():
    return render_template("signup.html")

@app.route("/compare-page")
def compare_page():
    return render_template("compare.html")

@app.route("/history-page")
def history_page():
    return render_template("history.html")

# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
