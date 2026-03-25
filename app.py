from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import re
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------------
# Configuration
# -------------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------
# Helper Functions
# -------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Extract text from PDF
def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except:
        pass
    return text


# Extract text from DOCX
def extract_text_from_docx(filepath):
    try:
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""


# Extract text from any file
def extract_text(filepath):
    ext = filepath.split(".")[-1].lower()

    if ext == "pdf":
        return extract_text_from_pdf(filepath)

    elif ext == "docx":
        return extract_text_from_docx(filepath)

    elif ext == "txt":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

    return ""


# Extract keywords
def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    stop_words = {
        "the","and","for","with","that","this","from","have","will",
        "your","are","you","was","were","been","into","their","they",
        "them","about","there","which","this","these"
    }

    keywords = [w for w in words if w not in stop_words]

    return list(set(keywords))


# Calculate similarity score
def calculate_score(resume, job_desc):

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([resume, job_desc])

        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        score = round(similarity * 100, 2)
    except:
        score = 0

    resume_keywords = set(extract_keywords(resume))
    job_keywords = set(extract_keywords(job_desc))

    matching = list(resume_keywords & job_keywords)
    missing = list(job_keywords - resume_keywords)

    return score, matching, missing

def calculate_ats_score(score, resume_text, missing_keywords):

    ats_score = score

    # Penalize if too many missing skills
    if len(missing_keywords) > 10:
        ats_score -= 10

    # Check resume length
    word_count = len(resume_text.split())

    if word_count < 150:
        ats_score -= 10
    elif word_count > 800:
        ats_score -= 5

    return max(0, min(100, round(ats_score, 2)))
# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:
        resume_text = ""

        if "resume_file" in request.files:
            file = request.files["resume_file"]

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                file.save(filepath)
                resume_text = extract_text(filepath)

                if os.path.exists(filepath):
                    os.remove(filepath)

        if request.form.get("resume_text"):
            resume_text = request.form.get("resume_text")

        job_desc = request.form.get("job_description", "")

        if not resume_text.strip():
            return jsonify({"error": "Please upload a resume"}), 400

        if not job_desc.strip():
            return jsonify({"error": "Please enter job description"}), 400

        score, matching, missing = calculate_score(resume_text, job_desc)
        ats_score = calculate_ats_score(score, resume_text, missing)

        if score >= 80:
            level = "Excellent"
            color = "green"
        elif score >= 60:
            level = "Good"
            color = "blue"
        elif score >= 40:
            level = "Average"
            color = "orange"
        else:
            level = "Low"
            color = "red"

        return jsonify({
            "success": True,
            "match_score": score,
            "ats_score": ats_score,
            "match_level": level,
            "color": color,
            "matching_keywords": matching[:10],
            "missing_keywords": missing[:10],
            "matching_count": len(matching),
            "missing_count": len(missing),
            "resume_length": len(resume_text.split())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)