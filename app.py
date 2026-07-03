"""
ResumeIQ Pro v3.1 — AI-Powered Resume Screening Platform
Complete, production-ready Flask application
"""
import os, time, logging, csv, io
from pathlib import Path
from datetime import datetime, date
from functools import wraps

from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from extensions import db, login_manager, bcrypt, cors, limiter
from models import User, Analysis, Comparison, JobApplication, AuditLog
from config import config
from nlp_utils import (
    calculate_match_score, calculate_ats_score, calculate_readability,
    extract_skills_from_text, generate_recommendations, classify_match_level,
    get_score_color, detect_experience_level, estimate_salary_range,
    calculate_quantification_score, detect_career_gaps,
)
from document_utils import DocumentProcessor
from ai_service import get_ai_analyzer

app = Flask(__name__)
env = os.getenv("FLASK_ENV", "development")
app.config.from_object(config[env])

db.init_app(app)
login_manager.init_app(app)
bcrypt.init_app(app)
cors.init_app(app, resources={r"/api/*": {"origins": "*"}})
limiter.init_app(app)

Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ai_analyzer = get_ai_analyzer()


# ─── HELPERS ──────────────────────────────────────────────
def audit(action: str, user_id=None, details=None):
    try:
        log = AuditLog(
            user_id=user_id, action=action,
            ip_address=request.remote_addr,
            user_agent=(request.user_agent.string or "")[:255],
            details=details or {}
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        logger.error(f"Audit error: {e}")


def api_login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return wrapper


def current_user():
    uid = session.get("user_id")
    return db.session.get(User, uid) if uid else None


# ─── PAGE ROUTES ──────────────────────────────────────────
@app.route("/")
def index():
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

@app.route("/profile-page")
def profile_page():
    return render_template("profile.html")

@app.route("/jobs-page")
def jobs_page():
    return render_template("jobs.html")


# ─── AUTH ─────────────────────────────────────────────────
@app.route("/api/signup", methods=["POST"])
@limiter.limit("10 per hour")
def signup():
    data     = request.get_json() or {}
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    audit("signup", user.id, {"email": email})
    return jsonify({"message": "Account created", "user": user.to_dict()}), 201


@app.route("/api/login", methods=["POST"])
@limiter.limit("20 per minute")
def login():
    data     = request.get_json() or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        audit("login_failed", details={"email": email})
        return jsonify({"error": "Invalid email or password"}), 401
    if not user.is_active:
        return jsonify({"error": "Account is disabled"}), 403

    user.last_login = datetime.utcnow()
    db.session.commit()
    session.permanent = True
    session["user_id"]    = user.id
    session["user_email"] = user.email
    session["user_name"]  = user.name
    audit("login", user.id)
    return jsonify({"message": "Login successful", "user": user.to_dict()}), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    uid = session.get("user_id")
    if uid:
        audit("logout", uid)
    session.clear()
    return jsonify({"message": "Logged out"}), 200


@app.route("/api/me")
def me():
    user = current_user()
    if not user:
        return jsonify({"logged_in": False}), 200
    return jsonify({"logged_in": True, "user": user.to_dict()}), 200


@app.route("/api/forgot-password", methods=["POST"])
@limiter.limit("5 per hour")
def forgot_password():
    data  = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email required"}), 400
    user = User.query.filter_by(email=email).first()
    if user:
        token = user.generate_reset_token()
        db.session.commit()
        # In production: send reset email with token
        logger.info(f"Password reset token for {email}: {token}")
        audit("password_reset_requested", user.id)
    return jsonify({"message": "If that email exists, a reset link has been sent"}), 200


@app.route("/api/reset-password", methods=["POST"])
def reset_password():
    data     = request.get_json() or {}
    token    = data.get("token", "")
    new_pass = data.get("password", "")
    if not token or not new_pass:
        return jsonify({"error": "Token and password required"}), 400
    if len(new_pass) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    user = User.query.filter_by(reset_token=token).first()
    if not user or not user.verify_reset_token(token):
        return jsonify({"error": "Invalid or expired token"}), 400
    user.set_password(new_pass)
    user.reset_token = user.reset_token_expires = None
    db.session.commit()
    audit("password_reset", user.id)
    return jsonify({"message": "Password reset successfully"}), 200


# ─── PROFILE ──────────────────────────────────────────────
@app.route("/api/profile", methods=["PUT"])
@api_login_required
def update_profile():
    user = current_user()
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    user.name    = name
    user.company = data.get("company", "").strip() or None
    user.role    = data.get("role", "").strip() or None
    db.session.commit()
    session["user_name"] = name
    audit("profile_updated", user.id)
    return jsonify({"message": "Profile updated", "user": user.to_dict()}), 200


@app.route("/api/profile/password", methods=["PUT"])
@api_login_required
def change_password():
    user = current_user()
    data = request.get_json() or {}
    cur  = data.get("current_password", "")
    new  = data.get("new_password", "")
    if not cur or not new:
        return jsonify({"error": "Both passwords required"}), 400
    if not user.check_password(cur):
        return jsonify({"error": "Current password incorrect"}), 400
    if len(new) < 8:
        return jsonify({"error": "New password must be ≥ 8 characters"}), 400
    user.set_password(new)
    db.session.commit()
    audit("password_changed", user.id)
    return jsonify({"message": "Password changed"}), 200


@app.route("/api/profile", methods=["DELETE"])
@api_login_required
def delete_account():
    user = current_user()
    try:
        db.session.delete(user)
        db.session.commit()
        session.clear()
        return jsonify({"message": "Account deleted"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to delete account"}), 500


@app.route("/api/stats")
@api_login_required
def user_stats():
    user  = current_user()
    total = Analysis.query.filter_by(user_id=user.id).count()
    avg_m = db.session.query(db.func.avg(Analysis.match_score)).filter_by(user_id=user.id).scalar() or 0
    avg_a = db.session.query(db.func.avg(Analysis.ats_score)).filter_by(user_id=user.id).scalar() or 0
    best  = db.session.query(db.func.max(Analysis.match_score)).filter_by(user_id=user.id).scalar() or 0
    return jsonify({
        "total_analyses": total,
        "avg_match_score": round(avg_m, 1),
        "avg_ats_score":   round(avg_a, 1),
        "best_score":      round(best,  1),
        "member_since":    user.created_at.isoformat() if user.created_at else None,
    }), 200


# ─── CORE ANALYSIS ────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
@limiter.limit("30 per hour")
def analyze_resume():
    start = time.time()
    resume_text  = ""
    resume_fname = None

    if "resume_file" in request.files:
        f = request.files["resume_file"]
        if f and f.filename and DocumentProcessor.allowed_file(f.filename):
            fname    = secure_filename(f.filename)
            fpath    = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            resume_fname = fname
            f.save(fpath)
            resume_text = DocumentProcessor.extract_text(fpath)
            try:
                os.remove(fpath)
            except:
                pass

    if not resume_text:
        resume_text = request.form.get("resume_text", "").strip()

    job_desc    = request.form.get("job_description", "").strip()
    job_title   = request.form.get("job_title", "").strip()
    job_company = request.form.get("job_company", "").strip()

    valid, err = DocumentProcessor.validate_resume_content(resume_text)
    if not valid:
        return jsonify({"error": err}), 400
    if not job_desc or len(job_desc) < 50:
        return jsonify({"error": "Job description too short (min 50 chars)"}), 400

    try:
        match_score, matching_kw, missing_kw = calculate_match_score(resume_text, job_desc)
        ats_scores  = calculate_ats_score(resume_text, job_desc)
        stats       = calculate_readability(resume_text)
        level       = classify_match_level(match_score)
        resume_skills = sorted(extract_skills_from_text(resume_text))
        job_skills    = sorted(extract_skills_from_text(job_desc))
        missing_skills = sorted(set(job_skills) - set(resume_skills))
        recommendations = generate_recommendations(missing_kw, missing_skills, ats_scores, stats)
        exp_info     = detect_experience_level(resume_text)
        salary       = estimate_salary_range(set(resume_skills), exp_info)
        quant        = calculate_quantification_score(resume_text)
        gaps         = detect_career_gaps(resume_text)

        result = {
            "match_score":       match_score,
            "level":             level,
            "color":             get_score_color(match_score),
            "ats_scores":        ats_scores,
            "matching_keywords": matching_kw[:20],
            "missing_keywords":  missing_kw[:20],
            "resume_skills":     resume_skills[:25],
            "job_skills":        job_skills[:25],
            "missing_skills":    missing_skills[:20],
            "recommendations":   recommendations,
            "stats":             stats,
            "experience_info":   exp_info,
            "salary_estimate":   salary,
            "quantification":    quant,
            "career_gaps":       gaps,
        }

        if ai_analyzer.is_available():
            try:
                ai = ai_analyzer.analyze_resume(resume_text, job_desc, result)
                result.update(ai)
            except Exception as e:
                logger.error(f"AI error: {e}")

        ms = int((time.time() - start) * 1000)
        result["processing_time_ms"] = ms

        user = current_user()
        if user:
            a = Analysis(
                user_id=user.id,
                resume_filename=resume_fname,
                resume_text_snippet=resume_text[:500],
                job_title=job_title, job_company=job_company,
                job_description_snippet=job_desc[:200],
                match_score=match_score, ats_score=ats_scores["overall"], level=level,
                ats_breakdown=ats_scores,
                matching_keywords=matching_kw[:20], missing_keywords=missing_kw[:20],
                resume_skills=resume_skills[:25], missing_skills=missing_skills[:20],
                recommendations=recommendations, stats=stats,
                experience_info=exp_info, salary_estimate=salary,
                quantification=quant, career_gaps=gaps,
                ai_summary=result.get("ai_summary"),
                ai_recommendations=result.get("ai_recommendations"),
                strengths=result.get("strengths"), weaknesses=result.get("weaknesses"),
                interview_tips=result.get("interview_tips"),
                processing_time_ms=ms,
            )
            db.session.add(a)
            db.session.commit()
            result["analysis_id"] = a.id
            audit("analyze", user.id, {"match": match_score, "ats": ats_scores["overall"]})

        return jsonify(result), 200

    except Exception as e:
        logger.exception(f"Analysis error: {e}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500


# ─── COMPARE ──────────────────────────────────────────────
@app.route("/api/compare", methods=["POST"])
@limiter.limit("20 per hour")
def compare_resumes():
    job_desc    = request.form.get("job_description", "").strip()
    resumes     = []
    resume_names = []

    for key in ["resume_a", "resume_b"]:
        text = ""
        name = f"Resume {key[-1].upper()}"
        fkey = f"{key}_file"
        if fkey in request.files:
            f = request.files[fkey]
            if f and f.filename and DocumentProcessor.allowed_file(f.filename):
                fname = secure_filename(f.filename)
                name  = fname
                fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                f.save(fpath)
                text = DocumentProcessor.extract_text(fpath)
                try: os.remove(fpath)
                except: pass
        if not text:
            text = request.form.get(key, "").strip()
            if not text:
                name = f"Resume {key[-1].upper()}"
        resumes.append(text)
        resume_names.append(name)

    if not resumes[0] or not resumes[1]:
        return jsonify({"error": "Please provide both resumes"}), 400
    if not job_desc:
        return jsonify({"error": "Please provide a job description"}), 400

    try:
        results = []
        for rt in resumes:
            ms, _, _ = calculate_match_score(rt, job_desc)
            ats      = calculate_ats_score(rt, job_desc)
            skills   = sorted(extract_skills_from_text(rt))[:20]
            exp      = detect_experience_level(rt)
            results.append({
                "match_score":  ms,
                "ats_score":    ats["overall"],
                "level":        classify_match_level(ms),
                "skills":       skills,
                "ats_breakdown":ats,
                "experience":   exp,
            })

        winner = "A" if results[0]["match_score"] >= results[1]["match_score"] else "B"
        resp = {
            "resume_a": results[0], "resume_b": results[1],
            "winner": winner,
            "resume_a_name": resume_names[0],
            "resume_b_name": resume_names[1],
        }

        user = current_user()
        if user:
            c = Comparison(
                user_id=user.id,
                resume_a_name=resume_names[0], resume_b_name=resume_names[1],
                job_description_snippet=job_desc[:200],
                resume_a_score=results[0]["match_score"], resume_b_score=results[1]["match_score"],
                resume_a_ats=results[0]["ats_score"],   resume_b_ats=results[1]["ats_score"],
                winner=winner, comparison_data=resp,
            )
            db.session.add(c)
            db.session.commit()
            audit("compare", user.id)

        return jsonify(resp), 200
    except Exception as e:
        logger.exception(f"Compare error: {e}")
        return jsonify({"error": "Comparison failed"}), 500


# ─── HISTORY ──────────────────────────────────────────────
@app.route("/api/history")
@api_login_required
def get_history():
    user = current_user()
    page = request.args.get("page", 1, type=int)
    pg   = Analysis.query.filter_by(user_id=user.id) \
               .order_by(Analysis.created_at.desc()) \
               .paginate(page=page, per_page=20, error_out=False)
    return jsonify({
        "analyses":     [a.to_dict() for a in pg.items],
        "total":        pg.total,
        "pages":        pg.pages,
        "current_page": page,
    }), 200


@app.route("/api/history/<int:aid>")
@api_login_required
def get_analysis(aid):
    user = current_user()
    a    = Analysis.query.filter_by(id=aid, user_id=user.id).first()
    if not a:
        return jsonify({"error": "Not found"}), 404
    return jsonify(a.to_dict(include_details=True)), 200


@app.route("/api/history/<int:aid>", methods=["DELETE"])
@api_login_required
def delete_analysis(aid):
    user = current_user()
    a    = Analysis.query.filter_by(id=aid, user_id=user.id).first()
    if not a:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(a)
    db.session.commit()
    audit("delete_analysis", user.id, {"id": aid})
    return jsonify({"message": "Deleted"}), 200


@app.route("/api/export/history")
@api_login_required
def export_history():
    user     = current_user()
    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).all()
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Date", "Job Title", "Company", "Match %", "ATS %", "Level", "Skills Found", "Time (ms)"])
    for a in analyses:
        w.writerow([
            a.created_at.strftime("%Y-%m-%d %H:%M"),
            a.job_title or "", a.job_company or "",
            a.match_score, a.ats_score, a.level,
            len(a.resume_skills or []),
            a.processing_time_ms or 0,
        ])
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv", as_attachment=True,
        download_name=f"resumeiq_history_{datetime.now().strftime('%Y%m%d')}.csv"
    )


# ─── JOB TRACKER (full CRUD, DB-backed) ───────────────────
@app.route("/api/jobs", methods=["GET"])
@api_login_required
def get_jobs():
    user   = current_user()
    status = request.args.get("status")
    q      = JobApplication.query.filter_by(user_id=user.id)
    if status:
        q = q.filter_by(status=status)
    jobs = q.order_by(JobApplication.created_at.desc()).all()
    # Status summary
    all_jobs = JobApplication.query.filter_by(user_id=user.id).all()
    summary = {}
    for j in all_jobs:
        summary[j.status] = summary.get(j.status, 0) + 1
    return jsonify({"jobs": [j.to_dict() for j in jobs], "summary": summary}), 200


@app.route("/api/jobs", methods=["POST"])
@api_login_required
def create_job():
    user = current_user()
    data = request.get_json() or {}
    title   = data.get("title", "").strip()
    company = data.get("company", "").strip()
    if not title or not company:
        return jsonify({"error": "Title and company are required"}), 400

    applied_date = None
    deadline     = None
    try:
        if data.get("applied_date"):
            applied_date = date.fromisoformat(data["applied_date"])
        if data.get("deadline"):
            deadline = date.fromisoformat(data["deadline"])
    except ValueError:
        pass

    j = JobApplication(
        user_id=user.id, title=title, company=company,
        location=data.get("location", "").strip() or None,
        url=data.get("url", "").strip() or None,
        salary=data.get("salary", "").strip() or None,
        status=data.get("status", "saved"),
        notes=data.get("notes", "").strip() or None,
        applied_date=applied_date, deadline=deadline,
    )
    db.session.add(j)
    db.session.commit()
    audit("job_created", user.id, {"title": title, "company": company})
    return jsonify(j.to_dict()), 201


@app.route("/api/jobs/<int:jid>", methods=["PUT"])
@api_login_required
def update_job(jid):
    user = current_user()
    j    = JobApplication.query.filter_by(id=jid, user_id=user.id).first()
    if not j:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json() or {}
    if "title"    in data: j.title    = data["title"].strip()
    if "company"  in data: j.company  = data["company"].strip()
    if "location" in data: j.location = data["location"].strip() or None
    if "url"      in data: j.url      = data["url"].strip() or None
    if "salary"   in data: j.salary   = data["salary"].strip() or None
    if "status"   in data: j.status   = data["status"]
    if "notes"    in data: j.notes    = data["notes"].strip() or None
    try:
        if data.get("applied_date"):
            j.applied_date = date.fromisoformat(data["applied_date"])
        if data.get("deadline"):
            j.deadline = date.fromisoformat(data["deadline"])
    except ValueError:
        pass
    db.session.commit()
    return jsonify(j.to_dict()), 200


@app.route("/api/jobs/<int:jid>", methods=["DELETE"])
@api_login_required
def delete_job(jid):
    user = current_user()
    j    = JobApplication.query.filter_by(id=jid, user_id=user.id).first()
    if not j:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(j)
    db.session.commit()
    return jsonify({"message": "Deleted"}), 200


# ─── AI FEATURES ──────────────────────────────────────────
@app.route("/api/ai/improve", methods=["POST"])
@api_login_required
@limiter.limit("10 per hour")
def ai_improve():
    data        = request.get_json() or {}
    resume_text = data.get("resume_text", "")
    focus_area  = data.get("focus_area", "overall")
    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400
    if not ai_analyzer.is_available():
        return jsonify({"error": "AI service unavailable — add ANTHROPIC_API_KEY to .env"}), 503
    suggestions = ai_analyzer.suggest_improvements(resume_text, focus_area)
    audit("ai_improve", current_user().id, {"focus": focus_area})
    return jsonify({"suggestions": suggestions}), 200


@app.route("/api/ai/cover-letter", methods=["POST"])
@api_login_required
@limiter.limit("5 per hour")
def ai_cover_letter():
    data        = request.get_json() or {}
    resume_text = data.get("resume_text", "")
    job_desc    = data.get("job_description", "")
    tone        = data.get("tone", "professional")
    if not resume_text or not job_desc:
        return jsonify({"error": "Resume and job description required"}), 400
    if not ai_analyzer.is_available():
        return jsonify({"error": "AI service unavailable — add ANTHROPIC_API_KEY to .env"}), 503
    letter = ai_analyzer.generate_cover_letter(resume_text, job_desc, tone)
    audit("cover_letter", current_user().id)
    return jsonify({"cover_letter": letter}), 200


# ─── REPORT ───────────────────────────────────────────────
@app.route("/api/download-report", methods=["POST"])
@limiter.limit("10 per hour")
def download_report():
    from report_generator import generate_pdf_report
    data = request.get_json() or {}
    try:
        pdf_path = generate_pdf_report(data)
        user = current_user()
        if user:
            audit("download_report", user.id)
        return send_file(
            pdf_path, mimetype="application/pdf",
            as_attachment=True, download_name="ResumeIQ_Report.pdf"
        )
    except Exception as e:
        logger.exception(f"Report error: {e}")
        return jsonify({"error": "Failed to generate report"}), 500


# ─── HEALTH ───────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status":       "healthy",
        "version":      app.config["APP_VERSION"],
        "ai_available": ai_analyzer.is_available(),
    }), 200


# ─── ERRORS ───────────────────────────────────────────────
@app.errorhandler(404)
def e404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render_template("404.html"), 404

@app.errorhandler(500)
def e500(e):
    db.session.rollback()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("500.html"), 500

@app.errorhandler(413)
def e413(e):
    return jsonify({"error": "File too large (max 10 MB)"}), 413


# ─── INIT ─────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logger.info("Database ready")
    app.run(debug=app.config.get("DEBUG", False), host="0.0.0.0", port=5000)
