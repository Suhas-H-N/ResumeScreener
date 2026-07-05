# ResumeIQ Pro v3.1 — AI-Powered Resume Screening Platform

A complete Flask web application for resume analysis, ATS scoring, skill gap detection, and job application tracking — with optional Claude AI integration.

---

## Features

### Core Analysis
- **Match Score** — TF-IDF cosine similarity between resume and job description
- **ATS Score** — 5-dimension breakdown: keyword density, skill match, length, sections, action verbs
- **Skill Gap Analysis** — 200+ tech + soft skills detected
- **Keyword Analysis** — matched vs. missing keywords highlighted
- **Readability Metrics** — Flesch-Kincaid grade + reading ease

### Enhanced Insights
- **Experience Level Detection** — Junior / Mid / Senior / Executive from resume text
- **Salary Estimation** — US market ranges based on skills + experience level
- **Quantification Scorer** — detects %, $, multipliers, team sizes in achievements
- **Career Gap Detector** — flags employment gaps with actionable advice
- **Resume Comparison** — side-by-side analysis of two candidates
- **PDF Report Download** — professional branded PDF of analysis results

### AI Features (requires `ANTHROPIC_API_KEY`)
- **AI Summary** — Claude analyses the resume against the role
- **Strengths & Weaknesses** — AI-identified pros/cons
- **Cover Letter Generator** — tailored cover letters in 4 tones
- **Resume Improver** — AI suggestions per section

### Job Application Tracker
- Full CRUD — add, edit, delete, status update
- **7 status stages**: Saved → Applied → Phone Screen → Interview → Offer → Rejected → Accepted
- **Kanban board** view with status columns
- All data persisted in database (not localStorage)

### User System
- Sign up / login / logout
- Profile management (name, company, role)
- Password change
- Account deletion
- Session-based auth

---

## Quick Start

```bash
# 1. Clone / extract project
cd ResumeScreener

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your Anthropic API key to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 5. Run
python app.py
```

Open http://localhost:5000

The SQLite database is created automatically on first run.

---

## File Structure

```
ResumeScreener/
├── app.py              # All routes — auth, analyze, compare, history, jobs, AI, report
├── models.py           # User, Analysis, Comparison, JobApplication, AuditLog
├── config.py           # Dev/prod config (SQLite dev, no Redis needed)
├── extensions.py       # Flask extensions (login, bcrypt, cors, limiter)
├── nlp_utils.py        # TF-IDF scoring, ATS, readability, salary, gap detection
├── ai_service.py       # Claude API wrapper (claude-sonnet-4-6)
├── document_utils.py   # PDF/DOCX/TXT extraction
├── report_generator.py # ReportLab PDF generation
├── requirements.txt    # Minimal, no Redis/Celery/Sentry
├── .env                # Environment variables
├── static/
│   ├── style.css       # Full design system (navy/gold theme + dark mode)
│   └── app.js          # Complete SPA frontend (no framework)
└── templates/
    ├── index.html      # Analyzer with score rings, tabs, AI panel
    ├── compare.html    # Side-by-side resume comparison
    ├── history.html    # Paginated history with detail modal
    ├── profile.html    # Profile + stats + password change
    ├── jobs.html       # Kanban job tracker (DB-backed)
    ├── login.html
    ├── signup.html
    ├── 404.html
    └── 500.html
```

---

## What Was Fixed vs Original

| Problem | Fix |
|---|---|
| `extensions.py` hard-required Redis — crashed on startup | Replaced with `storage_uri="memory://"` — zero config |
| `config.py` referenced Redis, Celery, Sentry everywhere | Stripped to minimal dev config |
| `ai_service.py` used invalid model `claude-sonnet-4-20250514` | Fixed to `claude-sonnet-4-6` |
| `/api/jobs` routes **didn't exist** — entire jobs page was broken | Full CRUD implemented (GET/POST/PUT/DELETE) |
| Job tracker used localStorage — wiped on browser clear | DB-persisted `JobApplication` model |
| `requirements.txt` missing flask-login, flask-bcrypt, etc. | Complete accurate requirements |
| `models.py` missing `JobApplication` model | Added with all fields + `to_dict()` |
| Kanban had no real data source | Renders live from `/api/jobs` |
| Job status summary hardcoded | Live counts from DB per status |
| Score display was plain text number | Animated SVG score rings |
| ATS breakdown had no visualization | Animated progress bars per dimension |
| AI section missing when no API key | Graceful fallback with setup instructions |
| Password strength only on signup | Also on profile security tab |
| History detail showed nothing | Full modal with skills, recs, AI summary |
| Cover letter modal wired to nothing | Full AI generation flow |
| Theme toggle added but not wired | localStorage-persisted dark mode |
| `flask-migrate` import crash (unused) | Removed entirely |
| `flask-jwt-extended` + `flask-mail` imported but unused | Removed from extensions |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | dev key | Change in production |
| `DATABASE_URL` | sqlite:///resumeiq.db | Any SQLAlchemy URL |
| `ANTHROPIC_API_KEY` | _(empty)_ | Optional — enables AI features |
| `FLASK_ENV` | development | Set `production` to disable debug |
