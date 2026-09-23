# ResumeScreener

ResumeScreener is a Flask-based resume analysis and applicant screening application built to help recruiters, hiring teams, and job seekers evaluate resumes against job descriptions. It combines ATS-style scoring, skill matching, readability checks, comparison tools, AI-powered insights, and application tracking in a single dashboard.

## Overview

The app can:

- Analyze a resume against a job description
- Calculate match score and ATS adequacy
- Detect missing skills and keyword gaps
- Score resume readability and structure quality
- Estimate experience level and salary range
- Compare two resumes side-by-side
- Store analysis history for logged-in users
- Track job applications through a Kanban-style workflow
- Optionally use Anthropic AI for summary and improvement suggestions

## Features

### Core screening
- Match score evaluation based on resume-to-job relevance
- ATS score breakdown across key hiring signals
- Skill extraction and gap detection
- Missing keyword highlighting
- Readability and content quality checks

### Advanced insights
- Experience level detection
- Salary estimate suggestions
- Quantification analysis of achievements
- Career gap detection warnings
- Resume comparison between two candidates
- PDF report generation for analysis results

### AI-powered features
- Resume summary against the target role
- Strengths and weaknesses analysis
- Cover letter generation
- Improvement suggestions by focus area
- Optional integration via `ANTHROPIC_API_KEY`

### User and job tracking
- Signup, login, logout, and session-based auth
- Profile management and password updates
- Analysis history with per-user records
- Full job application tracker with CRUD support
- Status board from Saved to Accepted

## Tech stack

- Python 3
- Flask
- SQLAlchemy
- Flask-Login
- Flask-Bcrypt
- Flask-Limiter
- Flask-CORS
- ReportLab
- PyMuPDF / document parsing support
- SQLite (default development database)
- Optional Anthropic API integration

## Project structure

```text
ResumeScreener/
├── app.py                 # Flask routes and application logic
├── config.py              # App configuration
├── extensions.py          # Flask extension setup
├── models.py              # SQLAlchemy models for users, analyses, jobs, and logs
├── nlp_utils.py           # ATS, scoring, and text-processing utilities
├── ai_service.py          # Optional Claude AI integration
├── document_utils.py      # Resume file parsing and validation
├── report_generator.py    # PDF report generation
├── requirements.txt       # Python dependencies
├── .env                   # Local environment variables
├── static/
│   ├── app.js             # Frontend logic
│   └── style.css          # Styling and layout
├── templates/
│   ├── 404.html
│   ├── 500.html
│   ├── compare.html
│   ├── history.html
│   ├── index.html
│   ├── jobs.html
│   ├── login.html
│   ├── profile.html
│   ├── signup.html
│   └── ...
├── README.md
└── resumeiq.db            # Local SQLite DB (created on first run)
```

## Quick start

```bash
# 1. Clone the repository
cd ResumeScreener

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file with required variables
cat > .env <<'EOF'
SECRET_KEY=your-secret-key
FLASK_ENV=development
DATABASE_URL=sqlite:///resumeiq.db
ANTHROPIC_API_KEY=your_api_key_here
EOF

# 5. Run the app
python app.py
```

Then open:

```text
http://localhost:5000
```

The SQLite database is created automatically when the app starts for the first time.

## Environment variables

| Variable | Required | Description |
|---|---:|---|
| `SECRET_KEY` | Yes | Secret key used by Flask sessions and security features |
| `DATABASE_URL` | No | Database connection string; defaults to SQLite for local dev |
| `ANTHROPIC_API_KEY` | No | Enables AI-based resume summary and improvement features |
| `FLASK_ENV` | No | Set to `development` or `production` |

## Notes

- AI-powered features are optional. If `ANTHROPIC_API_KEY` is not set, the app still works for core resume analysis features.
- The project is designed for local development and demo use, with SQLite as the default database backend.
- For production deployment, you should configure a stronger secret key, use a managed database, and secure your environment variables.

## License

This project is currently provided as-is for personal or educational use. If you plan to use it in production or distribute it publicly, confirm the appropriate licensing requirements before deployment.

## Contribution

Contributions, improvements, and bug fixes are welcome. If you want to extend the project, consider improving:

- job matching algorithm quality
- document parsing support
- UI/UX flow for recruiter workflows
- AI summarization prompts and accuracy
- deployment and production configuration
