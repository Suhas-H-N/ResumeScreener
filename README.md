# 🚀 ResumeIQ Pro

### AI-Powered Resume Analysis & ATS Optimization Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

---

## 📌 Overview

ResumeIQ Pro is a modern AI-powered resume screening and optimization platform designed to help job seekers improve their resumes for Applicant Tracking Systems (ATS) and increase their chances of landing interviews.

The platform combines Natural Language Processing (NLP), Machine Learning, and Claude AI to deliver intelligent resume evaluation, ATS scoring, skill-gap analysis, keyword optimization, and personalized recommendations.

---

## ✨ Features

### 🔍 Resume Analysis

- Parse resumes from PDF, DOCX, and TXT formats
- Extract structured resume information
- Analyze resumes against job descriptions
- Generate ATS compatibility scores
- Evaluate readability and content quality

### 🤖 AI-Powered Intelligence

- Claude AI integration
- Personalized improvement suggestions
- Resume strengths and weaknesses analysis
- Keyword optimization recommendations
- AI-generated cover letters

### 📊 Analytics & Reporting

- ATS score breakdown
- Skill gap analysis
- Missing keyword detection
- Resume comparison system
- PDF report generation
- CSV export support

### 🔐 Authentication & Security

- Secure user registration and login
- Password hashing with Bcrypt
- Session management
- Password recovery workflow
- API rate limiting
- Audit logging

### 🎨 User Experience

- Responsive design
- Drag-and-drop file upload
- Real-time feedback
- Interactive dashboard
- Modern UI/UX

---

## 🏗️ Technology Stack

| Category | Technologies |
|-----------|-------------|
| Backend | Python, Flask |
| AI | Claude AI |
| NLP | Scikit-learn, TF-IDF |
| Database | PostgreSQL, SQLite |
| Authentication | Flask-Login, Bcrypt |
| Reporting | ReportLab |
| Containerization | Docker |
| Caching | Redis |
| Frontend | HTML, CSS, JavaScript |

---

## 🎯 Core Functionalities

### ATS Score Calculation

ResumeIQ Pro evaluates resumes based on:

- Keyword Relevance
- Skill Coverage
- Resume Structure
- Readability
- ATS Compatibility
- Experience Alignment

### Resume Matching

The platform uses:

- TF-IDF Vectorization
- Cosine Similarity

to calculate the match percentage between a resume and a job description.

### Skill Gap Analysis

Automatically identifies:

- Matching Skills
- Missing Skills
- Recommended Skills
- Industry-Relevant Keywords

---

## 📸 Workflow

```text
Upload Resume
      │
      ▼
Extract Content
      │
      ▼
Analyze Job Description
      │
      ▼
Calculate ATS Score
      │
      ▼
Generate AI Insights
      │
      ▼
Provide Recommendations
      │
      ▼
Export Report
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (Optional)
- Redis 7+ (Optional)
- Docker (Recommended)

---

### Installation

```bash
# Clone Repository
git clone https://github.com/yourusername/resumeiq-pro.git

# Navigate into project
cd resumeiq-pro

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Run database migrations
flask db upgrade

# Start application
python app.py
```

Application will be available at:

```bash
http://localhost:5000
```

---

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

Check logs:

```bash
docker-compose logs -f
```

Stop services:

```bash
docker-compose down
```

---

## 📁 Project Structure

```text
ResumeIQ-Pro/
│
├── app.py
├── config.py
├── models.py
├── extensions.py
├── ai_service.py
├── nlp_utils.py
├── document_utils.py
├── report_generator.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── static/
│   ├── style.css
│   └── app.js
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── history.html
│   └── compare.html
│
├── migrations/
├── tests/
└── docs/
```

---

## 🔗 API Endpoints

### Authentication

```http
POST   /api/signup
POST   /api/login
POST   /api/logout
GET    /api/me
POST   /api/forgot-password
POST   /api/reset-password
```

### Resume Analysis

```http
POST   /api/analyze
POST   /api/compare
GET    /api/history
GET    /api/history/<id>
DELETE /api/history/<id>
GET    /api/export/history
```

### AI Features

```http
POST   /api/ai/improve
POST   /api/ai/cover-letter
```

### Utilities

```http
POST   /api/download-report
GET    /api/stats
GET    /health
```

---

## 🧪 Testing

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=. --cov-report=html
```

Run integration tests:

```bash
pytest tests/integration/
```

---

## 🔒 Security Features

- Password Hashing (Bcrypt)
- CSRF Protection
- SQL Injection Prevention
- Secure Session Management
- Rate Limiting
- Input Validation
- File Upload Restrictions
- HTTPS Support

---

## 📈 Performance Optimization

- Redis Caching
- Optimized Database Queries
- Connection Pooling
- Lazy Loading
- Static Asset Optimization
- Dockerized Deployment

---

## 🤝 Contributing

Contributions are welcome.

### Steps

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

## 🌟 Future Enhancements

### Version 3.1

- Email Verification
- Analytics Dashboard
- Resume Templates
- LinkedIn Integration
- Chrome Extension

### Version 3.2

- Team Collaboration
- Public API
- Mobile Applications
- Multi-language Support

### Version 4.0

- Enterprise Dashboard
- Job Recommendation Engine
- Salary Insights
- Advanced AI Models

---

## 📊 Why ResumeIQ Pro?

✅ ATS Optimization

✅ AI-Powered Resume Review

✅ Skill Gap Detection

✅ Resume-to-Job Matching

✅ Cover Letter Generation

✅ Professional Reports

✅ Secure Authentication

✅ Docker Ready

---

## 👨‍💻 Author

### Suhas H N

Software Developer | AI Enthusiast | Full Stack Developer

**GitHub:** https://github.com/Suhas-H-N

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, consider giving it a star on GitHub.