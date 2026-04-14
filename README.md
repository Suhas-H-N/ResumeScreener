# 🤖 ResumeIQ v2.0 — AI Resume Screener

An AI-powered resume screening and ATS analysis web application built with Python, Flask, and scikit-learn.

---

## ✅ Features

| Feature | v1 | v2 |
|---------|----|----|
| Resume upload (PDF, DOCX, TXT) | ✅ | ✅ |
| Paste resume / job description | ✅ | ✅ |
| TF-IDF cosine similarity match score | ✅ | ✅ |
| Keyword matching & missing keywords | ✅ | ✅ |
| PDF report download | ✅ | ✅ (redesigned) |
| **Multi-dimension ATS scoring** | ❌ | ✅ |
| **Animated ATS progress bars** | ❌ | ✅ |
| **Skill taxonomy (40+ tech + soft skills)** | ❌ | ✅ |
| **Resume statistics (words, sentences, readability)** | ❌ | ✅ |
| **Animated score ring** | ❌ | ✅ |
| **Smart recommendations engine** | basic | ✅ upgraded |
| **Side-by-side resume comparison** | ❌ | ✅ |
| **Analysis history page** | ❌ | ✅ |
| **Server-side session auth** | localStorage only | ✅ Flask sessions |
| **Toast notifications** (no more alert() popups) | ❌ | ✅ |
| **Password strength meter** | ❌ | ✅ |
| **Drag & Drop upload** | basic | ✅ |
| Dark / Light theme toggle | ✅ | ✅ |
| Responsive design | ✅ | ✅ |

---

## 📁 Project Structure

```
ResumeIQ/
├── app.py               ← Flask backend (all routes, NLP, scoring)
├── requirements.txt
├── users.json           ← User accounts (auto-created)
├── history.json         ← Analysis history (auto-created)
├── uploads/             ← Temp file uploads (auto-created)
│
├── templates/
│   ├── index.html       ← Main analyzer page
│   ├── login.html       ← Sign in
│   ├── signup.html      ← Create account
│   ├── compare.html     ← Side-by-side comparison (NEW)
│   └── history.html     ← Analysis history (NEW)
│
└── static/
    └── style.css        ← Full design system
```

---

## ⚙️ How to Run

### 1. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /signup | Create account |
| POST | /login | Sign in (server session) |
| POST | /logout | Sign out |
| GET  | /me | Check current session |
| POST | /analyze | Analyze resume vs JD |
| POST | /compare | Compare two resumes |
| GET  | /history | Get user's history (auth required) |
| DELETE | /history/:id | Delete one history entry |
| POST | /download-report | Generate PDF report |

---

## 🧠 How ATS Scoring Works

The ATS score is calculated across 5 weighted dimensions:

| Dimension | Weight | What it checks |
|-----------|--------|----------------|
| Keyword Density | 30% | How many JD keywords appear in resume |
| Skill Match | 30% | Tech/soft skills from JD found in resume |
| Section Structure | 15% | Presence of Experience, Education, Skills, etc. |
| Action Verbs | 10% | Impact words: Led, Built, Optimized, Delivered… |
| Resume Length | 15% | 300–1200 words = optimal for ATS |

---

## 👤 Author

**Suhas H N** · [github.com/Suhas-H-N](https://github.com/Suhas-H-N)
