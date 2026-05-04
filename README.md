# 🧠 ResumeIQ Pro v3.0 - AI-Powered Resume Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> **Professional resume screening platform powered by Claude AI and advanced NLP**

ResumeIQ Pro is a fully-featured, production-ready resume analysis platform that helps job seekers optimize their resumes for Applicant Tracking Systems (ATS) and specific job postings. Enhanced with Claude AI for intelligent insights and recommendations.

## ✨ Key Features

### 🎯 Core Analysis
- **Smart Resume Parsing** - Extract text from PDF, DOCX, TXT formats
- **TF-IDF Matching** - Calculate semantic similarity between resume and job description
- **ATS Scoring** - Multi-dimensional scoring across 5+ metrics
- **Skill Extraction** - Identify 200+ technical and soft skills
- **Keyword Analysis** - Find matching and missing keywords
- **Readability Metrics** - Flesch-Kincaid scoring

### 🤖 AI-Powered Features
- **Claude AI Integration** - Deep resume analysis with actionable insights
- **Personalized Recommendations** - AI-generated improvement suggestions
- **Strengths & Weaknesses** - Intelligent assessment of resume quality
- **Cover Letter Generation** - Auto-generate tailored cover letters
- **Interview Preparation** - AI-powered interview tips

### 📊 Advanced Capabilities
- **Side-by-Side Comparison** - Compare two resumes against a job
- **Analysis History** - Track all past analyses with search/filter
- **PDF Report Generation** - Professional downloadable reports
- **CSV Export** - Export history for external analysis
- **Real-time Processing** - Fast analysis (<2 seconds typical)

### 🔐 Security & Authentication
- **User Accounts** - Secure registration and login
- **Password Hashing** - Bcrypt encryption
- **Session Management** - Secure cookie-based sessions
- **Email Verification** - (Coming soon)
- **Password Reset** - Forgot password flow
- **Rate Limiting** - Protection against abuse
- **Audit Logging** - Track all user actions

### 🎨 Modern UI/UX
- **Responsive Design** - Works on desktop, tablet, mobile
- **Dark Mode Ready** - Easy theme switching
- **Smooth Animations** - Professional transitions
- **Drag & Drop Upload** - Intuitive file upload
- **Real-time Feedback** - Live score updates
- **Loading States** - Clear progress indicators

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (or SQLite for development)
- Redis 7+ (optional, for rate limiting)
- Node.js 18+ (optional, for frontend builds)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resumeiq-pro.git
cd resumeiq-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env and add your settings
# Especially important: ANTHROPIC_API_KEY for AI features

# Initialize database
flask db init
flask db migrate
flask db upgrade

# Create admin user (optional)
flask create-admin

# Run development server
python app.py
```

Visit `http://localhost:5000` in your browser!

### Docker Setup (Recommended for Production)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f web

# Stop containers
docker-compose down
```

## 📋 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Core Settings
FLASK_ENV=development
SECRET_KEY=your-super-secret-key-here
DEBUG=True

# Database
DATABASE_URL=sqlite:///resumeiq.db
# For PostgreSQL: postgresql://user:password@localhost/resumeiq

# Anthropic AI (Required for AI features)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# Email (for password reset)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password

# Redis (for rate limiting - optional)
REDIS_URL=redis://localhost:6379/0

# File Upload
MAX_CONTENT_LENGTH=10485760  # 10MB
UPLOAD_FOLDER=uploads
```

## 🏗️ Architecture

```
ResumeIQ-Pro/
├── app.py                 # Main Flask application
├── config.py              # Configuration management
├── models.py              # Database models
├── extensions.py          # Flask extensions
├── nlp_utils.py           # NLP processing logic
├── ai_service.py          # Claude AI integration
├── document_utils.py      # File processing
├── report_generator.py    # PDF report generation
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Multi-container setup
├── static/
│   ├── style.css          # Modern CSS
│   └── app.js             # Frontend JavaScript
├── templates/
│   ├── index.html         # Main analysis page
│   ├── login.html         # Login page
│   ├── signup.html        # Registration page
│   ├── history.html       # Analysis history
│   └── compare.html       # Resume comparison
├── migrations/            # Database migrations
├── tests/                 # Unit and integration tests
└── docs/                  # Additional documentation
```

## 🔧 API Endpoints

### Authentication
```
POST   /api/signup          - Create new account
POST   /api/login           - User login
POST   /api/logout          - User logout
GET    /api/me              - Get current user
POST   /api/forgot-password - Request password reset
POST   /api/reset-password  - Reset password with token
```

### Analysis
```
POST   /api/analyze         - Analyze resume (Rate limited: 20/hour)
POST   /api/compare         - Compare two resumes (Rate limited: 10/hour)
GET    /api/history         - Get analysis history
GET    /api/history/<id>    - Get specific analysis
DELETE /api/history/<id>    - Delete analysis
GET    /api/export/history  - Export history as CSV
```

### AI Features
```
POST   /api/ai/improve         - Get AI improvement suggestions (Rate limited: 5/hour)
POST   /api/ai/cover-letter    - Generate cover letter (Rate limited: 3/hour)
```

### Utilities
```
POST   /api/download-report - Generate PDF report
GET    /api/stats           - Get user statistics
GET    /health              - Health check
```

## 📊 Database Schema

### Users
- **id** - Primary key
- **name** - Full name
- **email** - Unique email address
- **password_hash** - Hashed password
- **email_verified** - Verification status
- **subscription_tier** - free, pro, enterprise
- **created_at** - Registration timestamp

### Analysis
- **id** - Primary key
- **user_id** - Foreign key to users
- **match_score** - Overall match percentage
- **ats_score** - ATS compatibility score
- **ats_breakdown** - JSON with detailed metrics
- **matching_keywords** - JSON array
- **missing_keywords** - JSON array
- **resume_skills** - JSON array
- **missing_skills** - JSON array
- **recommendations** - JSON array
- **ai_summary** - AI-generated summary
- **ai_recommendations** - JSON array
- **created_at** - Analysis timestamp

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_analysis.py

# Run integration tests
pytest tests/integration/
```

## 🚢 Deployment

### Heroku

```bash
heroku create resumeiq-pro
heroku addons:create heroku-postgresql:hobby-dev
heroku addons:create heroku-redis:hobby-dev
heroku config:set ANTHROPIC_API_KEY=your-key
git push heroku main
heroku run flask db upgrade
```

### AWS EC2

```bash
# SSH into instance
ssh -i key.pem ubuntu@your-instance

# Clone and setup
git clone https://github.com/yourusername/resumeiq-pro.git
cd resumeiq-pro
docker-compose up -d

# Configure nginx reverse proxy
sudo nano /etc/nginx/sites-available/resumeiq
```

### DigitalOcean App Platform

1. Connect GitHub repository
2. Set environment variables
3. Configure build command: `pip install -r requirements.txt`
4. Configure run command: `gunicorn app:app`
5. Deploy!

## 🔒 Security Best Practices

- ✅ All passwords hashed with bcrypt
- ✅ CSRF protection enabled
- ✅ SQL injection prevention via ORM
- ✅ Rate limiting on sensitive endpoints
- ✅ Input validation and sanitization
- ✅ Secure session management
- ✅ File upload restrictions
- ✅ HTTPS in production (configure nginx)

## 📈 Performance Optimization

- **Caching**: Redis for frequently accessed data
- **Database**: Indexed queries, connection pooling
- **File Processing**: Async processing with Celery (optional)
- **Frontend**: Minified CSS/JS, lazy loading
- **CDN**: Static assets served via CDN in production

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Claude AI** by Anthropic - For intelligent resume analysis
- **Flask** - Web framework
- **scikit-learn** - Machine learning utilities
- **ReportLab** - PDF generation
- **PostgreSQL** - Database
- **Redis** - Caching and rate limiting

## 📞 Support

- **Email**: support@resumeiq.pro
- **Documentation**: [docs.resumeiq.pro](https://docs.resumeiq.pro)
- **Issues**: [GitHub Issues](https://github.com/yourusername/resumeiq-pro/issues)

## 🗺️ Roadmap

### v3.1 (Next Release)
- [ ] Email verification
- [ ] Advanced analytics dashboard
- [ ] Resume template library
- [ ] LinkedIn integration
- [ ] Chrome extension

### v3.2
- [ ] Team collaboration features
- [ ] API for third-party integrations
- [ ] Mobile apps (iOS/Android)
- [ ] Multi-language support
- [ ] Video resume analysis

### v4.0
- [ ] Enterprise features
- [ ] White-label solution
- [ ] Advanced ML models
- [ ] Salary insights
- [ ] Job matching engine

---

**Built with ❤️ using Claude AI** | [Star on GitHub](https://github.com/yourusername/resumeiq-pro) ⭐
