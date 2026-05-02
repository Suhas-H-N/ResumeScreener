# 🚀 Quick Start Guide - ResumeIQ Pro

## Installation (5 minutes)

### Option 1: Automated Setup (Recommended)

```bash
cd ResumeIQ-Pro
chmod +x setup.sh
./setup.sh
```

The script will:
- ✓ Check Python version
- ✓ Create virtual environment
- ✓ Install all dependencies  
- ✓ Setup database
- ✓ Create admin user (optional)

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Initialize database
export FLASK_APP=app.py
flask db init
flask db migrate
flask db upgrade

# 5. Run application
python app.py
```

## Configuration

### Required Settings

Edit `.env` file:

```bash
# Required for AI features
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Secret key for sessions (generate random string)
SECRET_KEY=your-super-secret-key-change-this
```

### Optional Settings

```bash
# Database (default: SQLite)
DATABASE_URL=sqlite:///resumeiq.db

# Email (for password reset)
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password

# Redis (for rate limiting)
REDIS_URL=redis://localhost:6379/0
```

## First Run

1. Start the application:
   ```bash
   python app.py
   ```

2. Visit: `http://localhost:5000`

3. Create account or use as guest

4. Upload resume + job description

5. Get instant analysis!

## Key Features Overview

### 1. Resume Analysis
- Upload PDF/DOCX or paste text
- Get match score (0-100%)
- ATS compatibility score
- Missing keywords and skills
- AI-powered recommendations

### 2. AI Insights (Requires ANTHROPIC_API_KEY)
- Executive summary
- Strengths & weaknesses
- Interview preparation tips
- Cover letter generation

### 3. Resume Comparison
- Compare 2 resumes side-by-side
- See which performs better
- Identify unique strengths

### 4. History & Export
- Track all analyses
- Export to CSV
- Download PDF reports

## Common Issues

### "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### "AI features not working"
- Check ANTHROPIC_API_KEY in .env
- Verify key is valid
- Check API quota

### "Database error"
```bash
# Reset database
rm resumeiq.db
flask db upgrade
```

### "Port 5000 already in use"
```bash
# Use different port
flask run --port 5001
```

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## Production Deployment

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for:
- Heroku deployment
- AWS deployment
- DigitalOcean deployment
- SSL configuration
- Environment setup

## Getting Help

- 📖 Full documentation: README.md
- 🐛 Report issues: GitHub Issues
- 💬 Questions: Create a discussion

## Next Steps

1. ✅ Explore all features
2. ✅ Read full README.md
3. ✅ Check API documentation
4. ✅ Star the repo!

**Happy analyzing! 🎯**
