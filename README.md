AI Resume Screener is a web application that analyzes a candidate's resume and compares it with a job description to determine how well the resume matches the job requirements.

The system extracts text from resumes (PDF, DOCX, TXT), processes the content using Natural Language Processing (NLP), and calculates a similarity score between the resume and job description.

Features:

• Upload resume files (PDF, DOCX, TXT)
• Paste resume text manually
• Enter job description for comparison
• AI-based similarity score using TF-IDF and Cosine Similarity
• Keyword matching and missing keyword detection
• Resume statistics such as word count
• Clean and responsive web interface

Tech Stack

Backend
• Python
• Flask

Machine Learning / NLP
• Scikit-learn (TF-IDF Vectorizer, Cosine Similarity)

Document Processing
• PyPDF2
• python-docx

Frontend
• HTML
• CSS
• JavaScript

📂 Project Structure
ResumeScreener
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates
│   └── index.html
│
├── static
│   └── style.css
│
└── uploads