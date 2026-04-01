🤖 AI Resume Screener is a web application that analyzes a candidate's resume and compares it with a job description to determine how well the resume matches the job requirements.

The system extracts text from resumes (PDF, DOCX, TXT), processes the content using Natural Language Processing (NLP), and calculates a similarity score between the resume and job description.

🚀 Features:
• Upload resume files (PDF, DOCX, TXT)
• Paste resume text manually
• Enter job description for comparison
• AI-based similarity score using TF-IDF and Cosine Similarity
• Keyword matching and missing keyword detection
• Resume statistics such as word count
• Clean and responsive web interface

🛠 Tech Stack

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

⚙️ Installation

1️. Clone the repository:
git clone https://github.com/YOUR_USERNAME/ResumeScreener.git

2️. Navigate to the project folder:
cd ResumeScreener

3️. Install dependencies:
pip install -r requirements.txt

4. Run the application:
python app.py

📊 How It Works:
User uploads a resume or pastes resume text
User enters the job description
System extracts keywords from both texts
TF-IDF vectorization converts text to numerical representation
Cosine similarity calculates the match score
Matching and missing keywords are displayed

📈 Future Improvements:
• Advanced NLP using spaCy
• Skill extraction and classification
• Resume ranking system for multiple candidates
• ATS-style scoring system
• Dashboard for recruiters

👨‍💻 Author:
Suhas H N

⭐ Support
If you like this project, consider giving it a star ⭐ on GitHub.