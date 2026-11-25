# Bulk resume screening

The bulk screening page lets you evaluate multiple candidate resumes against one shared job description and ranks successful analyses by match score (then ATS score).

## Use it

1. Start the Flask app and open `http://localhost:5000/static/bulk-screen.html`.
2. Enter a job title (optional) and paste a job description of at least 50 characters.
3. Choose up to 20 PDF, DOCX, DOC, or TXT resumes (10 MB maximum per file), then select **Screen resumes**.
4. Review the ranked results and optionally download them as CSV.

Each resume is sent through the existing `/api/analyze` endpoint individually. When signed in, analyses are also stored in the account's history. Screening continues if an individual file fails; the results table reports the failure.
