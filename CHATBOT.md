# AI Career Coach

ResumeScreener includes a conversational AI career coach at `/static/chatbot.html`. Sign in first, then open that URL to ask follow-up questions about resumes, job applications, interviews, and career development. The coach uses the existing Anthropic integration and `ANTHROPIC_API_KEY` configuration; it does not retain chat history after the browser session.

The coach is subject to the same authentication and rate limit as the AI resume-improvement endpoint. Avoid entering sensitive personal information, and verify important advice.
