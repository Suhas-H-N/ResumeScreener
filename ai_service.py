"""AI Service — uses claude-sonnet-4-6 (current model string)"""
import os, json, re, logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client  = None
        if self.api_key and self.api_key != "sk-ant-your-key-here":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Anthropic init error: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def analyze_resume(self, resume_text: str, job_description: str, basic_analysis: Dict) -> Dict:
        if not self.is_available():
            return self._empty_result()
        try:
            prompt = f"""You are a senior recruiter and resume expert. Analyze this resume against the job description.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

BASIC ANALYSIS:
- Match Score: {basic_analysis.get('match_score', 0)}%
- ATS Score: {basic_analysis.get('ats_scores', {}).get('overall', 0)}%
- Missing Skills: {', '.join(basic_analysis.get('missing_skills', [])[:8])}

Respond ONLY with valid JSON, no markdown, no explanation:
{{
  "summary": "2-3 sentence executive assessment of fit for this role",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["gap 1", "gap 2", "gap 3"],
  "recommendations": ["specific action 1", "specific action 2", "specific action 3", "specific action 4"],
  "interview_tips": ["prep tip 1", "prep tip 2", "prep tip 3"],
  "improvement_priority": "single most important thing to improve"
}}"""

            msg = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._parse(msg.content[0].text)
        except Exception as e:
            logger.error(f"AI analyze error: {e}")
            return self._empty_result()

    def suggest_improvements(self, resume_text: str, focus_area: str) -> str:
        if not self.is_available():
            return "AI service unavailable. Set ANTHROPIC_API_KEY to enable."
        try:
            prompt = f"""As a resume expert, give 4 specific improvements for the '{focus_area}' section of this resume.

RESUME:
{resume_text[:2000]}

Be concrete, actionable, and give brief examples. Under 250 words."""
            msg = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text
        except Exception as e:
            logger.error(f"AI improve error: {e}")
            return "Unable to generate suggestions at this time."

    def generate_cover_letter(self, resume_text: str, job_description: str, tone: str = "professional") -> str:
        if not self.is_available():
            return "AI service unavailable. Set ANTHROPIC_API_KEY to enable."
        try:
            prompt = f"""Write a compelling {tone} cover letter (3-4 paragraphs, ~250 words).

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

Open with enthusiasm, highlight 2-3 specific achievements, explain fit, close with call to action.
Avoid generic phrases. Make it specific to this role."""
            msg = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text
        except Exception as e:
            logger.error(f"Cover letter error: {e}")
            return "Unable to generate cover letter at this time."

    def _parse(self, text: str) -> Dict:
        # Strip markdown fences if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        try:
            data = json.loads(m.group(0) if m else text)
            return {
                "ai_summary":         data.get("summary", ""),
                "ai_recommendations": data.get("recommendations", [])[:5],
                "strengths":          data.get("strengths", [])[:5],
                "weaknesses":         data.get("weaknesses", [])[:5],
                "interview_tips":     data.get("interview_tips", [])[:3],
                "improvement_priority": data.get("improvement_priority", ""),
            }
        except Exception as e:
            logger.error(f"AI parse error: {e}")
            return self._empty_result()

    def _empty_result(self):
        return {
            "ai_summary": None, "ai_recommendations": [],
            "strengths": [], "weaknesses": [], "interview_tips": [],
            "improvement_priority": ""
        }


_analyzer = None
def get_ai_analyzer() -> AIAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = AIAnalyzer()
    return _analyzer
