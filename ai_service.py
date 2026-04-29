"""
AI Service using Anthropic Claude API
Provides enhanced resume analysis capabilities
"""
import os
from typing import Dict, List, Optional
from anthropic import Anthropic
import logging

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """Enhanced resume analysis using Claude AI"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                self.client = Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return self.client is not None
    
    def analyze_resume(
        self,
        resume_text: str,
        job_description: str,
        basic_analysis: Dict
    ) -> Dict[str, any]:
        """
        Enhance basic analysis with AI insights
        
        Args:
            resume_text: Full resume content
            job_description: Job posting description
            basic_analysis: Results from traditional NLP analysis
            
        Returns:
            Dictionary with AI-enhanced insights
        """
        if not self.is_available():
            return {
                "ai_summary": None,
                "ai_recommendations": [],
                "strengths": [],
                "weaknesses": [],
                "interview_tips": []
            }
        
        try:
            prompt = self._build_analysis_prompt(resume_text, job_description, basic_analysis)
            
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            return self._parse_ai_response(response_text)
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {
                "ai_summary": None,
                "ai_recommendations": [],
                "strengths": [],
                "weaknesses": [],
                "interview_tips": []
            }
    
    def _build_analysis_prompt(
        self,
        resume_text: str,
        job_description: str,
        basic_analysis: Dict
    ) -> str:
        """Build the prompt for Claude"""
        return f"""You are an expert resume reviewer and career coach. Analyze this resume against the job description and provide detailed insights.

**Resume:**
{resume_text[:3000]}

**Job Description:**
{job_description[:2000]}

**Basic Analysis Results:**
- Match Score: {basic_analysis.get('match_score', 0)}%
- ATS Score: {basic_analysis.get('ats_scores', {}).get('overall', 0)}%
- Missing Keywords: {', '.join(basic_analysis.get('missing_keywords', [])[:10])}
- Missing Skills: {', '.join(basic_analysis.get('missing_skills', [])[:10])}

Provide your analysis in the following JSON format:

{{
  "summary": "A 2-3 sentence executive summary of how well this resume matches the role",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
  "recommendations": [
    "specific actionable recommendation 1",
    "specific actionable recommendation 2",
    "specific actionable recommendation 3"
  ],
  "interview_tips": [
    "tip 1 for potential interview",
    "tip 2 for potential interview"
  ],
  "improvement_priority": "The single most important thing to improve on this resume"
}}

Focus on:
1. How well the experience aligns with job requirements
2. Quality of achievement descriptions (quantified results)
3. Skill gaps that matter most
4. Resume structure and ATS optimization
5. How to position the candidate for this specific role

Return ONLY the JSON, no other text."""
    
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse Claude's JSON response"""
        import json
        import re
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            json_str = json_match.group(0) if json_match else response
        
        try:
            data = json.loads(json_str)
            return {
                "ai_summary": data.get("summary", ""),
                "ai_recommendations": data.get("recommendations", [])[:5],
                "strengths": data.get("strengths", [])[:5],
                "weaknesses": data.get("weaknesses", [])[:5],
                "interview_tips": data.get("interview_tips", [])[:3],
                "improvement_priority": data.get("improvement_priority", "")
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return {
                "ai_summary": response[:500],  # Return raw text as fallback
                "ai_recommendations": [],
                "strengths": [],
                "weaknesses": [],
                "interview_tips": []
            }
    
    def suggest_improvements(self, resume_text: str, focus_area: str) -> str:
        """Get specific improvement suggestions for a focus area"""
        if not self.is_available():
            return "AI service unavailable"
        
        try:
            prompt = f"""As a resume expert, provide specific improvements for this resume section.

**Resume Excerpt:**
{resume_text[:2000]}

**Focus Area:** {focus_area}

Provide 3-5 specific, actionable suggestions to improve this area. Be concrete and give examples.
Keep response under 300 words."""

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"AI suggestion error: {e}")
            return "Unable to generate suggestions at this time."
    
    def generate_cover_letter(
        self,
        resume_text: str,
        job_description: str,
        tone: str = "professional"
    ) -> str:
        """Generate a tailored cover letter"""
        if not self.is_available():
            return "AI service unavailable"
        
        try:
            prompt = f"""Write a compelling cover letter based on this resume and job description.

**Resume:**
{resume_text[:2000]}

**Job Description:**
{job_description[:1500]}

**Tone:** {tone}

Write a professional cover letter (3-4 paragraphs, ~250 words) that:
1. Opens with enthusiasm for the specific role
2. Highlights 2-3 most relevant achievements from the resume
3. Explains why the candidate is a great fit
4. Closes with a call to action

Make it personal and specific to this role. Avoid generic phrases."""

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Cover letter generation error: {e}")
            return "Unable to generate cover letter at this time."


# Singleton instance
_ai_analyzer = None

def get_ai_analyzer() -> AIAnalyzer:
    """Get or create AI analyzer instance"""
    global _ai_analyzer
    if _ai_analyzer is None:
        _ai_analyzer = AIAnalyzer()
    return _ai_analyzer
