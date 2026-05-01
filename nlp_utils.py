"""
Enhanced NLP utilities for resume analysis
Improved version with better algorithms
"""
import re
import string
from collections import Counter
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# Expanded skill taxonomy
TECH_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "ruby", "php", "scala", "r", "perl", "dart", "elixir",
    "haskell", "clojure", "lua", "julia", "objective-c", "vb.net", "matlab",
    
    # Web Frontend
    "react", "angular", "vue", "svelte", "nextjs", "nuxt", "gatsby", "redux",
    "mobx", "webpack", "vite", "babel", "html", "html5", "css", "css3",
    "sass", "scss", "less", "tailwind", "bootstrap", "material-ui", "chakra",
    
    # Web Backend
    "nodejs", "express", "django", "flask", "fastapi", "spring", "spring boot",
    "asp.net", "laravel", "rails", "ruby on rails", "graphql", "rest", "soap",
    
    # Mobile
    "android", "ios", "react native", "flutter", "xamarin", "ionic",
    
    # Data Science & ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "plotly", "opencv", "nltk", "spacy",
    "machine learning", "deep learning", "neural networks", "nlp",
    "computer vision", "data science", "data analysis", "statistics",
    
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra",
    "dynamodb", "elasticsearch", "oracle", "mssql", "sqlite", "mariadb",
    "neo4j", "couchdb", "firebase",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "circleci", "travis ci", "ci/cd", "linux", "unix", "bash", "powershell",
    
    # Big Data
    "hadoop", "spark", "kafka", "airflow", "flink", "hive", "pig",
    
    # Tools & Others
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "figma",
    "sketch", "adobe xd", "postman", "swagger", "api", "microservices",
    "agile", "scrum", "kanban", "testing", "unit testing", "tdd",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "collaboration", "problem solving",
    "critical thinking", "analytical", "creativity", "innovation", "adaptability",
    "flexibility", "time management", "organization", "attention to detail",
    "presentation", "public speaking", "writing", "negotiation", "mentoring",
    "coaching", "strategic thinking", "decision making", "conflict resolution",
    "emotional intelligence", "customer service", "stakeholder management",
}

ALL_SKILLS = TECH_SKILLS | SOFT_SKILLS

# Common stop words (expanded)
STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "are", "was", "were",
    "have", "has", "had", "will", "would", "can", "could", "should",
    "may", "might", "shall", "been", "being", "from", "into", "onto",
    "upon", "your", "their", "our", "which", "when", "where", "how",
    "what", "who", "why", "also", "than", "then", "just", "but", "not",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "too", "very", "about", "after",
    "before", "during", "through", "between", "under", "over", "above",
}

# Action verbs for ATS scoring
ACTION_VERBS = {
    "achieved", "accomplished", "accelerated", "adapted", "administered",
    "analyzed", "architected", "assembled", "assessed", "automated",
    "built", "balanced", "boosted", "coordinated", "created", "collaborated",
    "conducted", "consolidated", "designed", "developed", "delivered",
    "directed", "drove", "engineered", "enhanced", "established",
    "evaluated", "executed", "facilitated", "founded", "generated",
    "guided", "headed", "implemented", "improved", "increased", "initiated",
    "integrated", "launched", "led", "managed", "maximized", "mentored",
    "migrated", "modernized", "monitored", "negotiated", "optimized",
    "orchestrated", "organized", "overhauled", "pioneered", "planned",
    "produced", "programmed", "reduced", "refactored", "restructured",
    "revamped", "scaled", "spearheaded", "streamlined", "strengthened",
    "supervised", "transformed", "validated", "wrote",
}


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Preserve + and # for skills like C++, C#
    text = re.sub(r'[^\w\s\+\#\-\.]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """Extract meaningful keywords from text"""
    cleaned = clean_text(text)
    words = re.findall(r'\b[a-z][a-z0-9\+\#\-\.]{2,}\b', cleaned)
    return {w for w in words if w not in STOP_WORDS and len(w) >= min_length}


def extract_skills_from_text(text: str) -> Set[str]:
    """Extract skills from text using pattern matching"""
    text_lower = text.lower()
    found_skills = set()
    
    for skill in ALL_SKILLS:
        # Use word boundaries for exact matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    return found_skills


def extract_action_verbs(text: str) -> List[str]:
    """Extract action verbs from text"""
    text_lower = text.lower()
    found_verbs = []
    
    for verb in ACTION_VERBS:
        pattern = r'\b' + re.escape(verb) + r'\b'
        matches = re.findall(pattern, text_lower)
        found_verbs.extend(matches)
    
    return found_verbs


def calculate_match_score(resume: str, job_desc: str) -> Tuple[float, List[str], List[str]]:
    """
    Calculate TF-IDF based match score
    
    Returns:
        Tuple of (score, matching_keywords, missing_keywords)
    """
    try:
        # TF-IDF with bigrams for better context
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=500
        )
        vectors = vectorizer.fit_transform([resume, job_desc])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    except Exception as e:
        logger.error(f"TF-IDF calculation error: {e}")
        score = 0.0
    
    # Keyword matching
    resume_kw = extract_keywords(resume)
    job_kw = extract_keywords(job_desc)
    
    matching = sorted(list(resume_kw & job_kw))
    missing = sorted(list(job_kw - resume_kw))
    
    return round(score, 1), matching, missing


def calculate_ats_score(resume_text: str, job_desc: str) -> Dict[str, int]:
    """
    Enhanced ATS scoring with multiple dimensions
    
    Dimensions:
    1. Keyword density (30%)
    2. Skill match (30%)
    3. Section structure (15%)
    4. Action verbs (10%)
    5. Resume length (15%)
    """
    scores = {}
    
    # 1. Keyword density
    job_kw = extract_keywords(job_desc)
    resume_kw = extract_keywords(resume_text)
    
    if job_kw:
        overlap = len(resume_kw & job_kw) / len(job_kw)
        scores['keyword_density'] = round(min(overlap * 100, 100))
    else:
        scores['keyword_density'] = 0
    
    # 2. Skill match
    job_skills = extract_skills_from_text(job_desc)
    resume_skills = extract_skills_from_text(resume_text)
    
    if job_skills:
        matched = len(resume_skills & job_skills)
        scores['skill_match'] = round(min(matched / len(job_skills) * 100, 100))
    else:
        scores['skill_match'] = 50
    
    # 3. Resume length (300-1200 words optimal for ATS)
    word_count = len(resume_text.split())
    if 300 <= word_count <= 1200:
        scores['length'] = 100
    elif word_count < 150:
        scores['length'] = 30
    elif word_count < 300:
        scores['length'] = 60
    elif word_count > 1500:
        scores['length'] = 70
    else:
        scores['length'] = 85
    
    # 4. Section structure
    section_keywords = [
        "experience", "education", "skills", "projects",
        "summary", "objective", "certifications", "achievements",
        "awards", "publications", "volunteer"
    ]
    text_lower = resume_text.lower()
    found_sections = sum(1 for s in section_keywords if s in text_lower)
    scores['sections'] = round((found_sections / len(section_keywords)) * 100)
    
    # 5. Action verbs
    found_verbs = extract_action_verbs(resume_text)
    unique_verbs = len(set(found_verbs))
    scores['action_verbs'] = round(min(unique_verbs / 8 * 100, 100))
    
    # Weighted overall score
    weights = {
        'keyword_density': 0.30,
        'skill_match': 0.30,
        'length': 0.15,
        'sections': 0.15,
        'action_verbs': 0.10
    }
    
    overall = sum(scores[k] * weights[k] for k in scores)
    scores['overall'] = round(overall, 1)
    
    return scores


def calculate_readability(text: str) -> Dict[str, any]:
    """
    Calculate readability metrics
    Uses Flesch-Kincaid and other metrics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    char_count = len(text)
    
    # Average words per sentence
    avg_words_per_sentence = word_count / sent_count
    
    # Average syllables per word (approximation)
    def count_syllables(word):
        word = word.lower()
        count = len(re.findall(r'[aeiou]+', word))
        return max(count, 1)
    
    syllable_count = sum(count_syllables(w) for w in words)
    avg_syllables = syllable_count / max(word_count, 1)
    
    # Flesch Reading Ease (0-100, higher = easier)
    fre = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables
    fre = max(0, min(100, fre))
    
    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables - 15.59
    fk_grade = max(0, fk_grade)
    
    return {
        'word_count': word_count,
        'sentence_count': sent_count,
        'char_count': char_count,
        'avg_words_per_sentence': round(avg_words_per_sentence, 1),
        'flesch_reading_ease': round(fre, 1),
        'flesch_kincaid_grade': round(fk_grade, 1),
        'readability_level': _get_readability_level(fre)
    }


def _get_readability_level(fre_score: float) -> str:
    """Convert Flesch Reading Ease to level"""
    if fre_score >= 80:
        return "Very Easy"
    elif fre_score >= 60:
        return "Easy"
    elif fre_score >= 50:
        return "Fairly Easy"
    elif fre_score >= 30:
        return "Difficult"
    else:
        return "Very Difficult"


def generate_recommendations(
    missing_keywords: List[str],
    missing_skills: List[str],
    ats_scores: Dict[str, int],
    stats: Dict[str, any]
) -> List[str]:
    """Generate actionable recommendations"""
    tips = []
    
    # Skill gaps
    if missing_skills:
        top_skills = missing_skills[:5]
        tips.append(f"📚 Add these in-demand skills: {', '.join(top_skills)}")
    
    # Keyword optimization
    if missing_keywords and ats_scores.get('keyword_density', 100) < 60:
        top_kw = missing_keywords[:4]
        tips.append(f"🔑 Include job-specific keywords: {', '.join(top_kw)}")
    
    # Section structure
    if ats_scores.get('sections', 100) < 70:
        tips.append("📋 Add clear section headers: Experience, Education, Skills, Projects, Certifications")
    
    # Action verbs
    if ats_scores.get('action_verbs', 100) < 60:
        tips.append("💪 Use strong action verbs: Led, Developed, Optimized, Achieved, Delivered, Transformed")
    
    # Resume length
    word_count = stats.get('word_count', 0)
    if ats_scores.get('length', 100) < 70:
        if word_count < 300:
            tips.append("📝 Resume is too short — expand your experience with quantified achievements")
        elif word_count > 1500:
            tips.append("✂️ Resume may be too long — focus on most relevant 10-15 years of experience")
    
    # Quantification
    tips.append("📊 Quantify achievements with metrics (e.g., 'Increased revenue by 40%', 'Led team of 12')")
    
    # ATS compatibility
    tips.append("🤖 Keep formatting clean — avoid tables, text boxes, headers/footers, and images for ATS")
    
    # Readability
    if stats.get('flesch_reading_ease', 50) < 40:
        tips.append("📖 Simplify language — use clear, concise sentences for better readability")
    
    return tips[:8]  # Return top 8 recommendations


def classify_match_level(score: float) -> str:
    """Classify match score into levels"""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Low"


def get_score_color(score: float) -> str:
    """Get color for score visualization"""
    if score >= 80:
        return "#10b981"  # Green
    elif score >= 60:
        return "#f59e0b"  # Amber
    elif score >= 40:
        return "#f97316"  # Orange
    else:
        return "#ef4444"  # Red
