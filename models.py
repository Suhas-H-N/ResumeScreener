"""
Database models for ResumeIQ Pro
"""
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import secrets
from extensions import db


class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Profile
    company = db.Column(db.String(100))
    role = db.Column(db.String(100))
    avatar_url = db.Column(db.String(255))
    
    # Email verification
    email_verified = db.Column(db.Boolean, default=False)
    verification_token = db.Column(db.String(100), unique=True)
    verification_sent_at = db.Column(db.DateTime)
    
    # Password reset
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expires = db.Column(db.DateTime)
    
    # Account status
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Subscription (for future premium features)
    subscription_tier = db.Column(db.String(20), default="free")  # free, pro, enterprise
    subscription_expires = db.Column(db.DateTime)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    analyses = db.relationship("Analysis", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    comparisons = db.relationship("Comparison", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def generate_verification_token(self):
        """Generate email verification token"""
        self.verification_token = secrets.token_urlsafe(32)
        self.verification_sent_at = datetime.utcnow()
        return self.verification_token
    
    def generate_reset_token(self, expires_in=3600):
        """Generate password reset token"""
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
        return self.reset_token
    
    def verify_reset_token(self, token):
        """Verify password reset token"""
        if self.reset_token != token:
            return False
        if self.reset_token_expires < datetime.utcnow():
            return False
        return True
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "company": self.company,
            "role": self.role,
            "email_verified": self.email_verified,
            "subscription_tier": self.subscription_tier,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class Analysis(db.Model):
    """Analysis history model"""
    __tablename__ = "analyses"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Resume info
    resume_filename = db.Column(db.String(255))
    resume_text_snippet = db.Column(db.Text)  # First 500 chars for quick reference
    
    # Job description
    job_title = db.Column(db.String(200))
    job_company = db.Column(db.String(200))
    job_description_snippet = db.Column(db.Text)
    
    # Scores
    match_score = db.Column(db.Float, nullable=False)
    ats_score = db.Column(db.Float, nullable=False)
    level = db.Column(db.String(20))  # Excellent, Good, Average, Low
    
    # Detailed results (stored as JSON)
    ats_breakdown = db.Column(db.JSON)  # keyword_density, skill_match, etc.
    matching_keywords = db.Column(db.JSON)
    missing_keywords = db.Column(db.JSON)
    resume_skills = db.Column(db.JSON)
    missing_skills = db.Column(db.JSON)
    recommendations = db.Column(db.JSON)
    stats = db.Column(db.JSON)  # word_count, readability, etc.
    
    # AI insights (if using Claude)
    ai_summary = db.Column(db.Text)
    ai_recommendations = db.Column(db.JSON)
    
    # Metadata
    processing_time_ms = db.Column(db.Integer)  # How long the analysis took
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def to_dict(self, include_details=False):
        """Convert analysis to dictionary"""
        base = {
            "id": self.id,
            "date": self.created_at.isoformat() if self.created_at else None,
            "match_score": self.match_score,
            "ats_score": self.ats_score,
            "level": self.level,
            "job_snippet": self.job_description_snippet or "",
            "job_title": self.job_title,
            "job_company": self.job_company,
        }
        
        if include_details:
            base.update({
                "ats_breakdown": self.ats_breakdown,
                "matching_keywords": self.matching_keywords,
                "missing_keywords": self.missing_keywords,
                "resume_skills": self.resume_skills,
                "missing_skills": self.missing_skills,
                "recommendations": self.recommendations,
                "stats": self.stats,
                "ai_summary": self.ai_summary,
                "ai_recommendations": self.ai_recommendations,
            })
        
        return base


class Comparison(db.Model):
    """Resume comparison history"""
    __tablename__ = "comparisons"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Comparison data
    resume_a_name = db.Column(db.String(255))
    resume_b_name = db.Column(db.String(255))
    
    job_description_snippet = db.Column(db.Text)
    
    # Results
    resume_a_score = db.Column(db.Float)
    resume_b_score = db.Column(db.Float)
    resume_a_ats = db.Column(db.Float)
    resume_b_ats = db.Column(db.Float)
    winner = db.Column(db.String(1))  # 'A' or 'B'
    
    # Detailed results
    comparison_data = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def to_dict(self):
        """Convert comparison to dictionary"""
        return {
            "id": self.id,
            "date": self.created_at.isoformat() if self.created_at else None,
            "resume_a_name": self.resume_a_name,
            "resume_b_name": self.resume_b_name,
            "resume_a_score": self.resume_a_score,
            "resume_b_score": self.resume_b_score,
            "winner": self.winner,
        }


class APIKey(db.Model):
    """API keys for programmatic access"""
    __tablename__ = "api_keys"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    
    key_hash = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(100))  # User-defined name for the key
    
    is_active = db.Column(db.Boolean, default=True)
    last_used = db.Column(db.DateTime)
    
    # Rate limiting per key
    requests_per_hour = db.Column(db.Integer, default=100)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime)
    
    user = db.relationship("User", backref="api_keys")


class AuditLog(db.Model):
    """Audit log for tracking important events"""
    __tablename__ = "audit_logs"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), index=True)
    
    action = db.Column(db.String(100), nullable=False)  # login, analyze, export, etc.
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    details = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    user = db.relationship("User", backref="audit_logs")
