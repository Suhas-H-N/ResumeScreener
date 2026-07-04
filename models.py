"""Database models for ResumeIQ Pro"""
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import secrets
from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # Profile
    company   = db.Column(db.String(100))
    role      = db.Column(db.String(100))

    # Password reset
    reset_token         = db.Column(db.String(100), unique=True)
    reset_token_expires = db.Column(db.DateTime)

    # Account
    is_active  = db.Column(db.Boolean, default=True)
    is_admin   = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)

    # Relationships
    analyses    = db.relationship("Analysis",   backref="user", lazy="dynamic", cascade="all, delete-orphan")
    comparisons = db.relationship("Comparison", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    jobs        = db.relationship("JobApplication", backref="user", lazy="dynamic", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_reset_token(self, expires_in=3600):
        self.reset_token         = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
        return self.reset_token

    def verify_reset_token(self, token):
        return (
            self.reset_token == token
            and self.reset_token_expires
            and self.reset_token_expires > datetime.utcnow()
        )

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "company":    self.company,
            "role":       self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat()  if self.last_login  else None,
        }


class Analysis(db.Model):
    __tablename__ = "analyses"

    id      = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    resume_filename       = db.Column(db.String(255))
    resume_text_snippet   = db.Column(db.Text)

    job_title             = db.Column(db.String(200))
    job_company           = db.Column(db.String(200))
    job_description_snippet = db.Column(db.Text)

    match_score = db.Column(db.Float, nullable=False)
    ats_score   = db.Column(db.Float, nullable=False)
    level       = db.Column(db.String(20))

    ats_breakdown      = db.Column(db.JSON)
    matching_keywords  = db.Column(db.JSON)
    missing_keywords   = db.Column(db.JSON)
    resume_skills      = db.Column(db.JSON)
    missing_skills     = db.Column(db.JSON)
    recommendations    = db.Column(db.JSON)
    stats              = db.Column(db.JSON)
    experience_info    = db.Column(db.JSON)
    salary_estimate    = db.Column(db.JSON)
    quantification     = db.Column(db.JSON)
    career_gaps        = db.Column(db.JSON)

    ai_summary         = db.Column(db.Text)
    ai_recommendations = db.Column(db.JSON)
    strengths          = db.Column(db.JSON)
    weaknesses         = db.Column(db.JSON)
    interview_tips     = db.Column(db.JSON)

    processing_time_ms = db.Column(db.Integer)
    created_at         = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self, include_details=False):
        base = {
            "id":           self.id,
            "date":         self.created_at.isoformat() if self.created_at else None,
            "match_score":  self.match_score,
            "ats_score":    self.ats_score,
            "level":        self.level,
            "job_title":    self.job_title,
            "job_company":  self.job_company,
            "job_snippet":  self.job_description_snippet or "",
        }
        if include_details:
            base.update({
                "ats_breakdown":     self.ats_breakdown,
                "matching_keywords": self.matching_keywords,
                "missing_keywords":  self.missing_keywords,
                "resume_skills":     self.resume_skills,
                "missing_skills":    self.missing_skills,
                "recommendations":   self.recommendations,
                "stats":             self.stats,
                "experience_info":   self.experience_info,
                "salary_estimate":   self.salary_estimate,
                "quantification":    self.quantification,
                "career_gaps":       self.career_gaps,
                "ai_summary":        self.ai_summary,
                "ai_recommendations":self.ai_recommendations,
                "strengths":         self.strengths,
                "weaknesses":        self.weaknesses,
                "interview_tips":    self.interview_tips,
            })
        return base


class Comparison(db.Model):
    __tablename__ = "comparisons"

    id      = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    resume_a_name = db.Column(db.String(255))
    resume_b_name = db.Column(db.String(255))
    job_description_snippet = db.Column(db.Text)

    resume_a_score = db.Column(db.Float)
    resume_b_score = db.Column(db.Float)
    resume_a_ats   = db.Column(db.Float)
    resume_b_ats   = db.Column(db.Float)
    winner         = db.Column(db.String(1))
    comparison_data = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id":            self.id,
            "date":          self.created_at.isoformat() if self.created_at else None,
            "resume_a_name": self.resume_a_name,
            "resume_b_name": self.resume_b_name,
            "resume_a_score":self.resume_a_score,
            "resume_b_score":self.resume_b_score,
            "winner":        self.winner,
        }


class JobApplication(db.Model):
    """Job application tracker — persisted in DB"""
    __tablename__ = "job_applications"

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    title       = db.Column(db.String(200), nullable=False)
    company     = db.Column(db.String(200), nullable=False)
    location    = db.Column(db.String(200))
    url         = db.Column(db.String(500))
    salary      = db.Column(db.String(100))
    status      = db.Column(db.String(30), default="saved")
    # saved | applied | phone_screen | interview | offer | rejected | accepted
    notes       = db.Column(db.Text)
    applied_date = db.Column(db.Date)
    deadline    = db.Column(db.Date)

    created_at  = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id":           self.id,
            "title":        self.title,
            "company":      self.company,
            "location":     self.location,
            "url":          self.url,
            "salary":       self.salary,
            "status":       self.status,
            "notes":        self.notes,
            "applied_date": self.applied_date.isoformat()  if self.applied_date else None,
            "deadline":     self.deadline.isoformat()       if self.deadline     else None,
            "created_at":   self.created_at.isoformat()    if self.created_at   else None,
        }


class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), index=True)
    action     = db.Column(db.String(100), nullable=False)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    details    = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
