"""Configuration — dev works with SQLite + memory rate limiting, no Redis needed"""
import os
from pathlib import Path

basedir = Path(__file__).parent

class Config:
    APP_NAME    = "ResumeIQ Pro"
    APP_VERSION = "3.1.0"
    SECRET_KEY  = os.getenv("SECRET_KEY", "resumeiq-dev-secret-2026-change-in-prod")

    SQLALCHEMY_DATABASE_URI      = os.getenv("DATABASE_URL", f"sqlite:///{basedir / 'resumeiq.db'}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO              = False

    UPLOAD_FOLDER        = str(basedir / "uploads")
    MAX_CONTENT_LENGTH   = 10 * 1024 * 1024   # 10 MB
    ALLOWED_EXTENSIONS   = {"pdf", "docx", "txt", "doc"}

    ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")
    PAGINATION_PER_PAGE  = 20

class DevelopmentConfig(Config):
    DEBUG   = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG   = False
    TESTING = False

class TestingConfig(Config):
    TESTING = True
    DEBUG   = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

config = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
    "default":     DevelopmentConfig,
}
