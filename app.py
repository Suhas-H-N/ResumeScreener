"""
ResumeIQ Pro v3.0 — AI-Powered Resume Screening Platform
Enhanced with Claude AI, advanced NLP, and production-ready features

Author: Enhanced by Claude
"""
import os
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import extensions
from extensions import db, migrate, login_manager, mail, bcrypt, cors, limiter, jwt
from models import User, Analysis, Comparison, AuditLog
from config import config
from nlp_utils import (
    calculate_match_score,
    calculate_ats_score,
    calculate_readability,
    extract_skills_from_text,
    generate_recommendations,
    classify_match_level,
    get_score_color,
    detect_experience_level,
    estimate_salary_range,
    calculate_quantification_score,
    detect_career_gaps,
)
from document_utils import DocumentProcessor
from ai_service import get_ai_analyzer

# Initialize Flask app
app = Flask(__name__)

# Load configuration
env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Initialize extensions with app
db.init_app(app)
migrate.init_app(app, db)
login_manager.init_app(app)
mail.init_app(app)
bcrypt.init_app(app)
cors.init_app(app, resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}})
limiter.init_app(app)
jwt.init_app(app)

# Create upload directory
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AI analyzer
ai_analyzer = get_ai_analyzer()


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def audit_log(action: str, user_id: int = None, details: dict = None):
    """Create audit log entry"""
    try:
        log = AuditLog(
            user_id=user_id,
            action=action,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string[:255],
            details=details or {}
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        logger.error(f"Audit log error: {e}")


def login_required_api(f):
    """Decorator for API routes that require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get current logged-in user"""
    user_id = session.get('user_id')
    if user_id:
        return User.query.get(user_id)
    return None


# ═══════════════════════════════════════════════════════════
# AUTHENTICATION ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/api/signup', methods=['POST'])
@limiter.limit("5 per hour")
def signup():
    """Register new user account"""
    data = request.get_json()
    
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    # Validation
    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    try:
        # Create user
        user = User(name=name, email=email)
        user.set_password(password)
        
        # Generate verification token
        user.generate_verification_token()
        
        db.session.add(user)
        db.session.commit()
        
        # TODO: Send verification email
        # send_verification_email(user)
        
        audit_log('signup', user.id, {'email': email})
        
        return jsonify({
            'message': 'Account created successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Signup error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """User login"""
    data = request.get_json()
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    user = User.query.filter_by(email=email).first()
    
    if not user or not user.check_password(password):
        audit_log('login_failed', details={'email': email})
        return jsonify({'error': 'Invalid email or password'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is disabled'}), 403
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Set session
    session['user_id'] = user.id
    session['user_email'] = user.email
    session['user_name'] = user.name
    
    audit_log('login', user.id)
    
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict()
    }), 200


@app.route('/api/logout', methods=['POST'])
def logout():
    """User logout"""
    user_id = session.get('user_id')
    if user_id:
        audit_log('logout', user_id)
    
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200


@app.route('/api/me', methods=['GET'])
def get_current_user_info():
    """Get current user info"""
    user = get_current_user()
    
    if not user:
        return jsonify({'logged_in': False}), 200
    
    return jsonify({
        'logged_in': True,
        'user': user.to_dict()
    }), 200


@app.route('/api/forgot-password', methods=['POST'])
@limiter.limit("3 per hour")
def forgot_password():
    """Request password reset"""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    user = User.query.filter_by(email=email).first()
    
    # Always return success to prevent email enumeration
    if user:
        token = user.generate_reset_token()
        db.session.commit()
        
        # TODO: Send password reset email
        # send_password_reset_email(user, token)
        
        audit_log('password_reset_requested', user.id)
    
    return jsonify({
        'message': 'If that email exists, a reset link has been sent'
    }), 200


@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    """Reset password with token"""
    data = request.get_json()
    
    token = data.get('token', '')
    new_password = data.get('password', '')
    
    if not token or not new_password:
        return jsonify({'error': 'Token and password required'}), 400
    
    if len(new_password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    user = User.query.filter_by(reset_token=token).first()
    
    if not user or not user.verify_reset_token(token):
        return jsonify({'error': 'Invalid or expired reset token'}), 400
    
    # Update password
    user.set_password(new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.session.commit()
    
    audit_log('password_reset', user.id)
    
    return jsonify({'message': 'Password reset successfully'}), 200


# ═══════════════════════════════════════════════════════════
# CORE ANALYSIS ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("20 per hour")
def analyze_resume():
    """
    Analyze resume against job description
    Enhanced with AI insights
    """
    start_time = time.time()
    
    # Extract resume text
    resume_text = ""
    resume_filename = None
    
    if 'resume_file' in request.files:
        file = request.files['resume_file']
        if file and file.filename and DocumentProcessor.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            resume_filename = filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            resume_text = DocumentProcessor.extract_text(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
    
    if not resume_text:
        resume_text = request.form.get('resume_text', '').strip()
    
    job_description = request.form.get('job_description', '').strip()
    job_title = request.form.get('job_title', '').strip()
    job_company = request.form.get('job_company', '').strip()
    
    # Validate inputs
    is_valid, error_msg = DocumentProcessor.validate_resume_content(resume_text)
    if not is_valid:
        return jsonify({'error': error_msg}), 400
    
    if not job_description or len(job_description) < 50:
        return jsonify({'error': 'Please enter a valid job description (minimum 50 characters)'}), 400
    
    try:
        # Run NLP analysis
        match_score, matching_kw, missing_kw = calculate_match_score(resume_text, job_description)
        ats_scores = calculate_ats_score(resume_text, job_description)
        stats = calculate_readability(resume_text)
        level = classify_match_level(match_score)
        
        # Extract skills
        resume_skills = sorted(list(extract_skills_from_text(resume_text)))
        job_skills = sorted(list(extract_skills_from_text(job_description)))
        missing_skills = sorted(list(set(job_skills) - set(resume_skills)))
        
        # Generate recommendations
        recommendations = generate_recommendations(
            missing_kw, missing_skills, ats_scores, stats
        )

        # New enhanced features
        experience_info = detect_experience_level(resume_text)
        salary_estimate = estimate_salary_range(set(resume_skills), experience_info)
        quantification = calculate_quantification_score(resume_text)
        career_gaps = detect_career_gaps(resume_text)

        # Build basic result
        result = {
            'match_score': match_score,
            'level': level,
            'color': get_score_color(match_score),
            'ats_scores': ats_scores,
            'matching_keywords': matching_kw[:20],
            'missing_keywords': missing_kw[:20],
            'resume_skills': resume_skills[:25],
            'job_skills': job_skills[:25],
            'missing_skills': missing_skills[:20],
            'recommendations': recommendations,
            'stats': stats,
            'experience_info': experience_info,
            'salary_estimate': salary_estimate,
            'quantification': quantification,
            'career_gaps': career_gaps,
        }
        
        # AI Enhancement (if available)
        if ai_analyzer.is_available():
            try:
                ai_insights = ai_analyzer.analyze_resume(
                    resume_text, job_description, result
                )
                result.update(ai_insights)
            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                result['ai_summary'] = None
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        result['processing_time_ms'] = processing_time
        
        # Save to history if logged in
        user = get_current_user()
        if user:
            try:
                analysis = Analysis(
                    user_id=user.id,
                    resume_filename=resume_filename,
                    resume_text_snippet=resume_text[:500],
                    job_title=job_title,
                    job_company=job_company,
                    job_description_snippet=job_description[:200],
                    match_score=match_score,
                    ats_score=ats_scores['overall'],
                    level=level,
                    ats_breakdown=ats_scores,
                    matching_keywords=matching_kw[:20],
                    missing_keywords=missing_kw[:20],
                    resume_skills=resume_skills[:25],
                    missing_skills=missing_skills[:20],
                    recommendations=recommendations,
                    stats=stats,
                    ai_summary=result.get('ai_summary'),
                    ai_recommendations=result.get('ai_recommendations'),
                    processing_time_ms=processing_time
                )
                db.session.add(analysis)
                db.session.commit()
                
                result['analysis_id'] = analysis.id
                
                audit_log('analyze_resume', user.id, {
                    'match_score': match_score,
                    'ats_score': ats_scores['overall']
                })
            except Exception as e:
                db.session.rollback()
                logger.error(f"Failed to save analysis: {e}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500


@app.route('/api/compare', methods=['POST'])
@limiter.limit("10 per hour")
def compare_resumes():
    """
    Compare two resumes side-by-side
    """
    job_description = request.form.get('job_description', '').strip()
    resumes = []
    resume_names = []
    
    for key in ['resume_a', 'resume_b']:
        text = ""
        name = None
        
        # Try file upload first
        file_key = f"{key}_file"
        if file_key in request.files:
            file = request.files[file_key]
            if file and file.filename and DocumentProcessor.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                name = filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                text = DocumentProcessor.extract_text(filepath)
                try:
                    os.remove(filepath)
                except:
                    pass
        
        # Fallback to text input
        if not text:
            text = request.form.get(key, '').strip()
            name = f"Resume {key[-1].upper()}"
        
        resumes.append(text)
        resume_names.append(name)
    
    # Validate inputs
    if not resumes[0] or not resumes[1]:
        return jsonify({'error': 'Please provide both resumes'}), 400
    
    if not job_description:
        return jsonify({'error': 'Please provide a job description'}), 400
    
    try:
        results = []
        
        for resume_text in resumes:
            match_score, _, _ = calculate_match_score(resume_text, job_description)
            ats_scores = calculate_ats_score(resume_text, job_description)
            skills = sorted(list(extract_skills_from_text(resume_text)))[:20]
            
            results.append({
                'match_score': match_score,
                'ats_score': ats_scores['overall'],
                'level': classify_match_level(match_score),
                'skills': skills,
                'ats_breakdown': ats_scores
            })
        
        winner = 'A' if results[0]['match_score'] >= results[1]['match_score'] else 'B'
        
        response = {
            'resume_a': results[0],
            'resume_b': results[1],
            'winner': winner,
            'resume_a_name': resume_names[0],
            'resume_b_name': resume_names[1]
        }
        
        # Save comparison if logged in
        user = get_current_user()
        if user:
            try:
                comparison = Comparison(
                    user_id=user.id,
                    resume_a_name=resume_names[0],
                    resume_b_name=resume_names[1],
                    job_description_snippet=job_description[:200],
                    resume_a_score=results[0]['match_score'],
                    resume_b_score=results[1]['match_score'],
                    resume_a_ats=results[0]['ats_score'],
                    resume_b_ats=results[1]['ats_score'],
                    winner=winner,
                    comparison_data=response
                )
                db.session.add(comparison)
                db.session.commit()
                
                audit_log('compare_resumes', user.id)
            except Exception as e:
                db.session.rollback()
                logger.error(f"Failed to save comparison: {e}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return jsonify({'error': 'Comparison failed'}), 500


# ═══════════════════════════════════════════════════════════
# HISTORY & DATA MANAGEMENT
# ═══════════════════════════════════════════════════════════

@app.route('/api/history', methods=['GET'])
@login_required_api
def get_history():
    """Get user's analysis history"""
    user = get_current_user()
    page = request.args.get('page', 1, type=int)
    per_page = app.config['PAGINATION_PER_PAGE']
    
    pagination = Analysis.query.filter_by(user_id=user.id) \
        .order_by(Analysis.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'analyses': [a.to_dict() for a in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    }), 200


@app.route('/api/history/<int:analysis_id>', methods=['GET'])
@login_required_api
def get_analysis_detail(analysis_id):
    """Get detailed analysis results"""
    user = get_current_user()
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    return jsonify(analysis.to_dict(include_details=True)), 200


@app.route('/api/history/<int:analysis_id>', methods=['DELETE'])
@login_required_api
def delete_analysis(analysis_id):
    """Delete an analysis from history"""
    user = get_current_user()
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    try:
        db.session.delete(analysis)
        db.session.commit()
        
        audit_log('delete_analysis', user.id, {'analysis_id': analysis_id})
        
        return jsonify({'message': 'Analysis deleted'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Delete analysis error: {e}")
        return jsonify({'error': 'Failed to delete analysis'}), 500


@app.route('/api/export/history', methods=['GET'])
@login_required_api
def export_history():
    """Export history as CSV"""
    import csv
    import io
    
    user = get_current_user()
    analyses = Analysis.query.filter_by(user_id=user.id) \
        .order_by(Analysis.created_at.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Date', 'Job Title', 'Company', 'Match Score', 'ATS Score',
        'Level', 'Skills Found', 'Processing Time (ms)'
    ])
    
    # Write data
    for analysis in analyses:
        writer.writerow([
            analysis.created_at.strftime('%Y-%m-%d %H:%M'),
            analysis.job_title or '',
            analysis.job_company or '',
            analysis.match_score,
            analysis.ats_score,
            analysis.level,
            len(analysis.resume_skills or []),
            analysis.processing_time_ms or 0
        ])
    
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'resumeiq_history_{datetime.now().strftime("%Y%m%d")}.csv'
    )


# ═══════════════════════════════════════════════════════════
# AI-POWERED FEATURES
# ═══════════════════════════════════════════════════════════

@app.route('/api/ai/improve', methods=['POST'])
@login_required_api
@limiter.limit("5 per hour")
def ai_improve_resume():
    """Get AI-powered improvement suggestions"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    focus_area = data.get('focus_area', 'overall')
    
    if not resume_text:
        return jsonify({'error': 'Resume text required'}), 400
    
    if not ai_analyzer.is_available():
        return jsonify({'error': 'AI service unavailable'}), 503
    
    try:
        suggestions = ai_analyzer.suggest_improvements(resume_text, focus_area)
        
        audit_log('ai_improve', get_current_user().id, {'focus_area': focus_area})
        
        return jsonify({'suggestions': suggestions}), 200
    except Exception as e:
        logger.error(f"AI improve error: {e}")
        return jsonify({'error': 'Failed to generate suggestions'}), 500


@app.route('/api/ai/cover-letter', methods=['POST'])
@login_required_api
@limiter.limit("3 per hour")
def generate_cover_letter():
    """Generate AI-powered cover letter"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    job_description = data.get('job_description', '')
    tone = data.get('tone', 'professional')
    
    if not resume_text or not job_description:
        return jsonify({'error': 'Resume and job description required'}), 400
    
    if not ai_analyzer.is_available():
        return jsonify({'error': 'AI service unavailable'}), 503
    
    try:
        cover_letter = ai_analyzer.generate_cover_letter(
            resume_text, job_description, tone
        )
        
        audit_log('generate_cover_letter', get_current_user().id)
        
        return jsonify({'cover_letter': cover_letter}), 200
    except Exception as e:
        logger.error(f"Cover letter generation error: {e}")
        return jsonify({'error': 'Failed to generate cover letter'}), 500


# ═══════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════

@app.route('/api/download-report', methods=['POST'])
@limiter.limit("10 per hour")
def download_report():
    """Generate and download PDF report"""
    from report_generator import generate_pdf_report
    
    data = request.get_json()
    
    try:
        pdf_path = generate_pdf_report(data)
        
        user = get_current_user()
        if user:
            audit_log('download_report', user.id)
        
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='ResumeIQ_Report.pdf'
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'error': 'Failed to generate report'}), 500


# ═══════════════════════════════════════════════════════════
# PAGE ROUTES (Frontend)
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main analyzer page"""
    return render_template('index.html')


@app.route('/login-page')
def login_page():
    """Login page"""
    return render_template('login.html')


@app.route('/signup-page')
def signup_page():
    """Signup page"""
    return render_template('signup.html')


@app.route('/compare-page')
def compare_page():
    """Compare resumes page"""
    return render_template('compare.html')


@app.route('/history-page')
def history_page():
    """Analysis history page"""
    return render_template('history.html')


@app.route('/profile-page')
def profile_page():
    """User profile page"""
    return render_template('profile.html')


@app.route('/jobs-page')
def jobs_page():
    """Job tracker page"""
    return render_template('jobs.html')


# ═══════════════════════════════════════════════════════════
# PROFILE MANAGEMENT
# ═══════════════════════════════════════════════════════════

@app.route('/api/profile', methods=['PUT'])
@login_required_api
def update_profile():
    """Update user profile info"""
    user = get_current_user()
    data = request.get_json()

    name = data.get('name', '').strip()
    company = data.get('company', '').strip()
    role = data.get('role', '').strip()

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    try:
        user.name = name
        user.company = company or None
        user.role = role or None
        db.session.commit()

        # Update session name
        session['user_name'] = name

        audit_log('profile_updated', user.id)
        return jsonify({'message': 'Profile updated', 'user': user.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Profile update error: {e}")
        return jsonify({'error': 'Update failed'}), 500


@app.route('/api/profile/password', methods=['PUT'])
@login_required_api
def change_password():
    """Change user password"""
    user = get_current_user()
    data = request.get_json()

    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')

    if not current_password or not new_password:
        return jsonify({'error': 'Both current and new password required'}), 400

    if not user.check_password(current_password):
        return jsonify({'error': 'Current password is incorrect'}), 400

    if len(new_password) < 8:
        return jsonify({'error': 'New password must be at least 8 characters'}), 400

    try:
        user.set_password(new_password)
        db.session.commit()
        audit_log('password_changed', user.id)
        return jsonify({'message': 'Password changed successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Password change error: {e}")
        return jsonify({'error': 'Password change failed'}), 500


@app.route('/api/profile', methods=['DELETE'])
@login_required_api
def delete_account():
    """Delete user account"""
    user = get_current_user()
    try:
        db.session.delete(user)
        db.session.commit()
        session.clear()
        return jsonify({'message': 'Account deleted'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Account delete error: {e}")
        return jsonify({'error': 'Failed to delete account'}), 500


# ═══════════════════════════════════════════════════════════
# HEALTH CHECK & ADMIN
# ═══════════════════════════════════════════════════════════

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': app.config['APP_VERSION'],
        'ai_available': ai_analyzer.is_available()
    }), 200


@app.route('/api/stats', methods=['GET'])
@login_required_api
def get_user_stats():
    """Get user statistics"""
    user = get_current_user()
    
    total_analyses = Analysis.query.filter_by(user_id=user.id).count()
    avg_match_score = db.session.query(db.func.avg(Analysis.match_score)) \
        .filter_by(user_id=user.id).scalar() or 0
    avg_ats_score = db.session.query(db.func.avg(Analysis.ats_score)) \
        .filter_by(user_id=user.id).scalar() or 0
    
    return jsonify({
        'total_analyses': total_analyses,
        'avg_match_score': round(avg_match_score, 1),
        'avg_ats_score': round(avg_ats_score, 1),
        'member_since': user.created_at.isoformat() if user.created_at else None
    }), 200


# ═══════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    db.session.rollback()
    logger.error(f"Internal error: {error}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


@app.errorhandler(413)
def file_too_large(error):
    """413 error handler for file size"""
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413


# ═══════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ═══════════════════════════════════════════════════════════

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print("Database initialized!")


@app.cli.command()
def create_admin():
    """Create admin user"""
    email = input("Admin email: ")
    name = input("Admin name: ")
    password = input("Admin password: ")
    
    user = User(name=name, email=email, is_admin=True, email_verified=True)
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    
    print(f"Admin user created: {email}")


# ═══════════════════════════════════════════════════════════
# RUN APPLICATION
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(
        debug=app.config['DEBUG'],
        host='0.0.0.0',
        port=5000
    )
