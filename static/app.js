// ResumeIQ Pro - Enhanced JavaScript Application

// State management
const state = {
    currentUser: null,
    analysisResults: null,
    resumeFile: null,
    resumeText: '',
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeAuth();
    initializeFileUpload();
    initializeFormSubmission();
    initializeUserMenu();
    checkUserSession();
});

// ═══════════════════════════════════════════════════════════
// AUTHENTICATION
// ═══════════════════════════════════════════════════════════

async function checkUserSession() {
    try {
        const response = await fetch('/api/me');
        const data = await response.json();
        
        if (data.logged_in) {
            state.currentUser = data.user;
            updateUIForLoggedInUser(data.user);
        } else {
            updateUIForGuest();
        }
    } catch (error) {
        console.error('Session check error:', error);
        updateUIForGuest();
    }
}

function updateUIForLoggedInUser(user) {
    const loginLink = document.getElementById('loginLink');
    const userMenu = document.getElementById('userMenu');
    const userName = document.getElementById('userName');
    
    if (loginLink) loginLink.style.display = 'none';
    if (userMenu) userMenu.style.display = 'block';
    if (userName) userName.textContent = user.name;
}

function updateUIForGuest() {
    const loginLink = document.getElementById('loginLink');
    const userMenu = document.getElementById('userMenu');
    
    if (loginLink) loginLink.style.display = 'inline-flex';
    if (userMenu) userMenu.style.display = 'none';
}

function initializeAuth() {
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
}

async function handleLogout(e) {
    e.preventDefault();
    
    try {
        const response = await fetch('/api/logout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            state.currentUser = null;
            window.location.href = '/';
        }
    } catch (error) {
        console.error('Logout error:', error);
        showNotification('Logout failed', 'error');
    }
}

function initializeUserMenu() {
    const userMenuButton = document.getElementById('userMenuButton');
    const userDropdown = document.getElementById('userDropdown');
    
    if (userMenuButton && userDropdown) {
        userMenuButton.addEventListener('click', (e) => {
            e.stopPropagation();
            userDropdown.classList.toggle('show');
        });
        
        document.addEventListener('click', () => {
            userDropdown.classList.remove('show');
        });
    }
}

// ═══════════════════════════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════════════════════════

function initializeFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('resumeFile');
    const removeBtn = document.getElementById('removeFileBtn');
    
    if (!uploadArea || !fileInput) return;
    
    // Click to upload
    uploadArea.addEventListener('click', (e) => {
        if (e.target.closest('.btn-remove')) return;
        fileInput.click();
    });
    
    // File selected
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });
    
    // Remove file
    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeFile();
        });
    }
}

function handleFileSelect() {
    const fileInput = document.getElementById('resumeFile');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(pdf|docx|txt|doc)$/i)) {
        showNotification('Please upload a PDF, DOCX, or TXT file', 'error');
        removeFile();
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('File size must be less than 10MB', 'error');
        removeFile();
        return;
    }
    
    state.resumeFile = file;
    displayFilePreview(file);
    
    // Clear text input if file is uploaded
    const resumeText = document.getElementById('resumeText');
    if (resumeText) resumeText.value = '';
}

function displayFilePreview(file) {
    const uploadArea = document.getElementById('uploadArea');
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    const preview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    placeholder.style.display = 'none';
    preview.style.display = 'flex';
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
}

function removeFile() {
    const fileInput = document.getElementById('resumeFile');
    const uploadArea = document.getElementById('uploadArea');
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    const preview = document.getElementById('filePreview');
    
    fileInput.value = '';
    state.resumeFile = null;
    
    placeholder.style.display = 'block';
    preview.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ═══════════════════════════════════════════════════════════
// FORM SUBMISSION & ANALYSIS
// ═══════════════════════════════════════════════════════════

function initializeFormSubmission() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', handleAnalyze);
    }
    
    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', resetForm);
    }
    
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', downloadReport);
    }
}

async function handleAnalyze() {
    // Gather inputs
    const resumeText = document.getElementById('resumeText').value.trim();
    const jobDescription = document.getElementById('jobDescription').value.trim();
    const jobTitle = document.getElementById('jobTitle').value.trim();
    const jobCompany = document.getElementById('jobCompany').value.trim();
    
    // Validation
    if (!state.resumeFile && !resumeText) {
        showNotification('Please upload a resume or paste resume text', 'error');
        return;
    }
    
    if (!jobDescription || jobDescription.length < 50) {
        showNotification('Please enter a valid job description (minimum 50 characters)', 'error');
        return;
    }
    
    // Prepare form data
    const formData = new FormData();
    
    if (state.resumeFile) {
        formData.append('resume_file', state.resumeFile);
    } else {
        formData.append('resume_text', resumeText);
    }
    
    formData.append('job_description', jobDescription);
    formData.append('job_title', jobTitle);
    formData.append('job_company', jobCompany);
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        state.analysisResults = data;
        displayResults(data);
        showNotification('Analysis complete!', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(error.message || 'Analysis failed. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

function showLoading() {
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Scroll to loading
    document.getElementById('loadingState').scrollIntoView({
        behavior: 'smooth',
        block: 'center'
    });
}

function hideLoading() {
    document.getElementById('loadingState').style.display = 'none';
}

// ═══════════════════════════════════════════════════════════
// DISPLAY RESULTS
// ═══════════════════════════════════════════════════════════

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Display scores
    displayScoreCards(data);
    
    // Display ATS breakdown
    displayATSBreakdown(data.ats_scores);
    
    // Display AI insights if available
    if (data.ai_summary) {
        displayAIInsights(data);
    }
    
    // Display skills
    displaySkills(data);
    
    // Display keywords
    displayKeywords(data);
    
    // Display recommendations
    displayRecommendations(data.recommendations);
    
    // Display stats
    displayStats(data.stats);
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }, 100);
}

function displayScoreCards(data) {
    // Match Score
    const matchScore = document.getElementById('matchScore');
    const matchLevel = document.getElementById('matchLevel');
    const matchScoreBar = document.getElementById('matchScoreBar');
    
    matchScore.textContent = data.match_score + '%';
    matchLevel.textContent = data.level;
    matchScoreBar.style.width = data.match_score + '%';
    
    // ATS Score
    const atsScore = document.getElementById('atsScore');
    const atsScoreBar = document.getElementById('atsScoreBar');
    
    atsScore.textContent = data.ats_scores.overall + '%';
    atsScoreBar.style.width = data.ats_scores.overall + '%';
    
    // Skills Match
    const skillMatchScore = document.getElementById('skillMatchScore');
    const skillMatchBar = document.getElementById('skillMatchBar');
    
    skillMatchScore.textContent = data.ats_scores.skill_match + '%';
    skillMatchBar.style.width = data.ats_scores.skill_match + '%';
}

function displayATSBreakdown(atsScores) {
    const container = document.getElementById('atsMetrics');
    container.innerHTML = '';
    
    const metrics = [
        { label: 'Keyword Density', value: atsScores.keyword_density, weight: '30%' },
        { label: 'Skill Match', value: atsScores.skill_match, weight: '30%' },
        { label: 'Resume Length', value: atsScores.length, weight: '15%' },
        { label: 'Section Structure', value: atsScores.sections, weight: '15%' },
        { label: 'Action Verbs', value: atsScores.action_verbs, weight: '10%' },
    ];
    
    metrics.forEach(metric => {
        const metricEl = document.createElement('div');
        metricEl.className = 'metric-item';
        metricEl.innerHTML = `
            <div class="metric-label">${metric.label} (${metric.weight})</div>
            <div class="metric-value">${metric.value}%</div>
        `;
        container.appendChild(metricEl);
    });
}

function displayAIInsights(data) {
    const section = document.getElementById('aiInsightsSection');
    const summary = document.getElementById('aiSummary');
    const strengthsList = document.getElementById('strengthsList');
    const weaknessesList = document.getElementById('weaknessesList');
    
    section.style.display = 'block';
    summary.textContent = data.ai_summary || '';
    
    // Strengths
    strengthsList.innerHTML = '';
    (data.strengths || []).forEach(strength => {
        const li = document.createElement('li');
        li.textContent = strength;
        strengthsList.appendChild(li);
    });
    
    // Weaknesses
    weaknessesList.innerHTML = '';
    (data.weaknesses || []).forEach(weakness => {
        const li = document.createElement('li');
        li.textContent = weakness;
        weaknessesList.appendChild(li);
    });
}

function displaySkills(data) {
    const foundSkills = document.getElementById('foundSkills');
    const missingSkills = document.getElementById('missingSkills');
    
    foundSkills.innerHTML = '';
    missingSkills.innerHTML = '';
    
    // Found skills
    (data.resume_skills || []).slice(0, 20).forEach(skill => {
        const tag = document.createElement('span');
        tag.className = 'skill-tag';
        tag.textContent = skill;
        foundSkills.appendChild(tag);
    });
    
    // Missing skills
    (data.missing_skills || []).slice(0, 15).forEach(skill => {
        const tag = document.createElement('span');
        tag.className = 'skill-tag';
        tag.textContent = skill;
        missingSkills.appendChild(tag);
    });
}

function displayKeywords(data) {
    const matched = document.getElementById('matchedKeywords');
    const missing = document.getElementById('missingKeywords');
    
    matched.innerHTML = '';
    missing.innerHTML = '';
    
    // Matched keywords
    (data.matching_keywords || []).slice(0, 25).forEach(kw => {
        const tag = document.createElement('span');
        tag.className = 'keyword-tag';
        tag.textContent = kw;
        matched.appendChild(tag);
    });
    
    // Missing keywords
    (data.missing_keywords || []).slice(0, 25).forEach(kw => {
        const tag = document.createElement('span');
        tag.className = 'keyword-tag';
        tag.textContent = kw;
        missing.appendChild(tag);
    });
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsList');
    container.innerHTML = '';
    
    (recommendations || []).forEach(rec => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.textContent = rec;
        container.appendChild(item);
    });
}

function displayStats(stats) {
    const container = document.getElementById('statsGrid');
    container.innerHTML = '';
    
    const statItems = [
        { label: 'Word Count', value: stats.word_count },
        { label: 'Sentences', value: stats.sentence_count },
        { label: 'Avg Words/Sentence', value: stats.avg_words_per_sentence },
        { label: 'Reading Ease', value: stats.flesch_reading_ease },
        { label: 'Grade Level', value: stats.flesch_kincaid_grade },
        { label: 'Readability', value: stats.readability_level },
    ];
    
    statItems.forEach(stat => {
        const item = document.createElement('div');
        item.className = 'stat-item';
        item.innerHTML = `
            <div class="stat-item-label">${stat.label}</div>
            <div class="stat-item-value">${stat.value}</div>
        `;
        container.appendChild(item);
    });
}

// ═══════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════

function resetForm() {
    // Clear inputs
    document.getElementById('resumeText').value = '';
    document.getElementById('jobDescription').value = '';
    document.getElementById('jobTitle').value = '';
    document.getElementById('jobCompany').value = '';
    
    // Remove file
    removeFile();
    
    // Hide results
    document.getElementById('resultsSection').style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function downloadReport() {
    if (!state.analysisResults) {
        showNotification('No analysis results to download', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/download-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.analysisResults)
        });
        
        if (!response.ok) throw new Error('Failed to generate report');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ResumeIQ_Report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showNotification('Report downloaded successfully!', 'success');
        
    } catch (error) {
        console.error('Download error:', error);
        showNotification('Failed to download report', 'error');
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style it
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        borderRadius: '0.5rem',
        color: 'white',
        fontWeight: '600',
        zIndex: '9999',
        animation: 'slideInRight 0.3s ease',
        boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
    });
    
    // Set color based on type
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };
    notification.style.background = colors[type] || colors.info;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 4000);
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
