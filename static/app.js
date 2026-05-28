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
// ═══════════════════════════════════════════════════════════
// DISPLAY RESULTS
// ═══════════════════════════════════════════════════════════

let atsRadarInstance = null;

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    displayScoreCards(data);
    displayInfoBadges(data);
    displayTrackerCta(data);
    displayATSBreakdown(data.ats_scores);
    drawATSRadar(data.ats_scores);
    if (data.ai_summary) displayAIInsights(data);
    displaySkills(data);
    displayKeywords(data);
    displayRecommendations(data.recommendations);
    displayStats(data.stats);
    initCopyButtons(data);

    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function displayScoreCards(data) {
    document.getElementById('matchScore').textContent = data.match_score + '%';
    document.getElementById('matchLevel').textContent = data.level;
    document.getElementById('matchScoreBar').style.width = data.match_score + '%';
    document.getElementById('atsScore').textContent = data.ats_scores.overall + '%';
    document.getElementById('atsScoreBar').style.width = data.ats_scores.overall + '%';
    document.getElementById('skillMatchScore').textContent = data.ats_scores.skill_match + '%';
    document.getElementById('skillMatchBar').style.width = data.ats_scores.skill_match + '%';
}

function displayInfoBadges(data) {
    const container = document.getElementById('infoBadges');
    if (!container) return;

    const exp = data.experience_info || {};
    const sal = data.salary_estimate || {};
    const quant = data.quantification || {};
    const gaps = data.career_gaps || {};

    // Quantification class
    const quantClass = 'quant-' + (quant.rating || 'fair').toLowerCase().replace(/\s+/g, '-').replace('/', '').replace('no-metrics', 'no-metrics');

    container.innerHTML = `
        <div class="info-badge">
            <div class="badge-icon navy"><i class="fas fa-user-graduate"></i></div>
            <div>
                <div class="badge-label">Experience Level</div>
                <div class="badge-value">${exp.level || '—'}</div>
                <div class="badge-note">${exp.years_range || ''}</div>
            </div>
        </div>
        <div class="info-badge">
            <div class="badge-icon gold"><i class="fas fa-dollar-sign"></i></div>
            <div>
                <div class="badge-label">Salary Estimate</div>
                <div class="badge-value">${sal.range || '—'}</div>
                <div class="badge-note">${sal.currency || ''} · Informational</div>
            </div>
        </div>
        <div class="info-badge">
            <div class="badge-icon green"><i class="fas fa-hashtag"></i></div>
            <div>
                <div class="badge-label">Quantification</div>
                <div class="badge-value">
                    <span class="quant-pill ${quantClass}">${quant.rating || '—'}</span>
                </div>
                <div class="badge-note">${quant.total_metrics_found || 0} metric${quant.total_metrics_found !== 1 ? 's' : ''} found</div>
            </div>
        </div>
        <div class="info-badge">
            <div class="badge-icon ${gaps.has_potential_gaps ? 'warning' : 'green'}">
                <i class="fas fa-${gaps.has_potential_gaps ? 'exclamation-triangle' : 'check'}"></i>
            </div>
            <div>
                <div class="badge-label">Career Timeline</div>
                <div class="badge-value">${gaps.has_potential_gaps ? 'Gap Detected' : 'Continuous'}</div>
                <div class="badge-note">${gaps.career_span_years ? gaps.career_span_years + ' yr span' : 'No data'}</div>
            </div>
        </div>
    `;

    // Show gap alert if gaps found
    if (gaps.has_potential_gaps && gaps.advice) {
        const existing = document.getElementById('gapAlertBox');
        if (existing) existing.remove();
        const alert = document.createElement('div');
        alert.id = 'gapAlertBox';
        alert.className = 'gap-alert';
        alert.style.marginBottom = '20px';
        alert.innerHTML = `<i class="fas fa-exclamation-triangle"></i> <span><strong>Employment Gap Notice:</strong> ${escHtml(gaps.advice)}</span>`;
        container.insertAdjacentElement('afterend', alert);
    }
}

function displayTrackerCta(data) {
    const cta = document.getElementById('trackerCta');
    const btn = document.getElementById('addToTrackerBtn');
    if (!cta || !btn) return;

    const title = document.getElementById('jobTitle')?.value?.trim();
    const company = document.getElementById('jobCompany')?.value?.trim();

    if (title || company) {
        cta.style.display = 'flex';
        btn.onclick = () => {
            const job = {
                id: 'job_' + Date.now(),
                title: title || 'Untitled Role',
                company: company || 'Unknown Company',
                status: 'applied',
                date: new Date().toISOString().split('T')[0],
                notes: `Match: ${data.match_score}% · ATS: ${data.ats_scores?.overall}% · Added from ResumeIQ`,
                createdAt: Date.now()
            };
            const existing = JSON.parse(localStorage.getItem('resumeiq_jobs') || '[]');
            existing.push(job);
            localStorage.setItem('resumeiq_jobs', JSON.stringify(existing));
            showNotification('Added to Job Tracker!', 'success');
            btn.innerHTML = '<i class="fas fa-check"></i> Added!';
            btn.disabled = true;
            setTimeout(() => { btn.innerHTML = '<i class="fas fa-plus"></i> Add to Job Tracker'; btn.disabled = false; }, 3000);
        };
    } else {
        cta.style.display = 'none';
    }
}

function displayATSBreakdown(atsScores) {
    const container = document.getElementById('atsMetrics');
    container.innerHTML = '';

    const metrics = [
        { label: 'Keyword Density', value: atsScores.keyword_density, weight: '30%' },
        { label: 'Skill Match',     value: atsScores.skill_match,     weight: '30%' },
        { label: 'Resume Length',   value: atsScores.length,          weight: '15%' },
        { label: 'Section Structure', value: atsScores.sections,      weight: '15%' },
        { label: 'Action Verbs',    value: atsScores.action_verbs,    weight: '10%' },
    ];

    metrics.forEach(metric => {
        const el = document.createElement('div');
        el.className = 'metric-item';
        el.innerHTML = `
            <div class="metric-label">${metric.label}</div>
            <div class="metric-value">${metric.value}%</div>
            <div class="metric-bar"><div class="metric-bar-fill" style="width:${metric.value}%"></div></div>
            <div class="metric-weight">Weight: ${metric.weight}</div>
        `;
        container.appendChild(el);
    });
}

function drawATSRadar(atsScores) {
    const canvas = document.getElementById('atsRadarChart');
    if (!canvas) return;
    if (atsRadarInstance) { atsRadarInstance.destroy(); atsRadarInstance = null; }

    atsRadarInstance = new Chart(canvas, {
        type: 'radar',
        data: {
            labels: ['Keywords', 'Skills', 'Length', 'Structure', 'Action Verbs'],
            datasets: [{
                data: [
                    atsScores.keyword_density,
                    atsScores.skill_match,
                    atsScores.length,
                    atsScores.sections,
                    atsScores.action_verbs
                ],
                backgroundColor: 'rgba(184,146,42,0.15)',
                borderColor: '#b8922a',
                pointBackgroundColor: '#0d1b2a',
                pointBorderColor: '#b8922a',
                borderWidth: 2,
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    min: 0, max: 100,
                    ticks: { display: false, stepSize: 25 },
                    grid: { color: 'rgba(0,0,0,0.07)' },
                    pointLabels: { font: { size: 10, family: "'IBM Plex Sans', sans-serif" }, color: '#4a5568' }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function displayAIInsights(data) {
    const section = document.getElementById('aiInsightsSection');
    section.style.display = 'block';
    document.getElementById('aiSummary').textContent = data.ai_summary || '';

    const strengthsList = document.getElementById('strengthsList');
    strengthsList.innerHTML = '';
    (data.strengths || []).forEach(s => {
        const li = document.createElement('li'); li.textContent = s; strengthsList.appendChild(li);
    });

    const weaknessesList = document.getElementById('weaknessesList');
    weaknessesList.innerHTML = '';
    (data.weaknesses || []).forEach(w => {
        const li = document.createElement('li'); li.textContent = w; weaknessesList.appendChild(li);
    });

    const tipsSection = document.getElementById('interviewTipsSection');
    const tipsList = document.getElementById('interviewTipsList');
    const tips = data.interview_tips || [];
    if (tips.length > 0) {
        tipsSection.style.display = 'block';
        tipsList.innerHTML = '';
        tips.forEach(t => { const li = document.createElement('li'); li.textContent = t; tipsList.appendChild(li); });
    } else {
        tipsSection.style.display = 'none';
    }

    const prioritySection = document.getElementById('improvementPrioritySection');
    if (data.improvement_priority) {
        prioritySection.style.display = 'flex';
        document.getElementById('improvementPriorityText').textContent = data.improvement_priority;
    } else {
        prioritySection.style.display = 'none';
    }
}

function displaySkills(data) {
    const foundSkills = document.getElementById('foundSkills');
    const missingSkills = document.getElementById('missingSkills');
    foundSkills.innerHTML = '';
    missingSkills.innerHTML = '';

    (data.resume_skills || []).slice(0, 20).forEach(skill => {
        const tag = document.createElement('span');
        tag.className = 'skill-tag';
        tag.textContent = skill;
        foundSkills.appendChild(tag);
    });

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

    // Assign weight dots based on position (first = higher weight)
    const matchedList = (data.matching_keywords || []).slice(0, 25);
    matchedList.forEach((kw, i) => {
        const tag = document.createElement('span');
        tag.className = 'keyword-tag';
        const dotClass = i < matchedList.length * 0.33 ? 'kw-high' : i < matchedList.length * 0.66 ? 'kw-medium' : 'kw-low';
        tag.innerHTML = `<span class="kw-weight ${dotClass}"></span>${escHtml(kw)}`;
        matched.appendChild(tag);
    });

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
    (recommendations || []).forEach((rec, i) => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.innerHTML = `<div class="rec-number">${i + 1}</div><div>${escHtml(rec)}</div>`;
        container.appendChild(item);
    });
}

function displayStats(stats) {
    const container = document.getElementById('statsGrid');
    container.innerHTML = '';

    const statItems = [
        { label: 'Word Count',        value: stats.word_count,              sub: 'Ideal: 450–700' },
        { label: 'Sentences',          value: stats.sentence_count,          sub: '' },
        { label: 'Avg Words/Sentence', value: stats.avg_words_per_sentence,  sub: 'Ideal: 15–20' },
        { label: 'Reading Ease',       value: stats.flesch_reading_ease,     sub: 'Out of 100' },
        { label: 'Grade Level',        value: stats.flesch_kincaid_grade,    sub: 'Education level' },
        { label: 'Readability',        value: stats.readability_level,       sub: '' },
    ];

    statItems.forEach(stat => {
        const item = document.createElement('div');
        item.className = 'stat-item';
        item.innerHTML = `
            <div class="stat-item-label">${stat.label}</div>
            <div class="stat-item-value">${stat.value}</div>
            ${stat.sub ? `<div class="stat-item-sub">${stat.sub}</div>` : ''}
        `;
        container.appendChild(item);
    });
}

function initCopyButtons(data) {
    const copyMissing = document.getElementById('copyMissingSkills');
    if (copyMissing) {
        copyMissing.onclick = () => copyToClipboard(
            (data.missing_skills || []).join(', '),
            copyMissing, 'Skills copied!'
        );
    }
    const copyMissingKw = document.getElementById('copyMissingKw');
    if (copyMissingKw) {
        copyMissingKw.onclick = () => copyToClipboard(
            (data.missing_keywords || []).join(', '),
            copyMissingKw, 'Keywords copied!'
        );
    }
}

function copyToClipboard(text, btn, message) {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        const original = btn.innerHTML;
        btn.classList.add('copied');
        btn.innerHTML = '<i class="fas fa-check"></i> ' + message;
        setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = original; }, 2000);
    }).catch(() => {
        showNotification('Copy failed — please copy manually', 'error');
    });
}

function escHtml(s) {
    return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
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
    const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', warning: 'fa-exclamation-triangle', info: 'fa-info-circle' };
    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${escHtml(message)}</span>`;
    document.body.appendChild(notif);
    setTimeout(() => {
        notif.style.opacity = '0';
        notif.style.transform = 'translateY(8px)';
        notif.style.transition = 'all 0.3s ease';
        setTimeout(() => notif.remove(), 320);
    }, 3800);
}
