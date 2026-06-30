/* ═══════════════════════════════════════════════════
   ResumeIQ Pro v3.1 — Complete Frontend Application
═══════════════════════════════════════════════════ */

const state = {
    currentUser: null,
    analysisResults: null,
    resumeFile: null,
    resumeText: '',
    jobs: [],
};

/* ── BOOT ─────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    checkUserSession();
    initFileUpload();
    initUserMenu();
    initTheme();

    const page = document.body.dataset.page;
    if (page === 'index')   initAnalyzer();
    if (page === 'compare') initCompare();
    if (page === 'history') initHistory();
    if (page === 'profile') initProfile();
    if (page === 'jobs')    initJobs();
    if (page === 'login')   initLogin();
    if (page === 'signup')  initSignup();
});

/* ── THEME ────────────────────────────────────────── */
function initTheme() {
    const btn  = document.getElementById('themeToggle');
    const html = document.documentElement;
    const saved = localStorage.getItem('riq_theme') || 'light';
    html.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
    btn?.addEventListener('click', () => {
        const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', next);
        localStorage.setItem('riq_theme', next);
        updateThemeIcon(next);
    });
}
function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    if (btn) btn.innerHTML = theme === 'dark' ? '☀️' : '🌙';
}

/* ── SESSION ──────────────────────────────────────── */
async function checkUserSession() {
    try {
        const r = await fetch('/api/me');
        const d = await r.json();
        if (d.logged_in) {
            state.currentUser = d.user;
            updateNavForUser(d.user);
        } else {
            updateNavForGuest();
        }
    } catch { updateNavForGuest(); }
}

function updateNavForUser(user) {
    document.getElementById('loginLink')?.style.setProperty('display','none');
    document.getElementById('userMenu')?.style.setProperty('display','block');
    const un = document.getElementById('userName');
    if (un) un.textContent = user.name.split(' ')[0];
}
function updateNavForGuest() {
    document.getElementById('loginLink')?.style.setProperty('display','inline-flex');
    document.getElementById('userMenu')?.style.setProperty('display','none');
}

function initUserMenu() {
    const btn = document.getElementById('userMenuButton');
    const dd  = document.getElementById('userDropdown');
    btn?.addEventListener('click', e => { e.stopPropagation(); dd?.classList.toggle('show'); });
    document.addEventListener('click', () => dd?.classList.remove('show'));
    document.getElementById('logoutBtn')?.addEventListener('click', async e => {
        e.preventDefault();
        await fetch('/api/logout', { method: 'POST' });
        window.location.href = '/';
    });
}

/* ── NOTIFY ───────────────────────────────────────── */
function notify(msg, type = 'info', duration = 4500) {
    const icons = { success:'fa-check-circle', error:'fa-exclamation-circle', info:'fa-info-circle', warning:'fa-exclamation-triangle' };
    const el = document.createElement('div');
    el.className = `notification ${type}`;
    el.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${msg}</span>`;
    document.body.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 400); }, duration);
}

/* ── FILE UPLOAD ──────────────────────────────────── */
function initFileUpload() {
    const area   = document.getElementById('uploadArea');
    const input  = document.getElementById('resumeFile');
    const remove = document.getElementById('removeFileBtn');
    if (!area || !input) return;

    area.addEventListener('click', e => { if (!e.target.closest('.btn-remove')) input.click(); });
    input.addEventListener('change', () => handleFile(input.files[0]));
    area.addEventListener('dragover',  e => { e.preventDefault(); area.classList.add('drag-over'); });
    area.addEventListener('dragleave', ()  => area.classList.remove('drag-over'));
    area.addEventListener('drop', e => {
        e.preventDefault(); area.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
    remove?.addEventListener('click', e => { e.stopPropagation(); clearFile(); });
}

function handleFile(file) {
    if (!file) return;
    const ok = /\.(pdf|docx|doc|txt)$/i.test(file.name);
    if (!ok) { notify('Please upload PDF, DOCX, or TXT', 'error'); return; }
    if (file.size > 10 * 1024 * 1024) { notify('File must be under 10 MB', 'error'); return; }
    state.resumeFile = file;

    const ph = document.querySelector('.upload-placeholder');
    const pv = document.getElementById('filePreview');
    const fn = document.getElementById('fileName');
    const fs = document.getElementById('fileSize');
    if (ph) ph.style.display = 'none';
    if (pv) pv.style.display = 'flex';
    if (fn) fn.textContent = file.name;
    if (fs) fs.textContent = fmtSize(file.size);
    document.getElementById('resumeText')?.setAttribute('disabled', true);
}

function clearFile() {
    const input = document.getElementById('resumeFile');
    const area  = document.getElementById('uploadArea');
    const ph    = area?.querySelector('.upload-placeholder');
    const pv    = document.getElementById('filePreview');
    if (input) input.value = '';
    state.resumeFile = null;
    if (ph) ph.style.display = 'block';
    if (pv) pv.style.display = 'none';
    document.getElementById('resumeText')?.removeAttribute('disabled');
}

function fmtSize(b) {
    if (b < 1024)       return b + ' B';
    if (b < 1024*1024)  return (b/1024).toFixed(1) + ' KB';
    return (b/(1024*1024)).toFixed(1) + ' MB';
}

/* ── TABS ─────────────────────────────────────────── */
function initTabs(containerSel) {
    const container = document.querySelector(containerSel);
    if (!container) return;
    container.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            container.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const pane = container.querySelector(`#${btn.dataset.tab}`);
            if (pane) pane.classList.add('active');
        });
    });
}

/* ── MODAL ────────────────────────────────────────── */
function openModal(id) { document.getElementById(id)?.classList.add('open'); }
function closeModal(id) { document.getElementById(id)?.classList.remove('open'); }
document.addEventListener('click', e => {
    if (e.target.classList.contains('modal-overlay')) e.target.classList.remove('open');
    if (e.target.classList.contains('modal-close')) e.target.closest('.modal-overlay')?.classList.remove('open');
});

/* ─────────────────────────────────────────────────────────
   ANALYZER (index.html)
───────────────────────────────────────────────────────── */
function initAnalyzer() {
    document.getElementById('analyzeForm')?.addEventListener('submit', submitAnalysis);

    // Tab on textarea vs file
    document.getElementById('resumeText')?.addEventListener('input', e => {
        state.resumeText = e.target.value;
    });
}

async function submitAnalysis(e) {
    e.preventDefault();
    const jd = document.getElementById('jobDescription')?.value.trim();
    if (!jd || jd.length < 50) { notify('Job description must be at least 50 characters', 'error'); return; }
    if (!state.resumeFile && !state.resumeText.trim()) {
        notify('Please upload a resume or paste resume text', 'error'); return;
    }

    const fd = new FormData();
    if (state.resumeFile) {
        fd.append('resume_file', state.resumeFile);
    } else {
        fd.append('resume_text', state.resumeText);
    }
    fd.append('job_description', jd);
    fd.append('job_title',   document.getElementById('jobTitle')?.value.trim() || '');
    fd.append('job_company', document.getElementById('jobCompany')?.value.trim() || '');

    showLoading(true);
    try {
        const r    = await fetch('/api/analyze', { method: 'POST', body: fd });
        const data = await r.json();
        if (!r.ok) { notify(data.error || 'Analysis failed', 'error'); return; }
        state.analysisResults = data;
        renderResults(data);
        document.getElementById('resultsSection')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch { notify('Network error. Please try again.', 'error'); }
    finally   { showLoading(false); }
}

function showLoading(on) {
    const btn  = document.getElementById('analyzeBtn');
    const spin = document.getElementById('loadingSpinner');
    if (btn)  { btn.disabled = on; btn.querySelector('.btn-text').textContent = on ? 'Analyzing…' : 'Analyze Resume'; }
    if (spin) spin.style.display = on ? 'flex' : 'none';
}

function renderResults(d) {
    const sec = document.getElementById('resultsSection');
    if (!sec) return;
    sec.style.display = 'block';

    // Score ring
    const pct = d.match_score;
    renderScoreRing('matchRing', pct, d.color, `${pct}%`, d.level);

    // ATS ring
    const ats = d.ats_scores?.overall || 0;
    renderScoreRing('atsRing', ats, atsColor(ats), `${ats}%`, 'ATS Score');

    // ATS breakdown bars
    renderAtsBars(d.ats_scores);

    // Skills
    renderTagList('resumeSkillsList', d.resume_skills || [], 'skill-tag');
    renderTagList('missingSkillsList', d.missing_skills || [], 'skill-tag missing');

    // Keywords
    renderTagList('matchingKwList', d.matching_keywords || [], 'keyword-tag matched');
    renderTagList('missingKwList',  d.missing_keywords  || [], 'keyword-tag missing');

    // Stats
    const st = d.stats || {};
    setText('wordCount',  st.word_count || 0);
    setText('sentCount',  st.sentence_count || 0);
    setText('readScore',  st.flesch_reading_ease || 0);
    setText('readLevel',  st.readability_level || '—');
    setText('procTime',   `${d.processing_time_ms || 0}ms`);

    // Recommendations
    renderRecommendations(d.recommendations || []);

    // Insights row
    renderInsights(d);

    // Quantification
    renderQuant(d.quantification);

    // Career gaps
    renderGaps(d.career_gaps);

    // AI section
    renderAI(d);

    // Show tabs
    initTabs('#resultsTabs');
    document.querySelector('#resultsTabs .tab-btn')?.click();
}

function renderScoreRing(id, pct, color, label, sub) {
    const wrap = document.getElementById(id);
    if (!wrap) return;
    const r   = 52;
    const circ = 2 * Math.PI * r;
    const offset = circ - (pct / 100) * circ;
    wrap.innerHTML = `
      <svg class="score-ring" viewBox="0 0 130 130">
        <circle class="ring-track" cx="65" cy="65" r="${r}"/>
        <circle class="ring-fill" cx="65" cy="65" r="${r}"
          stroke="${color}" stroke-dasharray="${circ}" stroke-dashoffset="${circ}"
          id="${id}_fill"/>
        <text class="ring-label" x="65" y="68" text-anchor="middle">${label}</text>
        <text class="ring-sub"   x="65" y="82" text-anchor="middle">${sub}</text>
      </svg>`;
    setTimeout(() => {
        const el = document.getElementById(`${id}_fill`);
        if (el) el.style.strokeDashoffset = offset;
    }, 100);
}

function renderAtsBars(ats) {
    const container = document.getElementById('atsBars');
    if (!container || !ats) return;
    const items = [
        { label: 'Keyword Density', key: 'keyword_density', weight: '30%' },
        { label: 'Skill Match',     key: 'skill_match',     weight: '30%' },
        { label: 'Resume Length',   key: 'length',           weight: '15%' },
        { label: 'Sections',        key: 'sections',         weight: '15%' },
        { label: 'Action Verbs',    key: 'action_verbs',     weight: '10%' },
    ];
    container.innerHTML = items.map(it => {
        const v = ats[it.key] || 0;
        const c = v >= 70 ? '#16a34a' : v >= 40 ? '#d97706' : '#dc2626';
        return `<div class="quant-row">
          <span style="width:140px;font-size:0.8rem;font-weight:600;color:var(--slate)">${it.label} <small style="color:var(--slate-light)">${it.weight}</small></span>
          <div class="quant-bar-wrap"><div class="quant-bar-fill" style="width:0%;background:${c}" data-w="${v}"></div></div>
          <span class="quant-label">${v}%</span>
        </div>`;
    }).join('');
    setTimeout(() => {
        container.querySelectorAll('.quant-bar-fill').forEach(el => {
            el.style.width = el.dataset.w + '%';
        });
    }, 100);
}

function renderTagList(id, items, cls) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!items.length) { el.innerHTML = '<span style="color:var(--slate-light);font-size:0.85rem;">None found</span>'; return; }
    el.innerHTML = items.map(t => `<span class="skill-tag ${cls}">${t}</span>`).join('');
}

function renderRecommendations(recs) {
    const el = document.getElementById('recsList');
    if (!el) return;
    el.innerHTML = recs.map((r, i) => `
      <div class="recommendation-item">
        <span class="rec-num">${i + 1}</span>
        <span>${r}</span>
      </div>`).join('');
}

function renderInsights(d) {
    const el = document.getElementById('insightsRow');
    if (!el) return;
    const exp = d.experience_info || {};
    const sal = d.salary_estimate || {};
    el.innerHTML = `
      <div class="insight-card">
        <div class="ic-label">Experience Level</div>
        <div class="ic-value" style="color:${exp.color || 'var(--navy)'}">${exp.level || '—'}</div>
        <div class="ic-sub">${exp.years_range || ''}</div>
      </div>
      <div class="insight-card">
        <div class="ic-label">Salary Estimate</div>
        <div class="ic-value">${sal.range || '—'}</div>
        <div class="ic-sub">${sal.note ? sal.note.split('.')[0] + '.' : ''}</div>
      </div>
      <div class="insight-card">
        <div class="ic-label">Word Count</div>
        <div class="ic-value">${d.stats?.word_count || 0}</div>
        <div class="ic-sub">Optimal: 300–1,200</div>
      </div>
      <div class="insight-card">
        <div class="ic-label">Skills Found</div>
        <div class="ic-value">${(d.resume_skills || []).length}</div>
        <div class="ic-sub">Job needs: ${(d.job_skills || []).length}</div>
      </div>`;
}

function renderQuant(q) {
    const el = document.getElementById('quantRow');
    if (!el || !q) return;
    const c = q.score >= 60 ? '#16a34a' : q.score >= 30 ? '#d97706' : '#dc2626';
    el.innerHTML = `
      <div class="quant-row">
        <span style="width:120px;font-size:0.85rem;font-weight:600;">Metrics Score</span>
        <div class="quant-bar-wrap"><div class="quant-bar-fill" style="width:0%;background:${c}" data-w="${q.score}"></div></div>
        <span class="quant-label">${q.score}% · ${q.rating}</span>
      </div>
      <p style="font-size:0.8125rem;color:var(--slate);margin-top:0.5rem;">
        Found <strong>${q.total_metrics_found}</strong> quantified metrics (%, $, multipliers, team sizes).
        ${q.score < 60 ? 'Add measurable achievements like "Increased revenue by 35%" or "Led team of 8".' : 'Great use of quantified results!'}
      </p>`;
    setTimeout(() => {
        el.querySelectorAll('.quant-bar-fill').forEach(b => b.style.width = b.dataset.w + '%');
    }, 100);
}

function renderGaps(gaps) {
    const el = document.getElementById('gapInfo');
    if (!el || !gaps) return;
    if (!gaps.has_potential_gaps) {
        el.innerHTML = `<p style="color:#16a34a;font-size:0.875rem;">✓ ${gaps.advice}</p>`;
        return;
    }
    const gList = (gaps.gaps_found || []).map(g =>
        `<li style="font-size:0.8125rem;">Gap ~${g.duration_years} yr: ${g.from}–${g.to}</li>`
    ).join('');
    el.innerHTML = `<p style="color:var(--warning);font-size:0.875rem;margin-bottom:0.5rem;">⚠ ${gaps.advice}</p><ul>${gList}</ul>`;
}

function renderAI(d) {
    const sec = document.getElementById('aiSection');
    if (!sec) return;

    if (!d.ai_summary) {
        sec.innerHTML = `<div class="ai-badge" style="margin-bottom:1rem;"><i class="fas fa-robot"></i> AI Insights</div>
          <p style="color:var(--slate-light);font-size:0.875rem;">Add ANTHROPIC_API_KEY to your .env file to enable AI-powered insights, cover letter generation, and personalised recommendations.</p>`;
        return;
    }

    const priority = d.improvement_priority ? `
      <div class="priority-banner">
        <span class="priority-icon">🎯</span>
        <div><div class="priority-label">Top Priority</div>
        <div class="priority-text">${d.improvement_priority}</div></div>
      </div>` : '';

    const strengths = (d.strengths || []).map(s => `<li>${s}</li>`).join('');
    const weaknesses = (d.weaknesses || []).map(w => `<li>${w}</li>`).join('');
    const tips = (d.interview_tips || []).map(t => `<li>${t}</li>`).join('');
    const recs = (d.ai_recommendations || []).map(r => `<li>${r}</li>`).join('');

    sec.innerHTML = `
      <div class="ai-badge" style="margin-bottom:1rem;"><i class="fas fa-robot"></i> AI-Powered Insights (Claude)</div>
      ${priority}
      <p style="font-size:0.9375rem;line-height:1.7;color:var(--navy);margin-bottom:1.2rem;">${d.ai_summary}</p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">
        <div>
          <h4 style="color:#16a34a;margin-bottom:0.5rem;font-size:0.875rem;">✓ Strengths</h4>
          <ul style="list-style:disc;padding-left:1.2rem;font-size:0.875rem;color:var(--slate);display:flex;flex-direction:column;gap:4px;">${strengths}</ul>
        </div>
        <div>
          <h4 style="color:#dc2626;margin-bottom:0.5rem;font-size:0.875rem;">✗ Gaps</h4>
          <ul style="list-style:disc;padding-left:1.2rem;font-size:0.875rem;color:var(--slate);display:flex;flex-direction:column;gap:4px;">${weaknesses}</ul>
        </div>
      </div>
      ${recs ? `<h4 style="margin-bottom:0.5rem;font-size:0.875rem;">📋 AI Recommendations</h4>
        <ul style="list-style:disc;padding-left:1.2rem;font-size:0.875rem;color:var(--slate);display:flex;flex-direction:column;gap:4px;margin-bottom:1rem;">${recs}</ul>` : ''}
      ${tips ? `<h4 style="margin-bottom:0.5rem;font-size:0.875rem;">🎤 Interview Prep</h4>
        <ul style="list-style:disc;padding-left:1.2rem;font-size:0.875rem;color:var(--slate);display:flex;flex-direction:column;gap:4px;">${tips}</ul>` : ''}
      <div style="margin-top:1.5rem;display:flex;gap:0.75rem;flex-wrap:wrap;">
        <button class="btn btn-primary btn-sm" onclick="openCoverLetterModal()"><i class="fas fa-file-alt"></i> Generate Cover Letter</button>
        <button class="btn btn-outline btn-sm" onclick="openImproveModal()"><i class="fas fa-magic"></i> Improve Resume</button>
        <button class="btn btn-outline btn-sm" onclick="downloadReport()"><i class="fas fa-download"></i> Download PDF</button>
      </div>`;
}

/* Cover Letter Modal */
function openCoverLetterModal() {
    const rt = state.resumeFile ? null : state.resumeText;
    if (!state.analysisResults) { notify('Run an analysis first', 'warning'); return; }
    openModal('coverLetterModal');
}

async function generateCoverLetter() {
    const tone = document.getElementById('clTone')?.value || 'professional';
    const rt   = state.resumeText || '[resume from uploaded file]';
    const jd   = document.getElementById('jobDescription')?.value || '';
    const out  = document.getElementById('coverLetterOutput');
    if (out) out.innerHTML = '<div class="ai-thinking"><div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div> Generating…</div>';
    try {
        const r = await fetch('/api/ai/cover-letter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resume_text: rt, job_description: jd, tone }),
        });
        const d = await r.json();
        if (!r.ok) { notify(d.error || 'Failed', 'error'); if(out) out.innerHTML=''; return; }
        if (out) out.innerHTML = `<div class="cover-letter-output">${d.cover_letter}</div>
          <button class="btn btn-outline btn-sm" style="margin-top:0.75rem;" onclick="copyText('${escJs(d.cover_letter)}')"><i class="fas fa-copy"></i> Copy</button>`;
    } catch { notify('Network error', 'error'); }
}

function openImproveModal() { openModal('improveModal'); }

async function getImprovements() {
    const focus = document.getElementById('improveFocus')?.value || 'overall';
    const rt    = state.resumeText || '[resume from uploaded file]';
    const out   = document.getElementById('improveOutput');
    if (out) out.innerHTML = '<div class="ai-thinking"><div class="ai-dot"></div><div class="ai-dot"></div><div class="ai-dot"></div> Thinking…</div>';
    try {
        const r = await fetch('/api/ai/improve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resume_text: rt, focus_area: focus }),
        });
        const d = await r.json();
        if (!r.ok) { notify(d.error || 'Failed', 'error'); return; }
        if (out) out.innerHTML = `<div style="white-space:pre-wrap;font-size:0.9rem;line-height:1.7;">${d.suggestions}</div>`;
    } catch { notify('Network error', 'error'); }
}

async function downloadReport() {
    if (!state.analysisResults) { notify('Run an analysis first', 'warning'); return; }
    notify('Generating PDF report…', 'info');
    try {
        const r = await fetch('/api/download-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.analysisResults),
        });
        if (!r.ok) { notify('Failed to generate report', 'error'); return; }
        const blob = await r.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = 'ResumeIQ_Report.pdf'; a.click();
        URL.revokeObjectURL(url);
    } catch { notify('Network error', 'error'); }
}

/* ─────────────────────────────────────────────────────────
   COMPARE (compare.html)
───────────────────────────────────────────────────────── */
function initCompare() {
    initFileUploadFor('uploadAreaA', 'resumeAFile', 'filePreviewA', 'fileNameA', 'fileSizeA', 'removeBtnA');
    initFileUploadFor('uploadAreaB', 'resumeBFile', 'filePreviewB', 'fileNameB', 'fileSizeB', 'removeBtnB');
    document.getElementById('compareForm')?.addEventListener('submit', submitCompare);
}

function initFileUploadFor(areaId, inputId, prevId, nameId, sizeId, removeId) {
    const area  = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    if (!area || !input) return;
    area.addEventListener('click', e => { if (!e.target.closest('.btn-remove')) input.click(); });
    input.addEventListener('change', () => {
        const f = input.files[0];
        if (!f) return;
        document.getElementById(prevId).style.display = 'flex';
        area.querySelector('.upload-placeholder').style.display = 'none';
        document.getElementById(nameId).textContent = f.name;
        document.getElementById(sizeId).textContent = fmtSize(f.size);
    });
    area.addEventListener('dragover', e => { e.preventDefault(); area.classList.add('drag-over'); });
    area.addEventListener('dragleave', () => area.classList.remove('drag-over'));
    area.addEventListener('drop', e => {
        e.preventDefault(); area.classList.remove('drag-over');
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event('change'));
    });
    document.getElementById(removeId)?.addEventListener('click', e => {
        e.stopPropagation(); input.value = '';
        document.getElementById(prevId).style.display = 'none';
        area.querySelector('.upload-placeholder').style.display = 'block';
    });
}

async function submitCompare(e) {
    e.preventDefault();
    const jd  = document.getElementById('compareJobDesc')?.value.trim();
    const raF = document.getElementById('resumeAFile');
    const rbF = document.getElementById('resumeBFile');
    const raT = document.getElementById('resumeAText')?.value.trim();
    const rbT = document.getElementById('resumeBText')?.value.trim();
    if (!jd) { notify('Job description required', 'error'); return; }
    if ((!raF?.files[0] && !raT) || (!rbF?.files[0] && !rbT)) {
        notify('Both resumes required', 'error'); return;
    }
    const fd = new FormData();
    fd.append('job_description', jd);
    if (raF?.files[0]) fd.append('resume_a_file', raF.files[0]); else fd.append('resume_a', raT);
    if (rbF?.files[0]) fd.append('resume_b_file', rbF.files[0]); else fd.append('resume_b', rbT);

    const btn = document.getElementById('compareBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Comparing…'; }
    try {
        const r = await fetch('/api/compare', { method: 'POST', body: fd });
        const d = await r.json();
        if (!r.ok) { notify(d.error || 'Failed', 'error'); return; }
        renderCompare(d);
        document.getElementById('compareResults')?.scrollIntoView({ behavior: 'smooth' });
    } catch { notify('Network error', 'error'); }
    finally { if (btn) { btn.disabled = false; btn.textContent = 'Compare Resumes'; } }
}

function renderCompare(d) {
    const sec = document.getElementById('compareResults');
    if (!sec) return;
    sec.style.display = 'block';

    const winner = d.winner === 'A' ? d.resume_a_name : d.resume_b_name;
    const labelA = d.resume_a_name || 'Resume A';
    const labelB = d.resume_b_name || 'Resume B';
    const ra = d.resume_a, rb = d.resume_b;
    const winA = d.winner === 'A';

    sec.innerHTML = `
      <div class="results-card">
        <h3 style="margin-bottom:1.25rem;">Comparison Results</h3>
        <div style="padding:1rem 1.5rem;background:linear-gradient(135deg,rgba(184,146,42,0.1),rgba(13,27,42,0.05));border:1px solid rgba(184,146,42,0.3);border-radius:var(--radius-lg);text-align:center;margin-bottom:1.5rem;">
          <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--slate-light);margin-bottom:6px;">Winner 🏆</div>
          <div style="font-size:1.3rem;font-weight:700;color:var(--navy)">${winner}</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
          ${renderCompareCard(labelA, ra, winA)}
          ${renderCompareCard(labelB, rb, !winA)}
        </div>
      </div>`;
}

function renderCompareCard(label, r, isWinner) {
    const skills = (r.skills || []).slice(0, 10).map(s => `<span class="skill-tag">${s}</span>`).join('');
    return `
      <div style="border:${isWinner ? '2px solid var(--gold)' : '1px solid var(--border)'};border-radius:var(--radius-lg);padding:1.25rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
          <span style="font-weight:700;">${label}</span>
          ${isWinner ? '<span style="font-size:0.75rem;font-weight:700;background:var(--gold-pale);color:var(--gold);padding:3px 10px;border-radius:100px;">WINNER</span>' : ''}
        </div>
        <div class="compare-metric"><span>Match Score</span><strong>${r.match_score}%</strong></div>
        <div class="compare-metric"><span>ATS Score</span><strong>${r.ats_score}%</strong></div>
        <div class="compare-metric"><span>Level</span><strong>${r.level}</strong></div>
        <div style="margin-top:0.75rem;"><div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">Top Skills</div>
          <div style="display:flex;flex-wrap:wrap;gap:4px;">${skills || '<span style="color:var(--slate-light);font-size:0.8rem;">None detected</span>'}</div>
        </div>
      </div>`;
}

/* ─────────────────────────────────────────────────────────
   HISTORY (history.html)
───────────────────────────────────────────────────────── */
function initHistory() {
    loadHistory();
}

async function loadHistory(page = 1) {
    const list = document.getElementById('historyList');
    if (!list) return;
    list.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--slate-light);">Loading…</div>';
    try {
        const r = await fetch(`/api/history?page=${page}`);
        if (r.status === 401) {
            list.innerHTML = '<div style="text-align:center;padding:2rem;"><p>Please <a href="/login-page">sign in</a> to view your history.</p></div>';
            return;
        }
        const d = await r.json();
        if (!d.analyses?.length) {
            list.innerHTML = '<div style="text-align:center;padding:3rem;color:var(--slate-light);"><i class="fas fa-history" style="font-size:2rem;display:block;margin-bottom:1rem;"></i><p>No analyses yet. <a href="/">Analyse your first resume!</a></p></div>';
            return;
        }
        list.innerHTML = d.analyses.map(a => renderHistoryItem(a)).join('');
        renderPagination(d.pages, page);
    } catch { list.innerHTML = '<p style="text-align:center;color:var(--danger);">Failed to load history</p>'; }
}

function renderHistoryItem(a) {
    const score = a.match_score;
    const cls   = score >= 70 ? 'success' : score >= 45 ? 'warning' : 'danger';
    const date  = new Date(a.date).toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric' });
    return `
      <div class="history-item" id="hist-${a.id}">
        <div class="history-item-main">
          <div class="history-score badge-${cls}">${score}%</div>
          <div class="history-info">
            <div class="history-title">${a.job_title || 'Untitled Role'} ${a.job_company ? `@ ${a.job_company}` : ''}</div>
            <div class="history-meta">
              <span><i class="fas fa-calendar-alt"></i> ${date}</span>
              <span><i class="fas fa-robot"></i> ATS: ${a.ats_score}%</span>
              <span class="level-badge level-${(a.level||'').toLowerCase()}">${a.level || '—'}</span>
            </div>
          </div>
        </div>
        <div class="history-actions">
          <button class="btn btn-outline btn-sm" onclick="viewAnalysis(${a.id})"><i class="fas fa-eye"></i> View</button>
          <button class="btn btn-danger btn-sm" onclick="deleteHistoryItem(${a.id})"><i class="fas fa-trash"></i></button>
        </div>
      </div>`;
}

async function viewAnalysis(id) {
    const r = await fetch(`/api/history/${id}`);
    const d = await r.json();
    if (!r.ok) { notify('Failed to load', 'error'); return; }

    const el = document.getElementById('analysisDetail');
    if (!el) return;
    const recs = (d.recommendations || []).map((r,i) => `<div class="recommendation-item"><span class="rec-num">${i+1}</span>${r}</div>`).join('');
    const matchSkills = (d.resume_skills || []).map(s => `<span class="skill-tag">${s}</span>`).join('');
    const misSkills   = (d.missing_skills|| []).map(s => `<span class="skill-tag missing">${s}</span>`).join('');

    el.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem;">
        <div class="insight-card"><div class="ic-label">Match Score</div><div class="ic-value" style="color:${getScoreColor(d.match_score)}">${d.match_score}%</div></div>
        <div class="insight-card"><div class="ic-label">ATS Score</div><div class="ic-value">${d.ats_score}%</div></div>
        <div class="insight-card"><div class="ic-label">Level</div><div class="ic-value">${d.level}</div></div>
        <div class="insight-card"><div class="ic-label">Words</div><div class="ic-value">${d.stats?.word_count || '—'}</div></div>
      </div>
      <h4 style="margin-bottom:0.5rem;">Skills Found</h4><div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:1rem;">${matchSkills || '—'}</div>
      <h4 style="margin-bottom:0.5rem;">Missing Skills</h4><div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:1rem;">${misSkills || '—'}</div>
      <h4 style="margin-bottom:0.75rem;">Recommendations</h4>${recs}
      ${d.ai_summary ? `<h4 style="margin-top:1rem;margin-bottom:0.5rem;">AI Summary</h4><p style="font-size:0.875rem;line-height:1.7;">${d.ai_summary}</p>` : ''}`;
    openModal('analysisModal');
}

async function deleteHistoryItem(id) {
    if (!confirm('Delete this analysis?')) return;
    const r = await fetch(`/api/history/${id}`, { method: 'DELETE' });
    if (r.ok) { document.getElementById(`hist-${id}`)?.remove(); notify('Deleted', 'success'); }
    else notify('Failed to delete', 'error');
}

async function exportHistory() {
    notify('Exporting CSV…', 'info');
    const r = await fetch('/api/export/history');
    if (!r.ok) { notify('Export failed', 'error'); return; }
    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'resumeiq_history.csv'; a.click();
    URL.revokeObjectURL(url);
}

function renderPagination(pages, current) {
    const el = document.getElementById('pagination');
    if (!el || pages <= 1) { if (el) el.innerHTML = ''; return; }
    el.innerHTML = Array.from({length: pages}, (_,i) => i+1).map(p =>
        `<button class="btn btn-sm ${p===current?'btn-primary':'btn-outline'}" onclick="loadHistory(${p})">${p}</button>`
    ).join('');
}

/* ─────────────────────────────────────────────────────────
   PROFILE (profile.html)
───────────────────────────────────────────────────────── */
function initProfile() {
    loadProfile();
    document.getElementById('profileForm')?.addEventListener('submit', saveProfile);
    document.getElementById('passwordForm')?.addEventListener('submit', savePassword);
    document.getElementById('deleteAccountBtn')?.addEventListener('click', deleteAccount);
    initTabs('#profileTabs');
    document.querySelector('#profileTabs .tab-btn')?.click();
}

async function loadProfile() {
    const r = await fetch('/api/me');
    const d = await r.json();
    if (!d.logged_in) { window.location.href = '/login-page'; return; }
    const u = d.user;
    setVal('profileName',    u.name    || '');
    setVal('profileEmail',   u.email   || '');
    setVal('profileCompany', u.company || '');
    setVal('profileRole',    u.role    || '');
    document.getElementById('profileAvatar').textContent = u.name?.[0]?.toUpperCase() || '?';
    document.getElementById('profileNameDisplay').textContent = u.name || '';
    document.getElementById('profileEmailDisplay').textContent = u.email || '';

    const r2 = await fetch('/api/stats');
    if (r2.ok) {
        const s = await r2.json();
        setText('statTotal',  s.total_analyses);
        setText('statAvg',    s.avg_match_score + '%');
        setText('statAts',    s.avg_ats_score + '%');
        setText('statBest',   s.best_score + '%');
    }
}

async function saveProfile(e) {
    e.preventDefault();
    const r = await fetch('/api/profile', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name:    document.getElementById('profileName').value.trim(),
            company: document.getElementById('profileCompany').value.trim(),
            role:    document.getElementById('profileRole').value.trim(),
        }),
    });
    const d = await r.json();
    if (r.ok) { notify('Profile saved!', 'success'); loadProfile(); }
    else notify(d.error || 'Failed', 'error');
}

async function savePassword(e) {
    e.preventDefault();
    const newP = document.getElementById('newPassword').value;
    const conf = document.getElementById('confirmPassword').value;
    if (newP !== conf) { notify('Passwords do not match', 'error'); return; }
    const r = await fetch('/api/profile/password', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            current_password: document.getElementById('currentPassword').value,
            new_password: newP,
        }),
    });
    const d = await r.json();
    if (r.ok) { notify('Password changed!', 'success'); e.target.reset(); }
    else notify(d.error || 'Failed', 'error');
}

async function deleteAccount() {
    if (!confirm('Delete your account? This cannot be undone.')) return;
    const r = await fetch('/api/profile', { method: 'DELETE' });
    if (r.ok) window.location.href = '/';
    else notify('Failed to delete account', 'error');
}

/* ─────────────────────────────────────────────────────────
   JOB TRACKER (jobs.html) — fully DB-backed
───────────────────────────────────────────────────────── */
const JOB_STATUSES = [
    { key:'saved',        label:'Saved',        emoji:'🔖' },
    { key:'applied',      label:'Applied',       emoji:'📨' },
    { key:'phone_screen', label:'Phone Screen',  emoji:'📞' },
    { key:'interview',    label:'Interview',     emoji:'💼' },
    { key:'offer',        label:'Offer',         emoji:'🎉' },
    { key:'rejected',     label:'Rejected',      emoji:'❌' },
    { key:'accepted',     label:'Accepted',      emoji:'✅' },
];

let editingJobId = null;

function initJobs() {
    loadJobs();
    document.getElementById('addJobBtn')?.addEventListener('click', () => {
        editingJobId = null;
        clearJobForm();
        openModal('jobModal');
    });
    document.getElementById('jobForm')?.addEventListener('submit', saveJob);
    document.getElementById('filterBar')?.addEventListener('click', e => {
        const btn = e.target.closest('.filter-btn');
        if (!btn) return;
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        loadJobs(btn.dataset.status);
    });
}

async function loadJobs(status = '') {
    const r = await fetch(`/api/jobs${status ? '?status=' + status : ''}`);
    if (r.status === 401) {
        document.getElementById('jobsContent').innerHTML =
            '<div style="text-align:center;padding:3rem;"><p>Please <a href="/login-page">sign in</a> to track jobs.</p></div>';
        return;
    }
    const d = await r.json();
    state.jobs = d.jobs || [];
    renderJobs(state.jobs);
    renderJobSummary(d.summary || {});
}

function renderJobSummary(summary) {
    const el = document.getElementById('jobSummary');
    if (!el) return;
    const total = Object.values(summary).reduce((a,b) => a+b, 0);
    el.innerHTML = `
      <div class="insight-card"><div class="ic-label">Total</div><div class="ic-value">${total}</div></div>
      ${JOB_STATUSES.slice(0,4).map(s => `
        <div class="insight-card"><div class="ic-label">${s.emoji} ${s.label}</div>
        <div class="ic-value">${summary[s.key] || 0}</div></div>`).join('')}`;
}

function renderJobs(jobs) {
    const el = document.getElementById('jobsContent');
    if (!el) return;

    if (!jobs.length) {
        el.innerHTML = '<div style="text-align:center;padding:3rem;color:var(--slate-light);"><i class="fas fa-briefcase" style="font-size:2rem;display:block;margin-bottom:1rem;"></i><p>No jobs tracked yet. Click <strong>+ Add Job</strong> to get started.</p></div>';
        return;
    }

    // Group by status for kanban
    const grouped = {};
    JOB_STATUSES.forEach(s => grouped[s.key] = []);
    jobs.forEach(j => { if (grouped[j.status]) grouped[j.status].push(j); });

    el.innerHTML = `<div class="kanban-wrap">
      ${JOB_STATUSES.map(s => `
        <div class="kanban-col">
          <div class="kanban-col-header">
            <span class="kanban-col-title" style="color:var(--slate)">${s.emoji} ${s.label}</span>
            <span class="kanban-col-count">${grouped[s.key].length}</span>
          </div>
          ${grouped[s.key].map(j => renderKanbanCard(j)).join('')}
        </div>`).join('')}
    </div>`;
}

function renderKanbanCard(j) {
    const date = j.applied_date ? new Date(j.applied_date).toLocaleDateString('en-US', {month:'short', day:'numeric'}) : '';
    return `
      <div class="kanban-card" onclick="editJob(${j.id})">
        <div class="kanban-card-title">${j.title}</div>
        <div class="kanban-card-company">${j.company}${j.location ? ' · ' + j.location : ''}</div>
        <div class="kanban-card-meta">
          ${j.salary ? `<span style="font-size:0.72rem;color:var(--gold);font-weight:600;">${j.salary}</span>` : '<span></span>'}
          ${date ? `<span style="font-size:0.72rem;color:var(--slate-light);">${date}</span>` : ''}
        </div>
        ${j.notes ? `<div style="font-size:0.75rem;color:var(--slate-light);margin-top:5px;font-style:italic;">${j.notes.substring(0,60)}${j.notes.length>60?'…':''}</div>` : ''}
      </div>`;
}

function editJob(id) {
    const j = state.jobs.find(j => j.id === id);
    if (!j) return;
    editingJobId = id;
    setVal('jobTitle',    j.title);
    setVal('jobCompany',  j.company);
    setVal('jobLocation', j.location || '');
    setVal('jobUrl',      j.url || '');
    setVal('jobSalary',   j.salary || '');
    setVal('jobStatus',   j.status);
    setVal('jobNotes',    j.notes || '');
    setVal('jobApplied',  j.applied_date || '');
    setVal('jobDeadline', j.deadline || '');
    document.getElementById('deleteJobBtn').style.display = 'inline-flex';
    openModal('jobModal');
}

function clearJobForm() {
    ['jobTitle','jobCompany','jobLocation','jobUrl','jobSalary','jobNotes','jobApplied','jobDeadline'].forEach(id => setVal(id,''));
    setVal('jobStatus', 'saved');
    const del = document.getElementById('deleteJobBtn');
    if (del) del.style.display = 'none';
}

async function saveJob(e) {
    e.preventDefault();
    const payload = {
        title:       document.getElementById('jobTitle')?.value.trim(),
        company:     document.getElementById('jobCompany')?.value.trim(),
        location:    document.getElementById('jobLocation')?.value.trim(),
        url:         document.getElementById('jobUrl')?.value.trim(),
        salary:      document.getElementById('jobSalary')?.value.trim(),
        status:      document.getElementById('jobStatus')?.value,
        notes:       document.getElementById('jobNotes')?.value.trim(),
        applied_date:document.getElementById('jobApplied')?.value,
        deadline:    document.getElementById('jobDeadline')?.value,
    };
    const url    = editingJobId ? `/api/jobs/${editingJobId}` : '/api/jobs';
    const method = editingJobId ? 'PUT' : 'POST';
    const r = await fetch(url, { method, headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const d = await r.json();
    if (r.ok) {
        notify(editingJobId ? 'Job updated!' : 'Job added!', 'success');
        closeModal('jobModal');
        loadJobs();
    } else notify(d.error || 'Failed', 'error');
}

async function deleteJob() {
    if (!editingJobId || !confirm('Delete this job?')) return;
    const r = await fetch(`/api/jobs/${editingJobId}`, { method: 'DELETE' });
    if (r.ok) { notify('Deleted', 'success'); closeModal('jobModal'); loadJobs(); }
    else notify('Failed', 'error');
}

/* ─────────────────────────────────────────────────────────
   LOGIN / SIGNUP
───────────────────────────────────────────────────────── */
function initLogin() {
    document.getElementById('loginForm')?.addEventListener('submit', async e => {
        e.preventDefault();
        const btn = e.target.querySelector('button[type=submit]');
        btn.disabled = true; btn.textContent = 'Signing in…';
        const r = await fetch('/api/login', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email:    document.getElementById('email').value.trim(),
                password: document.getElementById('password').value,
            }),
        });
        const d = await r.json();
        if (r.ok) window.location.href = '/';
        else { notify(d.error || 'Login failed', 'error'); btn.disabled=false; btn.textContent='Sign In'; }
    });

    document.getElementById('forgotBtn')?.addEventListener('click', async () => {
        const email = document.getElementById('email').value.trim();
        if (!email) { notify('Enter your email first', 'warning'); return; }
        const r = await fetch('/api/forgot-password', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ email }),
        });
        const d = await r.json();
        notify(d.message || 'Check your email', 'info');
    });
}

function initSignup() {
    const pwdInput = document.getElementById('password');
    const strEl    = document.getElementById('pwdStrength');
    pwdInput?.addEventListener('input', () => {
        const v = pwdInput.value;
        let score = 0;
        if (v.length >= 8)       score++;
        if (/[A-Z]/.test(v))     score++;
        if (/[0-9]/.test(v))     score++;
        if (/[^A-Za-z0-9]/.test(v)) score++;
        const colors = ['','#dc2626','#d97706','#16a34a','#1d4ed8'];
        const labels = ['','Weak','Fair','Good','Strong'];
        if (!v) { strEl.innerHTML=''; return; }
        strEl.innerHTML = `<div style="display:flex;gap:4px;margin-top:5px;">${[1,2,3,4].map(i=>`<div style="flex:1;height:3px;border-radius:3px;background:${i<=score?colors[score]:'var(--border-soft)'}"></div>`).join('')}</div>
          <span style="font-size:0.75rem;color:${colors[score]};font-weight:600;">${labels[score]}</span>`;
    });

    document.getElementById('signupForm')?.addEventListener('submit', async e => {
        e.preventDefault();
        const pwd  = document.getElementById('password').value;
        const conf = document.getElementById('confirmPassword').value;
        if (pwd !== conf) { notify('Passwords do not match', 'error'); return; }
        const btn = e.target.querySelector('button[type=submit]');
        btn.disabled = true; btn.textContent = 'Creating account…';
        const r = await fetch('/api/signup', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({
                name:     document.getElementById('name').value.trim(),
                email:    document.getElementById('email').value.trim(),
                password: pwd,
            }),
        });
        const d = await r.json();
        if (r.ok) { notify('Account created! Signing you in…','success'); window.location.href='/login-page'; }
        else { notify(d.error || 'Signup failed', 'error'); btn.disabled=false; btn.textContent='Create Account'; }
    });
}

/* ── UTILITIES ────────────────────────────────────────── */
function setText(id, val)  { const el = document.getElementById(id); if (el) el.textContent = val; }
function setVal(id, val)   { const el = document.getElementById(id); if (el) el.value = val; }
function atsColor(v)       { return v >= 70 ? '#16a34a' : v >= 40 ? '#d97706' : '#dc2626'; }
function getScoreColor(v)  { return v >= 80 ? '#16a34a' : v >= 60 ? '#d97706' : v >= 40 ? '#f97316' : '#dc2626'; }
function escJs(s)          { return s.replace(/'/g,"&#39;").replace(/\n/g,'\\n'); }
function copyText(text) {
    navigator.clipboard.writeText(text).then(() => notify('Copied!','success')).catch(() => notify('Copy failed','error'));
}

/* ── AI thinking dots in CSS ───────────── */
/* Injected via JS for portability */
const _aiStyle = document.createElement('style');
_aiStyle.textContent = `.ai-thinking{display:flex;align-items:center;gap:6px;padding:10px 14px;color:var(--slate);font-size:0.875rem;}
.ai-dot{width:6px;height:6px;border-radius:50%;background:var(--gold);animation:aiBounce 1.2s infinite ease-in-out;}
.ai-dot:nth-child(2){animation-delay:0.2s;}
.ai-dot:nth-child(3){animation-delay:0.4s;}
@keyframes aiBounce{0%,80%,100%{transform:scale(0.7);opacity:0.5;}40%{transform:scale(1.1);opacity:1;}}`;
document.head.appendChild(_aiStyle);
