"""
Phishing URL Detector – FastAPI Application
POST /check  →  {"url": "..."}  →  {"result": "phishing"|"legitimate", "confidence": 0.93, "features": {...}}
GET  /       →  Serves the HTML frontend
"""

import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features, FEATURE_NAMES

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python train.py")

model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phishing URL Detector",
    description="Classifies URLs as phishing or legitimate using ML + feature engineering.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class URLRequest(BaseModel):
    url: str


class URLResponse(BaseModel):
    url: str
    result: str          # "phishing" or "legitimate"
    confidence: float    # 0.0 – 1.0
    risk_level: str      # "high" | "medium" | "low"
    features: dict


# ── API Routes ────────────────────────────────────────────────────────────────
@app.post("/check", response_model=URLResponse)
async def check_url(request: URLRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    # Extract features
    feat_dict = extract_features(url)
    feat_vec  = np.array([[feat_dict[name] for name in FEATURE_NAMES]])

    # Predict
    proba     = model.predict_proba(feat_vec)[0]
    phish_prob = float(proba[1])
    label     = "phishing" if phish_prob >= 0.5 else "legitimate"
    confidence = phish_prob if label == "phishing" else 1 - phish_prob

    # Risk level
    if phish_prob >= 0.75:
        risk_level = "high"
    elif phish_prob >= 0.45:
        risk_level = "medium"
    else:
        risk_level = "low"

    return URLResponse(
        url=url,
        result=label,
        confidence=round(confidence, 4),
        risk_level=risk_level,
        features=feat_dict,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# ── HTML Frontend ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phishing URL Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --accent:    #ff3c6e;
    --safe:      #00e5a0;
    --warn:      #ffb800;
    --text:      #e8e8f0;
    --muted:     #5a5a7a;
    --mono:      'Space Mono', monospace;
    --sans:      'Syne', sans-serif;
  }

  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Grid background ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(255,60,110,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,60,110,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 820px;
    margin: 0 auto;
    padding: 60px 24px 80px;
  }

  /* ── Header ── */
  header {
    text-align: center;
    margin-bottom: 52px;
  }

  .badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 12px;
    margin-bottom: 20px;
    text-transform: uppercase;
  }

  h1 {
    font-size: clamp(2.2rem, 6vw, 3.6rem);
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 14px;
  }

  h1 span { color: var(--accent); }

  .subtitle {
    color: var(--muted);
    font-family: var(--mono);
    font-size: 13px;
    letter-spacing: 0.5px;
  }

  /* ── Input form ── */
  .input-wrap {
    display: flex;
    gap: 0;
    margin-bottom: 40px;
    border: 1px solid var(--border);
    background: var(--surface);
    transition: border-color 0.2s;
  }

  .input-wrap:focus-within { border-color: var(--accent); }

  #url-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    padding: 18px 20px;
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
    min-width: 0;
  }

  #url-input::placeholder { color: var(--muted); }

  #check-btn {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 18px 28px;
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    white-space: nowrap;
  }

  #check-btn:hover  { opacity: 0.85; }
  #check-btn:active { transform: scale(0.97); }
  #check-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Result card ── */
  #result { display: none; animation: slideIn 0.35s ease; }

  @keyframes slideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .verdict-card {
    border: 1px solid var(--border);
    background: var(--surface);
    padding: 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 24px;
  }

  .verdict-icon {
    font-size: 3rem;
    line-height: 1;
    flex-shrink: 0;
  }

  .verdict-label {
    font-size: 11px;
    font-family: var(--mono);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
    color: var(--muted);
  }

  .verdict-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
  }

  .verdict-value.phishing   { color: var(--accent); }
  .verdict-value.legitimate { color: var(--safe); }

  .confidence-bar-wrap {
    margin-top: 14px;
  }

  .confidence-bar-track {
    height: 4px;
    background: var(--border);
    margin-top: 6px;
  }

  .confidence-bar-fill {
    height: 100%;
    transition: width 0.6s ease;
  }

  .confidence-bar-fill.phishing   { background: var(--accent); }
  .confidence-bar-fill.legitimate { background: var(--safe); }

  .conf-row {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }

  /* ── Feature grid ── */
  .section-title {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }

  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 8px;
    margin-bottom: 32px;
  }

  .feature-chip {
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }

  .feature-name {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.3px;
  }

  .feature-val {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    color: var(--text);
  }

  .feature-val.flag { color: var(--accent); }

  /* ── Risk pill ── */
  .risk-pill {
    display: inline-block;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border: 1px solid;
  }

  .risk-pill.high   { color: var(--accent); border-color: var(--accent); }
  .risk-pill.medium { color: var(--warn);   border-color: var(--warn); }
  .risk-pill.low    { color: var(--safe);   border-color: var(--safe); }

  /* ── URL echo ── */
  .url-echo {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    word-break: break-all;
    padding: 12px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    margin-bottom: 24px;
  }

  /* ── Error ── */
  .error-msg {
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 12px;
    padding: 14px 18px;
    margin-bottom: 24px;
  }

  /* ── Quick test URLs ── */
  .quick-tests {
    margin-top: 40px;
    border-top: 1px solid var(--border);
    padding-top: 28px;
  }

  .quick-tests h3 {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
  }

  .test-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .test-pill {
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 12px;
    border: 1px solid var(--border);
    background: var(--surface);
    cursor: pointer;
    color: var(--muted);
    transition: all 0.15s;
  }

  .test-pill:hover { border-color: var(--accent); color: var(--text); }
  .test-pill.phish { border-left: 2px solid var(--accent); }
  .test-pill.legit { border-left: 2px solid var(--safe); }

  /* ── Spinner ── */
  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  footer {
    text-align: center;
    margin-top: 60px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="badge">ML Security Tool</div>
    <h1>Phishing URL<br><span>Detector</span></h1>
    <p class="subtitle">Random Forest · 18 engineered features · real-time classification</p>
  </header>

  <div class="input-wrap">
    <input id="url-input" type="text" placeholder="https://suspicious-site.com/login/verify"
           autocomplete="off" spellcheck="false">
    <button id="check-btn" onclick="checkURL()">Analyze</button>
  </div>

  <div id="result"></div>

  <div class="quick-tests">
    <h3>Quick Test URLs</h3>
    <div class="test-pills">
      <div class="test-pill legit" onclick="quickTest('https://www.github.com')">github.com ✓</div>
      <div class="test-pill legit" onclick="quickTest('https://www.google.com')">google.com ✓</div>
      <div class="test-pill legit" onclick="quickTest('https://www.wikipedia.org')">wikipedia.org ✓</div>
      <div class="test-pill phish" onclick="quickTest('http://paypal-secure-login.com/verify/account')">paypal-secure-login.com ⚠</div>
      <div class="test-pill phish" onclick="quickTest('http://192.168.1.1/login/paypal/secure')">192.168.1.1/paypal ⚠</div>
      <div class="test-pill phish" onclick="quickTest('http://amazon-account-update.net/signin')">amazon-account-update ⚠</div>
      <div class="test-pill phish" onclick="quickTest('http://secure.login.verify.update.paypal.tk/account')">paypal.tk (subdomain abuse) ⚠</div>
      <div class="test-pill phish" onclick="quickTest('http://legitimate.com@evil-phishing.net/login')">@-symbol trick ⚠</div>
    </div>
  </div>

</div>

<footer>
  Phishing Detector · Random Forest classifier · Feature engineering via regex
</footer>

<script>
  const input = document.getElementById('url-input');
  const btn   = document.getElementById('check-btn');
  const out   = document.getElementById('result');

  input.addEventListener('keydown', e => { if (e.key === 'Enter') checkURL(); });

  function quickTest(url) {
    input.value = url;
    checkURL();
  }

  async function checkURL() {
    const url = input.value.trim();
    if (!url) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing…';
    out.style.display = 'none';

    try {
      const resp = await fetch('/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });

      if (!resp.ok) {
        const err = await resp.json();
        showError(err.detail || 'Something went wrong.');
        return;
      }

      const data = await resp.json();
      renderResult(data);

    } catch(e) {
      showError('Could not reach the API. Is the server running?');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Analyze';
    }
  }

  function showError(msg) {
    out.innerHTML = `<div class="error-msg">⚠ ${msg}</div>`;
    out.style.display = 'block';
  }

  function renderResult(d) {
    const isPhish = d.result === 'phishing';
    const icon    = isPhish ? '🎣' : '✅';
    const pct     = Math.round(d.confidence * 100);

    let featureHTML = '';
    const flagged = getFlaggedFeatures(d.features);

    for (const [key, val] of Object.entries(d.features)) {
      const isFlagged = flagged.includes(key);
      const display   = typeof val === 'number' && !Number.isInteger(val)
                        ? val.toFixed(3) : val;
      featureHTML += `
        <div class="feature-chip">
          <span class="feature-name">${key.replace(/_/g,' ')}</span>
          <span class="feature-val ${isFlagged ? 'flag' : ''}">${display}</span>
        </div>`;
    }

    out.innerHTML = `
      <div class="verdict-card">
        <div class="verdict-icon">${icon}</div>
        <div style="flex:1">
          <div class="verdict-label">Classification Result</div>
          <div class="verdict-value ${d.result}">${d.result.toUpperCase()}</div>
          <div style="margin-top:8px">
            <span class="risk-pill ${d.risk_level}">Risk: ${d.risk_level}</span>
          </div>
          <div class="confidence-bar-wrap">
            <div class="conf-row">
              <span>Confidence</span>
              <span>${pct}%</span>
            </div>
            <div class="confidence-bar-track">
              <div class="confidence-bar-fill ${d.result}" style="width:${pct}%"></div>
            </div>
          </div>
        </div>
      </div>

      <div class="url-echo">🔗 ${escapeHtml(d.url)}</div>

      <div class="section-title">Extracted Features (18)</div>
      <div class="feature-grid">${featureHTML}</div>
    `;

    out.style.display = 'block';
    out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function getFlaggedFeatures(f) {
    const flags = [];
    if (f.has_at)           flags.push('has_at');
    if (f.has_ip)           flags.push('has_ip');
    if (f.subdomain_count > 2)  flags.push('subdomain_count');
    if (!f.uses_https)      flags.push('uses_https');
    if (f.phishing_keyword_count > 0) flags.push('phishing_keyword_count');
    if (f.is_suspicious_tld) flags.push('is_suspicious_tld');
    if (f.has_hex_encoding)  flags.push('has_hex_encoding');
    if (f.has_double_slash)  flags.push('has_double_slash');
    if (f.num_hyphens > 3)  flags.push('num_hyphens');
    if (f.url_length > 100) flags.push('url_length');
    return flags;
  }

  function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def frontend():
    return HTML
