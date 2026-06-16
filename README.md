# Phishing URL Detector 🎣

A production-ready ML system that classifies URLs as **phishing** or **legitimate** using feature engineering + Random Forest classification, served via FastAPI with a sleek dark-mode frontend.

---

## Architecture

```
URL (raw string)
     │
     ▼
┌─────────────────────────────┐
│   Feature Extraction        │  features.py
│   18 engineered features    │
│   (regex, heuristics)       │
└────────────┬────────────────┘
             │  feature vector [18 floats]
             ▼
┌─────────────────────────────┐
│   Random Forest             │  model.joblib
│   StandardScaler pipeline   │
│   300 estimators            │
└────────────┬────────────────┘
             │  (label, confidence)
             ▼
┌─────────────────────────────┐
│   FastAPI  POST /check      │  app.py
│   JSON response             │
└────────────┬────────────────┘
             │
             ▼
         HTML Frontend (GET /)
```

---

## Features Extracted (18 total)

| Feature | Description |
|---|---|
| `url_length` | Total character count |
| `num_dots` | Dots in the URL (subdomain/path depth) |
| `has_at` | `@` symbol (credential spoofing trick) |
| `has_ip` | IPv4 address used as hostname |
| `subdomain_count` | Number of subdomains |
| `uses_https` | Scheme is HTTPS |
| `url_entropy` | Shannon entropy of domain (randomness) |
| `num_hyphens` | Hyphens in domain |
| `path_length` | Length of URL path |
| `num_params` | Query parameter count |
| `phishing_keyword_count` | Known phishing keywords matched |
| `domain_length` | Length of domain |
| `is_suspicious_tld` | TLD not in trusted set (.com/.org/.net…) |
| `num_slashes` | Slashes in path |
| `has_hex_encoding` | Percent-encoded characters |
| `has_double_slash` | Double slash redirect trick |
| `digit_ratio` | Proportion of digits in domain |
| `has_fragment` | Fragment present (#) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Using embedded representative dataset (works out of the box):
python train.py

# Using the full UCI Phishing Dataset from Kaggle:
python train.py --csv path/to/phishing_urls.csv
```

### 3. Run the API server

```bash
uvicorn app:app --reload --port 8000
```

### 4. Open the frontend

Navigate to **http://localhost:8000** in your browser.

---

## API Reference

### `POST /check`

Classify a URL.

**Request:**
```json
{ "url": "http://paypal-secure-login.com/verify/account" }
```

**Response:**
```json
{
  "url": "http://paypal-secure-login.com/verify/account",
  "result": "phishing",
  "confidence": 0.98,
  "risk_level": "high",
  "features": {
    "url_length": 46,
    "num_dots": 3,
    "has_at": 0,
    "has_ip": 0,
    "subdomain_count": 0,
    "uses_https": 0,
    "url_entropy": 3.8791,
    "num_hyphens": 2,
    "path_length": 17,
    "num_params": 0,
    "phishing_keyword_count": 3,
    "domain_length": 24,
    "is_suspicious_tld": 0,
    "num_slashes": 3,
    "has_hex_encoding": 0,
    "has_double_slash": 0,
    "digit_ratio": 0.0,
    "has_fragment": 0
  }
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## Extending with Real Data

Download the UCI Phishing Website Dataset from Kaggle:
https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector

Then retrain:
```bash
python train.py --csv phishing_urls.csv
```

The model will automatically re-save to `model.joblib`.

---

## Project Structure

```
phishing_detector/
├── features.py     # URL feature extraction (18 features)
├── train.py        # Dataset builder + model training
├── app.py          # FastAPI server + HTML frontend
├── model.joblib    # Trained pipeline (auto-generated)
├── requirements.txt
└── README.md
```

---

## Technical Notes

- **Model**: `RandomForestClassifier` (300 trees, `class_weight="balanced"`) wrapped in a `StandardScaler` pipeline
- **No external lookups at inference time** – whois queries are slow; all features are computed from the URL string itself for sub-millisecond latency
- **Confidence score**: probability of the predicted class (not raw phishing probability), so it always reads as "X% confident this is [result]"
- **Risk levels**: High ≥75% phishing probability, Medium 45–75%, Low <45%
