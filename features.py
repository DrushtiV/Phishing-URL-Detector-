"""
URL Feature Extraction for Phishing Detection
Extracts 18 hand-crafted features from raw URLs using regex and heuristics.
"""

import re
import math
from urllib.parse import urlparse
from typing import Dict, Any


# ── Entropy helper ────────────────────────────────────────────────────────────
def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


# ── Suspicious word lists ─────────────────────────────────────────────────────
PHISHING_KEYWORDS = {
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "ebay", "amazon", "apple", "microsoft", "google",
    "password", "credential", "wallet", "support", "helpdesk", "alert",
    "suspended", "unusual", "activity", "click", "here", "free", "prize",
}

TRUSTED_TLDS = {".com", ".org", ".net", ".edu", ".gov", ".co", ".io"}


# ── IP address pattern ────────────────────────────────────────────────────────
_IP_PATTERN = re.compile(
    r"((\d{1,3}\.){3}\d{1,3})"  # IPv4
    r"|(\[\d{1,3}(\.\d{1,3}){3}\])"  # IPv4 in brackets
)

_HEX_IP_PATTERN = re.compile(r"0x[0-9a-fA-F]{8}")


def extract_features(url: str) -> Dict[str, Any]:
    """
    Extract 18 numerical features from a URL string.
    Returns a dict {feature_name: value}.
    """
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)

    scheme    = parsed.scheme.lower()
    netloc    = parsed.netloc.lower()
    path      = parsed.path
    query     = parsed.query
    fragment  = parsed.fragment
    full_url  = url.lower()

    # Strip port from netloc for domain work
    domain = netloc.split(":")[0] if ":" in netloc else netloc
    # Remove www. prefix
    domain_clean = re.sub(r"^www\.", "", domain)
    # Split into parts
    parts = domain_clean.split(".")

    # ── 1. URL length ──────────────────────────────────────────────────────
    url_length = len(url)

    # ── 2. Number of dots ─────────────────────────────────────────────────
    num_dots = url.count(".")

    # ── 3. Presence of @ symbol ───────────────────────────────────────────
    has_at = int("@" in url)

    # ── 4. IP address in hostname ─────────────────────────────────────────
    has_ip = int(bool(_IP_PATTERN.search(domain)) or bool(_HEX_IP_PATTERN.search(domain)))

    # ── 5. Subdomain count ────────────────────────────────────────────────
    # parts = [sub..., domain, tld]; subdomains are everything before last 2
    subdomain_count = max(len(parts) - 2, 0)

    # ── 6. HTTPS usage ────────────────────────────────────────────────────
    uses_https = int(scheme == "https")

    # ── 7. URL entropy (randomness) ───────────────────────────────────────
    url_entropy = round(_shannon_entropy(domain_clean), 4)

    # ── 8. Number of hyphens in domain ───────────────────────────────────
    num_hyphens = domain.count("-")

    # ── 9. Path length ────────────────────────────────────────────────────
    path_length = len(path)

    # ── 10. Number of query parameters ───────────────────────────────────
    num_params = len(query.split("&")) if query else 0

    # ── 11. Presence of suspicious keywords ──────────────────────────────
    tokens = set(re.split(r"[\W_]+", full_url))
    phishing_keyword_count = len(tokens & PHISHING_KEYWORDS)

    # ── 12. Domain length ─────────────────────────────────────────────────
    domain_length = len(domain_clean)

    # ── 13. Has suspicious TLD ────────────────────────────────────────────
    tld = "." + parts[-1] if parts else ""
    is_suspicious_tld = int(tld not in TRUSTED_TLDS)

    # ── 14. Number of slashes in path ────────────────────────────────────
    num_slashes = path.count("/")

    # ── 15. Presence of hex encoding ─────────────────────────────────────
    has_hex_encoding = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))

    # ── 16. Double slash in path (redirect trick) ─────────────────────────
    has_double_slash = int("//" in path)

    # ── 17. Digit ratio in domain ─────────────────────────────────────────
    if domain_clean:
        digit_ratio = round(sum(c.isdigit() for c in domain_clean) / len(domain_clean), 4)
    else:
        digit_ratio = 0.0

    # ── 18. Fragment present ──────────────────────────────────────────────
    has_fragment = int(bool(fragment))

    return {
        "url_length":            url_length,
        "num_dots":              num_dots,
        "has_at":                has_at,
        "has_ip":                has_ip,
        "subdomain_count":       subdomain_count,
        "uses_https":            uses_https,
        "url_entropy":           url_entropy,
        "num_hyphens":           num_hyphens,
        "path_length":           path_length,
        "num_params":            num_params,
        "phishing_keyword_count": phishing_keyword_count,
        "domain_length":         domain_length,
        "is_suspicious_tld":     is_suspicious_tld,
        "num_slashes":           num_slashes,
        "has_hex_encoding":      has_hex_encoding,
        "has_double_slash":      has_double_slash,
        "digit_ratio":           digit_ratio,
        "has_fragment":          has_fragment,
    }


FEATURE_NAMES = list(extract_features("http://example.com").keys())
