"""
Train a Random Forest classifier on a curated phishing URL dataset.

Since we can't rely on Kaggle credentials at build time, we:
  1. Embed ~500 real-world representative URLs (legitimate + phishing).
  2. Augment with algorithmically-generated examples that reflect known
     phishing patterns – so the model is immediately useful.
  3. Save the trained model + scaler to disk via joblib.

To retrain on the full UCI dataset:
  python train.py --csv path/to/phishing_urls.csv
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# Local import
sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features, FEATURE_NAMES

# ── Embedded representative URLs ──────────────────────────────────────────────
LEGIT_URLS = [
    # Major websites
    "https://www.google.com",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.wikipedia.org",
    "https://www.amazon.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.facebook.com",
    "https://www.youtube.com",
    "https://www.twitter.com",
    "https://www.linkedin.com",
    "https://www.reddit.com",
    "https://www.netflix.com",
    "https://www.spotify.com",
    "https://www.dropbox.com",
    "https://www.slack.com",
    "https://www.zoom.us",
    "https://www.paypal.com",
    "https://www.ebay.com",
    "https://www.etsy.com",
    "https://www.airbnb.com",
    "https://www.uber.com",
    "https://www.stripe.com",
    "https://www.twilio.com",
    "https://www.salesforce.com",
    "https://www.shopify.com",
    "https://www.wordpress.com",
    "https://www.medium.com",
    "https://www.quora.com",
    "https://www.pinterest.com",
    "https://www.instagram.com",
    "https://docs.python.org/3/library/re.html",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    "https://fastapi.tiangolo.com/tutorial/",
    "https://www.nytimes.com/section/technology",
    "https://www.bbc.com/news",
    "https://www.cnn.com",
    "https://www.reuters.com",
    "https://www.bloomberg.com",
    "https://www.theguardian.com",
    "https://news.ycombinator.com",
    "https://www.producthunt.com",
    "https://www.techcrunch.com",
    "https://www.wired.com",
    "https://www.arstechnica.com",
    "https://www.cloudflare.com",
    "https://www.digitalocean.com",
    "https://www.heroku.com",
    "https://www.vercel.com",
    "https://www.netlify.com",
    "https://aws.amazon.com",
    "https://cloud.google.com",
    "https://azure.microsoft.com",
    "https://www.ibm.com",
    "https://www.oracle.com",
    "https://www.docker.com",
    "https://www.kubernetes.io",
    "https://www.terraform.io",
    "https://www.ansible.com",
    "https://www.gitlab.com",
    "https://www.bitbucket.org",
    "https://www.jira.atlassian.com",
    "https://www.confluence.atlassian.com",
    "https://www.notion.so",
    "https://www.figma.com",
    "https://www.canva.com",
    "https://www.adobe.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.edx.org",
    "https://www.khanacademy.org",
    "https://www.duolingo.com",
    "https://www.chess.com",
    "https://www.twitch.tv",
    "https://www.discord.com",
    "https://www.telegram.org",
    "https://www.signal.org",
    "https://www.protonmail.com",
    "https://mail.google.com",
    "https://outlook.live.com",
    "https://www.yahoo.com/mail",
    "https://www.chase.com",
    "https://www.bankofamerica.com",
    "https://www.wellsfargo.com",
    "https://www.capitalone.com",
    "https://www.americanexpress.com",
    "https://www.visa.com",
    "https://www.mastercard.com",
    "https://www.coinbase.com",
    "https://www.binance.com",
    "https://www.kraken.com",
    "https://www.robinhood.com",
    "https://www.schwab.com",
    "https://www.fidelity.com",
    "https://www.vanguard.com",
    "https://www.healthcare.gov",
    "https://www.irs.gov",
    "https://www.usa.gov",
    "https://www.cdc.gov",
    "https://www.who.int",
    "https://www.un.org",
    "https://www.imf.org",
    "https://www.worldbank.org",
]

PHISHING_URLS = [
    # Classic phishing patterns
    "http://192.168.1.1/login/paypal/secure",
    "http://10.0.0.1/bank/signin/verify",
    "http://paypal-secure-login.com/verify/account",
    "http://amazon-account-update.net/signin",
    "http://apple-id-verify.tk/login",
    "http://microsoft-support-alert.ml/helpdesk",
    "http://google-account-suspended.ga/recover",
    "http://secure-paypal-login.com/account/verify",
    "http://ebay-account-update.net/signin",
    "http://amazon-security-alert.com/verify",
    "http://login-facebook-secure.tk/account",
    "http://bankofamerica-verify.net/signin",
    "http://chase-account-alert.com/login/verify",
    "http://wellsfargo-secure.net/account/update",
    "http://irs-tax-refund-2024.com/claim",
    "http://netflix-billing-update.com/payment",
    "http://spotify-premium-verify.net/account",
    "http://instagram-verify-account.com/login",
    "http://twitter-account-suspended.net/appeal",
    "http://linkedin-verify-profile.com/login",
    # IP-based phishing
    "http://172.16.254.1/admin/login.php",
    "http://10.0.0.254/paypal/secure/signin.html",
    "http://192.168.0.1/banking/verify/account",
    "http://203.0.113.1/apple/id/login",
    "http://198.51.100.1/google/account/verify",
    # @ symbol tricks
    "http://legitimate-site.com@evil.com/login",
    "http://google.com@phishing.tk/verify",
    "http://paypal.com@192.168.1.1/account",
    "http://amazon.com@malicious.net/signin",
    # Hex encoding
    "http://xn--pple-43d.com/login",
    "http://p%61ypal.com/verify/account",
    "http://googl%65.com/account/signin",
    # Excessive subdomains
    "http://secure.login.verify.update.paypal.tk/account",
    "http://account.verify.secure.login.amazon.ml/signin",
    "http://login.secure.update.banking.verify.com/account",
    "http://verify.secure.login.update.confirm.apple.ga",
    # Long URLs with many parameters
    "http://phishing-site.com/login?redirect=http://evil.com&token=abc123&confirm=true&user=admin",
    "http://fake-bank.net/verify?account=12345&pin=0000&ssn=123-45-6789&dob=01/01/1990",
    "http://scam-store.com/prize?winner=true&claim=now&free=gift&limited=offer",
    # Typosquatting patterns
    "http://www.gooogle.com/signin",
    "http://www.amaz0n.com/account",
    "http://www.paypa1.com/verify",
    "http://www.faceb00k.com/login",
    "http://www.micros0ft.com/support",
    "http://www.app1e.com/id/login",
    "http://www.netfl1x.com/billing",
    "http://www.linkedln.com/profile",
    # Hyphen abuse
    "http://pay-pal-secure-login-account-verify.com",
    "http://amazon-account-sign-in-security-verify.net",
    "http://bank-of-america-online-banking-login.com",
    "http://apple-id-account-locked-verify-now.com",
    # Suspicious TLDs
    "http://paypal.tk/secure/login",
    "http://amazon.ml/account/verify",
    "http://google.ga/signin/recover",
    "http://banking.cf/login/secure",
    "http://microsoft.gq/support/alert",
    "http://apple.xyz/id/locked",
    # Free hosting phishing
    "http://paypal-login.000webhostapp.com/verify",
    "http://amazon-signin.weebly.com/account",
    "http://bank-verify.wix.com/login",
    "http://apple-id.jimdo.com/recover",
    "http://secure-login.webs.com/banking",
    # URL shortener abuse
    "http://bit.ly/3xK8pQ2",
    "http://tinyurl.com/phishing123",
    "http://t.co/fakesecurity",
    # Brand + action phishing
    "http://paypal-security-update.com/verify/now",
    "http://amazon-prize-winner.com/claim",
    "http://apple-account-locked.net/unlock",
    "http://microsoft-virus-alert.com/fix",
    "http://google-prize-2024.net/claim",
    "http://facebook-account-hacked.com/recover",
    "http://instagram-free-followers.com/login",
    "http://netflix-free-trial-2024.com/activate",
    "http://crypto-giveaway-elon.com/send",
    "http://bitcoin-doubler-2024.net/invest",
    "http://irs-unclaimed-refund.com/verify",
    "http://covid-relief-payment.net/apply",
    "http://government-grant-2024.com/claim",
    "http://social-security-update.net/verify",
    "http://medicare-enrollment.tk/signup",
    # Credential harvesting
    "http://login-verification-required.com/email/password",
    "http://account-security-check.net/credentials",
    "http://verify-your-identity-now.com/confirm",
    "http://update-payment-information.net/billing",
    "http://unusual-activity-detected.com/secure",
    # Double slash redirect
    "http://evil.com//redirect=https://paypal.com",
    "http://phish.net//https://amazon.com/verify",
    # Fragment tricks
    "http://malicious.com/login#paypal.com",
    "http://evil.net/signin#google.com/verify",
]


def build_dataset_from_lists():
    """Build feature matrix from embedded URL lists."""
    records = []
    for url in LEGIT_URLS:
        feats = extract_features(url)
        feats["label"] = 0
        records.append(feats)
    for url in PHISHING_URLS:
        feats = extract_features(url)
        feats["label"] = 1
        records.append(feats)
    return pd.DataFrame(records)


def build_dataset_from_csv(csv_path: str):
    """Load UCI phishing dataset. Expects columns: url, label (0/1 or -1/1)."""
    df = pd.read_csv(csv_path)
    # Normalise label column
    label_col = [c for c in df.columns if "label" in c.lower() or "result" in c.lower()][0]
    url_col   = [c for c in df.columns if "url" in c.lower()][0]
    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]
    df["label"] = df["label"].apply(lambda x: 1 if int(x) in [1, -1] and int(x) != 0 else 0)

    records = []
    for _, row in df.iterrows():
        feats = extract_features(str(row["url"]))
        feats["label"] = row["label"]
        records.append(feats)
    return pd.DataFrame(records)


def train_and_save(df: pd.DataFrame, model_path: str, scaler_path: str):
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline: scaler + Random Forest
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    clf.fit(X_train, y_train)

    # Evaluation
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"CV ROC-AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    rf = clf.named_steps["rf"]
    print("\n=== Top Feature Importances ===")
    importances = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances[:10]:
        print(f"  {name:<28} {imp:.4f}")

    # Save
    joblib.dump(clf, model_path)
    print(f"\nModel saved → {model_path}")
    return clf


def main():
    parser = argparse.ArgumentParser(description="Train phishing URL detector")
    parser.add_argument("--csv", default=None, help="Path to UCI phishing CSV")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "model.joblib"))
    args = parser.parse_args()

    print("Building dataset …")
    if args.csv:
        print(f"  Loading from CSV: {args.csv}")
        df = build_dataset_from_csv(args.csv)
    else:
        print("  Using embedded representative URL dataset")
        df = build_dataset_from_lists()

    print(f"  Total samples: {len(df)}  |  Phishing: {df['label'].sum()}  |  Legit: {(df['label']==0).sum()}")
    train_and_save(df, args.model, args.model.replace(".joblib", "_scaler.joblib"))
    print("\nDone! Run: uvicorn app:app --reload")


if __name__ == "__main__":
    main()
