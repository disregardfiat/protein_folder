"""
CASP-compliant HTTP server: FASTA in → PDB out.
POST /predict with body = FASTA (or Content-Type: application/json with {"fasta": "..."})
  → 200 + PDB text (CASP format). If "email" param is set and SMTP is configured, also sends PDB by email.
GET /health → 200 OK (for reverse proxy / liveness).
SMTP (optional): set SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM; SMTP_USE_TLS=1 (default) for STARTTLS. When set, results are also emailed to the request's "email" address.
Run from repo root: gunicorn -w 1 -b 127.0.0.1:8050 casp_server:app
"""

from __future__ import annotations

import os
import re
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

from flask import Flask, request, Response

# Ensure repo root is on path when run via gunicorn
if __name__ != "__main__":
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from horizon_physics.proteins import hqiv_predict_structure

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB max FASTA

# Optional SMTP: send PDB to request's "email" address when set
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SMTP_FROM = os.environ.get("SMTP_FROM") or (SMTP_USER if SMTP_USER else "noreply@localhost")
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "1").strip().lower() in ("1", "true", "yes")


def _get_email_and_title() -> tuple[str | None, str | None]:
    """Extract email and title from request (form or JSON)."""
    email, title = None, None
    if request.form:
        email = request.form.get("email") or request.form.get("email_to") or request.form.get("results_email")
        title = request.form.get("title")
    if (email is None or title is None) and request.is_json:
        data = request.get_json(silent=True) or {}
        if email is None:
            email = data.get("email") or data.get("email_to") or data.get("results_email")
        if title is None:
            title = data.get("title")
    if email:
        email = email.strip()
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            email = None
    return email or None, (title.strip() if title and title.strip() else None)


def _send_pdb_email(to_email: str, pdb: str, title: str | None) -> None:
    """Send PDB as email attachment. No-op if SMTP not configured. Logs but does not raise on failure."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction: {title}" if title else "HQIV structure prediction"
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = formataddr(("HQIV CASP Server", SMTP_FROM))
    msg["To"] = to_email
    msg.attach(MIMEText("PDB model attached (CASP format).", "plain"))
    part = MIMEText(pdb, "plain")
    part.add_header("Content-Disposition", "attachment", filename="model.pdb")
    msg.attach(part)
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(SMTP_FROM, [to_email], msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP send failed: %s", e)


@app.route("/health", methods=["GET"])
def health():
    return Response("OK\n", status=200, mimetype="text/plain")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept FASTA in body, JSON, or form (sequence= / fasta= / title= / email= / inchi=). Return CASP-format PDB. Ligand targets: InChI accepted and ignored — we return protein-only (CAMEO allows this). If email param is set and SMTP configured, also sends PDB to that address."""
    fasta = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        fasta = data.get("fasta") or data.get("sequence")
    if fasta is None and request.form:
        # CAMEO: sequence, fasta, title, email, inchi (ligand); we use sequence/fasta only, return protein-only
        fasta = request.form.get("sequence") or request.form.get("fasta")
    if fasta is None:
        fasta = request.get_data(as_text=True)
    if not fasta or not fasta.strip():
        return Response("Missing FASTA in body, JSON 'fasta'/'sequence', or form 'sequence'/'fasta'\n", status=400, mimetype="text/plain")

    to_email, job_title = _get_email_and_title()

    try:
        pdb = hqiv_predict_structure(fasta.strip())
        if to_email:
            _send_pdb_email(to_email, pdb, job_title)
        return Response(pdb, status=200, mimetype="text/plain")
    except Exception as e:
        return Response(f"Prediction failed: {e}\n", status=500, mimetype="text/plain")


@app.route("/", methods=["GET", "POST"])
def index():
    """GET: info. POST: same as /predict (so base URL can be used as submission URL)."""
    if request.method == "POST":
        return predict()
    return Response(
        "HQIV CASP server. POST FASTA to / or /predict, GET /health for liveness.\n",
        status=200,
        mimetype="text/plain",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, threaded=True)
