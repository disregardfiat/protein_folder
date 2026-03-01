"""
CASP-compliant HTTP server: FASTA in → PDB out (full structure via HKE + funnel).
POST /predict with body = FASTA or form/JSON (sequence=, title=, email=).
Uses minimize_full_chain_hierarchical with cone funnel (stages 1–2) then Cartesian refinement (stage 3).
If "email" param is set and SMTP is configured, also sends PDB by email.
GET /health → 200 OK.
Env: SMTP_* for email; SMTP_CC_TO to CC when not recipient; CASP_OUTPUT_DIR (default ./casp_results) for pending/ and outputs/; USE_FAST_PREDICT=1 to skip HKE.
Run from repo root: gunicorn -w 1 -b 127.0.0.1:8050 casp_server:app
"""

from __future__ import annotations

import os
import random
import re
import shutil
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid, formatdate

from flask import Flask, request, Response

# Ensure repo root is on path when run via gunicorn
if __name__ != "__main__":
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from horizon_physics.proteins.casp_submission import _parse_fasta
from horizon_physics.proteins import full_chain_to_pdb
from horizon_physics.proteins.hierarchical import (
    minimize_full_chain_hierarchical,
    hierarchical_result_for_pdb,
)

# Fast path (geometric-only, no minimization): for testing or when USE_FAST_PREDICT=1
try:
    from horizon_physics.proteins import hqiv_predict_structure, hqiv_predict_structure_assembly
except Exception:
    hqiv_predict_structure = None
    hqiv_predict_structure_assembly = None

USE_FAST_PREDICT = os.environ.get("USE_FAST_PREDICT", "").strip().lower() in ("1", "true", "yes")
FUNNEL_RADIUS = float(os.environ.get("FUNNEL_RADIUS", "10.0"))
FUNNEL_RADIUS_EXIT = float(os.environ.get("FUNNEL_RADIUS_EXIT", "20.0"))
HKE_MAX_ITER = (
    int(os.environ.get("HKE_MAX_ITER_S1", "40")),
    int(os.environ.get("HKE_MAX_ITER_S2", "60")),
    int(os.environ.get("HKE_MAX_ITER_S3", "80")),
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB max FASTA

# Result persistence: pending/ (request .txt, then .pdb when done) → on success move both to outputs/
_output_base = os.environ.get("CASP_OUTPUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "casp_results")
PENDING_DIR = os.path.join(_output_base, "pending")
OUTPUTS_DIR = os.path.join(_output_base, "outputs")


def _ensure_output_dirs() -> None:
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _job_id() -> str:
    return f"{int(time.time())}_{random.randint(1000, 9999)}"


def _write_pending_txt(job_id: str, job_title: str | None, to_email: str | None, num_sequences: int, seq_lengths: list[int]) -> None:
    """Write pending/{job_id}.txt with POST summary (instructions / request info)."""
    _ensure_output_dirs()
    path = os.path.join(PENDING_DIR, f"{job_id}.txt")
    lines = [
        f"title={job_title or ''}",
        f"email={to_email or ''}",
        f"num_sequences={num_sequences}",
        f"lengths={seq_lengths}",
        f"received_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "POST /predict with sequence= (or multiple for assembly), title=, email=.",
        "When prediction completes, this file and the .pdb are moved to outputs/.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _move_to_outputs(job_id: str, pdb_content: str) -> None:
    """Write pending/{job_id}.pdb then move both .txt and .pdb to outputs/."""
    _ensure_output_dirs()
    pdb_path = os.path.join(PENDING_DIR, f"{job_id}.pdb")
    txt_path = os.path.join(PENDING_DIR, f"{job_id}.txt")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)
    for name in (f"{job_id}.txt", f"{job_id}.pdb"):
        src = os.path.join(PENDING_DIR, name)
        dst = os.path.join(OUTPUTS_DIR, name)
        if os.path.isfile(src):
            shutil.move(src, dst)

# Optional SMTP: send PDB to request's "email" address when set
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SMTP_FROM_RAW = os.environ.get("SMTP_FROM") or (SMTP_USER if SMTP_USER else "")
SMTP_FROM_DOMAIN = (os.environ.get("SMTP_FROM_DOMAIN") or "").strip()  # e.g. disregardfiat.tech — use for From when Gmail rejects SMTP host–derived domain
SMTP_CC_TO = (os.environ.get("SMTP_CC_TO") or os.environ.get("cc_to") or "").strip()  # CC this address on result emails when not the recipient (e.g. to monitor CAMEO)
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "1").strip().lower() in ("1", "true", "yes")


def _smtp_from_address() -> str:
    """Return RFC 5322–compliant From address (user@domain). Gmail rejects some domains; set SMTP_FROM or SMTP_FROM_DOMAIN to use a compliant address."""
    addr = (SMTP_FROM_RAW or "").strip()
    if re.match(r"^[^@]+@[^@]+\.[^@]+", addr):
        return addr
    if SMTP_USER and "@" in SMTP_USER:
        return SMTP_USER
    # Prefer explicit domain (Gmail-friendly, e.g. disregardfiat.tech)
    domain = SMTP_FROM_DOMAIN
    if not domain and SMTP_HOST:
        # Derive from SMTP host: mail.comodomodo.com.py -> comodomodo.com.py
        domain = (SMTP_HOST or "").strip()
        if domain.startswith("mail."):
            domain = domain[5:]
        elif "." in domain:
            domain = domain.split(".", 1)[1]
    if SMTP_USER and domain:
        return f"{SMTP_USER}@{domain}"
    if domain:
        return f"noreply@{domain}"
    return "noreply@localhost"


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
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    msg["To"] = to_email
    recipients = [to_email]
    if SMTP_CC_TO and re.match(r"[^@]+@[^@]+\.[^@]+", SMTP_CC_TO):
        cc_addr = SMTP_CC_TO.strip().lower()
        to_addr_lower = to_email.strip().lower()
        if cc_addr != to_addr_lower:
            msg["Cc"] = SMTP_CC_TO
            recipients.append(SMTP_CC_TO)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
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
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP send failed: %s", e)


def _sequence_from_input(raw: str) -> str:
    """Extract one-letter sequence from FASTA or raw sequence."""
    s = (raw or "").strip()
    if not s:
        return ""
    return _parse_fasta(s) if ">" in s or "\n" in s else "".join(c for c in s.upper() if c.isalpha())


def _predict_hke_single(sequence: str) -> str:
    """Run HKE + funnel for one chain; return PDB string (chain A)."""
    pos, z_list = minimize_full_chain_hierarchical(
        sequence,
        include_sidechains=False,
        funnel_radius=FUNNEL_RADIUS,
        funnel_stiffness=1.0,
        funnel_radius_exit=FUNNEL_RADIUS_EXIT,
        max_iter_stage1=HKE_MAX_ITER[0],
        max_iter_stage2=HKE_MAX_ITER[1],
        max_iter_stage3=HKE_MAX_ITER[2],
    )
    result = hierarchical_result_for_pdb(pos, z_list, sequence, include_sidechains=False)
    return full_chain_to_pdb(result, chain_id="A")


def _merge_pdb_chains(result_and_chain_ids: list[tuple[dict, str]]) -> str:
    """Merge multiple (result, chain_id) into one PDB (MODEL 1) with renumbered atom IDs."""
    from horizon_physics.proteins.casp_submission import AA_1to3
    lines = ["MODEL     1"]
    atom_id = 1
    for result, chain_id in result_and_chain_ids:
        backbone_atoms = result["backbone_atoms"]
        sequence = result["sequence"]
        include_sidechains = result.get("include_sidechains", False)
        if not backbone_atoms:
            continue
        idx = 0
        for res_id in range(1, result["n_res"] + 1):
            res_1 = sequence[res_id - 1]
            res_3 = AA_1to3.get(res_1, "UNK")
            n_atoms_this = (5 if res_1 != "G" else 4) if include_sidechains else 4
            for _ in range(n_atoms_this):
                name, xyz = backbone_atoms[idx]
                lines.append(
                    f"ATOM  {atom_id:5d}  {name:2s}  {res_3:3s} {chain_id}{res_id:4d}    "
                    f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}  1.00  0.00           "
                )
                atom_id += 1
                idx += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


def _predict_hke_assembly(sequences: list[str]) -> str:
    """Run HKE + funnel per chain; return one PDB with chain A, B, C, ...."""
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    results = []
    for i, seq in enumerate(sequences):
        pos, z_list = minimize_full_chain_hierarchical(
            seq,
            include_sidechains=False,
            funnel_radius=FUNNEL_RADIUS,
            funnel_stiffness=1.0,
            funnel_radius_exit=FUNNEL_RADIUS_EXIT,
            max_iter_stage1=HKE_MAX_ITER[0],
            max_iter_stage2=HKE_MAX_ITER[1],
            max_iter_stage3=HKE_MAX_ITER[2],
        )
        result = hierarchical_result_for_pdb(pos, z_list, seq, include_sidechains=False)
        cid = chain_ids[i] if i < len(chain_ids) else "A"
        results.append((result, cid))
    return _merge_pdb_chains(results)


@app.route("/health", methods=["GET"])
def health():
    return Response("OK\n", status=200, mimetype="text/plain")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept FASTA in body, JSON, or form (sequence= / fasta= / title= / email= / inchi=). Multi-chain: form/JSON can send multiple sequence values; returns one PDB with chain A,B,C,.... If email set and SMTP configured, also sends PDB to that address."""
    sequences = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        seq = data.get("fasta") or data.get("sequence")
        if isinstance(seq, list):
            sequences = [s.strip() for s in seq if s and str(s).strip()]
        elif seq:
            sequences = [str(seq).strip()]
    if sequences is None and request.form:
        # CAMEO: sequence can be repeated for assembly (sequence=...&sequence=...)
        sequences = request.form.getlist("sequence") or request.form.getlist("fasta")
        if not sequences:
            single = request.form.get("sequence") or request.form.get("fasta")
            if single:
                sequences = [single.strip()]
        else:
            sequences = [s.strip() for s in sequences if s and s.strip()]
    if sequences is None:
        raw = request.get_data(as_text=True)
        if raw and raw.strip():
            sequences = [raw.strip()]
    if not sequences:
        return Response("Missing FASTA/sequence in body, JSON, or form 'sequence'/'fasta'\n", status=400, mimetype="text/plain")

    # Normalize to one-letter sequences (strip FASTA headers etc.)
    seqs = [_sequence_from_input(s) for s in sequences]
    seqs = [s for s in seqs if s]
    if not seqs:
        return Response("No valid sequence found.\n", status=400, mimetype="text/plain")

    to_email, job_title = _get_email_and_title()
    job_id = _job_id()
    _write_pending_txt(job_id, job_title, to_email, len(seqs), [len(s) for s in seqs])

    try:
        if USE_FAST_PREDICT and hqiv_predict_structure is not None:
            if len(seqs) == 1:
                pdb = hqiv_predict_structure(sequences[0])
            else:
                pdb = hqiv_predict_structure_assembly(sequences)
        else:
            # Full structure: HKE + funnel null search
            if len(seqs) == 1:
                pdb = _predict_hke_single(seqs[0])
            else:
                pdb = _predict_hke_assembly(seqs)
        _move_to_outputs(job_id, pdb)
        if to_email:
            _send_pdb_email(to_email, pdb, job_title)
        return Response(pdb, status=200, mimetype="text/plain")
    except Exception as e:
        return Response(f"Prediction failed: {e}\n", status=500, mimetype="text/plain")


# Links for / and /help
REPO_URL = "https://github.com/disregardfiat/protein_folder"
PYHQIV_URL = "https://pypi.org/project/pyhqiv/"
PAPER_DOI_URL = "https://zenodo.org/records/18794890"


@app.route("/help", methods=["GET"])
def help_page():
    """API and reference links."""
    body = f"""HQIV CASP prediction server — help

Submit: POST / or POST /predict with FASTA or form/JSON (sequence=, title=, email=).
Multi-chain: send multiple sequence= values for assembly (chain A, B, C, ...).
Results: PDB in response body; if email is set and SMTP configured, also sent by email.

Endpoints:
  GET  /       — this info and links
  GET  /help   — this message
  GET  /health — liveness
  POST / or /predict — structure prediction (HKE + funnel)

References:
  Repository:  {REPO_URL}
  pyhqiv:      {PYHQIV_URL}
  Paper (DOI): {PAPER_DOI_URL}
"""
    return Response(body, status=200, mimetype="text/plain")


@app.route("/", methods=["GET", "POST"])
def index():
    """GET: info and links. POST: same as /predict (submission URL)."""
    if request.method == "POST":
        return predict()
    body = f"""HQIV CASP server — full structure (HKE + funnel)

  Repository:  {REPO_URL}
  pyhqiv:      {PYHQIV_URL}
  Paper (DOI): {PAPER_DOI_URL}

  POST / or /predict with sequence(s); GET /help for API details.
"""
    return Response(body, status=200, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, threaded=True)
