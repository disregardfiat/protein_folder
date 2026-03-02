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

import io
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import zipfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr, make_msgid, formatdate

from flask import Flask, request, Response

# Ensure repo root is on path when run via gunicorn
if __name__ != "__main__":
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from horizon_physics.proteins.casp_submission import _parse_fasta
from horizon_physics.proteins import (
    full_chain_to_pdb,
    full_chain_to_pdb_complex,
    minimize_full_chain,
)
from horizon_physics.proteins.hierarchical import (
    minimize_full_chain_hierarchical,
    hierarchical_result_for_pdb,
)
from horizon_physics.proteins.assembly_dock import run_two_chain_assembly, run_two_chain_assembly_hke

# Fast path (geometric-only, no minimization): for testing or when USE_FAST_PREDICT=1
try:
    from horizon_physics.proteins import hqiv_predict_structure, hqiv_predict_structure_assembly
except Exception:
    hqiv_predict_structure = None
    hqiv_predict_structure_assembly = None

USE_FAST_PREDICT = os.environ.get("USE_FAST_PREDICT", "").strip().lower() in ("1", "true", "yes")
FUNNEL_RADIUS = float(os.environ.get("FUNNEL_RADIUS", "10.0"))
FUNNEL_RADIUS_EXIT = float(os.environ.get("FUNNEL_RADIUS_EXIT", "20.0"))
# HKE uses finite-difference gradient (2 * n_dofs evals per step; ~572/step for 141 res). Funnel is on; bottleneck is FD, not funnel. Lower iters so server runs finish in ~10–20 min per chain.
HKE_MAX_ITER = (
    int(os.environ.get("HKE_MAX_ITER_S1", "15")),
    int(os.environ.get("HKE_MAX_ITER_S2", "25")),
    int(os.environ.get("HKE_MAX_ITER_S3", "50")),
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


MAX_ATTEMPTS = 2  # After this many attempts (initial + retries on restart), send failure email instead of retrying


def _write_pending_txt(job_id: str, job_title: str | None, to_email: str | None, num_sequences: int, seq_lengths: list[int], attempts: int = 1) -> None:
    """Write pending/{job_id}.txt with POST summary and attempts count."""
    _ensure_output_dirs()
    path = os.path.join(PENDING_DIR, f"{job_id}.txt")
    lines = [
        f"attempts={attempts}",
        f"title={job_title or ''}",
        f"email={to_email or ''}",
        f"num_sequences={num_sequences}",
        f"lengths={seq_lengths}",
        f"received_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "POST /predict with sequence= (or multiple for assembly), title=, email=.",
        "When prediction completes, this file, .request.json, and .pdb are moved to outputs/.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_pending_request(job_id: str, sequences: list[str], job_title: str | None, to_email: str | None) -> None:
    """Write pending/{job_id}.request.json so the job can be retried on restart."""
    _ensure_output_dirs()
    path = os.path.join(PENDING_DIR, f"{job_id}.request.json")
    with open(path, "w") as f:
        json.dump({"sequences": sequences, "title": job_title, "email": to_email}, f)


def _move_to_outputs(job_id: str, pdb_content: str) -> None:
    """Write pending/{job_id}.pdb then move .txt, .pdb, and .request.json to outputs/."""
    _ensure_output_dirs()
    pdb_path = os.path.join(PENDING_DIR, f"{job_id}.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)
    for name in (f"{job_id}.txt", f"{job_id}.pdb", f"{job_id}.request.json"):
        src = os.path.join(PENDING_DIR, name)
        dst = os.path.join(OUTPUTS_DIR, name)
        if os.path.isfile(src):
            shutil.move(src, dst)


def _move_to_outputs_assembly(
    job_id: str,
    pdb_a: str,
    pdb_b: str,
    pdb_complex: str,
) -> None:
    """Write chain A, chain B, complex PDBs and a ZIP; move all to outputs/."""
    _ensure_output_dirs()
    for name, content in [
        (f"{job_id}_chain_a.pdb", pdb_a),
        (f"{job_id}_chain_b.pdb", pdb_b),
        (f"{job_id}_complex.pdb", pdb_complex),
    ]:
        path = os.path.join(PENDING_DIR, name)
        with open(path, "w") as f:
            f.write(content)
    zip_path = os.path.join(PENDING_DIR, f"{job_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("chain_a.pdb", pdb_a)
        zf.writestr("chain_b.pdb", pdb_b)
        zf.writestr("complex.pdb", pdb_complex)
    to_move = [
        f"{job_id}.txt", f"{job_id}.request.json",
        f"{job_id}_chain_a.pdb", f"{job_id}_chain_b.pdb", f"{job_id}_complex.pdb", f"{job_id}.zip",
    ]
    for name in to_move:
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


def _send_assembly_email(
    to_email: str,
    pdb_a: str,
    pdb_b: str,
    pdb_complex: str,
    job_title: str | None,
) -> None:
    """Send 2-chain assembly result as a ZIP (chain_a.pdb, chain_b.pdb, complex.pdb)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("chain_a.pdb", pdb_a)
        zf.writestr("chain_b.pdb", pdb_b)
        zf.writestr("complex.pdb", pdb_complex)
    buf.seek(0)
    subject = f"HQIV assembly: {job_title}" if job_title else "HQIV 2-chain assembly (chain A, chain B, complex)"
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
    msg.attach(MIMEText("2-chain assembly: chain_a.pdb, chain_b.pdb, complex.pdb (ZIP attached).", "plain"))
    part = MIMEBase("application", "zip")
    part.set_payload(buf.getvalue())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename="assembly.zip")
    msg.attach(part)
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP assembly email failed: %s", e)


def _send_job_failure_email(to_email: str, job_id: str, job_title: str | None, error_message: str) -> None:
    """Send email on any single failure (includes error details)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction failed (job {job_id})"
    body = f"Job {job_id} (title={job_title or 'n/a'}) failed.\n\nError: {error_message}"
    msg = MIMEMultipart()
    msg["Subject"] = subject
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    msg["To"] = to_email
    recipients = [to_email]
    if SMTP_CC_TO and re.match(r"[^@]+@[^@]+\.[^@]+", SMTP_CC_TO):
        cc_addr = SMTP_CC_TO.strip().lower()
        if cc_addr != to_email.strip().lower():
            msg["Cc"] = SMTP_CC_TO
            recipients.append(SMTP_CC_TO)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
    msg.attach(MIMEText(body, "plain"))
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP failure-email send failed: %s", e)


def _send_failure_email(to_email: str, job_id: str, job_title: str | None) -> None:
    """Notify that the job failed after max attempts (no third try)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction failed (job {job_id})"
    body = f"Job {job_id} (title={job_title or 'n/a'}) did not complete after {MAX_ATTEMPTS} attempts. No further retries will be made."
    msg = MIMEMultipart()
    msg["Subject"] = subject
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    msg["To"] = to_email
    recipients = [to_email]
    if SMTP_CC_TO and re.match(r"[^@]+@[^@]+\.[^@]+", SMTP_CC_TO):
        cc_addr = SMTP_CC_TO.strip().lower()
        if cc_addr != to_email.strip().lower():
            msg["Cc"] = SMTP_CC_TO
            recipients.append(SMTP_CC_TO)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
    msg.attach(MIMEText(body, "plain"))
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP failure-email send failed: %s", e)


def _read_pending_attempts(txt_path: str) -> int:
    """Parse attempts=N from first lines of pending .txt; default 1."""
    try:
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("attempts="):
                    return max(1, int(line.split("=", 1)[1].strip() or "1"))
    except Exception:
        pass
    return 1


def _update_pending_attempts(txt_path: str, attempts: int) -> None:
    """Set attempts=N in the .txt file (rewrite first line)."""
    with open(txt_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("attempts="):
            lines[i] = f"attempts={attempts}\n"
            break
    else:
        lines.insert(0, f"attempts={attempts}\n")
    with open(txt_path, "w") as f:
        f.writelines(lines)


def process_pending_jobs() -> None:
    """Run once on startup: for each pending job without .pdb, retry once or send failure email after MAX_ATTEMPTS."""
    _ensure_output_dirs()
    if not os.path.isdir(PENDING_DIR):
        return
    for name in os.listdir(PENDING_DIR):
        if not name.endswith(".txt") or name.endswith(".failed.txt"):
            continue
        job_id = name[:-4]
        txt_path = os.path.join(PENDING_DIR, f"{job_id}.txt")
        req_path = os.path.join(PENDING_DIR, f"{job_id}.request.json")
        pdb_path = os.path.join(PENDING_DIR, f"{job_id}.pdb")
        if os.path.isfile(pdb_path):
            continue
        if not os.path.isfile(req_path):
            # Legacy: .txt only (no .request.json) — can't retry; send failure if email in .txt
            try:
                with open(txt_path) as f:
                    for line in f:
                        if line.strip().startswith("email="):
                            to_email = line.split("=", 1)[1].strip()
                            if to_email and re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
                                _send_failure_email(to_email, job_id, None)
                            break
            except Exception:
                pass
            try:
                shutil.move(txt_path, os.path.join(OUTPUTS_DIR, f"{job_id}.failed.txt"))
            except Exception:
                pass
            continue
        attempts = _read_pending_attempts(txt_path)
        if attempts >= MAX_ATTEMPTS:
            try:
                with open(req_path) as f:
                    req = json.load(f)
                to_email = (req.get("email") or "").strip()
                if to_email and re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
                    _send_failure_email(to_email, job_id, req.get("title"))
            except Exception as e:
                app.logger.warning("Pending job %s failure-email error: %s", job_id, e)
            try:
                shutil.move(txt_path, os.path.join(OUTPUTS_DIR, f"{job_id}.failed.txt"))
            except Exception:
                pass
            try:
                os.remove(req_path)
            except Exception:
                pass
            continue
        _update_pending_attempts(txt_path, attempts + 1)
        try:
            with open(req_path) as f:
                req = json.load(f)
        except Exception:
            continue
        sequences = req.get("sequences") or []
        seqs = [_sequence_from_input(s) for s in sequences if s]
        if not seqs:
            continue
        to_email = (req.get("email") or "").strip()
        job_title = req.get("title")
        try:
            if USE_FAST_PREDICT and hqiv_predict_structure is not None:
                pdb = hqiv_predict_structure_assembly(sequences) if len(seqs) > 1 else hqiv_predict_structure(sequences[0])
                _move_to_outputs(job_id, pdb)
                if to_email:
                    _send_pdb_email(to_email, pdb, job_title)
            elif len(seqs) == 2:
                assembly = None
                try:
                    assembly = _predict_hke_assembly_with_complex(sequences)
                except Exception as e:
                    app.logger.warning("HKE 2-chain failed (pending), falling back to Cartesian: %s", e)
                    assembly = _predict_cartesian_assembly_with_complex(sequences)
                if assembly is not None:
                    pdb_a, pdb_b, pdb_complex = assembly
                    _move_to_outputs_assembly(job_id, pdb_a, pdb_b, pdb_complex)
                    if to_email:
                        _send_assembly_email(to_email, pdb_a, pdb_b, pdb_complex, job_title)
                else:
                    pdb = _predict_hke_assembly(seqs)
                    _move_to_outputs(job_id, pdb)
                    if to_email:
                        _send_pdb_email(to_email, pdb, job_title)
            else:
                pdb = _predict_hke_assembly(seqs) if len(seqs) > 1 else _predict_hke_single(seqs[0])
                _move_to_outputs(job_id, pdb)
                if to_email:
                    _send_pdb_email(to_email, pdb, job_title)
        except Exception as e:
            app.logger.warning("Pending job %s retry failed: %s", job_id, e)
            if to_email and re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
                _send_job_failure_email(to_email, job_id, job_title, str(e))


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


def _predict_hke_assembly_with_complex(sequences: list[str]) -> tuple[str, str, str] | None:
    """
    For exactly 2 chains: run each chain through HKE-with-funnel on its own thread;
    map bond sites (placement); then run complex with HKE (no funnel) until
    max displacement per residue < 0.5 Å. Returns (pdb_chain_a, pdb_chain_b, pdb_complex) or None if not 2 chains.
    """
    if len(sequences) != 2:
        return None
    seq_a = _sequence_from_input(sequences[0])
    seq_b = _sequence_from_input(sequences[1])
    if not seq_a or not seq_b:
        return None
    result_a, result_b, result_complex = run_two_chain_assembly_hke(
        seq_a,
        seq_b,
        funnel_radius=FUNNEL_RADIUS,
        funnel_radius_exit=FUNNEL_RADIUS_EXIT,
        funnel_stiffness=1.0,
        hke_max_iter_s1=HKE_MAX_ITER[0],
        hke_max_iter_s2=HKE_MAX_ITER[1],
        hke_max_iter_s3=HKE_MAX_ITER[2],
        converge_max_disp_per_100_res=0.5,
        max_dock_iter=2000,
    )
    pdb_a = full_chain_to_pdb(result_a, chain_id="A")
    pdb_b = full_chain_to_pdb(result_b, chain_id="B")
    pdb_complex = full_chain_to_pdb_complex(
        result_complex["backbone_chain_a"],
        result_complex["backbone_chain_b"],
        result_a["sequence"],
        result_b["sequence"],
        chain_id_a="A",
        chain_id_b="B",
    )
    return (pdb_a, pdb_b, pdb_complex)


def _predict_cartesian_assembly_with_complex(sequences: list[str]) -> tuple[str, str, str] | None:
    """
    Fallback for 2-chain: Cartesian minimizer per chain (avoids HKE overflow on long chains),
    then placement + complex minimization. Same outputs as run_pending_assembly_job.py.
    """
    if len(sequences) != 2:
        return None
    seq_a = _sequence_from_input(sequences[0])
    seq_b = _sequence_from_input(sequences[1])
    if not seq_a or not seq_b:
        return None
    result_a = minimize_full_chain(
        seq_a, max_iter=100, long_chain_max_iter=80, include_sidechains=False
    )
    result_b = minimize_full_chain(
        seq_b, max_iter=100, long_chain_max_iter=80, include_sidechains=False
    )
    result_a, result_b, result_complex = run_two_chain_assembly(
        result_a, result_b, max_dock_iter=60, converge_max_disp_per_100_res=0.5
    )
    pdb_a = full_chain_to_pdb(result_a, chain_id="A")
    pdb_b = full_chain_to_pdb(result_b, chain_id="B")
    pdb_complex = full_chain_to_pdb_complex(
        result_complex["backbone_chain_a"],
        result_complex["backbone_chain_b"],
        result_a["sequence"],
        result_b["sequence"],
        chain_id_a="A",
        chain_id_b="B",
    )
    return (pdb_a, pdb_b, pdb_complex)


@app.route("/health", methods=["GET"])
def health():
    return Response("OK\n", status=200, mimetype="text/plain")


def _run_job_in_background(job_id: str) -> None:
    """Run prediction for job_id (read from pending .request.json); move to outputs and send email on success. No-op if .request.json missing or job already done."""
    req_path = os.path.join(PENDING_DIR, f"{job_id}.request.json")
    if not os.path.isfile(req_path):
        return
    try:
        with open(req_path) as f:
            req = json.load(f)
    except Exception:
        return
    sequences = req.get("sequences") or []
    seqs = [_sequence_from_input(s) for s in sequences if s]
    if not seqs:
        return
    to_email = (req.get("email") or "").strip()
    job_title = req.get("title")
    try:
        if USE_FAST_PREDICT and hqiv_predict_structure is not None:
            pdb = hqiv_predict_structure_assembly(sequences) if len(seqs) > 1 else hqiv_predict_structure(sequences[0])
            _move_to_outputs(job_id, pdb)
            if to_email:
                _send_pdb_email(to_email, pdb, job_title)
        elif len(seqs) == 2:
            assembly = None
            try:
                assembly = _predict_hke_assembly_with_complex(sequences)
            except Exception as e:
                app.logger.warning("HKE 2-chain failed, falling back to Cartesian: %s", e)
                assembly = _predict_cartesian_assembly_with_complex(sequences)
            if assembly is not None:
                pdb_a, pdb_b, pdb_complex = assembly
                _move_to_outputs_assembly(job_id, pdb_a, pdb_b, pdb_complex)
                if to_email:
                    _send_assembly_email(to_email, pdb_a, pdb_b, pdb_complex, job_title)
            else:
                pdb = _predict_hke_assembly(seqs)
                _move_to_outputs(job_id, pdb)
                if to_email:
                    _send_pdb_email(to_email, pdb, job_title)
        else:
            pdb = _predict_hke_assembly(seqs) if len(seqs) > 1 else _predict_hke_single(seqs[0])
            _move_to_outputs(job_id, pdb)
            if to_email:
                _send_pdb_email(to_email, pdb, job_title)
    except Exception as e:
        app.logger.warning("Background job %s failed: %s", job_id, e)
        if to_email and re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
            _send_job_failure_email(to_email, job_id, job_title, str(e))


@app.route("/predict", methods=["POST"])
def predict():
    """Accept FASTA in body, JSON, or form (sequence= / fasta= / title= / email=). Queues the job, returns 200 immediately with job_id; prediction runs in background. If email set and SMTP configured, PDB is sent by email when done."""
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
    _write_pending_request(job_id, sequences, job_title, to_email)

    # Run prediction in background; return 200 immediately once queued
    thread = threading.Thread(target=_run_job_in_background, args=(job_id,), daemon=True)
    thread.start()
    return Response(
        json.dumps({"status": "queued", "job_id": job_id}),
        status=200,
        mimetype="application/json",
    )


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


# On startup, run pending jobs once in background (retry or send failure email after MAX_ATTEMPTS)
def _start_pending_processor() -> None:
    t = threading.Thread(target=process_pending_jobs, daemon=True)
    t.start()


_start_pending_processor()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, threaded=True)
