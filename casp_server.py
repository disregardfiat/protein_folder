"""
CASP-compliant HTTP server: FASTA in → PDB out.
POST /predict with body = FASTA (or Content-Type: application/json with {"fasta": "..."})
  → 200 + PDB text (CASP format).
GET /health → 200 OK (for reverse proxy / liveness).
Run from repo root: gunicorn -w 1 -b 127.0.0.1:8050 casp_server:app
"""

from __future__ import annotations

import sys
from flask import Flask, request, Response

# Ensure repo root is on path when run via gunicorn
if __name__ != "__main__":
    import os
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from horizon_physics.proteins import hqiv_predict_structure

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB max FASTA


@app.route("/health", methods=["GET"])
def health():
    return Response("OK\n", status=200, mimetype="text/plain")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept FASTA in body or JSON {"fasta": "..."}; return CASP-format PDB."""
    fasta = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        fasta = data.get("fasta") or data.get("sequence")
    if fasta is None:
        fasta = request.get_data(as_text=True)
    if not fasta or not fasta.strip():
        return Response("Missing FASTA in body or JSON 'fasta'\n", status=400, mimetype="text/plain")
    try:
        pdb = hqiv_predict_structure(fasta.strip())
        return Response(pdb, status=200, mimetype="text/plain")
    except Exception as e:
        return Response(f"Prediction failed: {e}\n", status=500, mimetype="text/plain")


@app.route("/", methods=["GET"])
def index():
    return Response(
        "HQIV CASP server. POST FASTA to /predict, GET /health for liveness.\n",
        status=200,
        mimetype="text/plain",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, threaded=True)
