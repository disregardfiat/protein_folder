# Failure analysis: Jobs 1772446236_5442 and 1772446236_5706

## Summary

Both jobs were marked failed at **13:10** on 2026-03-02 when `process_pending_jobs` ran on server startup: `attempts` was already 2, so the server sent the generic "did not complete after 2 attempts" email and moved their `.txt` to `outputs/*.failed.txt`. The **underlying exception** from the first attempt is not in the current gunicorn log (the run that failed was earlier; logs may have been overwritten or not captured).

---

## Job 1772446236_5442 (title=2026-02-28_00000338_full_151)

| Field | Value |
|-------|--------|
| num_sequences | **3** |
| lengths | **[376, 475, 166]** |
| email | sd-3d__2026-02-28_00000338__full-151@cameo3d.org |
| received_utc | 2026-03-02T10:10:36Z |

**Code path:** `len(seqs) == 3` → `_predict_hke_assembly(seqs)`. No 2-chain assembly; each chain is folded with `minimize_full_chain_hierarchical`, then `_merge_pdb_chains` produces one PDB (chains A, B, C).

**Root cause (fixed):** No **(A+B)+C** pipeline existed for 3+ chains; the server only did fold-each-chain + merge. We now use **`_predict_hke_assembly_multichain`**: dock A+B, then (A+B)+C, etc., and output one PDB with chains A, B, C, …

**Other possible contributors:**

1. **Resource / runtime**  
   Three long chains (376, 475, 166 residues) mean three full HKE runs and large structures in memory. Possible OOM, or worker killed (timeout 259200s is large but not infinite).

2. **Numerical / convergence (long chains)**  
   Logic for 3+ chains is “fold each chain + merge”; if anything assumes exactly 2 chains or has an indexing/merge bug, it could raise.

3. **Numerical / convergence**  
   Long chains (e.g. 475) can stress the hierarchical minimizer; possible unhandled exception or bad state.

---

## Job 1772446236_5706 (title=2026-02-28_00000119_full_151)

| Field | Value |
|-------|--------|
| num_sequences | **1** |
| lengths | **[334]** |
| email | sd-3d__2026-02-28_00000119__full-151@cameo3d.org |
| received_utc | 2026-03-02T10:10:36Z |

**Code path:** `_predict_hke_single(seqs[0])` → `minimize_full_chain_hierarchical` (single chain, 334 residues).

**Likely causes:**

1. **Long-chain load**  
   334 residues is a heavy single run; possible memory use or runtime leading to kill/timeout.

2. **Exception in hierarchical minimizer**  
   e.g. in funnel, L-BFGS, or backbone placement for long chains.

3. **Worker killed**  
   OOM or external kill; no Python traceback in that case.

---

## What we don’t have

- **Exact exception:** The failure from the **first** attempt would have been sent by `_send_job_failure_email(to_email, job_id, job_title, str(e))`, so the CAMEO address should have received an email that includes the exception string. We don’t have that email or a copy of it.
- **Traceback in logs:** `app.logger.warning("Pending job %s retry failed: %s", job_id, e)` would have been emitted when the job failed; that goes to gunicorn’s stderr. The current `/tmp/casp_gunicorn.log` only shows bind errors from later restarts, not that run.
- **Request payload:** When marking a job as “failed after 2 attempts”, the server deletes `job_id.request.json`, so we can’t re-inspect the exact sequences/FASTA from disk.

---

## Recommendations

1. **Persist last exception for “max attempts”**  
   When a retry fails, write the exception (and optionally traceback) into the job’s `.txt` (e.g. `last_error=...`) or a small `job_id.last_error` file. When sending the final “did not complete after 2 attempts” email, include that text so the user (and you) see the real error.

2. **Persistent app log**  
   Ensure gunicorn (or a wrapper) logs worker stderr to a persistent, rotated file (e.g. under `casp_results/` or `/var/log`) so “Pending job X retry failed: …” and tracebacks survive restarts.

3. **Re-run / debug**  
   To see the exact error for similar jobs, re-submit a small test (e.g. one short chain and one 3-chain with short sequences) and watch the same log file and failure email.

4. **3-chain and long-chain testing**  
   Run `_predict_hke_assembly` for 3 chains and `_predict_hke_single` for a 334-residue (or similar) chain in a dev environment with a single worker and full stderr capture to reproduce any exception.
