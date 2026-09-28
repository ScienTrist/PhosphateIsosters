# PHIP — Phosphate Isostere Hit Picker

Lets colleagues rate, in a browser (no PyMOL install needed), how convincing
each candidate hit ligand is as a phosphate isostere for its paired
reference site. Collected 1-5 scores + optional comments are meant to be
joined back against `prolif_plif_score_no_vdw` / `prolif_plif_score_with_vdw`
/ `ref_n_bits_novdw` to pick the `min_ref_int` / PLIF-score thresholds for
ProLIF_v2.

Reviewers never see either PLIF score or `ref_n_bits_novdw` while rating —
showing them would anchor their judgment to the numbers being calibrated.

## Two PLIF scores

The pool carries **both** versions of `prolif_plif_score` for every pair,
because they answer different questions:

- **`prolif_plif_score_with_vdw`** — the original site-level score (from
  `results/plif_vs_tanimoto_comparison_full_manifest.csv`, the "main
  manifest" of scored pairs), including plain van-der-Waals contacts. This
  is the score `build_ifg_prolif_dataset.py`'s `--min-score` filters on.
- **`prolif_plif_score_no_vdw`** — a stricter, per-pair recompute done by
  `build_ifg_prolif_dataset.py` itself that excludes VdWContact-only
  matches (see that script's `ATOM_MARKING_EXCLUDED_INTERACTIONS`). This is
  also what drives the red/purple atom highlighting shown in the viewer,
  and what `build_pool.py`'s sampling weight is centered on.

Both come back in `/api/export` so you can compare reviewer ratings against
either.

## Which ifg dataset the pool is built from

`build_ifg_prolif_dataset.py`'s output filename doesn't encode its own
`--min-score`/`--min-ref-bits` settings (it always defaults to
`ifg_prolif_dataset_full_manifest.csv`), so re-runs with different
thresholds silently overwrite each other unless `--out` is given explicitly.
Two labeled datasets exist in `results/`:

- `ifg_prolif_dataset_full_manifest_minscore0.5_minrefbits2.csv` — the
  original dataset (candidates pre-filtered to site score ≥0.5 and
  reference sites with >2 non-VdW interaction bits). Only covers pairs
  already judged plausible by the pipeline's current thresholds.
- `ifg_prolif_dataset_full_manifest_minscore0_minrefbits1.csv` — broader
  dataset (score ≥0, ref bits >1) that also covers weaker/borderline pairs,
  needed so reviewer ratings can inform *where* the thresholds should sit
  rather than only tuning within a range already pre-filtered to "probably
  fine". **This is what `build_pool.py` uses by default.** See "Generating
  a new ifg dataset" below for how to build it (or regenerate it with
  different settings).

## What it shows

Each pair is the reference pocket (its own crystal frame) and the hit
pocket, TM-aligned into the reference's frame using the pipeline's own
saved TM-align transform, both loaded into one [3Dmol.js](https://3dmol.org)
viewer the reviewer can freely rotate/zoom/pan. The hit ligand is colored
using the same red (isosteric-matched atom) / purple (same group,
unmatched) classification the pipeline already computed
(`red_atom_names`/`purple_atom_names`, from the no-VdW recompute).

## One-time setup

```
pip install -r requirements.txt
cd backend
python build_pool.py            # builds data/review.db + data/aligned_hits/*.pdb
```

Requires `ifg_prolif_dataset_full_manifest_minscore0_minrefbits1.csv` to
already exist in `results/` — see "Generating a new ifg dataset" below if it
doesn't yet. Pass `--ifg-csv path\to\other.csv` to build the pool from a
different dataset instead.

Re-run `build_pool.py` any time the underlying CSV/manifest changes.
`--shared-core N` controls how many pairs are in the shared pool (default
200). Every reviewer's assignments are drawn exclusively from this shared
set (see app.py's `/api/start`), so all reviewers rate the *same*
structures — there's no separate per-reviewer individualized portion. The
draw always samples toward the pool's own median no-VdW
`prolif_plif_score` / `ref_n_bits_novdw` — the ambiguous middle of the
distribution — rather than uniformly, since that's the region that
actually decides where the threshold should sit, but it's not an
exclusive draw from the middle either: pairs from the rest of the
distribution can still be picked, just with lower weight.

## Step 0: TM-align transform file

`build_pool.py` resolves its TM-align transform file the same way as
`scripts/plip_isostere/common.py`, `scripts/plip_isostere/resolve_pairs.py`,
and `ProLIF_v2/scripts/run_prolif.py` do (each has its own
`_default_tmalign_json()`): prefer
`Phosphate-binding-site/results/tmalign_results_{RUN_ID}_clean.json` (a
from-scratch regeneration with no reuse of any older file), falling back to
the plain `tmalign_results_{RUN_ID}.json` only if the clean one isn't there.
It never looks at a `_corrected.json` file — that naming convention is
retired. See `results/archived_tmalign_results/README.md` for why: an
earlier attempt to patch the original alignment output for a pairing-logic
fix (`scripts/regenerate_tmalign_corrected.py`) silently dropped ~9,700
pairs it should have kept while still carrying ~17,900 pairs from an
abandoned pairing scheme. `scripts/regenerate_tmalign_clean.py` redid the
alignment from scratch under the current pairing logic instead, verified
complete against `full_manifest.csv`'s target pairs, and that `_clean.json`
is what's live in `results/` today.

If `full_manifest.csv` grows to include structures added *after* the
`_clean.json` snapshot, some new pairs might genuinely lack a transform —
regenerate with `scripts/regenerate_tmalign_clean.py` (needs WSL, since
`TMalign` runs through it on Windows) rather than reaching for the retired
`_corrected` scripts.

## Generating a new ifg dataset

`build_pool.py` needs an `ifg_prolif_dataset_*.csv` to build from.
`scripts/build_ifg_prolif_dataset.py` generates one from `full_manifest.csv`
+ the cached ProLIF fingerprint — it's a multi-hour, all-CPU-cores job
(re-runs one ProLIF pass per candidate pair), so run it yourself rather than
through this app, **after** Step 0 above. From PowerShell, in
`ProLIF_v2\scripts`:

```powershell
cd D:\AI\Master_thesis\Phosphate-binding-site\ProLIF_v2\scripts
python build_ifg_prolif_dataset.py full_manifest.csv `
    --min-score 0 `
    --min-ref-bits 1 `
    --out ..\results\ifg_prolif_dataset_full_manifest_minscore0_minrefbits1.csv `
    --error-log ..\results\ifg_prolif_dataset_errors_full_manifest_minscore0_minrefbits1.log
```

Notes:
- **Always pass `--out`** (and ideally `--error-log`) explicitly — without
  it, the script defaults to `ifg_prolif_dataset_full_manifest.csv` and will
  silently overwrite whichever dataset currently has that name.
- Lowering `--min-score`/`--min-ref-bits` widens the candidate pool
  (more unique reference sites need a fresh non-VdW-bits pre-pass, since
  `results/ref_novdw_bits_by_site_full_manifest.csv` only caches the sites
  needed by whatever run built it last — expect this pre-pass to take longer
  than the ~5 min the original min-score-0.5 run needed, since min-score-0
  pulls in ~665 unique reference sites vs. 383 before, and a noticeably
  longer main run too given more candidate pairs overall). The console
  prints a `[i/total ... ETA]` line per completed pair once the main run
  starts — if you're watching and see nothing yet, it's still in this
  silent pre-pass, not stuck.
- Once it finishes, `cd ..\review_app\backend` and re-run `python
  build_pool.py` to rebuild the review pool from the new dataset, then
  restart `uvicorn` so it picks up the new `review.db`.

## Running

```
REVIEW_ACCESS_CODE="pick-a-real-code" REVIEW_ADMIN_TOKEN="pick-a-real-token" \
REVIEW_SECONDS_PER_STRUCTURE=60 \
uvicorn app:app --host 0.0.0.0 --port 8000
```
(run from `backend/`). Set `REVIEW_SECONDS_PER_STRUCTURE` to your real
estimated seconds-per-structure once you have one — it drives the time
estimate shown on the landing page.

Without setting `REVIEW_ACCESS_CODE`/`REVIEW_ADMIN_TOKEN` the app runs with
obvious placeholder defaults and prints a warning — fine for local testing,
**do not** expose it to the internet without setting both.

## Exposing it to colleagues remotely

Since reviewers are remote, run the server as above, then tunnel port 8000,
e.g. with [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/):

```
cloudflared tunnel --url http://localhost:8000
```

or `ngrok http 8000`. Share the resulting URL + the access code you set
(out of band, e.g. Slack/email — not in the same message as the URL if you
want any real separation). The access code isn't real authentication, just
a filter against randoms who stumble on the tunnel URL.

## Collecting results

```
curl "http://<host>:8000/api/export?token=<REVIEW_ADMIN_TOKEN>" -o ratings.csv
```

Ratings are append-only (a reviewer re-rating a pair adds a new row, not an
overwrite) — dedupe/aggregate as you see fit when analyzing (e.g. keep
latest per `(reviewer_name, pair_id)`, or average). `is_shared=1` rows are
the ones every reviewer rated in common; use those to check inter-rater
agreement before trusting the rest of the sample for threshold-picking.

## Files

- `backend/build_pool.py` — data prep (CSV -> `data/review.db` + aligned PDBs)
- `backend/app.py` — FastAPI backend
- `backend/db.py` — SQLite connection helper + paths
- `static/` — frontend (plain HTML/CSS/JS + vendored `3Dmol-min.js`, no build step)
- `data/review.db` — SQLite: `pairs`, `reviewers`, `assignments`, `ratings`
- `data/aligned_hits/*.pdb` — hit pocket PDBs pre-transformed into ref frame, one per `(ref_site, hit_site)` pair
