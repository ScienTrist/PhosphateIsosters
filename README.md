# Structural Classification of Phosphate-Binding Pockets

Code from my master's thesis, *"Structural Classification of Phosphate-Binding
Pockets: Toward the Rational Design of Phosphate Isosteres."*

## What this is

A structural bioinformatics pipeline that mines the RCSB PDB for
phosphate-binding sites, groups them by structural and sequence similarity,
and scores candidate small molecules as potential phosphate **bioisosteres**
(structural mimics that could replace a phosphate group while preserving its
binding interactions — relevant to drug design, since phosphate groups are
common but often pharmacokinetically problematic).

## Pipeline overview

1. **Site discovery** (`src/rcsb_search.py`, `src/pdb_downloader.py`) —
   query the RCSB PDB API for structures containing phosphate-group ligands,
   download and cache the relevant coordinate files.
2. **Homology enrichment** (`src/mmseqs_handler.py`, `src/blast_enrichment.py`,
   `src/uniprot_mapping.py`) — cluster sequences with MMseqs2 and BLAST to
   group binding sites by protein family rather than treating every PDB
   entry as independent.
3. **Structural superposition** (`src/tmalign_handler.py`,
   `src/tmalign_wrapper.py`, `scripts/batch_tmalign_parallel.py`) — TM-align
   every candidate pair of binding-site structures to obtain a rigid
   transform into a common reference frame.
4. **Motif extraction** (`scripts/batch_motif_extraction.py`,
   `pocket_types/`) — extract the local (~10 Å) residue environment around
   each phosphate site as a standalone structure, then cluster pocket
   geometries and amino-acid composition into recurring binding motifs.
5. **Isostere / interaction scoring** (`src/interaction_utils.py`,
   `src/ligand_detection_2D.py`, `src/ligand_detection_3D.py`,
   `scripts/analyze_isosteres_simple.py`, `scripts/analyze_plif.py`,
   `ProLIF_v2/`) — identify non-phosphate ligands occupying the aligned
   pocket and score their protein-ligand interaction fingerprint (PLIF)
   similarity to the native phosphate interactions, using both a
   homebrew geometric fingerprint and [ProLIF](https://github.com/chemosim-lab/ProLIF).
6. **Human-in-the-loop validation** (`ProLIF_v2/review_app/`) — **PHIP**
   (Phosphate Isostere Hit Picker), a small FastAPI + SQLite + 3Dmol.js web
   app built so colleagues could rate candidate isostere hits (1–5,
   ref/hit structures rendered directly in the browser) without needing
   PyMOL installed. Used to collect ground-truth judgments for calibrating
   the interaction-similarity score thresholds against human intuition.

## Stack

Python, [gemmi](https://gemmi.readthedocs.io/) (structure I/O), TM-align,
MMseqs2, [ProLIF](https://github.com/chemosim-lab/ProLIF), RDKit,
FastAPI/SQLite (review app).

## What's not in this repo

This is a curated view of the pipeline code for portfolio purposes. Raw
downloaded structures, generated manifests/caches, and result tables are
excluded, both because they're multi-hundred-MB regenerable artifacts and
because the underlying findings are part of an unpublished thesis. Running
the pipeline end-to-end from `src/rcsb_search.py` onward regenerates all of
these locally.
