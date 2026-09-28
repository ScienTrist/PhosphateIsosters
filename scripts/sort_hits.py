#!/usr/bin/env python3
"""
Sorts all hit files to their correct destination in one pass:

  hits/apo/                  — no isostere ligand found
  hits/holo/                 — isostere ligand found, not PLIF-confirmed
  hits/holo/plif_confirmed/  — isostere ligand found AND PLIF-confirmed

Scans every known location (root, apo/, holo/, holo/plif_confirmed/) so
already-sorted files are re-checked and moved if the classification changed.

Classification priority:
  1. isostere_full_list.csv  (from analyze_isosteres_simple.py)
  2. plif_results.json       (from analyze_plif.py) for confirmed status
  3. Physical proximity check (fallback when pair absent from CSV)
"""
import os
import sys
import glob
import json
import shutil
import csv
import gemmi

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MOTIF_DIR    = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
HITS_DIR     = os.path.join(MOTIF_DIR, "hits")
REFS_DIR     = os.path.join(MOTIF_DIR, "references")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from constants import STANDARD_AA, IGNORE_LIGANDS


# ── helpers ───────────────────────────────────────────────────────────────────

def get_ref_path(ref_id, chain, ref_lig, ref_num):
    for ch in [chain, chain.lower()]:
        for ext in [".cif", ".pdb"]:
            p = os.path.join(REFS_DIR, f"ref_{ref_id}_{ch}_{ref_lig}_{ref_num}{ext}")
            if os.path.exists(p):
                return p
    return None


def get_ref_p_pos(ref_path, ref_lig_name):
    """Return P-atom positions of the reference ligand (falls back to all atoms)."""
    try:
        st = gemmi.read_structure(ref_path)
    except Exception:
        return []
    p_pos, all_pos = [], []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == ref_lig_name:
                    all_pos.extend(a.pos for a in res)
                    p_pos.extend(a.pos for a in res if a.element.name == "P")
    return p_pos if p_pos else all_pos


def has_isostere(hit_path, ref_p_pos, cutoff=3.0):
    """Return True if any non-phosphate HETATM heteroatom is within cutoff of a P-atom."""
    if not ref_p_pos:
        return False
    try:
        st = gemmi.read_structure(hit_path)
    except Exception:
        return False
    for model in st:
        for chain in model:
            if not any(c.isupper() for c in chain.name):
                continue
            for res in chain:
                if res.name in STANDARD_AA or res.name in IGNORE_LIGANDS:
                    continue
                if any(a.element.name == "P" for a in res):
                    continue
                if gemmi.find_tabulated_residue(res.name).is_amino_acid():
                    continue
                min_d = min(
                    (a.pos.dist(p) for a in res
                     if a.element.name not in ("C", "H")
                     for p in ref_p_pos),
                    default=999.0
                )
                if min_d < cutoff:
                    return True
    return False


# ── data loaders ──────────────────────────────────────────────────────────────

def load_isostere_csv(motif_dir):
    """Returns {(ref, hit): 'APO'|'HOLO'}. When a pair has both, HOLO wins."""
    path = os.path.join(motif_dir, "isostere_full_list.csv")
    states = {}
    if not os.path.exists(path):
        return states
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row["Ref_ID"].upper(), row["Hit_ID"].upper())
            if states.get(k) != "HOLO":
                states[k] = row["State"].upper()
    return states


def load_confirmed_pairs(motif_dir):
    """Returns (confirmed_set, use_site_keys).
    If plif_results.json has ref_chain/ref_lig/ref_num fields, confirmed_set contains
    5-part (ref, hit, chain, lig, num) keys for precise per-site matching.
    Otherwise falls back to 2-part (ref, hit) keys (old JSON format)."""
    path = os.path.join(motif_dir, "plif_results.json")
    if not os.path.exists(path):
        return set(), False
    with open(path) as f:
        data = json.load(f)
    if not data:
        return set(), False

    use_site_keys = "ref_chain" in data[0]
    confirmed = set()
    for r in data:
        if use_site_keys:
            confirmed.add((
                r["ref"].upper(), r["hit"].upper(),
                r.get("ref_chain", "").upper(),
                r.get("ref_lig",   "").upper(),
                str(r.get("ref_num", "")),
            ))
        else:
            confirmed.add((r["ref"].upper(), r["hit"].upper()))
    return confirmed, use_site_keys


# ── main sort ─────────────────────────────────────────────────────────────────

def sort_hits():
    apo_dir       = os.path.join(HITS_DIR, "apo")
    holo_dir      = os.path.join(HITS_DIR, "holo")
    confirmed_dir = os.path.join(holo_dir, "plif_confirmed")

    for d in [apo_dir, holo_dir, confirmed_dir]:
        os.makedirs(d, exist_ok=True)

    isostere_states = load_isostere_csv(MOTIF_DIR)
    confirmed_pairs, use_site_keys = load_confirmed_pairs(MOTIF_DIR)

    if not isostere_states:
        print("Warning: isostere_full_list.csv not found - falling back to physical check for all files.")
    if not confirmed_pairs:
        print("Warning: plif_results.json not found - no files will be placed in plif_confirmed/.")
    else:
        mode = "precise site-key" if use_site_keys else "pair-key (re-run analyze_plif.py for precise matching)"
        print(f"  PLIF confirmed pairs: {len(confirmed_pairs)} ({mode})")

    # Collect every hit file regardless of current location
    search_dirs = [
        HITS_DIR,
        apo_dir,
        holo_dir,
        confirmed_dir,
    ]
    all_files = []
    for d in search_dirs:
        all_files.extend(glob.glob(os.path.join(d, "hit_*.pdb")))
        all_files.extend(glob.glob(os.path.join(d, "hit_*.cif")))

    total = len(all_files)
    print(f"Found {total} hit files across all locations.")
    if total == 0:
        print("Nothing to sort.")
        return

    moved    = {"apo": 0, "holo": 0, "confirmed": 0, "already_correct": 0, "error": 0}
    physical = 0

    for i, fpath in enumerate(all_files):
        if i % 500 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] apo={moved['apo']} holo={moved['holo']} "
                  f"confirmed={moved['confirmed']} correct={moved['already_correct']}",
                  end="\r", flush=True)

        fname  = os.path.basename(fpath)
        stem   = fname.replace(".pdb", "").replace(".cif", "")
        parts  = stem.split("_")
        if len(parts) < 6:
            moved["error"] += 1
            continue

        ref_id, hit_id, chain, ref_lig, ref_num = parts[1], parts[2], parts[3], parts[4], parts[5]
        pair_key = (ref_id.upper(), hit_id.upper())
        site_key = (ref_id.upper(), hit_id.upper(), chain.upper(), ref_lig.upper(), ref_num)

        # Determine APO vs HOLO
        csv_state = isostere_states.get(pair_key)
        if csv_state == "HOLO":
            is_holo = True
        elif csv_state == "APO":
            is_holo = False
        else:
            # Not in CSV — physical check
            ref_path = get_ref_path(ref_id, chain, ref_lig, ref_num)
            ref_p_pos = get_ref_p_pos(ref_path, ref_lig) if ref_path else []
            is_holo = has_isostere(fpath, ref_p_pos)
            physical += 1

        # Determine confirmed status — use site key if JSON has those fields, else pair key
        match_key    = site_key if use_site_keys else pair_key
        is_confirmed = is_holo and (match_key in confirmed_pairs)

        # Resolve destination
        if is_confirmed:
            dest_dir = confirmed_dir
        elif is_holo:
            dest_dir = holo_dir
        else:
            dest_dir = apo_dir

        dest_path = os.path.join(dest_dir, fname)
        current_dir = os.path.normpath(os.path.dirname(fpath))

        if os.path.normpath(dest_dir) == current_dir:
            moved["already_correct"] += 1
            continue

        try:
            shutil.move(fpath, dest_path)
            if is_confirmed:
                moved["confirmed"] += 1
            elif is_holo:
                moved["holo"] += 1
            else:
                moved["apo"] += 1
        except Exception as e:
            print(f"\n  ERROR moving {fname}: {e}")
            moved["error"] += 1

    print()  # newline after progress line
    total_moved = moved["apo"] + moved["holo"] + moved["confirmed"]
    print(f"\nSort complete - {total_moved} files moved, {moved['already_correct']} already correct.")
    print(f"  -> apo/              {moved['apo']}")
    print(f"  -> holo/             {moved['holo']}")
    print(f"  -> holo/plif_confirmed/  {moved['confirmed']}")
    if moved["error"]:
        print(f"  Errors: {moved['error']}")
    if physical:
        print(f"  Physical proximity check used for {physical} files (not in isostere CSV).")

    # Final counts
    n_apo       = len(glob.glob(os.path.join(apo_dir, "hit_*.*")))
    n_holo      = len(glob.glob(os.path.join(holo_dir, "hit_*.*")))
    n_confirmed = len(glob.glob(os.path.join(confirmed_dir, "hit_*.*")))
    print(f"\nCurrent totals:  apo={n_apo}  holo={n_holo}  confirmed={n_confirmed}")


if __name__ == "__main__":
    sort_hits()
