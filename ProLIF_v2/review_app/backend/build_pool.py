"""
DEPRECATED -- do not run this as a script anymore. Use
build_pool_from_discovery.py instead, which builds review.db from
discover_candidates.py's own output (the no-VdW-consistent, single-source-
of-truth phosphate assignment) instead of the old full_manifest.csv/
plif_vs_tanimoto_comparison_full_manifest.csv/ifg_prolif_dataset_*.csv
chain this script reads. That old chain had three independent
re-derivations of "which reference phosphate does this hit match," which
could (and did) disagree with each other on multi-phosphate references --
see build_pool_from_discovery.py's own docstring.

This file is kept, and its helper functions (compute_binding_residues,
write_transformed_pdb, gaussian_weight, _actual_ligand_resname, etc.) are
still imported and reused by build_pool_from_discovery.py -- only running
IT AS A SCRIPT (`python build_pool.py`) is deprecated; main() below refuses
to run without --i-know-this-is-the-old-chain for exactly that reason. The
old full_manifest.csv/*_full_manifest.* data files themselves are left in
place, deliberately NOT archived/moved -- ~25 other analysis scripts
(pocket_types/*, generate_highlighted_pngs.py, explain_pair.py,
render_dataset_pngs.py, pymol_browse_homebrew_vs_prolif.py, etc.) still
depend on them for work unrelated to PHIP.

Original docstring, for the old (still-importable, no-longer-run) pipeline
this describes:

One-time (re-runnable) data-prep step for the isostere-validity review app.

Reads results/ifg_prolif_dataset_full_manifest_minscore0_minrefbits1.csv
(one row per ref<->hit site pair, already carrying a *no-VdW* recomputed
prolif_plif_score, ref_n_bits_novdw, and the red_atom_names/purple_atom_names
isosteric-atom highlight computed by the main pipeline -- see that script's
own docstring for why the recomputed score excludes VdWContact), joins
chain/resnum/resname info from full_manifest.csv, joins the *with-VdW*
site-level prolif_plif_score back in from
results/plif_vs_tanimoto_comparison_full_manifest.csv (the "main manifest"
of scored pairs -- this is the original, unfiltered score used to select
candidates in the first place, before the no-VdW recompute), applies the
pipeline's own TM-align (t, u) rigid-body transform to bring each hit pocket
into its reference's coordinate frame (pocket PDBs are written in
native/unaligned frame -- see extract_pockets.py), and writes:

  - review_app/data/review.db          SQLite `pairs` table the FastAPI
                                        backend reads from
  - review_app/data/aligned_hits/*.pdb one coordinate-transformed copy of
                                        each hit pocket PDB, keyed by
                                        pair_id (a hit can be paired with
                                        more than one reference, each with
                                        its own transform)

Sampling weight is a Gaussian kernel centered on the pool's own median
no-VdW prolif_plif_score and median ref_n_bits_novdw, so pairs need not
cluster at the extremes to be selected -- ambiguous middle-of-the-distribution
pairs (the ones that actually matter for picking a threshold) are
oversampled relative to a uniform draw, without a hard cutoff that would
exclude the extremes entirely.

Usage: python build_pool.py [--ifg-csv path] [--shared-core 40] [--seed 0]
"""
import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import sqlite3
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_APP_ROOT = os.path.dirname(SCRIPT_DIR)
PROLIF_V2_ROOT = os.path.dirname(REVIEW_APP_ROOT)
PHOSPHATE_ROOT = os.path.dirname(PROLIF_V2_ROOT)

RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")

# min-score 0 / min-ref-bits 1 -- broader than the original min-score
# 0.5 / min-ref-bits 2 dataset (kept at
# ifg_prolif_dataset_full_manifest_minscore0.5_minrefbits2.csv), so the
# review pool also covers pairs below the pipeline's current candidate
# thresholds, letting reviewer ratings inform where those thresholds should
# actually sit rather than only tuning within a range already pre-filtered
# to "probably fine". Generate it with build_ifg_prolif_dataset.py --min-score
# 0 --min-ref-bits 1 (see review_app/README.md) before running this script.
DEFAULT_IFG_CSV = os.path.join(RESULTS_DIR, "ifg_prolif_dataset_full_manifest_minscore0_minrefbits1.csv")
MAIN_MANIFEST_SCORES_CSV = os.path.join(RESULTS_DIR, "plif_vs_tanimoto_comparison_full_manifest.csv")
FULL_MANIFEST_CSV = os.path.join(PROLIF_V2_ROOT, "full_manifest.csv")
POCKETS_DIR = os.path.join(PROLIF_V2_ROOT, "data", "pockets")

_TMALIGN_RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"


def _default_tmalign_json():
    """Prefers the from-scratch clean regeneration over the plain (older,
    pre-single-assignment-pairing) file -- never the archived _corrected.json,
    see results/archived_tmalign_results/README.md for why. Same resolution
    order as ProLIF_v2/scripts/run_prolif.py and
    scripts/plip_isostere/resolve_pairs.py's own _default_tmalign_json()."""
    results_dir = os.path.join(PHOSPHATE_ROOT, "results")
    clean = os.path.join(results_dir, f"tmalign_results_{_TMALIGN_RUN_ID}_clean.json")
    plain = os.path.join(results_dir, f"tmalign_results_{_TMALIGN_RUN_ID}.json")
    if os.path.isfile(clean):
        return clean
    if os.path.isfile(plain):
        return plain
    raise FileNotFoundError(
        f"No TM-align results found for run_id={_TMALIGN_RUN_ID} (checked {clean}, {plain})"
    )


TMALIGN_JSON = _default_tmalign_json()

SCRIPTS_DIR = os.path.join(PROLIF_V2_ROOT, "scripts")
FP_PICKLE = os.path.join(RESULTS_DIR, "prolif_fingerprint_full_manifest.pkl")

DATA_DIR = os.path.join(REVIEW_APP_ROOT, "data")
ALIGNED_HITS_DIR = os.path.join(DATA_DIR, "aligned_hits")
DB_PATH = os.path.join(DATA_DIR, "review.db")
# Display-corrected copies of reference pockets -- see the matching
# docstring on REF_POCKETS_DIR in db.py for why this exists (canonical
# data/pockets/*.pdb shows phosphates in their neutral, fully-protonated
# form; this copy strips the terminal-oxygen hydrogens ProLIF's own scoring
# already treats as absent, so what's displayed matches what's scored).
REF_POCKETS_DIR = os.path.join(DATA_DIR, "ref_pockets_display")

MEDIAN_SCORE = 0.5   # prolif_plif_score is already 0-1; recomputed below from the pool anyway
SIGMA_SCORE = 0.28
SIGMA_BITS = 0.35     # applied to bits distance normalized by observed (max-min) range


def load_transform_index():
    with open(TMALIGN_JSON, "r") as f:
        return json.load(f)


def get_transformation(idx, ref_pdb, hit_pdb):
    for entry in idx.get(ref_pdb.upper(), []):
        if entry.get("hit", "").upper() == hit_pdb.upper():
            t = entry["transformation"]["t"]
            u = entry["transformation"]["u"]
            return t, u
    return None


def apply_transform(t, u, xyz):
    x, y, z = xyz
    return (
        t[0] + u[0][0] * x + u[0][1] * y + u[0][2] * z,
        t[1] + u[1][0] * x + u[1][1] * y + u[1][2] * z,
        t[2] + u[2][0] * x + u[2][1] * y + u[2][2] * z,
    )


def write_transformed_pdb(src_path, dst_path, t, u):
    with open(src_path, "r") as f:
        lines = f.readlines()
    out = []
    for line in lines:
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM") and len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                out.append(line)
                continue
            nx, ny, nz = apply_transform(t, u, (x, y, z))
            line = f"{line[:30]}{nx:8.3f}{ny:8.3f}{nz:8.3f}{line[54:]}"
        out.append(line)
    with open(dst_path, "w") as f:
        f.writelines(out)


def load_full_manifest():
    by_site = {}
    with open(FULL_MANIFEST_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_site[row["site_id"]] = row
    return by_site


def load_with_vdw_scores():
    """(ref_site, hit_site) -> site-level prolif_plif_score straight from
    the main pairwise-comparison manifest, i.e. WITH VdWContact bits
    included -- this is a different number than the ifg dataset's own
    recomputed (no-VdW) prolif_plif_score column."""
    scores = {}
    with open(MAIN_MANIFEST_SCORES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["prolif_plif_score"] != "":
                scores[(row["ref_site"], row["hit_site"])] = float(row["prolif_plif_score"])
    return scores


EXCLUDED_INTERACTIONS = {"VdWContact"}  # matches export_prolif_datawarrior.py's ATOM_MARKING_EXCLUDED_INTERACTIONS


def _residue_ids_to_rows(residue_ids):
    rows = {str(r): [r.name, r.number, r.chain] for r in residue_ids}
    return sorted(rows.values(), key=lambda r: (r[2] or "", r[1]))


def _actual_ligand_resname(pdb_path, resnum, chain):
    """Reads the REAL residue name written in a pocket PDB's own ATOM/HETATM
    records for this ligand instance, instead of trusting
    full_manifest.csv's lig_resname column.

    Legacy PDB format truncates 4-5 character extended CCD codes (e.g.
    "A1ACZ") to 3 characters on every write in this pipeline
    (extract_pockets.py) -- confirmed directly: hit_7GYK_A1ACZ_415's own
    pocket PDB has "A1A" at that resi/chain, not "A1ACZ". The manifest still
    carries the untruncated code, so a viewer selecting atoms by {resn:
    lig_resname} (as review_app/static/app.js does) matches zero atoms for
    any such ligand -- it renders nothing and looks broken with no error.
    resid+chain alone already uniquely identifies the ligand instance (same
    convention run_prolif.py's run_site()/phosphate_ifp.py's
    phosphate_group_ifps() rely on to avoid this exact truncation problem),
    so the first ATOM/HETATM record at that resi+chain IS the ligand
    regardless of what its resname says. Returns None (caller should fall
    back to the manifest value) if the file can't be read or no matching
    record is found."""
    target_resi = str(resnum).strip()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line[:6].strip() not in ("ATOM", "HETATM"):
                    continue
                if len(line) < 26:
                    continue
                if line[21].strip() == chain and line[22:26].strip() == target_resi:
                    return line[17:20].strip()
    except OSError:
        return None
    return None


def _pdb_atom_names(mol, idxs):
    """RDKit atom indices -> PDB atom name strings, via the same
    GetPDBResidueInfo().GetName() convention export_prolif_datawarrior.py's
    atom_names() uses -- these line up with the pocket PDB's own HETATM atom
    names as long as `mol` was built the same way (safe_molecule_from_mda on
    the site's ligand selection), same precondition red/purple atom names
    already rely on for the hit side."""
    names = []
    for i in sorted(idxs):
        info = mol.GetAtomWithIdx(i).GetPDBResidueInfo()
        names.append(info.GetName().strip() if info else f"idx{i}")
    return names


# ---- parallel per-site precompute workers ----
# Reference-site (ProLIF phosphate_group_ifps -- one fp.generate() pass per
# unique reference ligand) and hit-site (ligand + protein atom-name maps) setup
# are each independent per site: no site's result depends on any other site's,
# only candidates' pairing structure below does. That makes them the actual
# parallelizable part of this script -- confirmed the bottleneck by watching a
# sequential run spend 20+ minutes here (vs ~2-3 min before the protein
# atom-name lookup was added) while CPU sat pinned at ~100% on a single core.
#
# Module-level (not nested closures) and picklable by design: multiprocessing
# on Windows uses "spawn", which re-imports this module fresh in each worker
# process rather than forking the parent's memory -- a worker can only be
# handed something reachable as a plain module-level name, not a function
# capturing rp/pif/mda from an enclosing scope. _init_worker runs once per
# worker process (via ProcessPoolExecutor's initializer=) and stashes the
# heavy, read-only, shareable-per-process state (the sys.path fix-up, the
# module imports, the ProLIF fingerprint pickle) in module globals instead of
# passing them through IPC on every task.
_worker_rp = _worker_pif = _worker_mda = _worker_fp = None


def _init_worker(fp_pickle_path=None):
    """fp_pickle_path overrides the module-level FP_PICKLE -- needed (not just
    a convenience) because ProcessPoolExecutor uses Windows's "spawn" start
    method, which re-imports this module fresh in each worker rather than
    inheriting the parent's already-mutated globals; a caller that wants a
    different pickle (e.g. the discover_candidates.py-driven pool builder's
    fresh, old-chain-free one) MUST pass it here via initargs=, not by
    monkeypatching build_pool.FP_PICKLE in the parent process beforehand."""
    global _worker_rp, _worker_pif, _worker_mda, _worker_fp
    sys.path.insert(0, SCRIPTS_DIR)
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda
    import run_prolif as rp
    import phosphate_ifp as pif
    import prolif as plf
    _worker_rp, _worker_pif, _worker_mda = rp, pif, mda
    _worker_fp = plf.Fingerprint.from_pickle(fp_pickle_path or FP_PICKLE)


def _mol_atom_names_by_idx(mol):
    names = {}
    for i, a in enumerate(mol.GetAtoms()):
        info = a.GetPDBResidueInfo()
        if info:
            names[i] = info.GetName().strip()
    return names


def _phosphate_h_names_to_strip(lig_mol, groups):
    """PDB atom names of the specific hydrogens that should be removed to
    show a reference ligand's phosphate group(s) in their physiologically
    correct, anionic form -- the exact same rule phosphate_ifp.py's
    _deprotonate_ionizable_oxygens applies (a terminal, non-bridging oxygen
    that currently has an attached H gets deprotonated; a bridging P-O-P
    oxygen never had an H to begin with and is correctly left alone), just
    read out here as PDB atom names instead of being applied as an in-memory
    RDKit formal-charge edit. Covers every phosphate group on the ligand
    (not just whichever one a given pair happens to score against), since
    display should be correct regardless of which pairing is being viewed."""
    strip_names = set()
    for group in groups:
        for idx in group["atom_idxs"][1:]:  # [0] is the P atom itself
            atom = lig_mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != "O":
                continue
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "H":
                    info = nbr.GetPDBResidueInfo()
                    if info:
                        strip_names.add(info.GetName().strip())
    return strip_names


def _write_display_corrected_pdb(src_path, dst_path, lig_resnum, lig_chain, strip_atom_names):
    """Copies a pocket PDB, dropping:
    (1) the specific ligand hydrogen atom records named in strip_atom_names
        (matched by atom name + resi + chain, so a same-named H elsewhere in
        the file -- e.g. on a different residue -- is never touched), and
    (2) any ligand hydrogen sitting at the literal origin (0.000, 0.000,
        0.000) regardless of name -- a real atom in these native
        crystallographic-frame pocket files is never actually there;
        confirmed this is RDKit's AddHs(addCoords=True) silently failing to
        place a hydrogen (64/376 reference sites in the current pool have at
        least one), which 3Dmol.js's own bond perception then connects to
        whatever's nearest with a nonsensically long stick. Both are a
        pre-existing upstream data issue, not introduced by this fix.
    Also strips any CONECT record entry referencing a dropped atom's serial
    number, so nothing is left pointing at an atom that's no longer there.
    Text-level line removal rather than a full RDKit round-trip: preserves
    everything else in the file (protein, waters, formatting) byte-for-byte."""
    target_resi = str(lig_resnum).strip()
    with open(src_path, "r") as f:
        lines = f.readlines()

    dropped_serials = set()
    out = []
    for line in lines:
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM") and len(line) >= 54:
            name = line[12:16].strip()
            chain = line[21].strip()
            resi = line[22:26].strip()
            elem = (line[76:78].strip() if len(line) >= 78 else "") or name[:1]
            is_ligand_atom = chain == lig_chain and resi == target_resi
            is_named_strip = name in strip_atom_names and is_ligand_atom
            is_origin_h = elem.upper() == "H" and is_ligand_atom and line[30:54] == "   0.000   0.000   0.000"
            if is_named_strip or is_origin_h:
                try:
                    dropped_serials.add(int(line[6:11]))
                except ValueError:
                    pass
                continue
        out.append(line)

    if dropped_serials:
        filtered = []
        for line in out:
            if line[:6].strip() == "CONECT":
                raw = [line[i:i + 5].strip() for i in range(6, len(line.rstrip("\n")), 5)]
                serials = [s for s in raw if s.isdigit()]
                if not serials or int(serials[0]) in dropped_serials:
                    continue  # malformed, or the record's own atom was dropped
                kept = [s for s in serials[1:] if int(s) not in dropped_serials]
                if not kept:
                    continue  # nothing bonded remains once dropped neighbors are removed
                filtered.append("CONECT" + f"{serials[0]:>5}" + "".join(f"{s:>5}" for s in kept) + "\n")
            else:
                filtered.append(line)
        out = filtered

    with open(dst_path, "w") as f:
        f.writelines(out)


def _compute_ref_site_worker(args):
    """One reference site's independent setup: its phosphate-group ifps
    (groups), each group's flat atom-name list (ref_group_atoms_by_site --
    thick-stick rendering), each group's FRAGMENT atom-index -> name map
    (ref_frag_atom_names_by_site -- see that dict's docstring note in
    compute_binding_residues for why this can't reuse the whole ligand's
    indices), and this site's own protein atom-index -> name map. Also
    writes a display-corrected copy of this site's pocket PDB to
    REF_POCKETS_DIR (phosphate hydroxyl hydrogens stripped to their
    anionic form -- see _phosphate_h_names_to_strip) as a side effect, so
    the fix happens once per unique site rather than once per pair. Runs in a
    worker process; any exception here is caught and logged rather than
    killing the whole parallel batch over one malformed site -- the caller
    treats a caught failure the same as this site having nothing (empty
    dicts), same as the original sequential version already did for e.g. a
    missing pocket file."""
    ref_sid, ref_row = args
    rp, pif, mda, fp = _worker_rp, _worker_pif, _worker_mda, _worker_fp
    try:
        groups, err = pif.phosphate_group_ifps(ref_row, fp) if ref_row else (None, "no manifest row")
        groups = groups or {}

        atoms_by_p_idx = {}
        frag_names_by_p_idx = {}
        pocket_path = os.path.join(POCKETS_DIR, f"{ref_sid}.pdb")
        prot_names = {}
        if os.path.isfile(pocket_path):
            u = mda.Universe(pocket_path)
            if groups:
                ligand = u.select_atoms(
                    f"resid {ref_row['lig_resnum']} and chainID {ref_row['chain']} and not protein")
                if len(ligand) > 0:
                    lig_mol = rp.safe_molecule_from_mda(ligand, use_segid=False)
                    lig_groups = pif.get_ligand_phosphate_groups(lig_mol)
                    for group in lig_groups:
                        atoms_by_p_idx[group["p_idx"]] = _pdb_atom_names(lig_mol, group["atom_idxs"])
                        frag_names = {}
                        for new_idx, old_idx in enumerate(group["atom_idxs"]):
                            info = lig_mol.GetAtomWithIdx(old_idx).GetPDBResidueInfo()
                            frag_names[new_idx] = info.GetName().strip() if info else f"idx{old_idx}"
                        frag_names_by_p_idx[group["p_idx"]] = frag_names
                    strip_names = _phosphate_h_names_to_strip(lig_mol, lig_groups)
                    _write_display_corrected_pdb(
                        pocket_path, os.path.join(REF_POCKETS_DIR, f"{ref_sid}.pdb"),
                        ref_row["lig_resnum"], ref_row["chain"], strip_names)
            protein = rp.protein_selection(u)
            if len(protein) > 0:
                prot_mol = rp.safe_molecule_from_mda(protein, use_segid=False)
                prot_names = _mol_atom_names_by_idx(prot_mol)
        return ref_sid, groups, atoms_by_p_idx, frag_names_by_p_idx, prot_names, None
    except Exception as e:  # noqa: BLE001 -- one bad site shouldn't sink the batch
        return ref_sid, {}, {}, {}, {}, f"{type(e).__name__}: {e}"


def _compute_hit_site_worker(args):
    """One hit site's independent setup: its whole-ligand atom-index -> name
    map (fix_ionizable=True, matching export_prolif_datawarrior.py's
    build_hit_ligand_mol -- the convention red/purple atom names already
    depend on) and its whole-protein atom-index -> name map. No fingerprint
    needed: hits are never re-fingerprinted here, only the cached ifp's own
    metadata gets consulted later against these name maps."""
    hit_sid, hit_row = args
    rp, mda = _worker_rp, _worker_mda
    try:
        lig_names = {}
        prot_names = {}
        pocket_path = os.path.join(POCKETS_DIR, f"{hit_sid}.pdb")
        if os.path.isfile(pocket_path):
            u = mda.Universe(pocket_path)
            ligand = u.select_atoms(
                f"resid {hit_row['lig_resnum']} and chainID {hit_row['chain']} and not protein")
            if len(ligand) > 0:
                lig_mol = rp.safe_molecule_from_mda(ligand, use_segid=False, fix_ionizable=True)
                lig_names = _mol_atom_names_by_idx(lig_mol)
            protein = rp.protein_selection(u)
            if len(protein) > 0:
                prot_mol = rp.safe_molecule_from_mda(protein, use_segid=False)
                prot_names = _mol_atom_names_by_idx(prot_mol)
        return hit_sid, lig_names, prot_names, None
    except Exception as e:  # noqa: BLE001
        return hit_sid, {}, {}, f"{type(e).__name__}: {e}"


def compute_binding_residues(candidates, manifest, max_workers=None, forced_p_idx_by_pair=None,
                              fp_pickle_override=None):
    """For every pair, computes two intentionally DIFFERENT residue sets:

    fp_pickle_override: optional path to use instead of the module-level
    FP_PICKLE (both for this function's own direct load and for every worker
    process's -- see _init_worker's docstring for why a worker needs this
    passed explicitly rather than picking up a parent-process monkeypatch).

    forced_p_idx_by_pair: optional {(ref_site, hit_site): p_idx} -- when a
    pair has an entry here, that phosphate group is used directly instead of
    being re-derived by the "most bits matched" search below. For callers
    that already know which phosphate group produced the pair's score (e.g.
    the discover_candidates.py-driven pool builder, which computes score and
    display from the SAME greedy assignment), this guarantees the visual
    highlight can never disagree with the reported score -- the exact
    inconsistency an independent re-derivation risked (confirmed on
    multi-phosphate references where two different tie-break rules picked
    two different groups for the same pair).

    - hit_binding_residues: only residues that contribute to a MATCHED
      (non-VdW) interaction bit -- the same "mimicry" set
      matched_hit_atom_indices() (export_prolif_datawarrior.py) uses to
      produce red_atom_names/purple_atom_names and prolif_plif_score_no_vdw.
      Deliberately the matched intersection, not "any non-VdW interaction
      this ligand happens to make": showing a hit's incidental
      hydrophobic-pocket contacts as "of interest" when they play no role in
      the actual isosteric-mimicry score is misleading (confirmed on
      ref_4EOJ_ATP_301/hit_5JQ5_I74_302, score_no_vdw=0.0: the old
      per-side-independent version showed ILE10/PHE80/PHE82/LEU83 here, none
      of which contributed to that 0.0 score at all).
    - ref_binding_residues: ALL of the (pair-specific, best-matching)
      phosphate group's own non-VdW residues, NOT intersected with what the
      hit matched. A total-miss pair (hit_binding_residues empty) still
      needs a visible reference baseline -- otherwise a reviewer has no way
      to see what the hit *failed* to replicate, only that it failed
      something. Group selection still prefers the hit's best-matching group
      when there IS a match; when nothing matches at all, it falls back to
      the group with the most non-VdW bits overall (the reference's single
      richest/most representative group) rather than an arbitrary one.

    Because both are inherently pair-specific (a hit's matched residues
    depend on which reference it's being compared against, and a hit can be
    paired with several references -- 952/5832 sites in the full manifest),
    both are pair-keyed here, not site-keyed. Reuses
    run_prolif.py/phosphate_ifp.py's functions +
    export_prolif_datawarrior.py's exact matched-bit/group-selection
    convention rather than re-deriving any of the interaction logic.

    The expensive part (one ProLIF phosphate_group_ifps() pass per
    reference ligand) still only runs once per unique reference site and is
    cached; per-pair work is just residue-correspondence lookups + set
    intersections over already-computed ifps, no new ProLIF fp.generate()
    calls.

    Also returns ref_atom_links/hit_atom_links: [ligand_atom_name, resn,
    resi, chain, protein_atom_name] per detected atom pair, restricted to
    matched_bits on BOTH sides (unlike ref_residues/hit_residues above, which
    intentionally keep ref at the wider non-VdW context) -- these drive the
    dashed interaction lines in the viewer, and a line is a claim that this
    exact contact was actually replicated, the same thing prolif_plif_score's
    numerator counts. Lets the viewer draw a dashed line between the exact
    interacting ligand atom and the exact interacting residue ATOM (e.g. an
    Arg's NH1, not just "somewhere on this residue"). Both atom identities
    come from each bit's own ProLIF metadata (metadata["indices"]["ligand"]/
    ["protein"]); the ligand side reuses export_prolif_datawarrior.py's
    matched_hit_atom_indices() provenance, the protein side is new here.
    """
    sys.path.insert(0, SCRIPTS_DIR)
    import warnings
    warnings.filterwarnings("ignore")
    import prolif as plf
    import run_prolif as rp

    # rp.build_residue_correspondence() (called per-pair below) needs
    # common.TMALIGN_JSON configured -- common.py deliberately has no working
    # default anymore (see its own TMALIGN_JSON comment: the old hardcoded
    # _corrected.json default silently carried ~17,900 stale pairs). Every
    # other entry point (resolve_pairs.py, run_prolif.py, discover_
    # candidates.py) already calls set_tmalign_json() explicitly at its own
    # start; this function is the one place inside build_pool.py that
    # actually needs it, so it's set here rather than relying on either
    # caller (build_pool.py's own main(), or build_pool_from_discovery.py) to
    # remember to do it first.
    sys.path.insert(0, os.path.join(PHOSPHATE_ROOT, "scripts", "plip_isostere"))
    import common as plip_common
    plip_common.set_tmalign_json(TMALIGN_JSON)

    def canon_novdw_bits(ifp, protein_mapping=None):
        canon = rp.canonicalize_ifp_for_alignment(ifp, protein_mapping=protein_mapping)
        bits = {b for b in rp.flatten_canon_bits(canon) if b[1] not in EXCLUDED_INTERACTIONS}
        return canon, bits

    fp_pickle_path = fp_pickle_override or FP_PICKLE
    print(f"Loading ProLIF fingerprint pickle from {fp_pickle_path} ...")
    fp = plf.Fingerprint.from_pickle(fp_pickle_path)
    ifp_by_site = dict(zip(fp.site_ids, fp.ifp.values()))

    ref_sites = sorted({r["ref_site"] for r, *_ in candidates})
    hit_sites = sorted({r["hit_site"] for r, *_ in candidates})
    workers = max_workers or os.cpu_count() or 1

    print(f"Loading phosphate-group ifps + atom names for {len(ref_sites)} unique "
          f"reference sites and atom names for {len(hit_sites)} unique hit sites "
          f"({workers} worker processes) ...")
    ref_groups_by_site = {}           # ref_sid -> {p_idx: raw ifp}
    ref_group_atoms_by_site = {}      # ref_sid -> {p_idx: [pdb atom names]}
    ref_frag_atom_names_by_site = {}  # ref_sid -> {p_idx: {fragment atom idx: pdb atom name}}
    ref_prot_atom_names_by_site = {}  # ref_sid -> {protein atom idx: pdb atom name}
    hit_lig_atom_names_by_site = {}   # hit_sid -> {ligand atom idx: pdb atom name}
    hit_prot_atom_names_by_site = {}  # hit_sid -> {protein atom idx: pdb atom name}

    # Both site kinds are independent of each other and of pairing structure,
    # so one shared pool handles both task types; ProcessPoolExecutor doesn't
    # care that _compute_ref_site_worker and _compute_hit_site_worker differ.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(fp_pickle_path,)
    ) as pool:
        ref_futures = {
            pool.submit(_compute_ref_site_worker, (ref_sid, manifest.get(ref_sid))): ref_sid
            for ref_sid in ref_sites
        }
        n_done = 0
        for future in concurrent.futures.as_completed(ref_futures):
            ref_sid, groups, atoms_by_p_idx, frag_names_by_p_idx, prot_names, err = future.result()
            if err:
                print(f"  WARNING: reference site {ref_sid} failed ({err}), treating as no phosphate groups")
            ref_groups_by_site[ref_sid] = groups
            ref_group_atoms_by_site[ref_sid] = atoms_by_p_idx
            ref_frag_atom_names_by_site[ref_sid] = frag_names_by_p_idx
            ref_prot_atom_names_by_site[ref_sid] = prot_names
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(ref_sites)} reference sites done", flush=True)
        print(f"  {n_done}/{len(ref_sites)} reference sites done", flush=True)

        hit_futures = {
            pool.submit(_compute_hit_site_worker, (hit_sid, manifest.get(hit_sid))): hit_sid
            for hit_sid in hit_sites
        }
        n_done = 0
        for future in concurrent.futures.as_completed(hit_futures):
            hit_sid, lig_names, prot_names, err = future.result()
            if err:
                print(f"  WARNING: hit site {hit_sid} failed ({err}), treating as no atoms")
            hit_lig_atom_names_by_site[hit_sid] = lig_names
            hit_prot_atom_names_by_site[hit_sid] = prot_names
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(hit_sites)} hit sites done", flush=True)
        print(f"  {n_done}/{len(hit_sites)} hit sites done", flush=True)

    print(f"Computing matched (mimicking) residues on both sides for all "
          f"{len(candidates)} pairs ...")
    ref_residues_by_pair = {}
    hit_residues_by_pair = {}
    ref_n_bits_by_pair = {}  # (ref_site, hit_site) -> selected group's own non-VdW bit count
    ref_phosphate_atoms_by_pair = {}
    ref_atom_links_by_pair = {}
    hit_atom_links_by_pair = {}
    for i, (r, ref_m, hit_m, *_rest) in enumerate(candidates, 1):
        ref_sid, hit_sid = r["ref_site"], r["hit_site"]
        key = (ref_sid, hit_sid)
        groups = ref_groups_by_site.get(ref_sid) or {}
        hit_ifp = ifp_by_site.get(hit_sid)
        if not groups or hit_ifp is None:
            ref_residues_by_pair[key] = []
            hit_residues_by_pair[key] = []
            ref_phosphate_atoms_by_pair[key] = []
            ref_atom_links_by_pair[key] = []
            hit_atom_links_by_pair[key] = []
            continue

        mapping = rp.build_residue_correspondence(manifest[ref_sid], manifest[hit_sid], cutoff=rp.CA_MATCH_CUTOFF)
        if mapping is None:
            ref_residues_by_pair[key] = []
            hit_residues_by_pair[key] = []
            ref_phosphate_atoms_by_pair[key] = []
            ref_atom_links_by_pair[key] = []
            hit_atom_links_by_pair[key] = []
            continue
        canon_hit, hit_bits = canon_novdw_bits(hit_ifp, protein_mapping=mapping)

        forced_p_idx = (forced_p_idx_by_pair or {}).get(key)
        if forced_p_idx is not None and forced_p_idx in groups:
            best_ifp, best_p_idx = groups[forced_p_idx], forced_p_idx
        else:
            # Group selection: prefer whichever group has the most bits actually
            # matched by this hit (same convention as matched_hit_atom_indices),
            # tie-broken by the group's own total non-VdW bit count when nothing
            # matches at all (e.g. score_no_vdw=0) -- so a total-miss pair still
            # shows the reference's single richest/most-representative group
            # rather than an arbitrary one.
            best_ifp, best_p_idx, best_key = None, None, (-1, -1)
            for p_idx, ifp in groups.items():
                _, cand_bits = canon_novdw_bits(ifp)
                matched = len(cand_bits & hit_bits)
                cmp_key = (matched, len(cand_bits))
                if cmp_key > best_key:
                    best_ifp, best_p_idx, best_key = ifp, p_idx, cmp_key

        if best_ifp is None:
            ref_residues_by_pair[key] = []
            hit_residues_by_pair[key] = []
            ref_phosphate_atoms_by_pair[key] = []
            ref_atom_links_by_pair[key] = []
            hit_atom_links_by_pair[key] = []
            continue

        canon_ref, ref_bits = canon_novdw_bits(best_ifp)
        matched_bits = ref_bits & hit_bits
        ref_n_bits_by_pair[key] = len(ref_bits)

        # Ref side: shows the SELECTED GROUP's own real (non-VdW) residues,
        # not just the ones the hit happens to replicate -- a total miss
        # (hit_bits empty, matched_bits empty) should still let the reviewer
        # see what the reference actually does here, otherwise "the hit
        # matched nothing" has no visible baseline to compare against.
        # matched_bits stays used for the hit side below, and for deciding
        # WHICH group to show here in the first place.
        str_to_ref_id = {str(prot_id): prot_id for (_lig, prot_id) in canon_ref}
        ref_residues_by_pair[key] = _residue_ids_to_rows(
            {str_to_ref_id[b[0]] for b in ref_bits if b[0] in str_to_ref_id})

        # Hit side: matched_bits' residue strings are REF-position labels
        # (canon_hit used protein_mapping=mapping) -- invert hit->ref to find
        # which of the hit's OWN residues mapped there, then confirm that
        # specific hit residue really carries that interaction type in the
        # raw (pre-canonicalization) hit_ifp before crediting it.
        ref_str_to_hit_ids = {}
        for hit_id, ref_id in mapping.items():
            ref_str_to_hit_ids.setdefault(str(ref_id), []).append(hit_id)
        hit_raw_index = {(prot_id, name) for (_lig, prot_id), data in hit_ifp.items() for name in data}
        hit_residue_ids = {
            hid for (ref_str, itype) in matched_bits
            for hid in ref_str_to_hit_ids.get(ref_str, [])
            if (hid, itype) in hit_raw_index
        }
        hit_residues_by_pair[key] = _residue_ids_to_rows(hit_residue_ids)

        ref_phosphate_atoms_by_pair[key] = ref_group_atoms_by_site.get(ref_sid, {}).get(best_p_idx, [])

        # Atom-level interaction links: which SPECIFIC ligand atom connects to
        # which SPECIFIC residue ATOM, for drawing dashed interaction lines in
        # the viewer that land on the real interacting atom (e.g. an Arg's
        # NH1, a Ser's OG) instead of the residue's CA "root" -- residue-level
        # binding_residues above don't carry this. UNLIKE the residue lists
        # above (where ref intentionally shows the full non-VdW context, hit
        # shows the matched-only intersection), both ref and hit LINES are
        # restricted to matched_bits -- the same numerator prolif_plif_score
        # itself is built from. A line is a claim of "this exact contact was
        # replicated"; drawing one for an unmatched ref bit would show a
        # mimicry claim that wasn't actually detected, on top of already
        # being redundant with the ref_binding_residues context labels
        # covering the wider (non-VdW) set. Both the ligand AND protein atom
        # identity come straight off each metadata entry's own
        # ProLIF-detected atom indices ("indices"->"ligand"/"protein") -- one
        # line per actual detected atom-pair, not one line per (residue,
        # interaction-type) bit, so a bit backed by multiple atom pairs draws
        # all of them.
        ref_names_by_idx = ref_frag_atom_names_by_site.get(ref_sid, {}).get(best_p_idx, {})
        ref_prot_names_by_idx = ref_prot_atom_names_by_site.get(ref_sid, {})
        bit_to_metadata_ref = {}
        for (_lig, prot_id), data in canon_ref.items():
            for name, metadata in data.items():
                bit_to_metadata_ref[(str(prot_id), name)] = metadata

        ref_links = set()
        for bit in matched_bits:
            ref_str, itype = bit
            prot_id = str_to_ref_id.get(ref_str)
            if prot_id is None:
                continue
            for meta in bit_to_metadata_ref.get(bit, ()):
                lig_names = [ref_names_by_idx[i] for i in meta["indices"]["ligand"] if i in ref_names_by_idx]
                # protein side MUST use parent_indices, not indices: ProLIF's
                # metadata["indices"]["protein"] is local to that one
                # residue's own Residue submol (confirmed via prolif's own
                # get_mapindex()/GetUnsignedProp("mapindex") -- the property
                # it reads specifically exists to map a residue-local index
                # back to the parent molecule). Using raw "indices" here
                # first produced nonsense like Arg's HB3/HG11 (backbone-area
                # hydrogens) instead of the guanidinium NH1/NH2/NE actually
                # forming the interaction -- indices["ligand"] escapes this
                # because every ligand mol here (fragment or whole) is built
                # as a SINGLE residue, where local and global indices
                # coincide; the protein mol has many residues, so they don't.
                prot_names = [ref_prot_names_by_idx[i] for i in meta["parent_indices"]["protein"] if i in ref_prot_names_by_idx]
                for lig_nm in lig_names:
                    for prot_nm in (prot_names or [None]):
                        ref_links.add((lig_nm, prot_id.name, prot_id.number, prot_id.chain, prot_nm))
        ref_atom_links_by_pair[key] = [list(t) for t in sorted(ref_links, key=lambda t: (t[0], t[1], t[2], t[3], t[4] or ""))]

        hit_names_by_idx = hit_lig_atom_names_by_site.get(hit_sid, {})
        hit_prot_names_by_idx = hit_prot_atom_names_by_site.get(hit_sid, {})
        bit_to_metadata_hit = {}
        for (_lig, prot_id), data in canon_hit.items():
            for name, metadata in data.items():
                bit_to_metadata_hit[(str(prot_id), name)] = metadata

        hit_links = set()
        for bit in matched_bits:
            ref_str, itype = bit
            hids = [hid for hid in ref_str_to_hit_ids.get(ref_str, []) if (hid, itype) in hit_raw_index]
            if not hids:
                continue
            for meta in bit_to_metadata_hit.get(bit, ()):
                lig_names = [hit_names_by_idx[i] for i in meta["indices"]["ligand"] if i in hit_names_by_idx]
                # parent_indices, not indices -- see the matching comment on
                # the ref side above for why.
                prot_names = [hit_prot_names_by_idx[i] for i in meta["parent_indices"]["protein"] if i in hit_prot_names_by_idx]
                for lig_nm in lig_names:
                    for prot_nm in (prot_names or [None]):
                        for hid in hids:
                            hit_links.add((lig_nm, hid.name, hid.number, hid.chain, prot_nm))
        hit_atom_links_by_pair[key] = [list(t) for t in sorted(hit_links, key=lambda t: (t[0], t[1], t[2], t[3], t[4] or ""))]

        if i % 500 == 0:
            print(f"  {i}/{len(candidates)} pairs done")

    return (ref_residues_by_pair, hit_residues_by_pair, ref_phosphate_atoms_by_pair,
            ref_atom_links_by_pair, hit_atom_links_by_pair, ref_n_bits_by_pair)


def gaussian_weight(score, bits, median_score, median_bits, bits_range):
    d_score = (score - median_score) / SIGMA_SCORE if SIGMA_SCORE else 0.0
    bits_norm = (bits - median_bits) / bits_range if bits_range else 0.0
    d_bits = bits_norm / SIGMA_BITS if SIGMA_BITS else 0.0
    return math.exp(-0.5 * (d_score ** 2 + d_bits ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifg-csv", default=DEFAULT_IFG_CSV,
                     help="ifg_prolif_dataset CSV to build the pool from")
    ap.add_argument("--shared-core", type=int, default=200,
                     help="number of pairs every reviewer sees -- see the matching "
                          "argument in build_pool_from_discovery.py, the pipeline that "
                          "actually runs now")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-binding-residues", action="store_true",
                     help="skip the ProLIF binding-residue pass (faster iteration; "
                          "pool rows get empty ref/hit_binding_residues)")
    ap.add_argument("--workers", type=int, default=None,
                     help="worker processes for the per-site precompute pass "
                          "(default: os.cpu_count())")
    ap.add_argument("--i-know-this-is-the-old-chain", action="store_true",
                     help="required: this script reads the retired full_manifest.csv/"
                          "plif_vs_tanimoto_comparison_full_manifest.csv/ifg_prolif_dataset_*.csv "
                          "chain, which can disagree with itself on multi-phosphate references "
                          "(see the module docstring). Use build_pool_from_discovery.py instead "
                          "unless you specifically need this old chain for something.")
    args = ap.parse_args()
    if not args.i_know_this_is_the_old_chain:
        print("Error: build_pool.py is deprecated -- it reads the old, potentially "
              "self-inconsistent full_manifest.csv/plif_vs_tanimoto_comparison_full_manifest.csv/"
              "ifg_prolif_dataset_*.csv chain (see this file's module docstring).")
        print("Use build_pool_from_discovery.py instead, which builds review.db from "
              "discover_candidates.py's own output.")
        print("If you specifically need this old chain anyway, rerun with "
              "--i-know-this-is-the-old-chain.")
        sys.exit(1)
    rng = random.Random(args.seed)

    if not os.path.isfile(args.ifg_csv):
        print(f"Error: {args.ifg_csv} not found.")
        print("Generate it first with build_ifg_prolif_dataset.py --min-score 0 "
              "--min-ref-bits 1 (see review_app/README.md), or pass --ifg-csv "
              "to point at a different dataset.")
        sys.exit(1)

    os.makedirs(ALIGNED_HITS_DIR, exist_ok=True)
    os.makedirs(REF_POCKETS_DIR, exist_ok=True)
    manifest = load_full_manifest()
    transform_idx = load_transform_index()
    with_vdw_scores = load_with_vdw_scores()

    with open(args.ifg_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    skipped = {"no_score": 0, "zero_score": 0, "no_manifest_row": 0, "no_pocket_pdb": 0, "no_transform": 0}
    n_no_with_vdw = 0
    for r in rows:
        ref_site, hit_site = r["ref_site"], r["hit_site"]
        if r["prolif_plif_score"] == "" or r["ref_n_bits_novdw"] == "":
            skipped["no_score"] += 1
            continue
        # A pair with prolif_plif_score_no_vdw == 0.0 has literally zero
        # matched non-VdW interactions -- no red/purple atoms, no binding
        # residues on either side, nothing structurally ambiguous to review.
        # Not useful for deciding where the isosteric-validity threshold
        # should sit (that's about the ambiguous middle, not "clearly
        # nothing"), and their prevalence (52% of the broader min-score-0
        # dataset) was dominating the median-centered sampling weight,
        # skewing the shared core to ~90% empty-residue pairs. Excluded
        # entirely rather than just down-weighted.
        if float(r["prolif_plif_score"]) == 0.0:
            skipped["zero_score"] += 1
            continue
        ref_m, hit_m = manifest.get(ref_site), manifest.get(hit_site)
        if ref_m is None or hit_m is None:
            skipped["no_manifest_row"] += 1
            continue
        ref_pdb_path = os.path.join(POCKETS_DIR, f"{ref_site}.pdb")
        hit_pdb_path = os.path.join(POCKETS_DIR, f"{hit_site}.pdb")
        if not (os.path.isfile(ref_pdb_path) and os.path.isfile(hit_pdb_path)):
            skipped["no_pocket_pdb"] += 1
            continue
        transformation = get_transformation(transform_idx, r["ref_pdb"], r["hit_pdb"])
        if transformation is None:
            skipped["no_transform"] += 1
            continue
        with_vdw = with_vdw_scores.get((ref_site, hit_site))
        if with_vdw is None:
            n_no_with_vdw += 1
        candidates.append((r, ref_m, hit_m, transformation, with_vdw))

    scores = [float(r["prolif_plif_score"]) for r, _, _, _, _ in candidates]
    bits = [float(r["ref_n_bits_novdw"]) for r, _, _, _, _ in candidates]
    median_score = sorted(scores)[len(scores) // 2]
    median_bits = sorted(bits)[len(bits) // 2]
    bits_range = (max(bits) - min(bits)) or 1.0

    print(f"Using ifg dataset: {args.ifg_csv}")
    print(f"Candidates: {len(candidates)} / {len(rows)} rows "
          f"(skipped: {skipped})")
    if n_no_with_vdw:
        print(f"  {n_no_with_vdw} candidates have no with-VdW score match in "
              f"{MAIN_MANIFEST_SCORES_CSV} (kept anyway, with_vdw stored as NULL)")
    print(f"Pool median no-VdW prolif_plif_score={median_score:.3f}, median ref_n_bits_novdw={median_bits:.1f}")

    if args.skip_binding_residues:
        ref_residues_by_pair, hit_residues_by_pair, ref_phosphate_atoms_by_pair = {}, {}, {}
        ref_atom_links_by_pair, hit_atom_links_by_pair, ref_n_bits_by_pair = {}, {}, {}
    else:
        (ref_residues_by_pair, hit_residues_by_pair, ref_phosphate_atoms_by_pair,
         ref_atom_links_by_pair, hit_atom_links_by_pair, ref_n_bits_by_pair) = \
            compute_binding_residues(candidates, manifest, max_workers=args.workers)

    # Only `pairs` gets dropped and rebuilt -- reviewers/assignments/ratings
    # are preserved across reruns (CREATE TABLE IF NOT EXISTS) so re-running
    # this script to fix a bug or regenerate from a new dataset never wipes
    # out ratings colleagues have already submitted. An assignment can end
    # up pointing at a pair_id that no longer exists in the new `pairs` (only
    # if the underlying ifg dataset itself changed, not on a same-dataset
    # rerun) -- the API/frontend already handle an unknown pair_id as a
    # normal 404 rather than crashing, so this degrades gracefully.
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS pairs")
    conn.execute("""
        CREATE TABLE pairs (
            pair_id TEXT PRIMARY KEY,
            ref_site TEXT, hit_site TEXT,
            ref_pdb TEXT, hit_pdb TEXT,
            ref_chain TEXT, ref_resname TEXT, ref_resnum TEXT,
            hit_chain TEXT, hit_resname TEXT, hit_resnum TEXT,
            red_atom_names TEXT, purple_atom_names TEXT,
            ref_binding_residues TEXT, hit_binding_residues TEXT,
            ref_phosphate_atom_names TEXT,
            ref_atom_links TEXT, hit_atom_links TEXT,
            prolif_plif_score_no_vdw REAL, prolif_plif_score_with_vdw REAL,
            ref_n_bits_novdw REAL,
            weight REAL, is_shared INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviewers (
            reviewer_name TEXT PRIMARY KEY,
            requested_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            reviewer_name TEXT,
            pair_id TEXT,
            order_index INTEGER,
            PRIMARY KEY (reviewer_name, pair_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reviewer_name TEXT,
            pair_id TEXT,
            score INTEGER,
            wrong_reference_phosphate INTEGER DEFAULT 0,
            comment TEXT,
            submitted_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    actual_resname_cache = {}

    def get_actual_resname(site_id, pdb_path, resnum, chain, fallback):
        cache_key = (site_id, resnum, chain)
        if cache_key not in actual_resname_cache:
            actual_resname_cache[cache_key] = _actual_ligand_resname(pdb_path, resnum, chain)
        return actual_resname_cache[cache_key] or fallback

    inserted = []
    for r, ref_m, hit_m, (t, u), with_vdw in candidates:
        pair_id = f"{r['ref_site']}__{r['hit_site']}"
        score_no_vdw = float(r["prolif_plif_score"])
        nbits = float(r["ref_n_bits_novdw"])
        weight = gaussian_weight(score_no_vdw, nbits, median_score, median_bits, bits_range)
        red_names = [n.strip() for n in r["red_atom_names"].split(",") if n.strip()]
        purple_names = [n.strip() for n in r["purple_atom_names"].split(",") if n.strip()]
        pair_key = (r["ref_site"], r["hit_site"])
        ref_residues = ref_residues_by_pair.get(pair_key, [])
        hit_residues = hit_residues_by_pair.get(pair_key, [])
        ref_phosphate_names = ref_phosphate_atoms_by_pair.get(pair_key, [])
        ref_atom_links = ref_atom_links_by_pair.get(pair_key, [])
        hit_atom_links = hit_atom_links_by_pair.get(pair_key, [])

        dst = os.path.join(ALIGNED_HITS_DIR, f"{pair_id}.pdb")
        src = os.path.join(POCKETS_DIR, f"{r['hit_site']}.pdb")
        write_transformed_pdb(src, dst, t, u)

        ref_resname = get_actual_resname(
            r["ref_site"], os.path.join(POCKETS_DIR, f"{r['ref_site']}.pdb"),
            ref_m["lig_resnum"], ref_m["chain"], ref_m["lig_resname"])
        hit_resname = get_actual_resname(
            r["hit_site"], src, hit_m["lig_resnum"], hit_m["chain"], hit_m["lig_resname"])

        conn.execute(
            "INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                pair_id, r["ref_site"], r["hit_site"],
                r["ref_pdb"], r["hit_pdb"],
                ref_m["chain"], ref_resname, ref_m["lig_resnum"],
                hit_m["chain"], hit_resname, hit_m["lig_resnum"],
                json.dumps(red_names), json.dumps(purple_names),
                json.dumps(ref_residues), json.dumps(hit_residues),
                json.dumps(ref_phosphate_names),
                json.dumps(ref_atom_links), json.dumps(hit_atom_links),
                score_no_vdw, with_vdw, nbits, weight, 0,
            ),
        )
        inserted.append((pair_id, weight))

    # Weighted sample (no replacement) for the shared core every reviewer sees.
    pool = inserted[:]
    shared_ids = set()
    n_shared = min(args.shared_core, len(pool))
    for _ in range(n_shared):
        total = sum(w for _, w in pool)
        pick = rng.uniform(0, total)
        acc = 0.0
        for i, (pid, w) in enumerate(pool):
            acc += w
            if acc >= pick:
                shared_ids.add(pid)
                pool.pop(i)
                break

    if shared_ids:
        conn.executemany(
            "UPDATE pairs SET is_shared = 1 WHERE pair_id = ?",
            [(pid,) for pid in shared_ids],
        )
    conn.commit()
    conn.close()

    print(f"Wrote {len(inserted)} pairs to {DB_PATH}")
    print(f"Shared core (every reviewer rates these): {len(shared_ids)} pairs")
    print(f"Aligned hit PDBs written to {ALIGNED_HITS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
