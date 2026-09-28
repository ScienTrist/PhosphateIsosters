"""
Step 7 (ProLIF pipeline): full ProLIF-native candidate-mimic discovery +
interaction scoring, replacing scripts/batch_motif_extraction.py +
scripts/analyze_plif.py's homebrew calculate_spatial_similarity scorer.

For every TM-aligned (ref_site, hit_id) pairing in results/tmalign_results_
{run_id}[_clean].json, this enumerates every chemically-eligible candidate
mimic residue in the hit structure using the SAME eligibility rules
scripts/analyze_plif.py's _process_ref_site already applies (ported here
verbatim, not re-derived -- see find_eligible_candidates), fingerprints each
one independently with ProLIF (reusing ProLIF_v2's own per-site building
blocks: protonate_protein -> protonate_ligand -> extract_pockets -> run_prolif
.run_site_scoped), scores it against the reference phosphate group's own
ProLIF fingerprint using run_prolif.py's own prolif_plif_score formula
(len(ref_bits & hit_bits) / len(ref_bits), verified at run_prolif.py:979-981),
and greedily assigns candidates to reference phosphates using the exact same
structure scripts/analyze_plif.py's _process_ref_site already uses: sort all
(candidate, ref_phosphate) pairs by score descending, assign each candidate
and each phosphate at most once.

Candidate discovery does NOT depend on any results/motif_analysis/{hits,
references}/*.cif file (the ones batch_motif_extraction.py writes rename
chains for PDB-format safety -- see get_safe_chain_name -- which would lose
each candidate's true native chain, needed by protonate_ligand.py/
extract_pockets.py, both of which key strictly on native chain+resnum).
Eligible residues are found directly against the hit's own untouched
data/structures/{references,hits}/{PDB}.* file: a throwaway in-memory copy is
TM-aligned into the reference's frame purely for the geometric radius test
(find_eligible_candidates); the residues actually used downstream are the
UNTRANSFORMED ones from a fresh, separate read of that same file.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import gemmi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "plip_isostere"))

from constants import STANDARD_AA, IGNORE_LIGANDS  # noqa: E402
from utils import fmt_eta  # noqa: E402
import batch_motif_extraction as bme  # find_structure_file, get_unique_ref_sites  # noqa: E402
import common as plip_common  # set_tmalign_json/get_transformation  # noqa: E402
import protonate_protein  # noqa: E402
import protonate_ligand  # noqa: E402
import extract_pockets  # noqa: E402
import fetch_ligand_smiles  # noqa: E402

SMILES_CACHE_PATH = os.path.join(PROLIF_V2_ROOT, "ligand_smiles.json")
PROTEIN_PROTONATED_DIR = os.path.join(PROLIF_V2_ROOT, "data", "protein_protonated")
MAIN_STRUCT_DIRS = [
    os.path.join(PROJECT_ROOT, "data", "structures", "references"),
    os.path.join(PROJECT_ROOT, "data", "structures", "hits"),
]

CANDIDATE_RADIUS = 10.0  # same convention as batch_motif_extraction.py/extract_pockets.py
MIN_SCORE = 0.1  # same pre-greedy gate analyze_plif.py used
FINGERPRINT_WORKERS = min(8, os.cpu_count() or 4)  # same cap run_prolif.py/analyze_plif.py use


# ── geometry / eligibility (ported from analyze_plif.py / batch_motif_extraction.py) ──

def transform_structure_inplace(st, t, u):
    """Applies a TM-align (t, u) rigid-body transform to every atom in st, IN
    PLACE. Same convention as common.apply_transform / batch_motif_extraction
    .process_single_reference's own hit-transform step."""
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = atom.pos
                    atom.pos = gemmi.Position(
                        u[0][0] * p.x + u[0][1] * p.y + u[0][2] * p.z + t[0],
                        u[1][0] * p.x + u[1][1] * p.y + u[1][2] * p.z + t[1],
                        u[2][0] * p.x + u[2][1] * p.y + u[2][2] * p.z + t[2],
                    )


def find_eligible_candidates(hit_path, transformation, ref_lig_atoms, rejection_counts=None):
    """
    hit_path: path to the hit's raw structure (data/structures/hits|references/{HIT}.*).
    transformation: (t, u) from common.get_transformation(ref_id, hit_id).
    ref_lig_atoms: list of gemmi.Atom, the reference site's anchor ligand's own
      atoms (the same cluster[0] anchor batch_motif_extraction.get_unique_ref_sites
      groups, used as `lig_pos` there too).
    rejection_counts: optional dict this function ADDS TO (in place) -- tallies
      every residue examined into exactly one of "rej_has_own_P",
      "rej_is_amino_acid", "rej_in_ignore_list", "rej_too_far_from_ref_ligand",
      "rej_duplicate_symmetric_copy" (passed every other check but lost the
      by_key dedup below to a closer symmetric copy), or "eligible" (made it
      into the returned list). Lets run_discovery's stage_counts/run report
      show not just how many candidates survived, but which specific
      criterion rejected the rest -- see build_run_report.py.

    Returns [(chain_name, res), ...] -- native (untransformed) residues from a
    fresh read of hit_path, one per eligible candidate mimic, deduplicated by
    (resname, resnum), keeping whichever chain copy sits closest to the
    reference ligand for a symmetric multimer carrying more than one copy
    (same rationale as analyze_plif.py's own distance-based dedup of duplicate
    mimic_key hits, see scripts/analyze_plif.py's _process_ref_site tiebreak).

    Eligibility -- verbatim port of analyze_plif._process_ref_site's candidate
    loop, checked in this exact order (cheapest/most-disqualifying first):
      1. no phosphorus atom of its own
      2. not gemmi.find_tabulated_residue(name).is_amino_acid()
      3. not in IGNORE_LIGANDS or STANDARD_AA (src/constants.py)
      4. any atom within CANDIDATE_RADIUS (10 A) of any ref_lig_atoms position,
         evaluated in the ref-aligned frame

    REVISION -- dropped the original chain.name.isupper() filter (was step 1):
    carried over from analyze_plif.py's own hit-chain filter with no clear
    surviving rationale, and confirmed near-negligible in practice (1,650 of
    18.6M residues examined across a full run, 0.009%) -- removed rather than
    kept as unexplained dead weight. Every chain is now examined regardless of
    name casing.

    REVISION -- dropped the LINK_CUTOFF (2.5 A "within reach of a real
    phosphorus elsewhere in the hit") filter too: also near-negligible in
    practice (664 of 18.6M residues, 0.004%), and residues it would have
    caught mostly still fail the remaining checks anyway (e.g. still part of
    a larger disqualified entity) -- see the user-facing before/after rerun
    for exactly how many additionally became eligible.
    """
    def _tally(key):
        if rejection_counts is not None:
            rejection_counts[key] = rejection_counts.get(key, 0) + 1

    st_native = gemmi.read_structure(hit_path)
    st_aligned = gemmi.read_structure(hit_path)  # independent second read; see module docstring
    t, u = transformation
    transform_structure_inplace(st_aligned, t, u)

    ref_lig_pos = [a.pos for a in ref_lig_atoms]

    by_key = {}  # (resname, resnum) -> (dist, chain_name, res_native)
    for chain_native, chain_aligned in zip(st_native[0], st_aligned[0]):
        for res_native, res_aligned in zip(chain_native, chain_aligned):
            if any(a.element.name == "P" for a in res_aligned):
                _tally("rej_has_own_P")
                continue
            if gemmi.find_tabulated_residue(res_aligned.name).is_amino_acid():
                _tally("rej_is_amino_acid")
                continue
            if res_aligned.name in IGNORE_LIGANDS or res_aligned.name in STANDARD_AA:
                _tally("rej_in_ignore_list")
                continue
            dists = [a.pos.dist(rp) for a in res_aligned for rp in ref_lig_pos]
            if not dists or min(dists) > CANDIDATE_RADIUS:
                _tally("rej_too_far_from_ref_ligand")
                continue
            key = (res_aligned.name, res_aligned.seqid.num)
            d = min(dists)
            if key in by_key and d >= by_key[key][0]:
                _tally("rej_duplicate_symmetric_copy")
                continue
            if key in by_key:
                _tally("rej_duplicate_symmetric_copy")  # the copy this one displaces
            by_key[key] = (d, chain_native.name, res_native)

    if rejection_counts is not None:
        rejection_counts["eligible"] = rejection_counts.get("eligible", 0) + len(by_key)
    return [(chain, res) for _, chain, res in by_key.values()]


# ── SMILES cache: worker-local dict seeded from disk, best-effort persisted ──
# ProcessPoolExecutor workers each get their own copy at spawn time (pickled,
# not shared memory) -- a fetch made by one worker is never visible to a
# sibling worker already running, only to workers spawned/re-initialized
# afterward that reload from disk. This is an accepted, documented limitation,
# not a correctness bug: protonate_ligand.protonate_ligand() already has a
# template-free fallback for any resname with no cached SMILES (outcome ==
# "fallback"), so a missing cache entry costs bond-order/charge quality for
# that one ligand, never a crash or a missing result.

def _ensure_smiles(resname, smiles_cache):
    if resname in smiles_cache:
        return
    try:
        smiles_cache[resname] = fetch_ligand_smiles.fetch_smiles(resname)
    except Exception:
        return
    try:
        _persist_smiles_entry(resname, smiles_cache[resname])
    except Exception:
        pass  # best-effort only -- this worker's own in-memory cache is already updated


def _persist_smiles_entry(resname, smiles):
    """Best-effort read-merge-write into the shared ligand_smiles.json so later
    -spawned workers (or a future run) pick up this fetch. A race between two
    workers writing concurrently can drop one of the two new entries from the
    file -- acceptable, since each worker still holds its own copy correctly
    for its own remaining tasks (see module note above).

    tmp_path MUST be unique per writer (pid, here): os.replace() is only
    atomic with respect to the FINAL rename, not the write that precedes it.
    A single shared f"{SMILES_CACHE_PATH}.tmp" path let two concurrent
    worker processes both open() and write() into the SAME tmp file at once,
    interleaving their output into one corrupt blob before either one's
    os.replace() fired -- confirmed directly: ligand_smiles.json was found
    corrupted this way twice in one session (two different worker processes'
    tail ends spliced together mid-entry), both times recovered only by a
    manual line-by-line repair. A per-pid tmp path makes concurrent writers
    fully independent again -- back to just the pre-existing, accepted
    "one process's new entry can get silently overwritten by another's
    same-round-trip write" tradeoff, never file-level corruption."""
    on_disk = {}
    if os.path.exists(SMILES_CACHE_PATH):
        with open(SMILES_CACHE_PATH) as f:
            on_disk = json.load(f)
    on_disk[resname] = smiles
    tmp_path = f"{SMILES_CACHE_PATH}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(on_disk, f, indent=2, sort_keys=True)
    os.replace(tmp_path, SMILES_CACHE_PATH)


# ── per-site preparation (protonate ligand -> extract pocket -> fingerprint) ──

def build_ref_site_dict(ref_id, cluster):
    """Same site dict shape _process_ref_site_task builds -- factored out so
    the Phase 0 min-ref-bits pre-pass and Phase 2's real scoring construct an
    identical, content-addressed site_id for the same physical site."""
    best_lig, best_chain = cluster[0]
    return {
        "site_id": f"ref_{ref_id}_{best_lig.name}_{best_lig.seqid.num}",
        "role": "reference", "pdb_id": ref_id, "chain": best_chain,
        "lig_resname": best_lig.name, "lig_resnum": str(best_lig.seqid.num),
        "paired_with": "",
    }


def prepare_reference_site(ref_site, fp, pif, smiles_cache):
    """Once per unique reference site. Returns ({p_idx: ifp}, err)."""
    _ensure_smiles(ref_site["lig_resname"], smiles_cache)
    outcome, msg = protonate_ligand.protonate_ligand(ref_site, smiles_cache, search_dirs=MAIN_STRUCT_DIRS)
    if outcome == "failed":
        return None, f"ligand protonation failed: {msg}"
    if not extract_pockets.extract_pocket(ref_site):
        return None, "pocket extraction failed"
    import run_prolif as rp
    ifp, err, groups = rp.run_site_scoped(ref_site, fp, pif)
    if groups is None:
        return None, err or "no phosphate groups on reference ligand"
    return groups, None


# Same convention as export_prolif_datawarrior.py's ATOM_MARKING_EXCLUDED_INTERACTIONS:
# VdWContact is excluded when judging whether a reference site has "real" chemical
# signal, since it fires on almost any nearby heavy atom regardless of genuine
# complementarity -- a site whose only bits are VdWContact can never distinguish a
# good mimic from a bad one no matter which candidate is scored against it.
NOVDW_EXCLUDED_INTERACTIONS = {"VdWContact"}


def _drop_vdw_only_bits(bits):
    return {b for b in bits if b[1] not in NOVDW_EXCLUDED_INTERACTIONS}


def ref_busiest_group_novdw_bits(ref_groups_ifp):
    """Non-VdW interaction bit count on this reference site's busiest
    phosphate group -- the same site-level richness measure
    build_ifg_prolif_dataset.py's --min-ref-bits filter uses (see that
    module's docstring: in the full-manifest dataset, 41/388 reference sites
    with 0 such bits were responsible for 46% of all pairs coming back
    score=n/a, regardless of which mimic was tried). Takes an
    already-computed {p_idx: ifp} dict rather than recomputing it, since
    prepare_reference_site already produces exactly this."""
    import run_prolif as rp
    best_n = 0
    for ifp in ref_groups_ifp.values():
        bits = _drop_vdw_only_bits(rp.flatten_canon_bits(rp.canonicalize_ifp_for_alignment(ifp)))
        best_n = max(best_n, len(bits))
    return best_n


def prepare_candidate_site(chain, res, hit_id, ref_id, smiles_cache):
    """Builds this candidate's site dict using the same site_id scheme
    ProLIF_v2's existing pipeline already uses: hit_{pdb_id}_{resname}_{resnum}
    -- content-addressed, so re-discovering the same physical residue from a
    different reference pairing reuses the already-protonated/extracted files
    for free (protonate_ligand.py/extract_pockets.py's own skip-if-exists
    checks). Returns (site_dict, ok)."""
    site = {
        "site_id": f"hit_{hit_id}_{res.name}_{res.seqid.num}",
        "role": "hit", "pdb_id": hit_id, "chain": chain,
        "lig_resname": res.name, "lig_resnum": str(res.seqid.num),
        "paired_with": ref_id,
    }
    _ensure_smiles(res.name, smiles_cache)
    outcome, msg = protonate_ligand.protonate_ligand(site, smiles_cache, search_dirs=MAIN_STRUCT_DIRS)
    if outcome == "failed":
        return site, False
    return site, extract_pockets.extract_pocket(site)


# ── scoring: one (ref_site, hit_id) pairing at a time ──

def score_hit_against_ref_site(ref_site, ref_groups_ifp, hit_id, transformation,
                                ref_lig_atoms, ref_ca, fp, pif, smiles_cache, failures, hit_ifps_out,
                                eligibility_counts=None):
    """Returns a list of assignment dicts (one per greedily-assigned candidate)
    for this ONE (ref_site, hit) pairing. Appends a dict to `failures` (in
    place) for every candidate that dropped out along the way, so nothing is
    silently lost -- see run_discovery's failure log.

    ref_ca: this reference site's own {ResidueId: xyz} CA positions (from
    run_prolif.get_ca_positions(ref_site), computed once by the caller and
    reused across every hit) -- used to compute each candidate's local pocket
    RMSD (calculate_pocket_rmsd), independent of ProLIF scoring/matching.

    Matching/assignment is decided entirely on NO-VdW bits (VdWContact fires
    on almost any nearby heavy atom regardless of genuine complementarity, so
    it's too promiscuous a signal to decide which phosphate a candidate is
    actually mimicking -- same reasoning as ref_busiest_group_novdw_bits'
    docstring). A secondary with-VdW score is computed for the SAME winning
    (candidate, phosphate) pairing afterward, for display parity with the
    review app's existing dual no-VdW/with-VdW convention -- it never
    influences which candidate wins which phosphate.

    hit_ifps_out: dict this function ADDS TO (in place) -- every candidate's
    own whole-ligand ifp, keyed by its site_id, regardless of whether that
    candidate went on to win an assignment. Accumulated across all ref_sites
    by the caller and persisted as a fresh ProLIF fingerprint pickle at the
    end of run_discovery, so downstream consumers (the review app's pool
    builder) never need the old chain's prolif_fingerprint_full_manifest.pkl.

    eligibility_counts: optional dict this function ADDS TO (in place) --
    "n_hits_scored"/"n_eligible_candidates"/"n_hits_zero_candidates", tallying
    what find_eligible_candidates (the 10A/2.5A geometric pre-filter) actually
    did on this hit, before any protonation/scoring cost is paid for its
    candidates. Aggregated by the caller into run_discovery's stage_counts
    report -- see build_run_report.py."""
    import run_prolif as rp

    site_tag = f"{ref_site['site_id']} vs {hit_id}"
    hit_path = bme.find_structure_file(hit_id, PROJECT_ROOT)
    if hit_path is None:
        failures.append({"phase": "phase2_pairing", "site": site_tag, "reason": "hit structure file not found"})
        return []
    candidates = find_eligible_candidates(hit_path, transformation, ref_lig_atoms,
                                           rejection_counts=eligibility_counts)
    if eligibility_counts is not None:
        eligibility_counts["n_hits_scored"] = eligibility_counts.get("n_hits_scored", 0) + 1
        eligibility_counts["n_eligible_candidates"] = eligibility_counts.get("n_eligible_candidates", 0) + len(candidates)
        if not candidates:
            eligibility_counts["n_hits_zero_candidates"] = eligibility_counts.get("n_hits_zero_candidates", 0) + 1

    # {(chain, resname, resnum): {p_idx: {"score", "score_with_vdw", "n_ca_mapped"}}}
    cand_p_scores = {}
    for chain, res in candidates:
        cand_tag = f"{site_tag} :: {res.name}{res.seqid.num}/{chain}"
        cand_site, ok = prepare_candidate_site(chain, res, hit_id, ref_site["pdb_id"], smiles_cache)
        if not ok:
            failures.append({"phase": "phase2_candidate", "site": cand_tag, "reason": "candidate ligand protonation/pocket extraction failed"})
            continue
        cand_ifp, err, _ = rp.run_site_scoped(cand_site, fp, pif)
        if cand_ifp is None:
            failures.append({"phase": "phase2_candidate", "site": cand_tag, "reason": f"ProLIF fingerprinting failed: {err}"})
            continue
        hit_ifps_out[cand_site["site_id"]] = cand_ifp

        # Independent of ProLIF scoring/matching below -- a purely geometric
        # measure of how well this candidate's local environment superimposes
        # onto the reference pocket, using the SAME (ref_id, hit_id) TM-align
        # transform already in hand. Computed once per candidate (not once
        # per phosphate group -- it doesn't depend on which group is scored).
        hit_ca = rp.get_ca_positions(cand_site, transform=transformation)
        pocket_rmsd = rp.calculate_pocket_rmsd(ref_ca, hit_ca)

        mapping = rp.build_residue_correspondence(ref_site, cand_site)
        if mapping is None:  # no TM-align transform on file for (ref_pdb, hit_pdb)
            failures.append({"phase": "phase2_candidate", "site": cand_tag, "reason": "no TM-align transform on file for this (ref, hit) pair"})
            continue
        canon_hit_ifp = rp.canonicalize_ifp_for_alignment(cand_ifp, protein_mapping=mapping)
        hit_bits_all = rp.flatten_canon_bits(canon_hit_ifp)
        hit_bits = _drop_vdw_only_bits(hit_bits_all)
        if not hit_bits:
            failures.append({"phase": "phase2_candidate", "site": cand_tag, "reason": "candidate produced zero mapped non-VdW interaction bits (no aligned-residue overlap, or VdW-only)"})
            continue

        for p_idx, ref_ifp in ref_groups_ifp.items():
            canon_ref_ifp = rp.canonicalize_ifp_for_alignment(ref_ifp)
            ref_bits_all = rp.flatten_canon_bits(canon_ref_ifp)
            ref_bits = _drop_vdw_only_bits(ref_bits_all)
            if not ref_bits:
                continue
            # Identical formula to run_prolif.py's own prolif_plif_score (verified
            # at run_prolif.py:979-981): fraction of the reference's own
            # interactions this candidate replicated, via ProLIF's aligned-residue
            # bits instead of the homebrew distance-based ones -- computed here on
            # NO-VdW bits. This is the REPORTED score, not what decides the
            # greedy assignment below -- see n_matched.
            matched_bits = ref_bits & hit_bits
            score = len(matched_bits) / len(ref_bits)
            if score <= MIN_SCORE:
                continue
            score_with_vdw = len(ref_bits_all & hit_bits_all) / len(ref_bits_all) if ref_bits_all else None
            key = (chain, res.name, str(res.seqid.num))
            cand_p_scores.setdefault(key, {})[p_idx] = {
                "score": score, "n_matched": len(matched_bits), "score_with_vdw": score_with_vdw,
                "n_ca_mapped": len(mapping), "pocket_rmsd": pocket_rmsd,
            }

    # Greedy assignment -- identical structure to analyze_plif._process_ref_site,
    # and same RAW-COUNT criterion run_prolif.py's own multi-phosphate group
    # selection uses (see that module's ref_phosphate_p_idx comment): sort all
    # (candidate, p_idx) pairs by NUMBER of matched non-VdW bits descending,
    # each candidate and each phosphate used at most once. Raw count, not the
    # fraction/score above -- a near-empty phosphate group (e.g. 1 total
    # interaction) can get its one bit matched and "win" on fraction (1/1 =
    # 100%) against a much better-matched, busier group (e.g. 5/8 = 62.5%)
    # that replicated far more real chemistry. Raw count isn't fooled by
    # that: a bigger, better-matched group can never lose to a smaller one on
    # matched-bit count the way it could on fraction.
    flat = [(d["n_matched"], key, p_idx, d)
            for key, pdict in cand_p_scores.items() for p_idx, d in pdict.items()]
    flat.sort(key=lambda x: -x[0])
    used_c, used_p, assignments = set(), set(), []
    for n_matched, key, p_idx, d in flat:
        if key not in used_c and p_idx not in used_p:
            used_c.add(key)
            used_p.add(p_idx)
            assignments.append((key, p_idx, d))

    n_phosphates_mimicked = len(used_p)
    results = []
    for (chain, mimic_name, mimic_num), p_idx, d in assignments:
        results.append({
            "ref": ref_site["pdb_id"], "hit": hit_id,
            "ref_chain": ref_site["chain"], "ref_lig": ref_site["lig_resname"],
            "ref_num": ref_site["lig_resnum"], "ref_p_idx": p_idx,
            "mimic": mimic_name, "mimic_chain": chain, "mimic_num": mimic_num,
            "prolif_plif_score": d["score"], "prolif_plif_score_with_vdw": d["score_with_vdw"],
            "n_ca_mapped": d["n_ca_mapped"], "pocket_rmsd": d["pocket_rmsd"],
            "n_ref_phosphates": len(ref_groups_ifp),
            "n_phosphates_mimicked": n_phosphates_mimicked,
        })
    return results


# ── ProcessPoolExecutor worker glue (module-level for Windows spawn pickling) ──

_worker_fp = None
_worker_pif = None


def _init_worker(tmalign_json_path):
    """Runs once per worker process. Rebuilds everything a fresh spawned
    process needs: RDKit logger silenced (same deadlock fix run_prolif.py's
    own _init_worker documents -- 8 workers writing sanitizer warnings to a
    shared inherited stderr pipe hung the pool solid on Windows), its own
    Fingerprint instance, and plip_common.TMALIGN_JSON re-set -- module-level
    state set in the PARENT process before pool creation does NOT propagate to
    spawned children on Windows (each reimports common.py fresh with
    TMALIGN_JSON=None), so build_residue_correspondence's get_transformation()
    call would otherwise raise in every worker despite main_prolif.py having
    already called set_tmalign_json() once up front."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import prolif as plf
    import phosphate_ifp as pif
    from run_prolif import INTERACTIONS
    global _worker_fp, _worker_pif
    _worker_fp = plf.Fingerprint(
        INTERACTIONS, parameters={"VdWContact": {"preset": "rdkit"}}, use_segid=False, count=True,
    )
    _worker_pif = pif
    plip_common.set_tmalign_json(tmalign_json_path)


def _ref_site_novdw_bits_task(ref_id, cluster, smiles_cache):
    """Phase 0 worker: protonates+extracts+fingerprints just this reference
    site (prepare_reference_site) and returns (site_id, novdw_bits, err).
    Reference-side work is small relative to the full hit-side cost Phase 1/2
    would otherwise spend on every one of this site's hits, so doing it AGAIN
    in Phase 2 for whichever sites survive the filter is an accepted, bounded
    duplication -- the whole point is skipping the much larger hit-side work
    for sites that don't survive."""
    ref_site = build_ref_site_dict(ref_id, cluster)
    ref_groups, err = prepare_reference_site(ref_site, _worker_fp, _worker_pif, smiles_cache)
    if not ref_groups:
        return ref_site["site_id"], 0, err
    return ref_site["site_id"], ref_busiest_group_novdw_bits(ref_groups), None


def _process_ref_site_task(ref_id, cluster, hits, smiles_cache):
    """Returns (assignments, failures, hit_ifps, eligibility_counts) --
    failures is a flat list of dicts ({"phase", "site", "reason"}) covering
    both this reference site's own setup and every candidate that dropped out
    while scoring its hits, so run_discovery can persist a complete
    accounting, not just successes. eligibility_counts is
    score_hit_against_ref_site's find_eligible_candidates tally, aggregated
    across this ref site's hits.
    hit_ifps (site_id -> raw ProLIF ifp) is every candidate fingerprinted
    while scoring this reference site's hits -- see score_hit_against_ref_site's
    hit_ifps_out docstring."""
    import run_prolif as rp

    failures = []
    hit_ifps = {}
    eligibility_counts = {}
    ref_site = build_ref_site_dict(ref_id, cluster)
    ref_groups, err = prepare_reference_site(ref_site, _worker_fp, _worker_pif, smiles_cache)
    if not ref_groups:
        failures.append({"phase": "phase2_reference", "site": ref_site["site_id"], "reason": err or "no phosphate groups"})
        return [], failures, hit_ifps, eligibility_counts
    # Once per reference site (not per hit/candidate): already restricted to
    # the ~10A pocket around the reference ligand for free, since
    # extract_pockets.py only ever extracts that radius -- see
    # calculate_pocket_rmsd's docstring.
    ref_ca = rp.get_ca_positions(ref_site)
    ref_lig_atoms = list(cluster[0][0])
    out = []
    for hit_data in hits:
        hit_id = hit_data["hit"]
        t, u = hit_data["transformation"]["t"], hit_data["transformation"]["u"]
        out.extend(score_hit_against_ref_site(
            ref_site, ref_groups, hit_id, (t, u), ref_lig_atoms, ref_ca, _worker_fp, _worker_pif,
            smiles_cache, failures, hit_ifps, eligibility_counts
        ))
    return out, failures, hit_ifps, eligibility_counts


# ── top-level driver ──

def _write_failure_log(path, run_id, failures, step_label="Step 7 discovery"):
    """Overwrites the failure log with the current full accounting: a
    by-phase/by-reason count summary up top (the "map"), then every
    individual failure below. Called periodically during Phase 2 (so a crash
    or interruption doesn't lose everything) and once more at the very end.

    step_label: distinguishes which pipeline step wrote this log when reused
    by another step's failure accounting (e.g. cluster_isosteric_motifs.py's
    Step 8) -- same format, just a different header, rather than duplicating
    this whole function for one string."""
    from collections import Counter
    by_phase = Counter(f["phase"] for f in failures)
    # Bucket by reason's first ~60 chars so near-identical messages (e.g. the
    # same pdb2pqr crash signature recurring across many structures) collapse
    # into one summary line instead of flooding it with near-duplicates.
    by_reason = Counter((f["phase"], f["reason"][:60]) for f in failures)

    lines = []
    lines.append(f"{step_label} failure log -- run_id={run_id}")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total failures so far: {len(failures)}")
    lines.append("")
    lines.append("== By phase ==")
    for phase, count in by_phase.most_common():
        lines.append(f"  {phase:20} {count}")
    lines.append("")
    lines.append("== By phase + reason (top 30) ==")
    for (phase, reason), count in by_reason.most_common(30):
        lines.append(f"  [{count:4}] {phase:20} {reason}")
    lines.append("")
    lines.append("== Full listing ==")
    for f in failures:
        lines.append(f"  [{f['phase']}] {f['site']} :: {f['reason']}")

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp_path, path)

    # Structured sibling (same basename, .csv instead of .log) -- so a report
    # generator (e.g. build_run_report.py) can read exact per-phase/per-reason
    # data directly instead of re-parsing this human-readable text.
    import csv
    csv_path = os.path.splitext(path)[0] + ".csv"
    csv_tmp_path = csv_path + ".tmp"
    with open(csv_tmp_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "site", "reason"])
        w.writerows([f["phase"], f["site"], f["reason"]] for f in failures)
    os.replace(csv_tmp_path, csv_path)


def _protonate_ids(ids, label):
    """Shared ThreadPoolExecutor protonation driver for a set of PDB ids --
    used for both the Phase 0a reference-only pass and Phase 1's full pass.
    Returns (n_ok, failures: list of {"phase","site","reason"})."""
    print(f"{label}: protonating {len(ids)} unique structures "
          f"({FINGERPRINT_WORKERS} parallel pdb2pqr30 workers)...")
    t0 = time.time()
    n_ok = 0
    failures = []
    # ThreadPoolExecutor, not ProcessPoolExecutor: each call is a blocking
    # subprocess.run() of the external pdb2pqr30 binary (protonate_protein.py),
    # which releases the GIL while waiting on the child process -- same reason
    # PDBDownloader parallelizes its downloads with threads, not processes.
    with ThreadPoolExecutor(max_workers=FINGERPRINT_WORKERS) as pool:
        futures = {
            pool.submit(protonate_protein.protonate, pdb_id, MAIN_STRUCT_DIRS, PROTEIN_PROTONATED_DIR): pdb_id
            for pdb_id in ids
        }
        for i, future in enumerate(as_completed(futures), 1):
            pdb_id = futures[future]
            try:
                ok, msg = future.result()
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}: {e}"
            n_ok += ok
            if not ok:
                print(f"  [warn] {pdb_id}: {msg}")
                failures.append({"phase": "phase1_protonation", "site": pdb_id, "reason": msg})
            if (i % 50 == 0) or (i == len(ids)):
                eta = (time.time() - t0) / i * (len(ids) - i)
                print(f"  [{i}/{len(ids)}] {n_ok} ok so far (ETA {fmt_eta(eta)})")
    print(f"{label} done: {n_ok}/{len(ids)} protonated.")
    return n_ok, failures


def run_discovery(run_id, tmalign_json_path, min_tm_score=0.7, min_ref_bits=0, workers=FINGERPRINT_WORKERS):
    """min_ref_bits: if > 0, skips entire reference sites whose busiest
    phosphate group has <= this many non-VdW interaction bits, BEFORE
    protonating any of their hits -- same site-level richness measure and
    threshold semantics as build_ifg_prolif_dataset.py's --min-ref-bits (a
    site with too little real chemical signal can never produce a meaningful
    score against any hit, so there's no reason to pay for its hits' full
    protonation+extraction+fingerprint cost). 0 (default) processes every
    reference site, matching the original, unfiltered behavior."""
    plip_common.set_tmalign_json(tmalign_json_path)  # parent process, for Phase 0/1
    with open(tmalign_json_path) as f:
        tm_data = json.load(f)

    smiles_cache = {}
    if os.path.exists(SMILES_CACHE_PATH):
        with open(SMILES_CACHE_PATH) as f:
            smiles_cache = json.load(f)

    out_dir = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
    os.makedirs(out_dir, exist_ok=True)
    failures_log_path = os.path.join(out_dir, f"discovery_failures_{run_id}.log")
    all_failures = []
    # Exact in/ok/failed counts per stage, populated as run_discovery
    # progresses -- written alongside the discovery output at the end (see
    # build_run_report.py, which turns this into the full per-step report).
    stage_counts = {}

    # Build the raw (ref_id, cluster, valid_hits) task list once -- feeds both
    # the optional min_ref_bits pre-pass below and Phase 2 itself. Parses every
    # reference structure and runs chain-dedup + spatial site clustering
    # (get_unique_ref_sites) -- real work across thousands of structures with
    # no parallelism (cheap per-structure, but silent otherwise), so it gets
    # its own progress line rather than looking like a hang before Phase 0/1's
    # first print.
    candidate_refs = [ref_id for ref_id, hits in tm_data.items()
                       if any(h.get("tm_score_1", 0) >= min_tm_score for h in hits)]
    print(f"Building task list: parsing + clustering sites for {len(candidate_refs)} reference structures...")
    t0 = time.time()
    raw_tasks = []
    for i, ref_id in enumerate(candidate_refs, 1):
        valid_hits = [h for h in tm_data[ref_id] if h.get("tm_score_1", 0) >= min_tm_score]
        ref_path = bme.find_structure_file(ref_id, PROJECT_ROOT)
        if not ref_path:
            continue
        st_ref = gemmi.read_structure(ref_path)
        for cluster in bme.get_unique_ref_sites(st_ref):
            raw_tasks.append((ref_id, cluster, valid_hits))
        if i % 500 == 0 or i == len(candidate_refs):
            eta = (time.time() - t0) / i * (len(candidate_refs) - i)
            print(f"  [{i}/{len(candidate_refs)}] {len(raw_tasks)} sites found so far (ETA {fmt_eta(eta)})")
    print(f"Task list built: {len(raw_tasks)} reference sites.")
    stage_counts["task_list_build"] = {
        "reference_structures_with_tm_hits": len(candidate_refs),
        "reference_sites_found": len(raw_tasks),
    }

    if min_ref_bits > 0:
        # Phase 0a: reference proteins only (a small set relative to the full
        # ref+hit pool) -- extract_pocket() requires the protein already
        # protonated, so this has to happen before Phase 0b can fingerprint
        # anything.
        ref_ids_only = {ref_id.upper() for ref_id, _, _ in raw_tasks}
        ref_ok, ref_failures = _protonate_ids(ref_ids_only, "Phase 0a (reference proteins)")
        all_failures.extend(ref_failures)
        stage_counts["phase0a_reference_protonation"] = {
            "in": len(ref_ids_only), "ok": ref_ok, "failed": len(ref_failures),
        }

        # Phase 0b: fingerprint each reference site and screen by non-VdW bit
        # count. Reference-side work here is small relative to the hit-side
        # work it lets Phase 1/2 skip for whichever sites don't survive.
        print(f"\nPhase 0b: screening {len(raw_tasks)} reference sites for > {min_ref_bits} "
              f"non-VdW interaction bits ({workers} workers)...")
        t0 = time.time()
        bits_by_site_id = {}
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                  initargs=(tmalign_json_path,)) as pool:
            futures = {pool.submit(_ref_site_novdw_bits_task, ref_id, cluster, smiles_cache): (ref_id, cluster)
                       for ref_id, cluster, _ in raw_tasks}
            for i, future in enumerate(as_completed(futures), 1):
                ref_id, cluster = futures[future]
                try:
                    site_id, bits, err = future.result()
                except Exception as e:
                    site_id = build_ref_site_dict(ref_id, cluster)["site_id"]
                    bits, err = 0, f"{type(e).__name__}: {e}"
                bits_by_site_id[site_id] = bits
                if err:
                    all_failures.append({"phase": "phase0b_ref_screen", "site": site_id, "reason": err})
                if i % 50 == 0 or i == len(raw_tasks):
                    eta = (time.time() - t0) / i * (len(raw_tasks) - i)
                    print(f"  [{i}/{len(raw_tasks)}] (ETA {fmt_eta(eta)})")

        tasks = [t for t in raw_tasks
                 if bits_by_site_id.get(build_ref_site_dict(t[0], t[1])["site_id"], 0) > min_ref_bits]
        n_screen_errors = sum(1 for f in all_failures if f["phase"] == "phase0b_ref_screen")
        print(f"  kept {len(tasks)}/{len(raw_tasks)} reference sites "
              f"({len(raw_tasks) - len(tasks)} dropped for <= {min_ref_bits} non-VdW bits)")
        stage_counts["phase0b_reference_screening"] = {
            "in": len(raw_tasks), "kept": len(tasks),
            "dropped_low_signal": len(raw_tasks) - len(tasks) - n_screen_errors,
            "errored": n_screen_errors,
        }
        if all_failures:
            _write_failure_log(failures_log_path, run_id, all_failures)
    else:
        tasks = raw_tasks

    # Phase 1: dedup protein protonation -- every ref/hit PDB the SURVIVING
    # tasks touch, protonated exactly once regardless of how many
    # pairings/candidates reference it (references were already protonated in
    # Phase 0a when min_ref_bits > 0; protonate() skip-if-exists makes
    # re-listing them here harmless). Shared out_dir with ProLIF_v2's existing
    # pipeline (free cross-run cache).
    ref_hit_ids = set()
    for ref_id, cluster, hits in tasks:
        ref_hit_ids.add(ref_id.upper())
        ref_hit_ids.update(h["hit"].upper() for h in hits)

    phase1_ok, hit_failures = _protonate_ids(ref_hit_ids, "Phase 1")
    all_failures.extend(hit_failures)
    stage_counts["phase1_protonation"] = {
        "in": len(ref_hit_ids), "ok": phase1_ok, "failed": len(hit_failures),
    }
    if all_failures:
        _write_failure_log(failures_log_path, run_id, all_failures)
        print(f"  ({len(all_failures)} failures logged so far to {os.path.basename(failures_log_path)})")

    # Phase 2: per-reference-site discovery + scoring, parallelized over
    # ref_site (not hit) so each reference's protonation/pocket/phosphate-
    # group fingerprint is computed once and reused across all of its hits.
    # `tasks` was already built above (and possibly filtered by min_ref_bits).
    n_pairings = sum(len(t[2]) for t in tasks)
    print(f"\nPhase 2: {len(tasks)} reference sites, {n_pairings} (ref_site, hit) pairings to score...")
    n_failures_before_phase2 = len(all_failures)
    # find_eligible_candidates' geometric pre-filter tally (10A/2.5A rule --
    # see score_hit_against_ref_site's eligibility_counts docstring), summed
    # across every ref site's hits. Keys aren't fixed up front: besides the
    # three summary counts, find_eligible_candidates contributes whichever
    # "rej_*"/"eligible" keys it actually hit (see its own rejection_counts
    # docstring) -- accumulated dynamically via defaultdict rather than a
    # fixed key list, so a future new rejection category shows up here for
    # free instead of silently being dropped by a `for k in eligibility_totals`
    # loop that doesn't know about it yet.
    from collections import defaultdict
    eligibility_totals = defaultdict(int)

    all_assignments = []
    all_hit_ifps = {}  # site_id -> raw ProLIF ifp, every candidate ever fingerprinted
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                              initargs=(tmalign_json_path,)) as pool:
        futures = {pool.submit(_process_ref_site_task, ref_id, cluster, hits, smiles_cache): ref_id
                   for ref_id, cluster, hits in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            ref_id = futures[future]
            try:
                assignments, failures, hit_ifps, elig_counts = future.result()
                all_assignments.extend(assignments)
                all_failures.extend(failures)
                all_hit_ifps.update(hit_ifps)
                for k, v in elig_counts.items():
                    eligibility_totals[k] += v
            except Exception as e:
                all_failures.append({"phase": "phase2_reference", "site": ref_id,
                                      "reason": f"{type(e).__name__}: {e}"})
                print(f"  [error] {ref_id}: {type(e).__name__}: {e}")
            if i % 20 == 0 or i == len(tasks):
                eta = (time.time() - t0) / i * (len(tasks) - i)
                print(f"  [{i}/{len(tasks)}] {len(all_assignments)} assignments, "
                      f"{len(all_failures)} failures so far (ETA {fmt_eta(eta)})")
                if all_failures:
                    _write_failure_log(failures_log_path, run_id, all_failures)

    out_path = os.path.join(out_dir, f"prolif_discovery_results_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(all_assignments, f, indent=2)
    print(f"\nDone. {len(all_assignments)} assignments -> {out_path}")
    if all_failures:
        _write_failure_log(failures_log_path, run_id, all_failures)
        print(f"{len(all_failures)} total failures logged -> {failures_log_path}")

    n_phase2_failures = len(all_failures) - n_failures_before_phase2
    stage_counts["phase2_eligibility_filter"] = dict(eligibility_totals)
    stage_counts["phase2_scoring"] = {
        "reference_sites": len(tasks),
        "pairings_attempted": n_pairings,
        "assignments_produced": len(all_assignments),
        "failed": n_phase2_failures,
    }
    stage_counts_path = os.path.join(out_dir, f"discovery_run_summary_{run_id}.json")
    with open(stage_counts_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id, "min_tm_score": min_tm_score, "min_ref_bits": min_ref_bits,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": stage_counts,
            "total_failures": len(all_failures),
        }, f, indent=2)
    print(f"Wrote per-stage run summary -> {stage_counts_path}")

    # Fresh per-hit-site ProLIF fingerprint pickle -- same fp.ifp/fp.site_ids
    # shape run_prolif.py's own prolif_fingerprint_full_manifest.pkl uses (see
    # that script's own fp.ifp = ...; fp.site_ids = ... construction), but
    # sourced entirely from every candidate THIS run fingerprinted. Only
    # hit-side ifps: reference-side ifps are always recomputed fresh downstream
    # regardless (review_app/backend's pool builder recomputes
    # phosphate_group_ifps() per reference site directly, same as
    # build_pool.py's own _compute_ref_site_worker already did). This is what
    # lets the review app's atom-level highlighting be derived without ever
    # touching the old chain's prolif_fingerprint_full_manifest.pkl.
    import prolif as plf
    from run_prolif import INTERACTIONS
    fresh_fp = plf.Fingerprint(
        INTERACTIONS, parameters={"VdWContact": {"preset": "rdkit"}}, use_segid=False, count=True,
    )
    fresh_fp.site_ids = list(all_hit_ifps.keys())
    fresh_fp.ifp = dict(enumerate(all_hit_ifps.values()))
    fp_pickle_path = os.path.join(out_dir, f"prolif_fingerprint_from_discovery_{run_id}.pkl")
    fresh_fp.to_pickle(fp_pickle_path)
    print(f"Wrote fresh hit-fingerprint pickle ({len(all_hit_ifps)} sites) -> {fp_pickle_path}")

    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--tmalign", default=None, help="Override path to tmalign_results_*.json")
    parser.add_argument("--min-tm-score", type=float, default=0.7)
    parser.add_argument("--min-ref-bits", type=int, default=0,
                         help="skip reference sites whose busiest phosphate group has "
                              "<= this many non-VdW interaction bits (0 = process all)")
    args = parser.parse_args()

    tm_path = args.tmalign
    if tm_path is None:
        results_dir = os.path.join(PROJECT_ROOT, "results")
        clean = os.path.join(results_dir, f"tmalign_results_{args.run_id}_clean.json")
        plain = os.path.join(results_dir, f"tmalign_results_{args.run_id}.json")
        tm_path = clean if os.path.exists(clean) else plain
        if not os.path.exists(tm_path):
            raise FileNotFoundError(f"No TM-align results for run_id={args.run_id} (checked {clean}, {plain})")

    run_discovery(args.run_id, tm_path, min_tm_score=args.min_tm_score, min_ref_bits=args.min_ref_bits)
