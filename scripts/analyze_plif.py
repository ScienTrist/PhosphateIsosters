import os
import csv
import json
import gemmi
import glob
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from constants import DISTANCE_CUTOFFS, METALS, EC_NAMES, STANDARD_AA, IGNORE_LIGANDS
from interaction_utils import load_ec_map
from utils import dist as _dist


# ── geometry helpers ──────────────────────────────────────────────────────────

# ── interaction detection ─────────────────────────────────────────────────────

def _ligand_coordinating_metal_positions(st, ligand_atoms):
    """Positions of metal ions within METAL coordination distance of any of
    the given ligand atoms. Used to detect metal-mediated protein contacts --
    a His/Glu/Asp coordinating the same metal the ligand coordinates sits
    close to the ligand only because of that shared metal, not because it
    bonds the ligand directly, so it shouldn't also be counted as a separate
    H_BOND/SALT_BRIDGE (that would double-count one real contact -- protein
    to metal to ligand -- as if it were two)."""
    positions = []
    metal_cutoff = DISTANCE_CUTOFFS["METAL"]
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in METALS:
                    for m_atom in res:
                        if any(m_atom.pos.dist(la.pos) <= metal_cutoff for la in ligand_atoms):
                            positions.append(m_atom.pos)
        break
    return positions


# Protein side-chain/backbone atoms that can only ACCEPT a hydrogen bond
# (no attached H, ever, at physiological pH) for standard amino acids.
# Backbone carbonyl O is acceptor-only for every residue. Ambiguous atoms
# (Ser/Thr/Tyr -OH, His ring N, Cys -SH) are deliberately left out -- they
# can plausibly act as either donor or acceptor, so they're not restricted.
_ACCEPTOR_ONLY_SIDECHAIN = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"),
    ("GLN", "OE1"),
}


def _is_protein_acceptor_only(res_name, atom_name):
    """True if this protein atom can only accept a hydrogen bond, never
    donate one -- used to reject impossible acceptor-acceptor "H-bonds"
    that a pure distance/element check can't tell apart from real ones.
    E.g. a phosphate oxygen (itself acceptor-only once deprotonated) sitting
    near an Asp/Glu carboxylate oxygen because both coordinate the same
    site, not because either has a proton to offer the other."""
    if atom_name == "O":  # backbone carbonyl, any residue
        return True
    return (res_name, atom_name) in _ACCEPTOR_ONLY_SIDECHAIN


def get_interactions(st, ligand_name, ligand_num, phosphate_only=False):
    """
    Returns list of ((x,y,z), type, res_name, res_num, chain_name).
    Positions stored as plain tuples so results are picklable across processes.
    """
    interactions = []
    ligand_res   = None

    cell = st.cell if st.cell.a > 1.0 else gemmi.UnitCell(1000, 1000, 1000, 90, 90, 90)
    ns   = gemmi.NeighborSearch(st[0], cell, 5.0)

    for c_idx, chain in enumerate(st[0]):
        for r_idx, res in enumerate(chain):
            if res.name == ligand_name and str(res.seqid.num) == str(ligand_num):
                ligand_res = res
            else:
                for a_idx, atom in enumerate(res):
                    ns.add_atom(atom, c_idx, r_idx, a_idx)

    if ligand_res is None:
        return interactions

    if phosphate_only:
        p_atoms = [a for a in ligand_res if a.element.name == "P"]
        if not p_atoms:
            ligand_atoms = list(ligand_res)
        else:
            seen = set()
            ligand_atoms = []
            for p in p_atoms:
                ligand_atoms.append(p)
                seen.add(id(p))
                for a in ligand_res:
                    if id(a) not in seen and a.element.name in ("O", "N") and p.pos.dist(a.pos) < 2.1:
                        ligand_atoms.append(a)
                        seen.add(id(a))
    else:
        ligand_atoms = list(ligand_res)

    metal_mediators = _ligand_coordinating_metal_positions(st, ligand_atoms)
    metal_cutoff     = DISTANCE_CUTOFFS["METAL"]

    seen_keys = set()
    # A SALT_BRIDGE always satisfies the H_BOND geometry too (tighter cutoff
    # subset of the looser one, same atoms) -- it's not an independent
    # interaction, just the stronger case of the same electrostatic contact.
    # Collected separately from seen_keys and reduced after the loop so a
    # residue is credited with exactly one electrostatic interaction no
    # matter which ligand atom/type the neighbor search happens to visit
    # first: SALT_BRIDGE always wins over H_BOND regardless of discovery order,
    # and among same-type candidates the CLOSEST atom wins (also regardless of
    # discovery order) -- gemmi.NeighborSearch.find_atoms() doesn't guarantee a
    # stable visit order run to run, and a multi-atom side chain (e.g. ARG's
    # NE/NH1/NH2) can easily span more than calculate_spatial_similarity's 2.0 A
    # match threshold, so "whichever atom got found first" previously let the
    # *reported* position for an already-decided interaction jitter between
    # runs on identical input, occasionally flipping a borderline match. Stored
    # as (itype, pos, dist) instead of (itype, pos) so the closest-wins compare
    # has something to compare against.
    electrostatic_by_residue = {}
    for l_atom in ligand_atoms:
        l_elem = l_atom.element.name
        # find_atoms(pos, alt, ...), not find_neighbors(atom, ...): find_neighbors
        # implicitly restricts results to protein atoms sharing l_atom's own altloc
        # label, so any ligand atom modeled at a non-blank altloc (routine for
        # partial-occupancy/PanDDA-event ligands in fragment-screening depositions --
        # confirmed on WYY@5SOI, entirely altloc 'C' at 13% occupancy) silently finds
        # ZERO protein neighbors even when real, well-within-cutoff contacts exist,
        # since the protein's own altloc labels ('A'/'B') are assigned independently
        # and essentially never coincide with the ligand's. '\0' bypasses that
        # altloc-matching (verified directly: recovers a 2.77 A VAL49 backbone-N
        # contact that find_neighbors silently dropped). Interaction presence/absence
        # doesn't need altloc-pairing consistency the way refinement would.
        for mark in ns.find_atoms(l_atom.pos, "\0", min_dist=0.01, radius=5.0):
            cra    = mark.to_cra(st[0])
            p_atom = cra.atom
            p_res  = cra.residue
            if p_res.name in ("HOH", "DOD", "WAT"):
                continue
            p_chain= cra.chain
            dist   = l_atom.pos.dist(p_atom.pos)
            pos    = (p_atom.pos.x, p_atom.pos.y, p_atom.pos.z)
            is_metal_mediated = any(p_atom.pos.dist(mp) <= metal_cutoff for mp in metal_mediators)

            # Check all three types independently — a single atom pair can
            # generate both SALT_BRIDGE and H_BOND (e.g. ligand-O to ARG-NH1).
            candidate_keys = []
            if not is_metal_mediated and dist <= DISTANCE_CUTOFFS["SALT_BRIDGE"]:
                if l_elem in ("O", "P") and p_res.name in ("ARG", "LYS", "HIS"):
                    if p_atom.name in ("NZ", "NH1", "NH2", "NE", "ND1", "NE2"):
                        candidate_keys.append(("SALT_BRIDGE", p_res.name, p_res.seqid.num, p_chain.name))

            if not is_metal_mediated and dist <= DISTANCE_CUTOFFS["H_BOND"]:
                if l_elem in ("O", "N", "F") and p_atom.element.name in ("O", "N"):
                    # Reject impossible acceptor-acceptor pairs: a protein atom
                    # that can only accept (never donate) paired with a ligand
                    # oxygen. This project's phosphate/sulfonate/carboxylate
                    # mimic chemistry means ligand O atoms are overwhelmingly
                    # acceptor-only oxyanion oxygens too, so neither side has
                    # a proton to actually form the bond with.
                    acceptor_pair = _is_protein_acceptor_only(p_res.name, p_atom.name) and l_elem == "O"
                    if not acceptor_pair:
                        candidate_keys.append(("H_BOND", p_res.name, p_res.seqid.num, p_chain.name))

            if dist <= DISTANCE_CUTOFFS["METAL"] and p_res.name in METALS:
                candidate_keys.append(("METAL_COORD", p_res.name, p_res.seqid.num, p_chain.name))

            for key in candidate_keys:
                itype, res_name, res_num, chain_name = key
                if itype in ("SALT_BRIDGE", "H_BOND"):
                    res_key = (res_name, res_num, chain_name)
                    existing = electrostatic_by_residue.get(res_key)
                    if (existing is None
                            or (existing[0] == "H_BOND" and itype == "SALT_BRIDGE")
                            or (existing[0] == itype and dist < existing[2])):
                        electrostatic_by_residue[res_key] = (itype, pos, dist)
                elif key not in seen_keys:
                    seen_keys.add(key)
                    interactions.append((pos, itype, res_name, res_num, chain_name))

    for (res_name, res_num, chain_name), (itype, pos, _dist) in electrostatic_by_residue.items():
        interactions.append((pos, itype, res_name, res_num, chain_name))

    return interactions


def get_interactions_per_phosphate(st, ligand_name, ligand_num):
    """
    Returns {p_idx: [interactions]} — one entry per P atom in the ligand,
    containing only the protein contacts attributable to that phosphate group
    (the P itself plus O/N atoms within 2.1 Å).  Phosphates with no contacts
    are omitted from the dict.
    """
    ligand_res = None
    for chain in st[0]:
        for res in chain:
            if res.name == ligand_name and str(res.seqid.num) == str(ligand_num):
                ligand_res = res
                break
        if ligand_res is not None:
            break

    if ligand_res is None:
        return {}

    p_atoms = [a for a in ligand_res if a.element.name == "P"]
    if not p_atoms:
        return {}

    cell = st.cell if st.cell.a > 1.0 else gemmi.UnitCell(1000, 1000, 1000, 90, 90, 90)
    ns   = gemmi.NeighborSearch(st[0], cell, 5.0)

    for c_idx, chain in enumerate(st[0]):
        for r_idx, res in enumerate(chain):
            if res.name == ligand_name and str(res.seqid.num) == str(ligand_num):
                continue
            for a_idx, atom in enumerate(res):
                ns.add_atom(atom, c_idx, r_idx, a_idx)

    result = {}
    for p_idx, p_atom in enumerate(p_atoms):
        group_atoms = [p_atom]
        for a in ligand_res:
            if a.element.name in ("O", "N") and p_atom.pos.dist(a.pos) < 2.1:
                group_atoms.append(a)

        metal_mediators = _ligand_coordinating_metal_positions(st, group_atoms)
        metal_cutoff     = DISTANCE_CUTOFFS["METAL"]

        seen_keys  = set()
        # See get_interactions() for why electrostatic contacts are deduped
        # per-residue after the loop instead of inline, and why the closest
        # candidate (not the first-found one) wins among same-type matches.
        electrostatic_by_residue = {}
        interactions = []
        for l_atom in group_atoms:
            l_elem = l_atom.element.name
            # See get_interactions()'s comment on this same substitution -- altloc
            # bypass, not a behavior change beyond fixing the silent missed-neighbor bug.
            for mark in ns.find_atoms(l_atom.pos, "\0", min_dist=0.01, radius=5.0):
                cra    = mark.to_cra(st[0])
                n_atom = cra.atom
                n_res  = cra.residue
                if n_res.name in ("HOH", "DOD", "WAT"):
                    continue
                n_chain = cra.chain
                dist    = l_atom.pos.dist(n_atom.pos)
                pos     = (n_atom.pos.x, n_atom.pos.y, n_atom.pos.z)
                is_metal_mediated = any(n_atom.pos.dist(mp) <= metal_cutoff for mp in metal_mediators)

                candidate_keys = []
                if not is_metal_mediated and dist <= DISTANCE_CUTOFFS["SALT_BRIDGE"]:
                    if l_elem in ("O", "P") and n_res.name in ("ARG", "LYS", "HIS"):
                        if n_atom.name in ("NZ", "NH1", "NH2", "NE", "ND1", "NE2"):
                            candidate_keys.append(("SALT_BRIDGE", n_res.name, n_res.seqid.num, n_chain.name))
                if not is_metal_mediated and dist <= DISTANCE_CUTOFFS["H_BOND"]:
                    if l_elem in ("O", "N", "F") and n_atom.element.name in ("O", "N"):
                        acceptor_pair = _is_protein_acceptor_only(n_res.name, n_atom.name) and l_elem == "O"
                        if not acceptor_pair:
                            candidate_keys.append(("H_BOND", n_res.name, n_res.seqid.num, n_chain.name))
                if dist <= DISTANCE_CUTOFFS["METAL"] and n_res.name in METALS:
                    candidate_keys.append(("METAL_COORD", n_res.name, n_res.seqid.num, n_chain.name))

                for key in candidate_keys:
                    itype, res_name, res_num, chain_name = key
                    if itype in ("SALT_BRIDGE", "H_BOND"):
                        res_key = (res_name, res_num, chain_name)
                        existing = electrostatic_by_residue.get(res_key)
                        if (existing is None
                                or (existing[0] == "H_BOND" and itype == "SALT_BRIDGE")
                                or (existing[0] == itype and dist < existing[2])):
                            electrostatic_by_residue[res_key] = (itype, pos, dist)
                    elif key not in seen_keys:
                        seen_keys.add(key)
                        interactions.append((pos, itype, res_name, res_num, chain_name))

        for (res_name, res_num, chain_name), (itype, pos, _dist) in electrostatic_by_residue.items():
            interactions.append((pos, itype, res_name, res_num, chain_name))

        if interactions:
            result[p_idx] = interactions

    return result


def get_metal_positions(st):
    """Returns list of (pos, res_name, res_num, chain_name) for every metal ion in the structure."""
    positions = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in METALS:
                    for atom in res:
                        pos = (atom.pos.x, atom.pos.y, atom.pos.z)
                        positions.append((pos, res.name, res.seqid.num, chain.name))
    return positions


def get_phosphorus_positions(st):
    """Returns gemmi.Position for every P atom in the structure, any residue.
    Used to catch candidates that are covalently (or closely) linked to a real
    phosphate carried by a *different* residue -- e.g. a PDB depositing one
    continuous phosphorylated ligand as separate HET codes (ARA-TT7-4GL),
    where the phosphate-free fragment alone would otherwise look like a valid
    isostere. A per-residue phosphorus check can't catch this; this can,
    regardless of whether a LINK record documents the bond."""
    positions = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.element.name == "P":
                        positions.append(atom.pos)
        break
    return positions


def calculate_spatial_similarity(ref_inters, hit_inters, hit_metal_positions=None, mimic_positions=None):
    """
    hit_metal_positions: output of get_metal_positions(st_hit).  When provided,
    any reference METAL_COORD that finds no match in hit_inters gets a second
    chance: if any hit metal sits within 1.0 Å of the reference metal position
    it is counted as matched -- but ONLY if the candidate mimic itself
    (mimic_positions) has an atom within metal-coordination distance of that
    hit metal. Structural conservation of the metal alone isn't enough: the
    protein's own coordinating residues (His/Glu/Asp) will hold that metal in
    place regardless of what ligand is bound, so without this check a mimic
    that doesn't chelate the metal at all could still get METAL_COORD credit
    purely because the protein's pocket is conserved.
    """
    if not ref_inters:
        return 0.0, {}

    def _etype(t):
        return "ELECTROSTATIC" if t in ("SALT_BRIDGE", "H_BOND") else t

    matched_ref   = set()
    matched_pairs = {}
    metal_cutoff  = DISTANCE_CUTOFFS["METAL"]

    for i, (r_pos, r_type, r_res, r_num, r_chain) in enumerate(ref_inters):
        for h_inter in hit_inters:
            h_pos, h_type, h_res, h_num, h_chain = h_inter
            if _etype(r_type) == _etype(h_type) and _dist(r_pos, h_pos) < 2.0:
                matched_ref.add(i)
                matched_pairs[(r_type, r_res, r_num, r_chain)] = h_inter
                break

        # Fallback for unmatched METAL_COORD: check all metals in the hit structure
        # within 1.0 Å, regardless of ion type -- but only counts if the mimic
        # itself actually coordinates that metal.
        if i not in matched_ref and r_type == "METAL_COORD" and hit_metal_positions and mimic_positions:
            for m_pos, m_name, m_num, m_chain in hit_metal_positions:
                if _dist(r_pos, m_pos) < 1.0 and any(_dist(mp, m_pos) <= metal_cutoff for mp in mimic_positions):
                    matched_ref.add(i)
                    matched_pairs[(r_type, r_res, r_num, r_chain)] = (
                        m_pos, "METAL_COORD", m_name, m_num, m_chain
                    )
                    break

    return len(matched_ref) / len(ref_inters), matched_pairs


def get_ref_p_positions(st, lig_name, lig_num):
    positions = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == lig_name and str(res.seqid.num) == str(lig_num):
                    for atom in res:
                        if atom.element.name == "P":
                            positions.append((atom.pos.x, atom.pos.y, atom.pos.z))
    return positions


def has_phosphate_at_site(st_hit, ref_p_positions, cutoff=2.5):
    if not ref_p_positions:
        return False
    # Simple/free phosphorus-oxoanion species only -- verified individually against
    # the RCSB Chemical Component Dictionary: PO4=phosphate, PI=hydrogenphosphate,
    # 2HP=dihydrogenphosphate, PO3=phosphite. "HPO" and "H2P" were WRONGLY in this
    # set: HPO is actually "2-OXOHEPTYLPHOSPHONIC ACID" (C7H15O4P, DrugBank
    # DB07912) -- a phosphonic-acid drug-like molecule, not a free phosphate ion --
    # and H2P is a phosphonylated sugar derivative. Both are exactly the kind of
    # complex phosphate-mimic candidate this pipeline is trying to find, so leaving
    # them here was silently excluding legitimate isostere hits as if they were
    # "already a real phosphate at this site."
    phosphate_residues = {"PO4", "PI", "2HP", "PO3"}
    for model in st_hit:
        for chain in model:
            if not chain.name.isupper():
                continue
            for res in chain:
                if res.name in phosphate_residues:
                    for atom in res:
                        if atom.element.name == "P":
                            ap = (atom.pos.x, atom.pos.y, atom.pos.z)
                            if any(_dist(ap, rp) < cutoff for rp in ref_p_positions):
                                return True
    return False


def load_rmsd_data(motif_dir):
    """Keys by reference SITE (ref_id, hit_id, ref_lig, ref_num), not by mimic
    ligand name — Pocket_RMSD is a property of the ref/hit Cα superposition at
    that site, computed once per hit file regardless of which ligand ends up
    being the matched mimic, so keying by ligand name made real RMSDs
    unreachable whenever PLIF's mimic-matching found a ligand that
    analyze_isosteres_simple.py's own (stricter/different) ligand filter
    hadn't classified as the site's HOLO ligand."""
    rmsd_map = {}
    csv_path = os.path.join(motif_dir, "isostere_full_list.csv")
    if not os.path.exists(csv_path):
        return rmsd_map
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = f"{row['Ref_ID']}_{row['Hit_ID']}_{row['Ref_Lig']}_{row['Ref_Num']}"
                rmsd_map[key] = float(row["Pocket_RMSD"])
            except (KeyError, ValueError):
                pass
    return rmsd_map


# ── per-reference-site worker (module-level for pickling) ─────────────────────

def _find_ref_path(ref_dir, ref_id, chain, ref_lig, ref_num):
    for ch in [chain, chain.lower()]:
        for ext in [".cif", ".pdb"]:
            p = os.path.join(ref_dir, f"ref_{ref_id}_{ch}_{ref_lig}_{ref_num}{ext}")
            if os.path.exists(p):
                return p
    return None


def _process_ref_site(args):
    """
    Worker: loads reference structure ONCE, then scores all hit files for that site.
    Returns (results_list, verifications_list).
    """
    ref_site_key, hit_paths, ref_dir, rmsd_map, ec_map = args
    ref_id, chain, ref_lig, ref_num = ref_site_key

    ref_path = _find_ref_path(ref_dir, ref_id, chain, ref_lig, ref_num)
    if not ref_path:
        return [], []

    try:
        st_ref = gemmi.read_structure(ref_path)
    except Exception:
        return [], []

    ref_inters_per_p = get_interactions_per_phosphate(st_ref, ref_lig, ref_num)
    ref_p_positions  = get_ref_p_positions(st_ref, ref_lig, ref_num)
    n_ref_phosphates = len(ref_inters_per_p)

    if not ref_inters_per_p:
        return [], []

    major_ec = ec_map.get(ref_id.upper(), "no_EC")
    ec_name  = f"EC_{major_ec}_{EC_NAMES.get(major_ec, 'Unknown')}"

    results       = []
    verifications = []

    for hit_path in hit_paths:
        fname = os.path.basename(hit_path)
        parts = fname.replace(".pdb", "").replace(".cif", "").split("_")
        if len(parts) < 6:
            continue
        hit_id = parts[2]

        try:
            st_hit = gemmi.read_structure(hit_path)
        except Exception:
            continue

        if has_phosphate_at_site(st_hit, ref_p_positions):
            continue

        hit_metal_positions = get_metal_positions(st_hit)
        hit_p_positions      = get_phosphorus_positions(st_hit)
        link_cutoff          = DISTANCE_CUTOFFS["LINKED_PHOSPHATE"]

        # Score every candidate mimic against every reference phosphate.
        # {(res.name, seqid_num): {p_idx: {score, dist, p_ref_inters, hit_inters, matched_pairs, rmsd}}}
        mimic_p_scores = {}
        for model in st_hit:
            for c in model:
                if not c.name.isupper():
                    continue
                for res in c:
                    has_phosphorus = any(a.element.name == "P" for a in res)
                    is_amino_acid  = gemmi.find_tabulated_residue(res.name).is_amino_acid()

                    # Anything that itself contains phosphorus is excluded from mimic
                    # scoring — this is an isostere search, so a phosphorylated residue
                    # (free phosphate, nucleotide, or a modified amino acid like SEP/
                    # TPO/PTR/LLP) is never a candidate, only a genuinely phosphate-free
                    # residue/ligand can "mimic" the reference phosphate.
                    if has_phosphorus:
                        continue
                    # Also exclude candidates covalently (or near-covalently) linked to
                    # a real phosphate carried by a different residue -- a residue-local
                    # phosphorus check alone misses this (see get_phosphorus_positions).
                    if hit_p_positions and any(
                        a.pos.dist(p) < link_cutoff for a in res for p in hit_p_positions
                    ):
                        continue
                    if is_amino_acid:
                        continue
                    if res.name in IGNORE_LIGANDS or res.name in STANDARD_AA:
                        continue

                    hit_inters = get_interactions(st_hit, res.name, res.seqid.num)
                    mimic_key  = (res.name, str(res.seqid.num))
                    mimic_positions = [(a.pos.x, a.pos.y, a.pos.z) for a in res]

                    for p_idx, p_ref_inters in ref_inters_per_p.items():
                        score, matched_pairs = calculate_spatial_similarity(
                            p_ref_inters, hit_inters, hit_metal_positions, mimic_positions
                        )
                        if score > 0.1:
                            rmsd = rmsd_map.get(f"{ref_id}_{hit_id}_{ref_lig}_{ref_num}", 0.0)
                            # Closest-approach distance from any mimic atom to this
                            # reference phosphate's P position -- used ONLY to pick
                            # among duplicate copies of the SAME mimic (same res.name +
                            # seqid.num showing up in more than one chain, e.g. a
                            # symmetric multimer each carrying its own copy) below.
                            # Deliberately geometric, not PLIF-score-based: the whole
                            # point of this tiebreak is "which physical copy is actually
                            # sitting where the phosphate was," which is a position
                            # question, not an interaction-pattern-quality question --
                            # using `score` here (as before) meant a chain positioned
                            # slightly worse but coincidentally scoring a marginally
                            # higher spatial-similarity number could win over a copy
                            # that's the more obviously "correct" spatial occupant.
                            ref_p_pos = ref_p_positions[p_idx]
                            dist = min(
                                ((mx - ref_p_pos[0]) ** 2 + (my - ref_p_pos[1]) ** 2 + (mz - ref_p_pos[2]) ** 2) ** 0.5
                                for mx, my, mz in mimic_positions
                            )
                            p_dict = mimic_p_scores.setdefault(mimic_key, {})
                            if p_idx not in p_dict or dist < p_dict[p_idx]["dist"]:
                                p_dict[p_idx] = {
                                    "score": score, "dist": dist, "rmsd": rmsd,
                                    "p_ref_inters": p_ref_inters,
                                    "hit_inters": hit_inters,
                                    "matched_pairs": matched_pairs,
                                }

        # Greedy assignment: sort all (mimic, phosphate) candidates by score descending.
        # Each mimic and each phosphate can be used at most once.
        flat = [
            (data["score"], mimic_key, p_idx, data)
            for mimic_key, p_dict in mimic_p_scores.items()
            for p_idx, data in p_dict.items()
        ]
        flat.sort(key=lambda x: -x[0])

        used_mimics     = set()
        used_phosphates = set()
        assignments     = []
        for score, mimic_key, p_idx, data in flat:
            if mimic_key not in used_mimics and p_idx not in used_phosphates:
                used_mimics.add(mimic_key)
                used_phosphates.add(p_idx)
                assignments.append((mimic_key, p_idx, data))

        n_phosphates_mimicked = len(used_phosphates)

        for mimic_key, p_idx, data in assignments:
            mimic_name = mimic_key[0]
            results.append({
                "ref": ref_id, "hit": hit_id, "mimic": mimic_name,
                "ref_chain": chain, "ref_lig": ref_lig, "ref_num": ref_num,
                "ref_p_idx": p_idx,
                "score": data["score"], "rmsd": data["rmsd"], "ec": ec_name,
                "ref_inters": len(data["p_ref_inters"]),
                "hit_inters": len(data["hit_inters"]),
                "n_ref_phosphates": n_ref_phosphates,
                "n_phosphates_mimicked": n_phosphates_mimicked,
            })
            verifications.append({
                "ref_id": ref_id, "hit_id": hit_id, "mimic": mimic_name,
                "ref_p_idx": p_idx,
                "n_ref_phosphates": n_ref_phosphates,
                "n_phosphates_mimicked": n_phosphates_mimicked,
                "score": data["score"], "rmsd": data["rmsd"],
                "ref_inters": data["p_ref_inters"],
                "hit_inters": data["hit_inters"],
                "matched_pairs": {str(k): v for k, v in data["matched_pairs"].items()},
            })

    return results, verifications


# ── PLIF sort ─────────────────────────────────────────────────────────────────

def sort_plif_confirmed(motif_dir, results):
    hits_dir      = os.path.join(motif_dir, "hits")
    holo_dir      = os.path.join(hits_dir, "holo")
    confirmed_dir = os.path.join(holo_dir, "plif_confirmed")
    os.makedirs(confirmed_dir, exist_ok=True)

    # Use full site key when available (new JSON format) so only the specific
    # binding site that was confirmed gets promoted.
    use_site_keys  = results and "ref_chain" in results[0]
    if use_site_keys:
        confirmed_keys = {
            (r["ref"].upper(), r["hit"].upper(),
             r.get("ref_chain", "").upper(), r.get("ref_lig", "").upper(), str(r.get("ref_num", "")))
            for r in results
        }
    else:
        confirmed_keys = {(r["ref"].upper(), r["hit"].upper()) for r in results}

    # Scan ALL hit locations — PLIF can confirm files that analyze_isosteres_simple
    # put in apo/ (e.g. ligand just outside 3 Å P-atom cutoff but still H-bonding).
    all_hits = (glob.glob(os.path.join(hits_dir, "**", "hit_*.pdb"), recursive=True) +
                glob.glob(os.path.join(hits_dir, "**", "hit_*.cif"), recursive=True))

    to_confirmed = to_unconfirmed = 0
    for fpath in all_hits:
        fname  = os.path.basename(fpath)
        parts  = fname.replace(".pdb", "").replace(".cif", "").split("_")
        if len(parts) < 6:
            continue
        ref_id, hit_id, chain, ref_lig, ref_num = (parts[1].upper(), parts[2].upper(),
                                                    parts[3].upper(), parts[4].upper(), parts[5])
        match_key  = (ref_id, hit_id, chain, ref_lig, ref_num) if use_site_keys else (ref_id, hit_id)
        qualifies  = match_key in confirmed_keys
        in_confirm = os.path.normpath(os.path.dirname(fpath)) == os.path.normpath(confirmed_dir)

        if qualifies and not in_confirm:
            shutil.move(fpath, os.path.join(confirmed_dir, fname)); to_confirmed += 1
        elif not qualifies and in_confirm:
            shutil.move(fpath, os.path.join(holo_dir, fname));      to_unconfirmed += 1

    total_c = sum(1 for f in os.listdir(confirmed_dir) if f.endswith((".pdb", ".cif")))
    total_u = sum(1 for f in os.listdir(holo_dir)      if f.endswith((".pdb", ".cif")))
    print(f"\nPLIF sort: {to_confirmed} -> plif_confirmed/  |  {to_unconfirmed} moved back")
    print(f"           {total_c} confirmed  |  {total_u} unconfirmed")


# ── main analysis ─────────────────────────────────────────────────────────────

def run_plif_analysis(motif_dir, limit=None, workers=None):
    if workers is None:
        workers = min(8, os.cpu_count() or 4)

    ec_map   = load_ec_map(motif_dir)
    rmsd_map = load_rmsd_data(motif_dir)

    verify_dir = os.path.join(motif_dir, "plif_verification")
    os.makedirs(verify_dir, exist_ok=True)
    if limit:
        for f in glob.glob(os.path.join(verify_dir, "verify_*.txt")):
            os.remove(f)

    h_dir = os.path.join(motif_dir, "hits")
    r_dir = os.path.join(motif_dir, "references")

    all_hits = (glob.glob(os.path.join(h_dir, "**", "hit_*.pdb"), recursive=True) +
                glob.glob(os.path.join(h_dir, "**", "hit_*.cif"), recursive=True))

    if limit:
        random.seed(42)
        random.shuffle(all_hits)
        all_hits = all_hits[:limit]
        print(f"Limiting to {limit} files.")

    # Group by reference site so each ref is loaded only once
    site_to_hits = defaultdict(list)
    for path in all_hits:
        parts = os.path.basename(path).replace(".pdb","").replace(".cif","").split("_")
        if len(parts) >= 6:
            key = (parts[1], parts[3], parts[4], parts[5])  # ref_id, chain, lig, num
            site_to_hits[key].append(path)

    total_sites = len(site_to_hits)
    total_files = len(all_hits)
    print(f"Found {total_files} hit files across {total_sites} reference sites.")
    print(f"Processing with {workers} workers (ref loaded once per site).\n")
    print(f"{'Ref':<6} | {'Hit':<6} | {'Mimic':<8} | {'P':<2} | {'PLIF':<6} | {'RMSD':<6} | {'P mimicked'}")
    print("-" * 70)

    task_args = [
        (site_key, hits, r_dir, rmsd_map, ec_map)
        for site_key, hits in site_to_hits.items()
    ]

    all_results       = []
    all_verifications = []
    done = 0

    hits_done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_ref_site, a): a for a in task_args}
        for future in as_completed(futures):
            task = futures[future]
            site_key, hit_paths = task[0], task[1]
            hits_done += len(hit_paths)
            done += 1
            if done % 20 == 0 or done == total_sites:
                print(f"  [{hits_done}/{total_files} hits | {done}/{total_sites} sites]    ",
                      end="\r", flush=True)
            try:
                results, verifications = future.result()
                for r in results:
                    print(" " * 70, end="\r")
                    print(f"{r['ref']:<6} | {r['hit']:<6} | {r['mimic']:<8} | "
                          f"P{r['ref_p_idx']} | {r['score']:<6.2f} | {r['rmsd']:<6.2f} | "
                          f"{r['n_phosphates_mimicked']}/{r['n_ref_phosphates']} P mimicked")
                all_results.extend(results)
                all_verifications.extend(verifications)
            except Exception as e:
                print(f"\n  ERROR in site {site_key}: {e}")

    print()

    # Write JSON results
    out_path = os.path.join(motif_dir, "plif_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    # Write verification files
    for v in all_verifications:
        ref_id, hit_id, m_name = v["ref_id"], v["hit_id"], v["mimic"]
        p_idx = v["ref_p_idx"]
        vpath = os.path.join(verify_dir, f"verify_{ref_id}_{hit_id}_{m_name}_P{p_idx}.txt")
        with open(vpath, "w", encoding="utf-8") as vf:
            vf.write(f"Verification for {ref_id} (Ref) vs {hit_id} (Hit)\n")
            vf.write(f"Mimic: {m_name}  |  Phosphate group: P{p_idx}\n")
            vf.write(f"Phosphates mimicked: {v['n_phosphates_mimicked']} / {v['n_ref_phosphates']}\n")
            vf.write(f"PLIF Score: {v['score']:.2f}\n")
            vf.write(f"Pocket RMSD: {v['rmsd']:.2f}\n\n")
            vf.write("REF interactions (✓ = matched in hit):\n")
            for inter in sorted(v["ref_inters"], key=lambda x: (x[2], x[3])):
                pos, t, r, n, ch = inter
                match = v["matched_pairs"].get(str((t, r, n, ch)))
                if match:
                    h_pos, h_t, h_r, h_n, h_ch = match
                    vf.write(f"  ✓ {r}{n}{ch:<4}: {t} at ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
                             f"  <- {h_r}{h_n}{h_ch} {h_t}\n")
                else:
                    vf.write(f"  ✗ {r}{n}{ch:<4}: {t} at ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})\n")
            vf.write("\nHIT interactions:\n")
            for inter in sorted(v["hit_inters"], key=lambda x: (x[2], x[3])):
                pos, t, r, n, ch = inter
                vf.write(f"  {r}{n}{ch:<4}: {t} at ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})\n")

    print(f"\nDone! {len(all_results)} results -> plif_results.json")
    sort_plif_confirmed(motif_dir, all_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",       default=os.path.join(PROJECT_ROOT, "results", "motif_analysis"))
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--workers",   type=int, default=None)
    parser.add_argument("--sort-only", action="store_true",
                        help="Skip PLIF scoring; just re-sort files using existing plif_results.json")
    args = parser.parse_args()

    if args.sort_only:
        json_path = os.path.join(args.dir, "plif_results.json")
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found. Run without --sort-only first.")
        else:
            with open(json_path, encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} results from plif_results.json")
            sort_plif_confirmed(args.dir, results)
    else:
        run_plif_analysis(args.dir, limit=args.limit, workers=args.workers)
