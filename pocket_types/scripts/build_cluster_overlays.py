"""
Step 3 of the pocket_types subproject: for each cluster (composition,
geometric, or AA-type -- anything with the right CSV schema), aligns a
sample of up to 25 member sites onto one anchor site and packs the result
into ONE combined PyMOL .py script covering every cluster -- lets you
eyeball whether a grouping method's output actually looks alike
structurally, one cluster at a time, without leaving PyMOL.

REVISION -- one file instead of one-per-cluster: originally this wrote a
separate pymol_overlay_cluster_<id>.py per cluster (151 files once every
AA-type category got its own cluster_id). Switching clusters meant closing
PyMOL's script and `run`-ing a different file each time. Restructured into
a single script that computes and embeds every cluster's alignment/display
data once, then exposes a `show_cluster` PyMOL command (via cmd.extend) so
switching is just `show_cluster ARG1-LYS1` at the PyMOL prompt -- no
re-running the (slow, up-front) script, no hunting for filenames.
`list_clusters` prints every available label. See write_combined_script()
for the generated script's exact structure.

IMPORTANT: this only READS ProLIF_v2/data/pockets/*.pdb (the same pocket
files run_prolif.py/build_pocket_features.py already use) and never writes
to them or to any other existing structure/results file. The alignment
transform is baked as plain numbers into the generated PyMOL script and
applied in-memory at PyMOL load time via cmd.transform_selection() -- the
exact convention pymol_load_pair_simple.py already uses for ref/hit overlay
(see its own module docstring), applied here to ref/ref instead. All new
output lives under pocket_types/results/ only.

REVISION -- alignment method replaced entirely, TM-align dropped: the
original version TM-aligned each site's protein backbone within
POCKET_ALIGN_RADIUS (5A) of the ligand, on the reasoning that the full ~10A
pocket diluted the fit with bystander residues. That was still wrong in a
more basic way, confirmed directly by eye: TM-align optimizes the PROTEIN
backbone's fit, with no mechanism at all tying the LIGAND (the phosphate
group these overlays exist to compare) to any particular position --
overlaying AA-type categories built by classify_by_aa_type.py showed
phosphate groups nowhere near each other despite "successful" TM-align runs,
because two unrelated proteins' backbones can superpose reasonably well while
their ligands, sitting in different spots relative to that backbone, end up
scattered. Fixed by reusing geometric_subcluster.py's typed-point maximum-
clique Kabsch fit instead (site_typed_points() + best_clique_correspondence_
rmsd()) -- the SAME phosphate-forced-anchor method built for the geometric
sub-clustering step, imported directly rather than reimplemented. Since the
phosphate pair is unconditionally forced into every correspondence there,
the resulting rigid transform is guaranteed to place both sites' phosphate
groups at (or very near) the same point -- confirmed directly: 0.37A
residual on a real pair, sub-atomic-radius, i.e. visually coincident. This
also means no more WSL/TM-align dependency, no more radius-filtered temp
PDBs, no more tempfile.mkdtemp() scratch directory for this script at all --
the whole alignment step is now a pure Python/scipy computation over the
SAME ifp-derived point sets build_pocket_features.py/geometric_subcluster.py
already build. Pairs with fewer than MIN_MATCHED_POINTS_FOR_DISPLAY are
skipped, same "not enough evidence, don't guess" convention geometric_
subcluster.py uses for its own (stricter, clustering-purpose) MIN_MATCHED_
POINTS -- see that constant's own comment for why this one is deliberately
lower.

One anchor per cluster (star topology, not all-vs-all): every sampled site is
aligned directly onto whichever sampled site has the most total interaction
hits (build_pocket_features.py's per-site count) -- the best-resolved pocket
in the sample, so the reference frame isn't an unusually sparse/noisy site.

Display is restricted to just the ligand's phosphate group (not the whole
ligand -- e.g. not all of ATP's adenine+ribose+triphosphate) plus whichever
SPECIFIC ATOMS (not whole residues) actually contributed a non-excluded
interaction for that site -- i.e. exactly the evidence build_pocket_features.
site_feature_bag() counted into the composition vector, at atom-level
resolution. Residue/atom identity comes straight from the cached fingerprint
pickle's ifp metadata (ProLIF ResidueId + parent_indices), the same source
build_pocket_features.py/geometric_subcluster.py already read.

Optionally takes --labels-csv (cluster_id,category,slug -- written by
prepare_aa_type_overlay_assignments.py) to key show_cluster()'s dict and
name its PyMOL group after the category's slug instead of the bare
cluster_id, e.g. `show_cluster ARG1-LYS1` / group "ARG1-LYS1" rather than
`show_cluster cluster_7` / group "cluster_7" -- purely cosmetic (the
cluster_id numbering underneath is unchanged, and show_cluster() accepts
either), and optional because not every assignments CSV has a meaningful
category string to label with (e.g. pocket_cluster_assignments.csv's KMeans
cluster_id has no such label) -- falls back to "cluster_<id>" when omitted
or a cluster_id has no matching row.

Usage: python build_cluster_overlays.py --preset {aa_type,geometric}   # common case
   or: python build_cluster_overlays.py [--assignments path] [--pockets-dir path]
                                          [--sample-size N] [--labels-csv path]
                                          [--out-dir path] [--out-name name.py] [--force]

Skips the (slow) rebuild entirely if the output script already looks newer than its
inputs (assignments/labels/fingerprint) -- pass --force to rebuild anyway.
"""
import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(POCKET_TYPES_ROOT)
PROLIF_V2_ROOT = os.path.join(PROJECT_ROOT, "ProLIF_v2")
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")

sys.path.insert(0, os.path.join(PROLIF_V2_ROOT, "scripts"))
import run_prolif as rp  # noqa: E402
import phosphate_ifp as pif  # noqa: E402
import MDAnalysis as mda  # noqa: E402

sys.path.insert(0, SCRIPT_DIR)
from build_pocket_features import (  # noqa: E402
    _load_ifp_by_site, _manifest_reference_sites, _provenance,
    BACKBONE_ATOM_NAMES, DEFAULT_EXCLUDED_INTERACTIONS,
)
from geometric_subcluster import site_typed_points, best_clique_correspondence_rmsd  # noqa: E402

RANDOM_SEED = 42
POCKETS_DIR_DEFAULT = os.path.join(PROLIF_V2_ROOT, "data", "pockets")
# Deliberately LOWER than geometric_subcluster.MIN_MATCHED_POINTS (3): that
# value is chosen for a fully rotation-determined RMSD feeding the actual
# clustering decision. This is display-only -- many AA-type categories are,
# by construction, very sparse (e.g. "LYS:1" is often just phosphate + one
# NZ atom, 2 points total), so requiring 3 meant almost nothing could ever
# align (confirmed directly: the LYS:1 category aligned only 1/24 sampled
# sites at the stricter bar). 2 points (phosphate + one more) still
# guarantees phosphate overlap and correctly orients the approach direction;
# it just leaves one rotational degree of freedom (twist) unconstrained,
# an acceptable tradeoff for a picture, not for a number used in clustering.
MIN_MATCHED_POINTS_FOR_DISPLAY = 2

# Same selector convention pymol_load_pair.py / pymol_load_pair_simple.py already
# use, reused verbatim so pocket residues/ligand/metals/water render consistently
# with the rest of the project's PyMOL scripts.
METALS_SEL = "elem Ca+Mn+Zn+Mg+Fe+Cu+Co+Ni+Mo+W"
AA_SEL = "ALA+ARG+ASN+ASP+CYS+GLU+GLN+GLY+HIS+ILE+LEU+LYS+MET+PHE+PRO+SER+THR+TRP+TYR+VAL"
WAT_SEL = "HOH+DOD+WAT"

# Colored by RESIDUE TYPE, not by structure/object -- with up to 25 structures
# overlaid, per-object coloring answers "which structure is this" (not useful
# once you're checking whether a community is genuinely Arg-dominated vs
# Lys-dominated); per-type coloring, held FIXED across every object in the
# script, answers "which residue type is this", so e.g. every ARG across all
# 25 overlaid sites reads as the same color and visually clusters together
# if the community's geometry is real. Every canonical AA gets its own unique
# color; metals reuse some AA hues since they're rendered as spheres (shape
# already disambiguates them from AA sticks). Ligand (phosphate group) gets
# its own fixed neutral color, distinct from every residue color, since it's
# the shared reference point every site is aligned around.
RESIDUE_COLORS = {
    # basic -- most directly relevant to phosphate salt bridges
    "ARG": "red", "LYS": "orange", "HIS": "salmon",
    # acidic
    "ASP": "blue", "GLU": "marine",
    # polar / H-bonding
    "SER": "green", "THR": "limegreen", "ASN": "forest", "GLN": "teal",
    "TYR": "deepteal", "CYS": "yellow",
    # small / hydrophobic (mostly backbone-only relevance here)
    "GLY": "grey70", "ALA": "wheat", "PRO": "purple", "VAL": "lightblue",
    "LEU": "brown", "ILE": "olive", "MET": "violet", "PHE": "deeppurple",
    "TRP": "raspberry",
    # metals
    "MG": "green", "ZN": "grey50", "CA": "tan", "MN": "hotpink", "FE": "orange",
    "CU": "brown", "CO": "skyblue", "NI": "cyan", "MO": "magenta", "W": "pink",
}
DEFAULT_RESIDUE_COLOR = "grey80"  # fallback for any resname not in the table above
LIGAND_COLOR = "white"


def load_cluster_members(assignments_csv):
    by_cluster = defaultdict(list)
    with open(assignments_csv, newline="") as f:
        for row in csv.DictReader(f):
            by_cluster[int(row["cluster_id"])].append(row)
    return by_cluster


def load_cluster_labels(labels_csv):
    """{cluster_id: slug, ...} from prepare_aa_type_overlay_assignments.py's
    aa_type_overlay_labels.csv, or {} if labels_csv is None."""
    if not labels_csv:
        return {}
    with open(labels_csv, newline="") as f:
        return {int(row["cluster_id"]): row["slug"] for row in csv.DictReader(f)}


def sample_cluster(members, sample_size, rng):
    if len(members) <= sample_size:
        return list(members)
    return rng.sample(members, sample_size)


def pick_anchor(sample, points_by_site):
    """Picks whichever sampled site aligns to the MOST other sampled sites
    (ties broken by lowest average RMSD among those alignments), instead of
    whichever has the highest total_hits.

    total_hits measures how well-RESOLVED a single site's own crystal
    structure is -- it says nothing about whether that site is geometrically
    TYPICAL of the cluster. Since every other sampled site is aligned
    directly onto this one (star topology, not all-vs-all), a total_hits-
    picked anchor that happens to be an outlier doesn't just produce worse
    RMSDs -- any site that can't reach MIN_MATCHED_POINTS_FOR_DISPLAY against
    it is silently dropped from the picture entirely (see align_sample_onto_
    anchor's SKIP prints). A promiscuous/crowded pocket can rack up more
    total_hits than a cleaner, more representative one, making it a worse
    anchor despite "winning" on that metric.

    This does one all-vs-sample pre-pass instead: at most sample_size^2
    alignments (<=~300 for the default sample_size=25), the same scale
    geometric_subcluster.compute_comparable_pairs() already runs per cluster
    for the actual clustering decision, so it's cheap here too. Falls back to
    the old total_hits heuristic only if no sampled site has a usable point
    set or any comparable partner at all, so a cluster still gets SOME
    anchor rather than crashing outright."""
    site_ids = [r["site_id"] for r in sample if points_by_site.get(r["site_id"]) is not None]
    by_id = {r["site_id"]: r for r in sample}

    best_id, best_score = None, None
    for candidate_id in site_ids:
        candidate_points = points_by_site[candidate_id]
        n_aligned, rmsd_sum = 0, 0.0
        for other_id in site_ids:
            if other_id == candidate_id:
                continue
            rmsd, n_matched, u, t = best_clique_correspondence_rmsd(points_by_site[other_id], candidate_points)
            if n_matched >= MIN_MATCHED_POINTS_FOR_DISPLAY:
                n_aligned += 1
                rmsd_sum += rmsd
        avg_rmsd = rmsd_sum / n_aligned if n_aligned else float("inf")
        score = (n_aligned, -avg_rmsd)  # maximize how many members it keeps on screen, then minimize their avg RMSD
        if best_score is None or score > best_score:
            best_score, best_id = score, candidate_id

    if best_id is None:
        return max(sample, key=lambda r: int(r["total_hits"]))
    return by_id[best_id]


def pocket_selection_by_type(residues_by_type, prot_mol):
    """{resname: pymol_selection_string, ...}. Per residue: if its
    interaction is BACKBONE-only (via the SAME _provenance() classifier
    build_pocket_features.py/geometric_subcluster.py already use), show only
    its backbone atoms (BACKBONE_ATOM_NAMES) -- the side chain isn't part of
    what's actually interacting, so it shouldn't be shown as if it were.
    Otherwise (sidechain/metal/mixed provenance -- i.e. the side chain
    itself, or a metal, is genuinely involved) show the whole residue.
    residues_by_type is {resname: {(chain, number): [atom_idx, ...], ...}, ...}
    (atom indices into prot_mol, for the provenance classification)."""
    backbone_names = "+".join(sorted(BACKBONE_ATOM_NAMES))
    selections = {}
    for resname, by_residue in residues_by_type.items():
        by_chain_full, by_chain_backbone = defaultdict(list), defaultdict(list)
        for (chain, number), atom_idxs in by_residue.items():
            provenance = _provenance(prot_mol, atom_idxs, resname)
            target = by_chain_backbone if provenance == "backbone" else by_chain_full
            target[chain].append(number)
        parts = []
        for chain, numbers in sorted(by_chain_full.items()):
            resi_list = "+".join(str(n) for n in sorted(numbers))
            chain_expr = f"chain {chain} and " if chain else ""
            parts.append(f"({chain_expr}resi {resi_list})")
        for chain, numbers in sorted(by_chain_backbone.items()):
            resi_list = "+".join(str(n) for n in sorted(numbers))
            chain_expr = f"chain {chain} and " if chain else ""
            parts.append(f"({chain_expr}resi {resi_list} and name {backbone_names})")
        selections[resname] = " or ".join(parts) if parts else "none"
    return selections


def ligand_selection(atom_names):
    """PyMOL selection for just the given atom names, still guarded by the
    same not-protein/not-water/not-metal check the old whole-ligand selector
    used (so a same-named protein atom in another residue can't leak in)."""
    if not atom_names:
        return f"not resn {AA_SEL}+{WAT_SEL} and not ({METALS_SEL})"  # fallback: whole ligand
    name_list = "+".join(atom_names)
    return f"(not resn {AA_SEL}+{WAT_SEL} and not ({METALS_SEL})) and name {name_list}"


def site_display_selections(site_id, manifest_row, ifp, excluded_interactions, pockets_dir, phosphate_target):
    """Returns (pocket_sel_by_type, ligand_sel) for one site: pocket_sel_by_
    type is whole qualifying RESIDUES (see pocket_selection_by_type()),
    ligand_sel is the phosphate group ONLY (still atom-level -- see below).
    Computed together off ONE prot_mol/lig_mol rebuild rather than two.
    Returns (None, None) if the pocket/ligand/protein selection fails.

    phosphate_target is the EXACT PHOSPHATE_TYPE coordinate site_typed_points()
    already picked as this site's alignment anchor (points_by_site[sid][0][1]),
    not re-derived here. This used to be recomputed independently via its own
    "closest phosphate group to interacting-residue centroid" pass, with a
    comment claiming it used "the SAME picking rule" as geometric_subcluster.
    _ligand_phosphate_point -- it didn't: this function's target_coords had no
    atom-index dedup and didn't exclude hydrogens, while _ligand_phosphate_
    point's does both. For single-phosphate ligands that never mattered (only
    one candidate group exists), but for multi-phosphate ligands (ATP, ADP,
    ANP, GNP, GTP -- common in this reference set) the two computations could
    pick DIFFERENT phosphate groups, so the atoms actually rendered as "the
    phosphate" weren't always the ones the alignment transform had pinned
    together. Passing in the already-computed anchor position instead of
    re-deriving a second, independently-drifting copy fixes this by
    construction rather than by trying to keep two implementations in sync."""
    pocket_path = os.path.join(pockets_dir, f"{site_id}.pdb")
    if not ifp or not os.path.exists(pocket_path):
        return None, None
    u = mda.Universe(pocket_path)
    protein_ag = rp.protein_selection(u)
    ligand_ag = u.select_atoms(
        f"resid {manifest_row['lig_resnum']} and chainID {manifest_row['chain']} and not protein")
    if len(protein_ag) == 0 or len(ligand_ag) == 0:
        return None, None
    prot_mol = rp.safe_molecule_from_mda(protein_ag, use_segid=False)
    lig_mol = rp.safe_molecule_from_mda(ligand_ag, use_segid=False, fix_ionizable=True)

    # resname -> (chain, number) -> [atom_idx, ...] -- only atoms that were
    # actually part of a non-excluded interaction's metadata, not the whole
    # residue; kept as INDICES (not names) so _provenance() can classify
    # each residue's interaction as backbone vs sidechain/metal/mixed.
    residues_by_type = defaultdict(lambda: defaultdict(list))
    for (lig_id, prot_id), interactions in ifp.items():
        qualifying = [md for name, mds in interactions.items() if name not in excluded_interactions for md in mds]
        if not qualifying:
            continue
        key = (prot_id.chain, prot_id.number)
        for md in qualifying:
            for idx in md["parent_indices"]["protein"]:
                residues_by_type[prot_id.name][key].append(idx)

    if not residues_by_type:
        return None, None
    pocket_sel_by_type = pocket_selection_by_type(residues_by_type, prot_mol)

    # Ligand: JUST the phosphate group's own atoms (P + coordinating O/N),
    # not the whole ligand (e.g. not all of ATP's adenine+ribose+triphosphate
    # -- different ligands' non-phosphate parts share nothing and were never
    # part of what the clustering, composition or geometric, actually looked
    # at). If a ligand has more than one phosphate group (ATP/GTP), picks
    # whichever sits closest to phosphate_target -- the caller's already-
    # computed alignment anchor (see this function's docstring for why this
    # is passed in rather than re-derived here).
    groups = pif.get_ligand_phosphate_groups(lig_mol)
    ligand_names = None
    if groups:
        lig_conf = lig_mol.GetConformer()
        best_group, best_dist = None, None
        for g in groups:
            p_pos = lig_conf.GetAtomPosition(g["atom_idxs"][0])
            p_pos = np.array([p_pos.x, p_pos.y, p_pos.z])
            dist = 0.0 if phosphate_target is None else float(np.linalg.norm(p_pos - phosphate_target))
            if best_dist is None or dist < best_dist:
                best_group, best_dist = g, dist
        ligand_names = [lig_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().strip()
                         for idx in best_group["atom_idxs"]]

    return pocket_sel_by_type, ligand_selection(ligand_names)


def align_sample_onto_anchor(sample, anchor, points_by_site):
    """Returns {site_id: {"rmsd":, "n_matched":, "t":, "u":}} for every non-
    anchor site that aligned successfully (n_matched >= MIN_MATCHED_POINTS_
    FOR_DISPLAY); anchor itself is not included (it needs no transform).
    Alignment is geometric_subcluster.best_clique_correspondence_rmsd() --
    the phosphate-forced typed-point Kabsch fit, not TM-align -- see module
    docstring for why."""
    anchor_points = points_by_site.get(anchor["site_id"])
    if anchor_points is None:
        print(f"    SKIP whole cluster: anchor {anchor['site_id']} has no typed point set")
        return {}
    results = {}
    for row in sample:
        site_id = row["site_id"]
        if site_id == anchor["site_id"]:
            continue
        site_points = points_by_site.get(site_id)
        if site_points is None:
            print(f"    SKIP {site_id}: no typed point set")
            continue
        rmsd, n_matched, u, t = best_clique_correspondence_rmsd(site_points, anchor_points)
        if n_matched < MIN_MATCHED_POINTS_FOR_DISPLAY:
            print(f"    SKIP {site_id}: only {n_matched} matched points vs anchor "
                  f"(need >={MIN_MATCHED_POINTS_FOR_DISPLAY})")
            continue
        results[site_id] = {"rmsd": rmsd, "n_matched": n_matched, "t": t, "u": u}
    return results


def build_cluster_payload(cluster_id, label, anchor, aligned, meta_by_site, pocket_sel_by_type_by_site,
                           ligand_sel_by_site):
    """One cluster's worth of data for the combined script's CLUSTERS dict --
    everything show_cluster() needs at PyMOL runtime, with no filesystem
    paths baked in except site_id (POCKETS_DIR + site_id is joined at
    runtime in the generated script instead, so the whole CLUSTERS dict
    doesn't repeat the pockets directory once per site)."""
    member_entries = {}
    for site_id, xform in aligned.items():
        m = meta_by_site.get(site_id, {})
        t, u = xform["t"], xform["u"]
        matrix = [
            u[0][0], u[0][1], u[0][2], t[0],
            u[1][0], u[1][1], u[1][2], t[1],
            u[2][0], u[2][1], u[2][2], t[2],
            0.0, 0.0, 0.0, 1.0,
        ]
        member_entries[site_id] = {
            "pdb_id": m.get("pdb_id", ""), "ref_ligand": m.get("ref_ligand", ""),
            "total_hits": m.get("total_hits", "?"),
            "rmsd": xform["rmsd"], "n_matched": xform["n_matched"], "matrix": matrix,
            "sel_by_type": pocket_sel_by_type_by_site[site_id], "lig_sel": ligand_sel_by_site[site_id],
        }
    return {
        "cluster_id": cluster_id,
        "label": label,
        "anchor": {
            "site_id": anchor["site_id"], "pdb_id": anchor["pdb_id"], "ref_ligand": anchor["ref_ligand"],
            "total_hits": anchor["total_hits"],
            "sel_by_type": pocket_sel_by_type_by_site[anchor["site_id"]],
            "lig_sel": ligand_sel_by_site[anchor["site_id"]],
        },
        "members": member_entries,
    }


def write_combined_script(payloads_by_label, pockets_dir, out_path):
    """Writes ONE PyMOL script covering every cluster in payloads_by_label
    ({label: build_cluster_payload() result, ...}), plus a companion .pkl
    file holding the actual CLUSTERS/CLUSTER_ID_TO_LABEL data. The .py script
    defines two PyMOL commands (cmd.extend, so they're callable directly from
    the PyMOL prompt): show_cluster(label_or_id) loads+aligns+styles just
    that one cluster (clearing whatever was shown before), and
    list_clusters() prints every available label. Switching clusters is then
    just `show_cluster <label>` at the prompt -- no re-running this script,
    no per-cluster files to hunt through.

    CLUSTERS used to be embedded directly in the .py file as a literal Python
    dict via repr(). Confirmed directly to be the reason `run`ning this
    script took close to a minute: a single ~400KB literal expression is a
    known-slow shape for CPython's tokenizer/compiler to parse, regardless of
    how simple the data inside it is. Loading the identical data from a
    pickle file instead (a compact, C-optimized binary format meant for
    exactly this) is dramatically faster -- the .py file itself now only
    contains function definitions and small constants (RESIDUE_COLORS etc.),
    fast to parse regardless of how many clusters/sites the pickle holds.

    Still two-pass per cluster inside show_cluster() (load everything, ONE
    cmd.sync(), THEN transform/style) for the same reason the old one-file-
    per-cluster version was: PyMOL's cmd.load()/cmd.show() calls are queued
    against its own internal thread, and a per-object sync at the default
    1s timeout reliably falls behind mid-batch -- confirmed directly, not
    theoretical."""
    labels_sorted = sorted(payloads_by_label, key=lambda lbl: -len(payloads_by_label[lbl]["members"]) - 1)
    cluster_id_to_label = {p["cluster_id"]: lbl for lbl, p in payloads_by_label.items()}

    lines = []
    lines.append('"""')
    lines.append("Auto-generated by build_cluster_overlays.py -- one script covering every cluster's")
    lines.append("sampled reference sites (aligned onto one anchor site per cluster) for visual")
    lines.append("inspection. Display is restricted to each site's own actual interacting residues")
    lines.append("and metals, colored by residue type (no full-protein cartoon, no ligand -- see")
    lines.append("this file's own module docstring for why).")
    lines.append("")
    lines.append("Usage, in PyMOL's command line:")
    lines.append(f"    run {os.path.basename(out_path)}   # run ONCE -- just loads data, shows nothing yet")
    lines.append("    list_clusters                        # see every available label")
    lines.append("    show_cluster ARG1-LYS1               # or: show_cluster 7  (cluster_id also works)")
    lines.append("    show_cluster LYS1-METAL1+BB          # switch again whenever -- no re-running")
    lines.append('"""')
    lines.append("import os")
    lines.append("import pickle")
    lines.append("import time")
    lines.append("from pymol import cmd")
    lines.append("")
    lines.append(f"POCKETS_DIR = r'{pockets_dir}'")
    lines.append(f"AA_SEL = '{AA_SEL}'")
    lines.append(f"WAT_SEL = '{WAT_SEL}'")
    lines.append(f"METALS_SEL = '{METALS_SEL}'")
    lines.append(f"RESIDUE_COLORS = {RESIDUE_COLORS!r}")
    lines.append(f"DEFAULT_RESIDUE_COLOR = {DEFAULT_RESIDUE_COLOR!r}")
    lines.append(f"LIGAND_COLOR = {LIGAND_COLOR!r}")
    lines.append("METAL_RESNAMES = {'CA', 'MN', 'ZN', 'MG', 'FE', 'CU', 'CO', 'NI', 'MO', 'W'}")
    lines.append("")
    lines.append("def _wait_for_objects(expected_names, timeout=30.0, poll=0.05):")
    lines.append("    \"\"\"Polls cmd.get_object_list() until every name in expected_names is")
    lines.append("    present AND has a nonzero atom count, or timeout elapses. Replaces")
    lines.append("    cmd.sync(), which this PyMOL build doesn't reliably detect queue")
    lines.append("    completion with -- confirmed directly: cmd.sync(N) consistently consumed")
    lines.append("    its FULL requested N seconds regardless of how much (or how little) work")
    lines.append("    was actually pending, making a 25-object cluster's show_cluster() call")
    lines.append("    take ~90s of pure dead time (30+60) before any real work even started.")
    lines.append("")
    lines.append("    The atom-count check (not just name presence) matters: this build's")
    lines.append("    async command queue can apparently register an object's NAME before its")
    lines.append("    atom data is actually parsed in -- confirmed directly by a real crash,")
    lines.append("    \"Invalid selection name\" thrown on an object that get_object_list() had")
    lines.append("    just reported present. Requiring count_atoms > 0 closes that gap.\"\"\"")
    lines.append("    deadline = time.time() + timeout")
    lines.append("    expected = set(expected_names)")
    lines.append("    while time.time() < deadline:")
    lines.append("        present = set(cmd.get_object_list())")
    lines.append("        if expected.issubset(present) and all(cmd.count_atoms(n) > 0 for n in expected):")
    lines.append("            return")
    lines.append("        time.sleep(poll)")
    lines.append("    print(f\"  [warn] _wait_for_objects timed out after {timeout}s -- \"")
    lines.append("          f\"missing/empty: {expected - set(cmd.get_object_list())}\")")
    lines.append("")
    lines.append("")
    lines.append("def _wait_for_empty(timeout=30.0, poll=0.05):")
    lines.append("    \"\"\"Same race, other end: cmd.delete('all') is also queued async, so the")
    lines.append("    very next cmd.load() can fire before the delete has actually cleared the")
    lines.append("    scene. Without this, re-running show_cluster on the same cluster (same")
    lines.append("    object names as last time) can hit the exact same stale-name race as")
    lines.append("    _wait_for_objects, just on the delete side instead of the load side.\"\"\"")
    lines.append("    deadline = time.time() + timeout")
    lines.append("    while time.time() < deadline:")
    lines.append("        if not cmd.get_object_list():")
    lines.append("            return")
    lines.append("        time.sleep(poll)")
    lines.append("    print(f\"  [warn] _wait_for_empty timed out after {timeout}s -- \"")
    lines.append("          f\"still present: {cmd.get_object_list()}\")")
    lines.append("")
    lines.append("def _show(obj, pocket_sel_by_type, ligand_sel):")
    lines.append("    # Only the residues/metals that actually contributed a qualifying")
    lines.append("    # interaction for THIS site (pocket_sel_by_type is already atom-level --")
    lines.append("    # backbone-only where the interaction is backbone-only, see")
    lines.append("    # pocket_selection_by_type()) -- never a full-protein cartoon, never the")
    lines.append("    # ligand. With N sites from N different proteins overlaid, the ligands")
    lines.append("    # (different sizes/shapes, only their phosphate group is forced into the")
    lines.append("    # same point) stack into one dominant blob and add nothing the residue")
    lines.append("    # colors don't already show.")
    lines.append("    try: cmd.unbond(f'({obj} and ({METALS_SEL}))', obj)")
    lines.append("    except Exception: pass")
    lines.append("    cmd.hide('everything', obj)")
    lines.append("    for resname, sel in pocket_sel_by_type.items():")
    lines.append("        if sel == 'none':")
    lines.append("            continue")
    lines.append("        full_sel = f'{obj} and ({sel})'")
    lines.append("        color = RESIDUE_COLORS.get(resname, DEFAULT_RESIDUE_COLOR)")
    lines.append("        if resname in METAL_RESNAMES:")
    lines.append("            cmd.show('spheres', full_sel)")
    lines.append("            cmd.set('sphere_scale', 0.35, full_sel)")
    lines.append("            cmd.color(color, full_sel)")
    lines.append("        else:")
    lines.append("            cmd.show('sticks', full_sel)")
    lines.append("            cmd.color(color, f'{full_sel} and elem C')")
    lines.append("")

    pkl_path = os.path.splitext(out_path)[0] + "_data.pkl"
    lines.append(f"_DATA_PKL = r'{pkl_path}'")
    lines.append("with open(_DATA_PKL, 'rb') as _f:")
    lines.append("    _cache = pickle.load(_f)")
    lines.append("CLUSTER_ID_TO_LABEL = _cache['cluster_id_to_label']")
    lines.append("CLUSTERS = _cache['clusters']")
    lines.append("")

    lines.append("def show_cluster(label_or_id):")
    lines.append("    label = CLUSTER_ID_TO_LABEL.get(int(label_or_id)) if str(label_or_id).lstrip('-').isdigit() else label_or_id")
    lines.append("    payload = CLUSTERS.get(label)")
    lines.append("    if payload is None:")
    lines.append("        print(f\"No such cluster: {label_or_id!r} -- run list_clusters for valid labels/ids\")")
    lines.append("        return")
    lines.append("    _t_start = time.time()")
    lines.append("    cmd.delete('all')")
    lines.append("    _wait_for_empty(timeout=30.0)")
    lines.append('    cmd.bg_color("black")')
    lines.append("    _t_delete = time.time()")
    lines.append("    cmd.set('stick_radius', 0.06)")
    lines.append("    cmd.set('stick_ball', 0)")
    lines.append("    cmd.set('stick_ball_ratio', 1.0)")
    lines.append("    cmd.set('nonbonded_size', 0.1)")
    lines.append("    cmd.set('line_width', 1)")
    lines.append("")
    lines.append("    anchor = payload['anchor']")
    lines.append("    anchor_obj = 'anchor_' + anchor['site_id']")
    lines.append("    members = [anchor_obj]")
    lines.append("    # suspend_updates stops PyMOL re-rendering the viewport after EVERY")
    lines.append("    # single cmd.load()/cmd.show()/cmd.color() call -- confirmed directly to be")
    lines.append("    # the dominant cost for a 25-object cluster (cmd.sync(60) was hitting its")
    lines.append("    # full timeout without the load queue draining at all). try/finally so a")
    lines.append("    # mid-loop error can't leave the whole PyMOL session stuck looking frozen.")
    lines.append("    cmd.set('suspend_updates', 'on')")
    lines.append("    try:")
    lines.append("        # --- Pass 1: load every structure (styling happens after ALL loads finish) ---")
    lines.append("        print(f\"Anchor: {anchor['site_id']}  ({anchor['pdb_id']} {anchor['ref_ligand']}, \"")
    lines.append("              f\"{anchor['total_hits']} hits) -- native frame, everything else aligned onto this\")")
    lines.append("        cmd.load(os.path.join(POCKETS_DIR, anchor['site_id'] + '.pdb'), object=anchor_obj)")
    lines.append("        for site_id, m in payload['members'].items():")
    lines.append("            obj = 's_' + site_id")
    lines.append("            print(f\"{site_id}  ({m['pdb_id']} {m['ref_ligand']}, {m['total_hits']} hits) -- \"")
    lines.append("                  f\"RMSD {m['rmsd']:.2f} A over {m['n_matched']} matched atoms vs anchor\")")
    lines.append("            cmd.load(os.path.join(POCKETS_DIR, site_id + '.pdb'), object=obj)")
    lines.append("            members.append(obj)")
    lines.append("        _t_loaded = time.time()")
    lines.append("        print(f\"  [timing] {len(members)} cmd.load() calls: {_t_loaded - _t_delete:.1f}s\")")
    lines.append("")
    lines.append("        # --- Pass 2: wait for the whole load batch, then transform/style every object ---")
    lines.append("        _wait_for_objects(members, timeout=60.0)")
    lines.append("        _t_sync = time.time()")
    lines.append("        print(f\"  [timing] wait for {len(members)} objects to load: {_t_sync - _t_loaded:.1f}s\")")
    lines.append("        _show(anchor_obj, anchor['sel_by_type'], anchor['lig_sel'])")
    lines.append("        for site_id, m in payload['members'].items():")
    lines.append("            obj = 's_' + site_id")
    lines.append("            cmd.transform_selection(obj, m['matrix'], homogenous=1)")
    lines.append("            _show(obj, m['sel_by_type'], m['lig_sel'])")
    lines.append("        _t_styled = time.time()")
    lines.append("        print(f\"  [timing] transform+style {len(members)} objects: {_t_styled - _t_sync:.1f}s\")")
    lines.append("    finally:")
    lines.append("        cmd.set('suspend_updates', 'off')")
    lines.append("")
    lines.append("    group_name = f\"cluster_{payload['cluster_id']}\"")
    lines.append("    cmd.group(group_name, ' '.join(members))")
    lines.append("    cmd.deselect()")
    lines.append("    cmd.zoom('all')")
    lines.append("    _t_end = time.time()")
    lines.append("    print(f\"  [timing] group+zoom: {_t_end - _t_styled:.1f}s -- TOTAL show_cluster: {_t_end - _t_start:.1f}s\")")
    lines.append("    print(f\"Cluster {payload['cluster_id']} ({payload['label']}): {len(members)} \"")
    lines.append("          f\"structures overlaid onto anchor {anchor['site_id']}\")")
    lines.append("")

    lines.append("def list_clusters():")
    lines.append("    for label, payload in sorted(CLUSTERS.items(), key=lambda kv: -(len(kv[1]['members'])+1)):")
    lines.append("        n = len(payload['members']) + 1")
    lines.append("        print(f\"  cluster_{payload['cluster_id']:<4} {label:30s} {n:3d} sites shown\")")
    lines.append("")

    lines.append("cmd.extend('show_cluster', show_cluster)")
    lines.append("cmd.extend('list_clusters', list_clusters)")
    lines.append("")
    lines.append("print(f'Loaded {len(CLUSTERS)} clusters. Run list_clusters to see labels, "
                 "show_cluster <label> to display one.')")

    with open(pkl_path, "wb") as f:
        pickle.dump(
            {"cluster_id_to_label": cluster_id_to_label,
             "clusters": {label: payloads_by_label[label] for label in labels_sorted}},
            f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# {preset_name: (assignments_filename, labels_filename_or_None, out_filename)} -- lets
# --preset aa_type / --preset geometric stand in for the 3 flags people otherwise had to
# remember together (which assignments CSV goes with which labels CSV, and what to call
# the output). Bare filenames only; always resolved against RESULTS_DIR/args.out_dir
# below, same as the old hardcoded defaults were.
PRESETS = {
    "aa_type": ("aa_type_overlay_assignments.csv", "aa_type_overlay_labels.csv", "pymol_overlay_all_clusters.py"),
    "geometric": ("pocket_cluster_assignments.csv", None, "pymol_overlay_all_clusters.py"),
}


# Module-level (not nested in main()) because ProcessPoolExecutor on Windows
# uses "spawn": each worker re-imports this whole file, and a spawned
# process can only unpickle a function it can look up by module-level name --
# a closure/nested function isn't picklable at all. main() itself stays safe
# to keep using a pool because it's guarded by if __name__ == "__main__" at
# the bottom of this file, so a spawned worker re-importing the module never
# re-executes main() and re-spawns its own pool recursively (that failure
# mode is exactly what broke pocket_types/scripts/run_step7_8.py earlier --
# see its own history for the concrete RuntimeError this guards against).
def _compute_site_points(task):
    sid, manifest_row, ifp, excluded_interactions, pockets_dir = task
    return sid, site_typed_points(sid, manifest_row, ifp, excluded_interactions, pockets_dir)


def _compute_site_display(task):
    sid, manifest_row, ifp, excluded_interactions, pockets_dir, phosphate_target = task
    sel_by_type, lig_sel = site_display_selections(
        sid, manifest_row, ifp, excluded_interactions, pockets_dir, phosphate_target)
    return sid, sel_by_type, lig_sel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                         help="Shortcut for --assignments/--labels-csv/--out-name together "
                              "(overrides those three if also given). aa_type = classify_by_aa_type.py's "
                              "clustering; geometric = geometric_subcluster.py's KMeans clustering.")
    parser.add_argument("--assignments", default=os.path.join(RESULTS_DIR, "pocket_cluster_assignments.csv"))
    parser.add_argument("--pockets-dir", default=POCKETS_DIR_DEFAULT)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--fp-pickle", default=os.path.join(PROLIF_V2_ROOT, "results",
                                                              "prolif_fingerprint_full_manifest.pkl"))
    parser.add_argument("--manifest", default=os.path.join(PROLIF_V2_ROOT, "full_manifest.csv"))
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--out-name", default="pymol_overlay_all_clusters.py")
    parser.add_argument("--force", action="store_true",
                         help="Recompute even if the output script already looks up to date "
                              "(newer than the assignments/labels/fingerprint inputs).")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                         help="Parallel workers for per-site structure loading -- confirmed directly "
                              "to be the dominant cost (~0.89s/site building the MDAnalysis/RDKit "
                              "molecules), not the alignment math itself (~1ms/call), and each site's "
                              "load is independent of every other site's, so this parallelizes cleanly.")
    args = parser.parse_args()

    if args.preset:
        assignments_name, labels_name, out_name = PRESETS[args.preset]
        args.assignments = os.path.join(RESULTS_DIR, assignments_name)
        args.labels_csv = os.path.join(RESULTS_DIR, labels_name) if labels_name else None
        args.out_name = out_name

    out_path = os.path.join(args.out_dir, args.out_name)
    input_paths = [args.assignments, args.fp_pickle, args.manifest] + ([args.labels_csv] if args.labels_csv else [])
    if not args.force and os.path.exists(out_path):
        out_mtime = os.path.getmtime(out_path)
        stale = [p for p in input_paths if os.path.exists(p) and os.path.getmtime(p) > out_mtime]
        if not stale:
            print(f"{out_path} is already up to date with its inputs -- skipping rebuild (use --force to override).")
            return

    by_cluster = load_cluster_members(args.assignments)
    labels = load_cluster_labels(args.labels_csv)
    meta_by_site = {}
    for members in by_cluster.values():
        for row in members:
            meta_by_site[row["site_id"]] = row

    manifest_rows = _manifest_reference_sites(args.manifest)

    print(f"Loading cached ProLIF fingerprint from {args.fp_pickle} ...")
    ifp_by_site = _load_ifp_by_site(args.fp_pickle)

    rng = random.Random(RANDOM_SEED)
    t0 = time.time()

    # Every cluster's sample is decided up front so both parallel passes
    # below can batch ALL sites across ALL clusters into ONE shared worker
    # pool each, instead of spinning a pool up/down per cluster -- most
    # clusters are tiny (median size 1 site), so a per-cluster pool would
    # mostly just pay process-startup overhead for zero parallelism benefit.
    samples_by_cluster = {cid: sample_cluster(members, args.sample_size, rng)
                           for cid, members in by_cluster.items()}

    # Pass 1: build every sampled site's typed points in parallel. Confirmed
    # directly to be THE dominant cost of this whole script (~0.89s/site
    # building the MDAnalysis Universe + RDKit molecules, vs ~1ms/call for
    # the alignment math that consumes them) -- and it's embarrassingly
    # parallel, since one site's points depend only on that site's own
    # manifest row/ifp entry, nothing shared or mutated across sites.
    point_tasks, seen = [], set()
    for sample in samples_by_cluster.values():
        for row in sample:
            sid = row["site_id"]
            if sid in seen:
                continue  # a site can't be in more than one cluster's sample; defensive only
            seen.add(sid)
            point_tasks.append((sid, manifest_rows[sid], ifp_by_site.get(sid), DEFAULT_EXCLUDED_INTERACTIONS, args.pockets_dir))

    print(f"Building typed points for {len(point_tasks)} sites across {len(by_cluster)} clusters "
          f"({args.workers} parallel workers) ...")
    points_by_site = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (sid, pts) in enumerate(executor.map(_compute_site_points, point_tasks), 1):
            points_by_site[sid] = pts
            if i % 100 == 0 or i == len(point_tasks):
                print(f"  [{i}/{len(point_tasks)}] ({time.time()-t0:.0f}s elapsed)")

    # Pass 2: per-cluster anchor selection + alignment. Confirmed directly to
    # be cheap (well under a second even for the largest cluster's full
    # O(n^2) all-vs-sample pre-pass) -- no parallelization needed here.
    all_transforms = {}
    anchor_by_cluster, aligned_by_cluster = {}, {}
    for cluster_id in sorted(by_cluster):
        sample = samples_by_cluster[cluster_id]
        anchor = pick_anchor(sample, points_by_site)
        print(f"Cluster {cluster_id}: {len(by_cluster[cluster_id])} sites, sampled {len(sample)}, "
              f"anchor={anchor['site_id']} ({anchor['total_hits']} hits)")
        aligned = align_sample_onto_anchor(sample, anchor, points_by_site)
        print(f"  {len(aligned)}/{len(sample)-1} non-anchor sites aligned successfully")
        anchor_by_cluster[cluster_id] = anchor
        aligned_by_cluster[cluster_id] = aligned
        all_transforms[cluster_id] = {"anchor": anchor["site_id"], "sample_size": len(sample), "members": aligned}

    # Pass 3: build display selections in parallel for every (anchor + each
    # aligned member) across every cluster -- the SAME per-site MDAnalysis/
    # RDKit cost as Pass 1, so it gets the same parallel treatment. Can't be
    # merged into Pass 1: which sites even need a display selection isn't
    # known until Pass 2 decides who aligned.
    display_tasks, seen = [], set()
    for cluster_id, sample in samples_by_cluster.items():
        anchor, aligned = anchor_by_cluster[cluster_id], aligned_by_cluster[cluster_id]
        for row in [anchor] + [r for r in sample if r["site_id"] in aligned]:
            sid = row["site_id"]
            if sid in seen:
                continue
            seen.add(sid)
            phosphate_target = points_by_site[sid][0][1]  # site_typed_points()'s (PHOSPHATE_TYPE, coord, weight)[0]
            display_tasks.append((sid, manifest_rows[sid], ifp_by_site.get(sid), DEFAULT_EXCLUDED_INTERACTIONS,
                                   args.pockets_dir, phosphate_target))

    print(f"\nBuilding display selections for {len(display_tasks)} sites ({args.workers} parallel workers) ...")
    pocket_sel_by_type_by_site, ligand_sel_by_site = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (sid, sel_by_type, lig_sel) in enumerate(executor.map(_compute_site_display, display_tasks), 1):
            pocket_sel_by_type_by_site[sid] = sel_by_type or {}
            ligand_sel_by_site[sid] = lig_sel or ligand_selection(None)
            if i % 100 == 0 or i == len(display_tasks):
                print(f"  [{i}/{len(display_tasks)}] ({time.time()-t0:.0f}s elapsed)")

    # Pass 4: assemble each cluster's payload from the precomputed results.
    payloads_by_label = {}
    for cluster_id, sample in samples_by_cluster.items():
        anchor, aligned = anchor_by_cluster[cluster_id], aligned_by_cluster[cluster_id]
        label = labels.get(cluster_id, f"cluster_{cluster_id}")
        payloads_by_label[label] = build_cluster_payload(
            cluster_id, label, anchor, aligned, meta_by_site, pocket_sel_by_type_by_site, ligand_sel_by_site)

    transforms_path = os.path.join(args.out_dir, "cluster_overlay_transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(all_transforms, f, indent=2)
    print(f"\nWrote {transforms_path}")

    write_combined_script(payloads_by_label, args.pockets_dir, out_path)
    print(f"Wrote {out_path} ({len(payloads_by_label)} clusters)")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
