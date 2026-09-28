"""
Step 4 of pocket_types: geometric sub-clustering WITHIN one composition
cluster (meant for the dominant Arg/Lys salt-bridge cluster) using proper
local-site structural comparison, not the composition-vector approach.

Why a new method, not a sharper composition vector: cluster_pocket_types.py's
composition bag counts (resname, interaction_type, provenance) occurrences
per site -- it is deliberately ORDER/GEOMETRY-BLIND (see that module's own
docstring), so it cannot distinguish "these residues sit in a triangle
around the phosphate" from "these residues sit in a line". That's exactly
the kind of sub-structure being looked for within the dominant cluster (does
Arg-only vs Lys-only vs both, or backbone-contact geometry, actually
correspond to different 3D arrangements). No off-the-shelf package solves
this exact problem -- the academic tools that do (SPASM/PINTS/Query3D) are
old, non-pip-installable C/Fortran/Java tools; `rmsd`/`spyrmsd` on PyPI are
close but built for equal-atom-count same-molecule conformer comparison, not
unequal-size cross-structure local environments. Built instead on established
primitives, not a rolled-by-hand substitute:
  - networkx.find_cliques (Bron-Kerbosch, 1973): solving which point
    corresponds to which (type-constrained, and now via maximum clique over
    a distance-consistency graph -- see REVISION 2 below -- not brute-force
    permutation, which is what an earlier version of this script used).
  - scipy.spatial.transform.Rotation.align_vectors: the actual Kabsch fit
    (exact least-squares rotation for a GIVEN correspondence, closed-form,
    no initial-pose/iteration needed -- confirmed against its own docs:
    "Estimate a rotation to optimally align two sets of vectors ... solved
    with Kabsch algorithm").

Method:
1. Per site, build a small TYPED POINT SET:
   - One guaranteed "PHOSPHATE_P" point: the reference ligand's own
     phosphate P atom (every reference site has one, by construction --
     reused via phosphate_ifp.get_ligand_phosphate_groups(), same helper
     phosphate_ifp.py/build_pocket_features.py already use). If a ligand has
     more than one phosphate group (ATP/GTP), the group whose P atom sits
     closest to the interacting-residue centroid is picked -- a principled
     stand-in for "the group actually being recognized here" without
     needing to recompute which specific p_idx build_pocket_features.py's
     cached ifp came from.
   - One point per individual HEAVY contact atom (not one centroid per
     residue -- see revision note below), typed by (resname, atom_name) so
     e.g. an Arg's NH1 can only correspond to another site's Arg NH1, never
     its NH2 or a Lys NZ. Hydrogens excluded: their positions are added
     computationally by PDB2PQR, not experimentally observed, and less
     reproducible across structures than heavy-atom positions.

   REVISION: the first version of this used one CENTROID point per residue
   (mean position of all its qualifying contact atoms). That was wrong for
   this pipeline's purposes, confirmed directly, not theoretically: build_
   cluster_overlays.py's overlay display was tightened to show the actual
   specific contact atoms (e.g. just NH1/NH2 for an Arg salt bridge), and
   the resulting overlays looked visually poor despite passing this script's
   own RMSD<=1.5A threshold -- because a single centroid only constrains a
   residue's POSITION, not its ORIENTATION (a side chain has rotational
   freedom the centroid discards), the clustering was optimizing a fit that
   the display was never actually verifying. Per-atom points fix this by
   aligning on the exact same atoms that get displayed.

2. REVISION 2: correspondence is now found via MAXIMUM CLIQUE DETECTION
   (networkx.find_cliques, i.e. Bron-Kerbosch, 1973), not brute-force
   permutation enumeration -- switched after a real methods paper doing the
   same kind of local phosphate-environment comparison (491 PDB structures,
   atoms within 7A of the nearest phosphorus, distance-matrix representation,
   maximum clique for correspondence, Bron-Kerbosch by name) was cited as
   inspiration. The old permutation approach FORCED every same-type point
   into the correspondence whenever counts allowed it -- if 4 of a site's 5
   Arg residues were a good geometric fit and 1 was not, the bad one still
   got dragged into the RMSD, dragging the score down (and, downstream,
   still looking like a poor overlay despite "passing" the threshold). A
   maximum clique naturally excludes whichever points aren't mutually
   consistent with the rest, instead of an all-or-nothing per-type decision.

   Concretely: build a graph H whose NODES are candidate atom-atom pairings
   (i, j) -- points_a[i] and points_b[j] sharing a type AND both consistent
   with the mandatory phosphate-to-phosphate anchor pair (|d(P_a,a_i) -
   d(P_b,b_j)| <= DISTANCE_CONSISTENCY_THRESHOLD, the paper's condition (i)).
   Two candidate nodes get an EDGE in H if they're simultaneously usable:
   different indices on both sides (an injective mapping -- point i can't
   serve double duty as two different points' partner), and their own
   pairwise distance is internally consistent (|d(a_i,a_i') - d(b_j,b_j')|
   <= DISTANCE_CONSISTENCY_THRESHOLD). The MAXIMUM CLIQUE in H, plus the
   forced phosphate pair, is the correspondence. This is rotation/
   translation-invariant by construction -- no initial-pose guess needed,
   same as the old Kabsch-per-candidate approach, but the compatibility
   graph is built from pure pairwise DISTANCES, not coordinates, so no
   superposition happens until AFTER the correspondence is already decided.
   Once decided, Kabsch (scipy.spatial.transform.Rotation.align_vectors)
   gives the final rotation and RMSD for that one, already-fixed, mapping.

3. Pairs matching fewer than MIN_MATCHED_POINTS (a rigid rotation is only
   well-defined with >=3 non-collinear points) are NOT comparable -- and
   this turned out to be the empirically dominant case: on a 40-site test
   slice of the real dominant cluster, only 108/780 pairs (14%) shared
   >=3 residue types at all. Confirmed directly that this is NOT a small
   detail: a first version of this script scored every "not comparable"
   pair with a fixed large filler distance and fed the resulting mostly-
   identical dense matrix into scipy.cluster.hierarchy (average linkage) --
   with 84% of entries tied at the same filler value, fcluster's low-k
   search degenerated into one giant blob (everything merges together
   below the tie block) with no usable cut between k=2 and k=~14. "Not
   comparable" is a fundamentally different claim from "confirmed
   dissimilar", and conflating them broke the clustering.

4. Fixed by NOT building a dense matrix at all: only genuinely comparable,
   geometrically-close pairs become edges in a sparse similarity graph
   (networkx) over all sites in the cluster. REVISION 2 also tightened the
   final acceptance bar: SIMILARITY_THRESHOLD is now 1.0A (was 1.5A),
   matching the paper's own two-tier scheme -- DISTANCE_CONSISTENCY_THRESHOLD
   (1.5A) is the LOOSER bar used to build clique candidates, SIMILARITY_
   THRESHOLD (1.0A) is the STRICTER bar the resulting RMSD must clear to
   actually become a graph edge, mirroring the paper's condition (i) vs (iii).
   Plain connected components was tried first and rejected too -- verified
   directly on the same 40-site slice that it chains transitively (A~B~C
   merges A and C into one group even with no direct A-C evidence),
   swallowing up to 26/40 sites into one component even at a strict 0.5 A
   cutoff. Louvain community detection (networkx.algorithms.community.
   louvain_communities, modularity-based, not transitive-closure-based) on
   the identical graph instead gave a handful of real communities (sizes
   like 9/8/5/3-4) plus singletons -- the qualitatively sensible result the
   dense-matrix and connected-components approaches both failed to produce.
   Sites with no edge above the threshold end up as their own singleton
   community, which is the correct outcome for a site with no good
   geometric match in this cluster, not an error.

Scoped to ONE existing cluster's site_ids (--cluster-assignments +
--cluster-id), not the whole manifest -- this is O(n^2) pairwise geometry,
appropriate for a specific sub-population (e.g. the ~550-600-site dominant
cluster), not meant to replace the full-dataset composition clustering.

Usage: python geometric_subcluster.py --cluster-assignments path --cluster-id N
                                       [--manifest path] [--fp-pickle path]
                                       [--similarity-threshold A] [--out-dir path]
"""
import argparse
import csv
import os
import sys
import time
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from scipy.spatial.transform import Rotation

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
    _load_ifp_by_site, _manifest_reference_sites, _provenance, BACKBONE_ATOM_NAMES, DEFAULT_EXCLUDED_INTERACTIONS,
)

PHOSPHATE_TYPE = "PHOSPHATE_P"
MIN_MATCHED_POINTS = 3  # a rigid rotation needs >=3 non-collinear points to be well-defined
DISTANCE_CONSISTENCY_THRESHOLD = 1.5  # Angstrom -- looser bar for building clique candidates (paper's condition i)
SIMILARITY_THRESHOLD = 1.0  # Angstrom RMSD -- stricter final edge-acceptance bar (paper's condition iii)
# Angstrom -- deliberately more generous than DISTANCE_CONSISTENCY_THRESHOLD:
# this gates the nearest-same-type correspondence step of the iterative
# refinement below, run against EVERY point either site has, not just
# candidates feeding the initial clique search -- it doesn't need (and
# shouldn't use) as tight a bar as building that clique does.
COVERAGE_DISTANCE_THRESHOLD = 3.0
# The phosphate anchor gets this weight instead of the usual 1.0/(atoms in
# its own "residue") every other point gets -- ~2x a typical single-atom
# partner's weight, so it stays the dominant piece of evidence (it's the
# entire reason for the comparison) without being an unconditional lock.
# Previously the phosphate pair was forced into the fit with NO consistency
# check at all and used as the sole reference every other candidate's
# eligibility was filtered against -- a de facto rigid anchor even though
# the Kabsch fit itself never mathematically required exact coincidence.
# Weighting it instead of hard-coding it means the fit can trade a little
# phosphate deviation for a better overall fit when that's genuinely
# warranted, while still keeping it the single most influential point.
PHOSPHATE_WEIGHT = 2.0
# Hard cap on the correspondence<->refit loop below, purely as a safety net
# against oscillation -- convergence (the correspondence set stops changing
# between rounds) is expected to be reached well before this in practice.
MAX_REFINEMENT_ITERATIONS = 10


def _ligand_phosphate_point(lig_mol, residue_points):
    """Picks whichever phosphate group's P atom sits closest to the
    interacting-residue centroid (a stand-in for "the group actually being
    recognized here" -- see module docstring) and returns its 3D position,
    or None if the ligand has no phosphorus at all."""
    groups = pif.get_ligand_phosphate_groups(lig_mol)
    if not groups:
        return None
    conf = lig_mol.GetConformer()
    if residue_points:
        target = np.mean([p for _, p, _ in residue_points], axis=0)
    else:
        target = None
    best_pos, best_dist = None, None
    for g in groups:
        p_idx = g["atom_idxs"][0]
        pos = conf.GetAtomPosition(p_idx)
        pos = np.array([pos.x, pos.y, pos.z])
        dist = 0.0 if target is None else np.linalg.norm(pos - target)
        if best_dist is None or dist < best_dist:
            best_pos, best_dist = pos, dist
    return best_pos


def site_typed_points(site_id, manifest_row, ifp, excluded_interactions, pockets_dir):
    """Returns [(type_label, np.array([x,y,z]), weight), ...] for one site, or
    None if the pocket/ligand/protein can't be loaded. type_label is
    PHOSPHATE_TYPE, or (residue_label, atom_name, interaction_types) where
    residue_label is "BACKBONE" for a protein backbone atom (residue-agnostic
    -- a Ser backbone NH and a Lys backbone NH making the same contact are the
    same evidence regardless of which residue is there) or the resname for a
    genuine side-chain/metal contact atom, and interaction_types is a sorted
    tuple of every qualifying ProLIF interaction type (HBDonor, Cationic,
    MetalAcceptor, ...) that atom satisfies -- an atom can legitimately
    satisfy more than one (e.g. an Arg NH1 flagged as both Cationic and
    HBDonor), and the full set is what's correspondence-relevant, not just
    one of them. Without this, an Arg NH1 making a genuine salt bridge and an
    Arg NH1 merely close enough for a weak H-bond were typed identically and
    treated as equally good candidates, discarding real chemical-specificity
    information ProLIF's own fingerprint already computed. interaction_types
    is empty for a residue's non-interacting atoms (see below) -- one point
    per individual heavy atom, not one centroid per residue (see module
    docstring's REVISION note for why the centroid version was wrong).

    Every heavy atom of an interacting residue is included, not just the 1-3
    tip atoms ProLIF flagged as directly making the interaction: confirmed
    directly (rotated/flipped overlays in practice) that tip-atoms-only left
    matched correspondences too sparse to reliably constrain a rotation --
    exactly 2 matched points leaves one rotational axis completely free, and
    even 3 points are often crowded/near-collinear (interacting atoms all
    sit near the same small ligand region), which is numerically unstable
    even though it nominally clears MIN_MATCHED_POINTS. The residue's other
    atoms get interaction_types=() so they still only match another site's
    equally-non-interacting atoms of the same name -- contributing real
    geometric spread without being conflated with genuine chemical evidence.

    weight is 1/(number of atoms this point's OWN residue instance
    contributed), so every interacting partner -- whether it's a metal ion
    exposing exactly one atom or a residue exposing three or four -- sums to
    the same total weight of 1 once fed into a weighted Kabsch fit. Without
    this, best_clique_correspondence_rmsd's RMSD/rotation was implicitly an
    unweighted average over ATOMS, so a multi-atom residue (e.g. an Arg salt
    bridge contributing NH1+NH2+NE) pulled the fit 3-4x harder than an
    equally-real single-atom metal coordination -- confirmed directly against
    ProLIF_v2/run_prolif.py's own PROTEIN_SEL comment, which already treats
    metals as single-atom residues for this exact reason in a different
    module. The phosphate anchor gets PHOSPHATE_WEIGHT (deliberately more
    than a typical single-atom partner's weight of 1 -- see that constant's
    own comment for why: dominant but not an unconditional lock).
    Grouped by (chain, number, name) -- the same residue-instance key
    build_cluster_overlays.pocket_selection_by_type() already uses -- so two
    different residues that happen to share a resname (e.g. two ARGs) are
    never merged into the same weight group."""
    pocket_path = os.path.join(pockets_dir, f"{site_id}.pdb")
    if not ifp or not os.path.exists(pocket_path):
        return None
    u = mda.Universe(pocket_path)
    protein_ag = rp.protein_selection(u)
    ligand_ag = u.select_atoms(
        f"resid {manifest_row['lig_resnum']} and chainID {manifest_row['chain']} and not protein")
    if len(protein_ag) == 0 or len(ligand_ag) == 0:
        return None
    prot_mol = rp.safe_molecule_from_mda(protein_ag, use_segid=False)
    lig_mol = rp.safe_molecule_from_mda(ligand_ag, use_segid=False, fix_ionizable=True)

    conf = prot_mol.GetConformer()
    seen_idxs = set()
    atoms_by_residue = defaultdict(list)  # (chain, number, resname) -> [(type_label, coord), ...]

    # Every heavy atom of EACH interacting residue, not just the 1-3 tip
    # atoms ProLIF flagged as directly interacting -- built once per site
    # (not per residue) by scanning prot_mol's own PDB residue info, then
    # looked up below per residue_key. Confirmed directly (rotated/flipped
    # overlays): with only the tip atoms present, a matched correspondence
    # was often just the phosphate + 1 atom (one rotational axis completely
    # unconstrained) or 3 atoms crowded/near-collinear near the same small
    # ligand region (numerically unstable rotation even though nominally
    # "matched"). Including the whole residue gives the fit real 3D spread
    # to lock onto, without needing MORE distinct interacting residues.
    all_residue_idxs = defaultdict(list)
    for atom in prot_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (info.GetChainId().strip(), info.GetResidueNumber(), info.GetResidueName().strip())
        all_residue_idxs[key].append(atom.GetIdx())

    for (lig_id, prot_id), interactions in ifp.items():
        residue_key = (prot_id.chain, prot_id.number, prot_id.name)
        # Map each contact atom to the SET of qualifying interaction types it
        # satisfies, built BEFORE flattening to individual md's -- the old
        # code flattened straight to a list of md's here, discarding which
        # interaction_name each one came from before an atom's type label
        # was ever built.
        idx_to_types = defaultdict(set)
        for interaction_name, mds in interactions.items():
            if interaction_name in excluded_interactions:
                continue
            for md in mds:
                for idx in md["parent_indices"]["protein"]:
                    idx_to_types[idx].add(interaction_name)
        if not idx_to_types:
            continue

        # Fall back to just the interacting atoms if the residue lookup
        # somehow misses (e.g. a key-format mismatch) -- old behavior, not a
        # crash -- rather than silently dropping this residue's evidence.
        residue_idxs = all_residue_idxs.get(residue_key) or list(idx_to_types)

        # Classify this residue's interaction ONCE, using ALL of its own
        # interacting atoms together -- the SAME call pocket_selection_by_
        # type() already makes for display, not a per-atom check. If the
        # interaction is backbone-only, restrict the expansion to just this
        # residue's backbone atoms: its side chain isn't part of what's
        # interacting, a different structure could have an entirely
        # different residue type sitting in that same backbone-recognition
        # spot, and including that side chain as a fit constraint would
        # distort the alignment on conformational noise unrelated to the
        # actual match. Residues whose interaction genuinely involves the
        # side chain (sidechain/mixed/metal provenance) still expand to the
        # whole residue, unchanged.
        residue_provenance = _provenance(prot_mol, list(idx_to_types.keys()), prot_id.name)
        if residue_provenance == "backbone":
            residue_idxs = [
                idx for idx in residue_idxs
                if prot_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().strip() in BACKBONE_ATOM_NAMES
            ]

        for idx in residue_idxs:
            if idx in seen_idxs:
                continue  # same atom can contribute to >1 qualifying interaction
            seen_idxs.add(idx)
            atom = prot_mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "H":
                continue
            pos = conf.GetAtomPosition(idx)
            name = atom.GetPDBResidueInfo().GetName().strip()
            # Backbone-mediated recognition doesn't depend on which residue
            # is there -- a Ser backbone NH and a Lys backbone NH making the
            # same H-bond at the structurally equivalent position are the
            # same evidence, but typing by (resname, atom_name) would block
            # them from ever matching across sites purely on residue-name
            # mismatch. Reuse _provenance() (not a hand-rolled backbone-name
            # check) specifically because it already checks resname in
            # METALS before checking atom names -- a bare Ca2+ ion's PDB
            # atom name is literally "CA", identical to the alpha-carbon
            # backbone name, so a naive check would mislabel every calcium
            # coordination as a generic backbone atom.
            provenance = _provenance(prot_mol, [idx], prot_id.name)
            residue_label = "BACKBONE" if provenance == "backbone" else prot_id.name
            # Atoms ProLIF actually flagged keep their specific interaction
            # type(s) (unchanged from before -- still typed for chemical
            # specificity); every OTHER atom of this same residue gets an
            # empty interaction_types tuple instead, so it can only match
            # another residue's equally-non-interacting atoms of the same
            # name -- never conflated with atoms that are real chemical
            # evidence, just used to constrain orientation.
            interaction_types = idx_to_types.get(idx, set())
            type_label = (residue_label, name, tuple(sorted(interaction_types)))
            atoms_by_residue[residue_key].append((type_label, np.array([pos.x, pos.y, pos.z])))

    if not atoms_by_residue:
        return None

    atom_points = []  # [((resname, atom_name), coord, weight), ...]
    for pts in atoms_by_residue.values():
        w = 1.0 / len(pts)
        for type_label, coord in pts:
            atom_points.append((type_label, coord, w))

    phosphate_pos = _ligand_phosphate_point(lig_mol, atom_points)
    if phosphate_pos is None:
        return None

    return [(PHOSPHATE_TYPE, phosphate_pos, PHOSPHATE_WEIGHT)] + atom_points


def _nearest_same_type_pairs(points_a, points_b, u, t, threshold=COVERAGE_DISTANCE_THRESHOLD):
    """The correspondence half of the iterative refinement loop (TM-align's
    own alternation, adapted to be permutation-invariant -- see
    best_clique_correspondence_rmsd's docstring for why TM-align's actual
    dynamic-programming step doesn't apply to sequence-scattered pocket
    residues). For EVERY point in points_a, transforms it with (u, t) and
    finds the nearest point in points_b with the EXACT SAME type_label --
    tight matching (same residue_label, atom_name, AND interaction_types),
    not loosened, so correspondence never conflates a real chemical-evidence
    atom with a merely-structural one. If that nearest same-typed point is
    within `threshold` (deliberately more generous than
    DISTANCE_CONSISTENCY_THRESHOLD -- this runs against a whole point set
    under an ALREADY-FOUND transform, it isn't filtering candidates for a
    combinatorial search), it's kept as a correspondence pair.

    Returns [(i, j), ...] index pairs into points_a/points_b. NOT necessarily
    injective (two different i's can both find the same j as their nearest
    neighbor) -- the standard nearest-neighbor correspondence step used by
    ICP-style iterative refinement; refitting on this pulls in whatever the
    CURRENT transform actually explains, and injectivity tends to resolve
    itself as the fit improves across iterations."""
    u_matrix = np.array(u)
    t_vector = np.array(t)
    coords_b = np.array([c for _, c, _ in points_b])
    types_b = [tb for tb, _, _ in points_b]
    by_type = defaultdict(list)
    for j, tb in enumerate(types_b):
        by_type[tb].append(j)

    pairs = []
    for i, (type_label, coord, _) in enumerate(points_a):
        candidates = by_type.get(type_label)
        if not candidates:
            continue
        transformed = u_matrix @ coord + t_vector
        dists = [float(np.linalg.norm(transformed - coords_b[j])) for j in candidates]
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= threshold:
            pairs.append((i, candidates[best_idx]))
    return pairs


def _weighted_kabsch(points_a, points_b, pairs):
    """The refit half of the iterative refinement loop: a weighted Kabsch fit
    over an explicit list of (i, j) index pairs into points_a/points_b. Both
    the rotation (via scipy's weights= argument) and the reported RMSD use
    each pair's own weight (average of each side's residue-instance weight --
    see site_typed_points()'s docstring for why), so the number reported is
    actually what the returned rotation minimizes.

    Returns (rmsd, u, t) in this project's standard convention: x_b_frame =
    u @ x_a + t (same as common.py's apply_transform / every cmd.
    transform_selection() call elsewhere in pocket_types). Returns
    (0.0, None, None) if fewer than 2 pairs (no rotation is well-defined)."""
    if len(pairs) < 2:
        return 0.0, None, None
    coords_a = np.array([c for _, c, _ in points_a])
    coords_b = np.array([c for _, c, _ in points_b])
    weights_a = np.array([w for _, _, w in points_a])
    weights_b = np.array([w for _, _, w in points_b])

    A = coords_a[[p[0] for p in pairs]]
    B = coords_b[[p[1] for p in pairs]]
    # Per-pair weight is the average of each side's own residue-instance
    # weight, so a pair only gets full weight when BOTH sides agree it's a
    # single-atom partner; if one side's residue happens to expose more
    # atoms than the other for the same interaction, the pair is weighted
    # down accordingly rather than picking one side arbitrarily.
    pair_weights = np.array([(weights_a[i] + weights_b[j]) / 2.0 for i, j in pairs])
    w_sum = pair_weights.sum()
    A_mean = (pair_weights[:, None] * A).sum(axis=0) / w_sum
    B_mean = (pair_weights[:, None] * B).sum(axis=0) / w_sum
    A_c, B_c = A - A_mean, B - B_mean
    rotation, _ = Rotation.align_vectors(B_c, A_c, weights=pair_weights)
    sq_err = np.sum((rotation.apply(A_c) - B_c) ** 2, axis=1)
    rmsd = float(np.sqrt(np.sum(pair_weights * sq_err) / w_sum))
    u_matrix = rotation.as_matrix()
    t_vector = B_mean - u_matrix @ A_mean  # x_b = u @ x_a + t, so t = B_mean - u @ A_mean
    return rmsd, u_matrix.tolist(), t_vector.tolist()


def best_clique_correspondence_rmsd(points_a, points_b, distance_consistency_threshold=DISTANCE_CONSISTENCY_THRESHOLD):
    """Finds the best atom-atom correspondence between two typed point sets
    and iteratively refines it -- TM-align's own alternation between
    "given a superposition, find the best correspondence" and "given a
    correspondence, refit the superposition," adapted to be permutation-
    invariant: TM-align's actual correspondence step is dynamic programming
    over a SEQUENTIAL residue order, which assumes a monotonic, non-crossing
    alignment path -- exactly right for two whole protein chains, but wrong
    here, since these "sites" are scattered, non-contiguous residues that can
    come from completely unrelated folds with no shared sequence order at
    all. So the correspondence step here is the same maximum-clique matching
    this module always used (permutation-invariant by construction, mirrors
    the cited paper's method -- a distance-matrix compatibility graph +
    Bron-Kerbosch maximum clique), used ONLY to seed a chemically-sane
    initial transform; from there, _nearest_same_type_pairs()/
    _weighted_kabsch() alternate (TM-align's actual loop) until the
    correspondence stops changing or MAX_REFINEMENT_ITERATIONS is hit.

    points_a[0]/points_b[0] MUST be the PHOSPHATE_P anchor (site_typed_
    points()'s own convention) -- it's forced into the SEED correspondence
    unconditionally (every candidate node below is pre-filtered for distance-
    consistency with it), but afterward it's just another weighted point like
    any other in the refit loop -- see PHOSPHATE_WEIGHT's own comment for why
    it's no longer an unconditional lock.

    Returns (rmsd, n_matched, u, t) from the FINAL converged iteration, not
    the seed clique's own numbers -- the clique only proves a handful of
    points are mutually consistent; with as few as 2 points, one whole
    rotational axis is completely unconstrained, so a clique's own RMSD can
    be deceptively perfect (confirmed directly: 0.07A over 2 points) while
    saying nothing about whether the resulting transform explains anything
    else, including the rest of the very residues those 2 points belong to.
    n_matched is therefore the CONVERGED correspondence size, not a clique
    size -- it can be much larger than the seed clique for a genuinely good
    transform (lots of additional agreement gets revealed across iterations),
    and stays near the seed's own size for a spurious one (nothing else lines
    up no matter how many rounds run). u/t are None if the seed clique has
    <2 points (no rotation possible); otherwise u is a 3x3 rotation matrix
    (list of lists) and t a 3-vector (list), in this project's standard
    convention: x_b_frame = u @ x_a + t (same as common.py's apply_transform
    / every cmd.transform_selection() call elsewhere in pocket_types) -- so
    callers can hand (u, t) straight to a PyMOL script without reformatting.
    Caller gates on MIN_MATCHED_POINTS."""
    coords_a = np.array([c for _, c, _ in points_a])
    coords_b = np.array([c for _, c, _ in points_b])
    types_a = [t for t, _, _ in points_a]
    types_b = [t for t, _, _ in points_b]
    weights_a = np.array([w for _, _, w in points_a])
    weights_b = np.array([w for _, _, w in points_b])

    d_pa = np.linalg.norm(coords_a - coords_a[0], axis=1)
    d_pb = np.linalg.norm(coords_b - coords_b[0], axis=1)

    # Candidate nodes: same type, AND distance-to-phosphate-anchor consistent
    # (paper's condition (i), applied to the forced anchor pair first).
    nodes = [(i, j) for i in range(1, len(points_a)) for j in range(1, len(points_b))
             if types_a[i] == types_b[j] and abs(d_pa[i] - d_pb[j]) <= distance_consistency_threshold]

    H = nx.Graph()
    H.add_nodes_from(nodes)
    for idx1 in range(len(nodes)):
        i1, j1 = nodes[idx1]
        for idx2 in range(idx1 + 1, len(nodes)):
            i2, j2 = nodes[idx2]
            if i1 == i2 or j1 == j2:
                continue  # can't reuse the same point on either side (injective mapping)
            d_a = np.linalg.norm(coords_a[i1] - coords_a[i2])
            d_b = np.linalg.norm(coords_b[j1] - coords_b[j2])
            if abs(d_a - d_b) <= distance_consistency_threshold:
                H.add_edge(nodes[idx1], nodes[idx2])

    # Rank candidate cliques by SUMMED WEIGHT, not node count: a 3-atom Arg
    # match (weight 1/3 each = 1.0 total) shouldn't beat a 2-partner match
    # like {metal, Ser} (weight 1.0 + 1.0 = 2.0 total) just because it has
    # more nodes -- the atom-count bias this whole change is fixing would
    # otherwise still decide which correspondence gets FOUND, even after
    # fixing how a found correspondence gets FIT. No extra enumeration cost:
    # nx.find_cliques(H) is already walked in full to find the max either way.
    node_weight = {(i, j): (weights_a[i] + weights_b[j]) / 2.0 for i, j in nodes}
    best_clique = max(nx.find_cliques(H), key=lambda c: sum(node_weight[n] for n in c), default=[]) if nodes else []
    seed_pairs = [(0, 0)] + list(best_clique)

    rmsd, u, t = _weighted_kabsch(points_a, points_b, seed_pairs)
    if u is None:
        return 0.0, 1, None, None  # phosphate-only; caller's MIN_MATCHED_POINTS gate excludes this anyway

    # TM-align's actual loop: alternate correspondence <-> refit until the
    # correspondence stops changing. Seeding with the clique's transform
    # (rather than starting from nothing, or from an arbitrary initial pose)
    # is what keeps this from falling into ICP's classic failure mode of
    # converging on a geometrically-smooth but chemically-meaningless fit --
    # every candidate correspondence at every round is still type-matched.
    current_pairs = set(seed_pairs)
    for _ in range(MAX_REFINEMENT_ITERATIONS):
        new_pairs = _nearest_same_type_pairs(points_a, points_b, u, t)
        new_pairs_set = set(new_pairs)
        if not new_pairs_set or new_pairs_set == current_pairs:
            break  # nothing left to match, or converged
        new_rmsd, new_u, new_t = _weighted_kabsch(points_a, points_b, new_pairs)
        if new_u is None:
            break
        rmsd, u, t, current_pairs = new_rmsd, new_u, new_t, new_pairs_set

    return rmsd, len(current_pairs), u, t


def compute_comparable_pairs(site_ids, points_by_site):
    """Returns [(site_a, site_b, rmsd, n_matched), ...] for every pair with
    >=MIN_MATCHED_POINTS in common -- NOT a dense matrix (see module
    docstring for why a dense matrix with a filler value for incomparable
    pairs was tried and rejected)."""
    n = len(site_ids)
    pairs = []
    t0 = time.time()
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            rmsd, n_matched, _, _ = best_clique_correspondence_rmsd(
                points_by_site[site_ids[i]], points_by_site[site_ids[j]])
            if n_matched >= MIN_MATCHED_POINTS:
                pairs.append((site_ids[i], site_ids[j], rmsd, n_matched))
            done += 1
        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.time() - t0
            frac = done / total
            eta = elapsed / frac - elapsed if frac else 0
            print(f"  pairwise RMSD: {done}/{total} pairs ({frac:.1%}), "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")
    print(f"  {len(pairs)}/{total} pairs comparable (>={MIN_MATCHED_POINTS} matched points)")
    return pairs


def build_similarity_graph(site_ids, comparable_pairs, threshold):
    G = nx.Graph()
    G.add_nodes_from(site_ids)
    for a, b, rmsd, n_matched in comparable_pairs:
        if rmsd <= threshold:
            G.add_edge(a, b, weight=1.0 / (0.1 + rmsd), rmsd=rmsd, n_matched=n_matched)
    return G


def community_labels(G, site_ids, seed=42):
    """Louvain communities -- see module docstring for why this, not
    connected components (transitive chaining) or dense-matrix hierarchical
    clustering (broken by the high incomparable-pair rate)."""
    comms = louvain_communities(G, weight="weight", seed=seed)
    label_by_site = {}
    for cid, comm in enumerate(comms):
        for sid in comm:
            label_by_site[sid] = cid
    return [label_by_site[sid] for sid in site_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-assignments", default=os.path.join(RESULTS_DIR, "pocket_cluster_assignments.csv"))
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--manifest", default=os.path.join(PROLIF_V2_ROOT, "full_manifest.csv"))
    parser.add_argument("--fp-pickle", default=os.path.join(PROLIF_V2_ROOT, "results",
                                                              "prolif_fingerprint_full_manifest.pkl"))
    parser.add_argument("--pockets-dir", default=os.path.join(PROLIF_V2_ROOT, "data", "pockets"))
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD,
                         help="RMSD (Angstrom) cutoff for a similarity-graph edge")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--tag", default=None, help="output filename suffix; default: cluster_<id>")
    args = parser.parse_args()
    tag = args.tag or f"cluster_{args.cluster_id}"

    with open(args.cluster_assignments, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["cluster_id"]) == args.cluster_id]
    print(f"{len(rows)} sites in cluster {args.cluster_id} of {args.cluster_assignments}")

    manifest_rows = _manifest_reference_sites(args.manifest)
    print(f"Loading cached ProLIF fingerprint from {args.fp_pickle} ...")
    ifp_by_site = _load_ifp_by_site(args.fp_pickle)

    points_by_site = {}
    meta_by_site = {}
    n_fail = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        sid = row["site_id"]
        meta_by_site[sid] = row
        pts = site_typed_points(sid, manifest_rows[sid], ifp_by_site.get(sid),
                                 DEFAULT_EXCLUDED_INTERACTIONS, args.pockets_dir)
        if pts is None:
            n_fail += 1
            continue
        points_by_site[sid] = pts
        if i % 50 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {len(points_by_site)} ok, {n_fail} failed "
                  f"({time.time()-t0:.0f}s elapsed)")

    site_ids = list(points_by_site.keys())
    print(f"{len(site_ids)}/{len(rows)} sites produced a typed point set "
          f"({n_fail} failed to load/had no phosphate group)")

    print(f"Computing pairwise RMSDs ({len(site_ids)*(len(site_ids)-1)//2} candidate pairs) ...")
    comparable_pairs = compute_comparable_pairs(site_ids, points_by_site)

    G = build_similarity_graph(site_ids, comparable_pairs, args.similarity_threshold)
    n_isolated = sum(1 for sid in site_ids if G.degree(sid) == 0)
    print(f"Similarity graph (threshold={args.similarity_threshold} A): "
          f"{G.number_of_edges()} edges, {n_isolated}/{len(site_ids)} sites with no match at all")

    labels = community_labels(G, site_ids)
    n_communities = len(set(labels))
    sizes = sorted((list(labels).count(c) for c in set(labels)), reverse=True)
    print(f"\nFinal: {n_communities} Louvain communities, sizes (top 10) = {sizes[:10]}")

    assignments_path = os.path.join(args.out_dir, f"geometric_subcluster_assignments_{tag}.csv")
    with open(assignments_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "ref_ligand", "cluster_id", "total_hits"])
        w.writeheader()
        for sid, label in zip(site_ids, labels):
            m = meta_by_site[sid]
            w.writerow({"site_id": sid, "pdb_id": m["pdb_id"], "ref_ligand": m["ref_ligand"],
                        "cluster_id": int(label), "total_hits": m["total_hits"]})
    print(f"Wrote {assignments_path}")

    summary_path = os.path.join(args.out_dir, f"geometric_subcluster_summary_{tag}.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Geometric sub-clustering of cluster {args.cluster_id} "
                f"({len(site_ids)} sites, {n_communities} Louvain communities, "
                f"threshold={args.similarity_threshold} A)\n")
        f.write("Kabsch RMSD on type-constrained-correspondence local-environment point sets; "
                "edge only if RMSD <= threshold; Louvain community detection (not hierarchical -- "
                "see module docstring for why)\n")
        f.write("=" * 80 + "\n\n")
        for cid in sorted(set(labels), key=lambda c: -list(labels).count(c)):
            members = [sid for sid, lab in zip(site_ids, labels) if lab == cid]
            f.write(f"Community {cid}: {len(members)} sites\n")
            for sid in sorted(members, key=lambda s: -int(meta_by_site[s]["total_hits"]))[:8]:
                m = meta_by_site[sid]
                f.write(f"    {sid:28s} {m['pdb_id']:5s} {m['ref_ligand']:5s} ({m['total_hits']} hits)\n")
            f.write("\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
