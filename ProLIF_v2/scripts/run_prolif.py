"""
Step 6: runs ProLIF's interaction fingerprint on each of the 20 prepared
pockets (protein via PDB2PQR, ligand via RDKit template matching, merged and
sliced in extract_pockets.py). Output is ProLIF's native bitvector
representation (fp.to_dataframe()/to_bitvectors()) rather than a flattened
per-interaction table, plus Tanimoto similarity for each ref/hit pair named
in the manifest.

safe_molecule_from_mda/_extract_submol below are copied (not imported) from
the earlier scripts/plip_isostere/generate_prolif_example.py prototype --
that module does `from plip.structure.preparation import ...` at import time,
so importing it would pull in a PLIP dependency even to reuse these two
PLIP-independent helper functions. Copying avoids that entirely.

Reason these exist at all: prolif.Molecule.from_mda() segfaults natively in
RDKit 2025.09.5's GetMolFrags(asMols=True) (called via
prolif.utils.split_mol_by_residues) on some structures -- a native crash, not
a catchable exception -- so residues are split manually instead, bypassing
that code path.
"""
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import attrgetter

warnings.filterwarnings("ignore")


import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
# CLI override, e.g. `python run_prolif.py full_manifest.csv` -- same pattern
# batch_prepare_structures_parallel.py already uses. Defaults to the 50-pair
# sample so every existing call site (and this session's whole analysis/artifact
# trail, all built against the plain filenames below) keeps working unchanged.
MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
POCKETS_DIR = os.path.join(PROLIF_V2_ROOT, "data", "pockets")
RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Every output file below is a whole-run artifact (not per-site like pockets), so a
# full-manifest run must not silently overwrite the sample's results out from under
# the analysis already built on them -- suffix output filenames by manifest name,
# but only when it's NOT the default sample manifest, so sample runs keep writing
# the plain, already-referenced filenames exactly as before.
_manifest_tag = os.path.splitext(os.path.basename(MANIFEST_PATH))[0]
RUN_SUFFIX = "" if _manifest_tag == "sample_manifest" else f"_{_manifest_tag}"

# data/pockets/*.pdb is written in native (unaligned) frame for both ref and hit
# (extract_pockets.py no longer pre-superimposes -- see that module's docstring for
# why a site_id-keyed file can't hold a hit's alignment when it's paired with more
# than one reference). build_residue_correspondence below applies the
# (ref_id, hit_id)-specific transform itself, per pairing, on the hit's native
# coordinates -- so a hit shared across several references gets each one right.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "plip_isostere"))
import common  # noqa: E402
from common import get_transformation, apply_transform  # noqa: E402

_TMALIGN_RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"


def _default_tmalign_json():
    """Prefers the from-scratch clean regeneration over the plain (older,
    pre-single-assignment-pairing) file -- never the archived _corrected.json,
    see common.py's TMALIGN_JSON comment for why."""
    results_dir = os.path.join(PROJECT_ROOT, "results")
    clean = os.path.join(results_dir, f"tmalign_results_{_TMALIGN_RUN_ID}_clean.json")
    plain = os.path.join(results_dir, f"tmalign_results_{_TMALIGN_RUN_ID}.json")
    if os.path.exists(clean):
        return clean
    if os.path.exists(plain):
        return plain
    raise FileNotFoundError(f"No TM-align results found for run_id={_TMALIGN_RUN_ID} (checked {clean}, {plain})")

# Reused (not copied) from the main pipeline's own METALS set (src/constants.py) so
# this stays in sync with whatever the rest of the project considers a coordinating
# metal ion. MDAnalysis's "protein" selection keyword only matches standard amino
# acids -- a metal ion HETATM (e.g. the Mn2+ in 4AVL) is otherwise silently dropped
# before it ever reaches ProLIF's fingerprint generator, so ProLIF can never detect
# a MetalDonor/MetalAcceptor interaction with it, no matter how close it sits to the
# ligand. PROTEIN_SEL below is what actually goes into every u.select_atoms() call
# that used to just say "protein".
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from constants import METALS  # noqa: E402
from utils import fmt_eta  # noqa: E402

PROTEIN_SEL = "protein or resname " + " ".join(sorted(METALS))

# MDAnalysis's OWN internal bond-guesser (a third, separate vdW radii table from
# both ProLIF's VdWContact presets and this project's own) doesn't have these metals
# either -- and unlike ProLIF's VdWContact, which just fails to score that one atom
# pair, a missing radius here makes ag.guess_bonds() silently return ZERO bonds for
# the ENTIRE atom group it was asked to guess, not just the metal. Confirmed directly:
# every protein selection that included a metal ion came back as a fully-disconnected
# graph (0 bonds across 1000+ atoms), which is why every titratable residue's formal
# charge downstream was nonsense (RDKit's sanitizer assigns a charge equal to
# `0 - default_valence` to a disconnected atom, e.g. -3 on a lysine NZ that should be
# +1) -- despite VdWContact/MetalAcceptor still looking correct, since neither of
# those needs bond connectivity, only the charge-gated Cationic/Anionic detection
# does. Reuses the exact same radii ProLIF's own "rdkit" VdWContact preset already
# validated for these elements (see the Fingerprint construction in main()), rather
# than inventing a second set of numbers.
#
# W (tungsten) and friends below belong here too, not in METALS/IGNORE_LIGANDS
# (src/constants.py): WO4/WO5 tungstate shows up in this manifest as a HIT
# ligand -- a phosphate transition-state mimic in its own right (same role as
# vanadate), not a protein-coordinating ion to fold into PROTEIN_SEL. It hits
# this exact same missing-radius gap on the ligand side, though: guess_bonds()
# on a bare WO4^2-/WO5 ion (no pre-existing bonds in the raw HETATM record)
# needs a W radius to avoid the same all-or-nothing disconnected-graph failure
# described above.
#
# Auditing the full manifest (pipeline_failure_report.py) turned up seven more
# elements hitting the identical "vdw radii for types: X" crash, same root
# cause, just never enumerated here: AS (26 sites, mostly cacodylate/CAC -- a
# common crystallization buffer, not a phosphate mimic), V (11, VO3/VO4
# vanadate -- a phosphate transition-state mimic exactly like W), RH (11),
# RU (8), OS/MO/IR/TB (1 each, assorted metal-complex crystallization
# additives). All added uniformly rather than picking favorites -- whether a
# given ion is chemically *interesting* to this comparison is a question for
# analysis downstream, not a reason to leave its bonds unguessable upstream.
METAL_VDWRADII = {
    "MG": 2.2, "ZN": 2.1, "MN": 2.05, "CA": 2.4, "FE": 2.05, "CU": 2.0, "CO": 2.0, "NI": 2.0,
    "W": 2.1, "AS": 2.05, "V": 2.05, "RH": 2.0, "RU": 2.05, "OS": 2.0, "MO": 2.1, "IR": 2.0, "TB": 2.37,
}

INTERACTIONS = [
    "Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking",
    "Anionic", "Cationic", "CationPi", "PiCation",
    "XBDonor", "XBAcceptor", "MetalDonor", "MetalAcceptor", "VdWContact",
]

# max CA-CA distance (Angstrom), post-superposition, for two residues to be considered
# the "same" structural position. Typical for well-aligned pocket regions.
CA_MATCH_CUTOFF = 3.0

# Same cap-at-8 convention as scripts/analyze_plif.py's own ProcessPoolExecutor use
# (a structurally similar per-site CPU-bound task) and ProLIF_v2/scripts/
# batch_prepare_structures_parallel.py's N_WORKERS_DEFAULT -- ProcessPoolExecutor,
# not ThreadPoolExecutor (that script's choice), since per-site fingerprinting is
# CPU-bound RDKit/ProLIF geometry work in-process, not I/O/subprocess-bound; the
# GIL would prevent real parallelism across threads for this workload.
FINGERPRINT_WORKERS = min(8, os.cpu_count() or 4)


def _extract_submol(mol, idxs):
    """Atom-induced subgraph of `mol` on `idxs` (NOT Chem.PathToSubmol, which
    takes bond indices)."""
    conf = mol.GetConformer()
    frag = Chem.RWMol()
    old_to_new = {}
    for old_idx in idxs:
        old_to_new[old_idx] = frag.AddAtom(mol.GetAtomWithIdx(old_idx))
    idx_set = set(idxs)
    seen_bonds = set()
    for old_idx in idxs:
        for bond in mol.GetAtomWithIdx(old_idx).GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in idx_set and j in idx_set:
                key = (min(i, j), max(i, j))
                if key not in seen_bonds:
                    seen_bonds.add(key)
                    frag.AddBond(old_to_new[i], old_to_new[j], bond.GetBondType())
    new_conf = Chem.Conformer(len(idxs))
    for old_idx, new_idx in old_to_new.items():
        new_conf.SetAtomPosition(new_idx, conf.GetAtomPosition(old_idx))
    frag_mol = frag.GetMol()
    frag_mol.AddConformer(new_conf, assignId=True)
    try:
        Chem.SanitizeMol(frag_mol)
    except Chem.KekulizeException:
        ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY
        Chem.SanitizeMol(frag_mol, sanitizeOps=ops)
    except Chem.AtomValenceException:
        # Same root cause as safe_molecule_from_mda's inferrer fallback (a
        # distance-guessed bond occasionally over-valences an atom for specific
        # structures, e.g. 4AVL) -- surfacing here means it's a different atom
        # than the one caught there, since this sanitizes each residue submol in
        # isolation rather than the whole protein at once. Skipping just
        # SANITIZE_PROPERTIES (the valence check) keeps ring/conjugation/
        # aromaticity perception intact for the rest of the residue -- same
        # "degrade only the one check that's failing" approach as Kekulize above.
        ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
        Chem.SanitizeMol(frag_mol, sanitizeOps=ops)
    return frag_mol


def safe_molecule_from_mda(atomgroup, use_segid=False, **converter_kwargs):
    """Drop-in replacement for prolif.Molecule.from_mda() avoiding the
    GetMolFrags segfault: builds each residue as its own submol directly
    from manually grouped (resname, resid, chain, icode) atom indices."""
    try:
        has_bonds = len(atomgroup.bonds) > 0
    except NoDataError:
        has_bonds = False
    if not has_bonds:
        # Guess bonds ourselves, with metal radii supplied, rather than letting
        # convert_to.rdkit()'s own internal atomgroup.guess_bonds() call do it
        # blind -- see METAL_VDWRADII's comment for why that silently produces a
        # fully-disconnected molecule whenever the selection includes a metal ion.
        #
        # Checking len() == 0, not just catching NoDataError, matters here: once
        # any OTHER atomgroup from the same Universe (e.g. the ligand, always
        # converted before the protein in run_site()/phosphate_group_ifps()) has
        # already had guess_bonds() run on it, the Universe gains a `bonds`
        # TopologyAttr -- so `atomgroup.bonds` on the protein selection stops
        # raising NoDataError entirely, but comes back empty (0 bonds relevant to
        # protein atoms), silently skipping this branch and leaving the protein
        # fully disconnected. Confirmed directly: `protein.bonds` raised
        # NoDataError before the ligand was converted, then returned 0 (no
        # exception) right after -- the real pipeline always converts ligand
        # first, so the NoDataError-only check never actually fired in practice.
        atomgroup.guess_bonds(vdwradii=METAL_VDWRADII)
    per_residue_inference = False
    try:
        mol = atomgroup.convert_to.rdkit(**converter_kwargs)
    except AttributeError as e:
        if "No hydrogen atom could be found" not in str(e):
            raise
        # Monatomic ions (e.g. IOD, a bare I- ion) have no hydrogens and no
        # bonds to infer in the first place, so skipping bond-order inference
        # here is exact, not a guess -- unlike blanket-forcing every
        # conversion, which would silently hide a genuinely missing-H bug on
        # a real multi-atom ligand.
        mol = atomgroup.convert_to.rdkit(force=True, **converter_kwargs)
    except Chem.AtomValenceException:
        # MDAnalysis's own bond-order/formal-charge inferrer (the step that assigns
        # e.g. a protonated LYS's NZ +1, an ASP/GLU carboxylate -1 -- see
        # METAL_VDWRADII's comment for the fix that made those charges correct in
        # the first place) standardizes the WHOLE mol in one call, and occasionally
        # mis-standardizes a distance-guessed bond into an over-valent atom
        # somewhere in it. A confirmed, pre-existing PDB2PQR-hydrogen-placement/
        # bond-guessing edge case, unrelated to metals or phosphates -- but naively
        # falling back to inferrer=None for the whole mol here would silently drop
        # correct formal charges for every OTHER residue too, not just the one
        # actually at fault (confirmed on 4AVL: LYS134's own bonds are perfectly
        # normal ammonium geometry, yet it lost its +1 charge -- and with it,
        # ProLIF's Anionic/salt-bridge detection -- purely because TYR24's HB3
        # elsewhere in the same structure tripped the exception).
        #
        # Falls back to bonds-only here (inferrer=None, always succeeds -- it's
        # specifically charge/bond-order standardization that's fragile, not bond
        # guessing itself), then re-runs the same inferrer PER RESIDUE below
        # instead of on the whole mol at once. Confirmed this isolates the damage
        # correctly: applying it to just LYS134's 22-atom fragment still correctly
        # assigns NZ +1, on the very structure where the whole-mol version crashes
        # on a different, unrelated atom. Whole-mol inference is tried first (this
        # except branch, not always-per-residue) since it's cheaper and already
        # correct for the ~65% of structures that never hit this at all.
        mol = atomgroup.convert_to.rdkit(inferrer=None, **converter_kwargs)
        per_residue_inference = True
    for atom in mol.GetAtoms():
        atom.SetUnsignedProp("mapindex", atom.GetIdx())

    residue_atom_idxs = defaultdict(list)
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        key = (info.GetResidueName(), info.GetResidueNumber(), info.GetChainId(), info.GetInsertionCode())
        residue_atom_idxs[key].append(atom.GetIdx())

    import prolif as plf
    from prolif.residue import Residue
    residue_frags = [_extract_submol(mol, idxs) for idxs in residue_atom_idxs.values()]
    if per_residue_inference:
        from MDAnalysis.converters.RDKitInferring import MDAnalysisInferrer
        infer_one = MDAnalysisInferrer()
        inferred = []
        for frag in residue_frags:
            try:
                inferred.append(infer_one(frag))
            except Chem.AtomValenceException:
                # This one specific residue's local geometry still doesn't
                # standardize cleanly even in isolation (rare -- most residues
                # that trip the whole-mol version succeed once isolated, per
                # the LYS134 case above). Keep the bonds-only, uncharged
                # version rather than losing the whole structure over it.
                inferred.append(frag)
        residue_frags = inferred
    residues = [Residue(frag, use_segid=use_segid) for frag in residue_frags]
    residues.sort(key=attrgetter("resid"))
    return plf.Molecule(mol, use_segid=use_segid, residues=residues)


def _is_peptide_bonded(u, res, cutoff=1.45):
    """True if res's N or C atom sits within peptide-bond distance of a
    neighboring residue's C or N in the same chain. This is the discriminator
    between a genuine backbone residue (canonical or a modified/PTM one, e.g.
    KCX/ALY/TPO/SEP/PTR/CSO) and an unrelated ligand that merely happens to
    reuse atom names N/CA/C/O -- confirmed directly that name-alone is NOT a
    safe test: SAH, SAM, MTX, FOL (real cofactor/ligand molecules, not amino
    acids) all have atoms named N/CA/C/O too, and a name-only "has a backbone"
    check wrongly flagged all of them in a full-manifest scan. Actual polymer
    connectivity is what separates the two."""
    n_atoms = res.atoms.select_atoms("name N")
    c_atoms = res.atoms.select_atoms("name C")
    if len(n_atoms) == 0 or len(c_atoms) == 0:
        return False
    n_pos, c_pos = n_atoms[0].position, c_atoms[0].position
    chain = res.atoms.chainIDs[0]
    for other in u.residues:
        if other.resid == res.resid and other.resname == res.resname:
            continue
        if other.atoms.chainIDs[0] != chain:
            continue
        oc = other.atoms.select_atoms("name C")
        on = other.atoms.select_atoms("name N")
        if len(oc) and np.linalg.norm(n_pos - oc[0].position) < cutoff:
            return True
        if len(on) and np.linalg.norm(c_pos - on[0].position) < cutoff:
            return True
    return False


def protein_selection(u):
    """Returns the AtomGroup that should count as 'protein' for this pocket
    Universe: MDAnalysis's own `protein` keyword (standard amino acids) plus
    METALS plus any residue that's peptide-bonded to a neighbor in the same
    chain regardless of resname -- catches non-canonical/modified amino acids
    that `protein` doesn't recognize by name, which would otherwise be
    silently dropped from both fingerprinting (run_site) and residue
    correspondence (get_ca_positions), making that whole residue invisible to
    ProLIF even though it's structurally part of the polymer chain. See
    _is_peptide_bonded for why this needs an actual connectivity check rather
    than just testing for N/CA/C/O atom names."""
    protein = u.select_atoms(PROTEIN_SEL)
    protein_resnames = set(protein.residues.resnames)
    extra_resids = []
    for res in u.residues:
        name = res.resname
        if name in protein_resnames or name in METALS:
            continue
        atom_names = set(res.atoms.names)
        if {"N", "CA", "C", "O"} <= atom_names and _is_peptide_bonded(u, res):
            extra_resids.append((res.resid, res.atoms.chainIDs[0]))
    if not extra_resids:
        return protein
    extra_sel = " or ".join(f"(resid {rid} and chainID {ch})" for rid, ch in extra_resids)
    return protein | u.select_atoms(extra_sel)


def run_site(site, fp):
    pocket_path = os.path.join(POCKETS_DIR, f"{site['site_id']}.pdb")
    u = mda.Universe(pocket_path)
    # Not filtered by resname: legacy PDB format truncates 4-5 character extended CCD
    # codes to 3 characters on every write in this pipeline (see extract_pockets.py's
    # merge_ligand_into_protein docstring), so the manifest's full-length resname would
    # never match here for those ligands. resid+chainID alone already uniquely identifies
    # this site's ligand instance (that's how site_id itself is built).
    ligand = u.select_atoms(f"resid {site['lig_resnum']} and chainID {site['chain']} and not protein")
    protein = protein_selection(u)
    if len(ligand) == 0:
        return None, f"ligand selection matched 0 atoms in {pocket_path}"
    if len(protein) == 0:
        return None, f"protein selection matched 0 atoms in {pocket_path}"

    lig_mol = safe_molecule_from_mda(ligand, use_segid=False)
    prot_mol = safe_molecule_from_mda(protein, use_segid=False)
    # metadata=True (not False) because to_dataframe()/to_bitvectors() below expect the
    # sparse {interaction: [metadata, ...]} form, not a raw per-call numpy bitvector.
    ifp = fp.generate(lig_mol, prot_mol, residues=None, metadata=True)
    return ifp, None


def get_ca_positions(site, transform=None):
    """Maps each protein residue in a pocket PDB to its CA coordinate, keyed by the
    same ResidueId (name, number, chain) convention ProLIF uses in its ifp keys.

    `transform`, if given, is a (t, u) TM-align transform applied to each position
    before it's returned -- pocket PDBs are written in native (unaligned) frame
    (see extract_pockets.py), so a hit site needs its (ref_id, hit_id)-specific
    transform applied here to land in the reference's frame; the reference site
    itself is always called with transform=None (it's the fixed target)."""
    from prolif.residue import ResidueId

    pocket_path = os.path.join(POCKETS_DIR, f"{site['site_id']}.pdb")
    u = mda.Universe(pocket_path)
    positions = {}
    for atom in protein_selection(u).select_atoms("name CA"):
        resid = ResidueId(atom.resname, int(atom.resid), atom.chainID)
        pos = atom.position.astype(float)
        if transform is not None:
            t, u_mat = transform
            pos = np.array(apply_transform(t, u_mat, tuple(pos)))
        positions[resid] = pos
    return positions


def get_metal_positions(site, transform=None):
    """Maps each coordinating metal ion in a pocket PDB to its own atom position,
    keyed the same way get_ca_positions keys amino acids -- metals are single-atom
    residues (see PROTEIN_SEL), so there's no CA to anchor on; the ion's own
    position is the natural equivalent for correspondence-mapping purposes.

    `transform`, see get_ca_positions -- same convention."""
    from prolif.residue import ResidueId

    pocket_path = os.path.join(POCKETS_DIR, f"{site['site_id']}.pdb")
    u = mda.Universe(pocket_path)
    positions = {}
    for atom in u.select_atoms("resname " + " ".join(sorted(METALS))):
        resid = ResidueId(atom.resname, int(atom.resid), atom.chainID)
        pos = atom.position.astype(float)
        if transform is not None:
            t, u_mat = transform
            pos = np.array(apply_transform(t, u_mat, tuple(pos)))
        positions[resid] = pos
    return positions


def _nearest_within_cutoff(ref_positions, hit_positions, cutoff):
    """Maps each hit ResidueId to its nearest ref ResidueId (raw distance -- both
    sides are assumed already in a shared frame) within `cutoff`. Shared by the
    CA-based and metal-based correspondence searches in build_residue_correspondence,
    kept as two separate calls rather than one combined nearest-neighbor search over
    everything so a metal ion can never match to a nearby amino acid's CA (or vice
    versa) -- geometrically plausible (metals often sit within a few A of a
    coordinating residue) but chemically meaningless for interaction comparison."""
    if not ref_positions or not hit_positions:
        return {}
    ref_ids = list(ref_positions.keys())
    ref_xyz = np.array([ref_positions[r] for r in ref_ids])
    mapping = {}
    for hit_id, xyz in hit_positions.items():
        dists = np.linalg.norm(ref_xyz - np.array(xyz), axis=1)
        best = int(np.argmin(dists))
        if dists[best] <= cutoff:
            mapping[hit_id] = ref_ids[best]
    return mapping


def build_residue_correspondence(ref_site, hit_site, cutoff=CA_MATCH_CUTOFF):
    """Finds, for each hit-pocket residue AND metal ion, the nearest corresponding
    ref-pocket one (by raw distance) within `cutoff`. Returns {hit_ResidueId:
    ref_ResidueId}, or None if no TM-align transform is on file for this ref/hit
    pdb pair.

    data/pockets/*.pdb is written in native (unaligned) frame for both reference and
    hit (extract_pockets.py, Step 5) -- a hit's correct rigid-body transform is
    (ref_id, hit_id)-specific, and the same hit is frequently paired with several
    different references (952 of 5,832 sites in the full manifest), so no single
    pre-baked file could ever be correct for more than one of those pairings. The
    (t, u) transform is therefore fetched and applied here, per pairing, directly
    to the hit's native coordinates -- reference coordinates are never transformed
    (it's always the fixed target). An earlier version of this function assumed
    extract_pockets.py had already superimposed the hit and skipped re-applying the
    transform; that assumption silently broke for any multiply-paired hit (confirmed
    on hit_5XFW_MLI_405, paired with 6 references: its old pre-baked pocket sat 88.8 A
    from one paired reference's ligand and 113.6 A from another's).

    Metal ions (e.g. the two Mn2+ in 4AVL/5FDG) get their own nearest-distance
    correspondence alongside the CA-based one, not folded into it -- see
    _nearest_within_cutoff. Without this, PROTEIN_SEL correctly including metals so
    ProLIF can detect MetalAcceptor/MetalDonor interactions on them (run_prolif's
    Fingerprint construction) was necessary but not sufficient: the interactions
    were detected on both sides but canonicalize_ifp_for_alignment had no mapping
    entry to connect a ref metal to its hit counterpart, so they never landed in
    the aligned-residue Tanimoto's intersection -- confirmed on 4AVL/5FDG, where
    the two Mn ions sit just 0.19 A apart post-alignment but contributed 0 matched
    bits before this fix."""
    transformation = get_transformation(ref_site["pdb_id"], hit_site["pdb_id"])
    if transformation is None:
        return None

    ref_ca, hit_ca = get_ca_positions(ref_site), get_ca_positions(hit_site, transform=transformation)
    mapping = _nearest_within_cutoff(ref_ca, hit_ca, cutoff)
    ref_metal = get_metal_positions(ref_site)
    hit_metal = get_metal_positions(hit_site, transform=transformation)
    mapping.update(_nearest_within_cutoff(ref_metal, hit_metal, cutoff))
    return mapping


def calculate_pocket_rmsd(ref_ca, hit_ca, match_cutoff=5.0):
    """Local (pocket-only) CA-CA RMSD between a reference pocket and a hit
    structure already in the same (TM-aligned) frame -- ported from the old
    homebrew scripts/analyze_isosteres_simple.py's calculate_pocket_rmsd (not
    imported: that script pulls in the whole homebrew scorer this pipeline is
    replacing), same algorithm.

    Matches each reference-pocket CA to its nearest hit CA by raw 3D distance
    rather than identical (chain, residue_number): both sides are already
    superposed via TM-align, and homologs are frequently numbered differently
    (indels, different constructs) even when the fold occupies the same
    position post-superposition. Only matches closer than match_cutoff count
    toward the RMSD, so a hit that only locally covers part of the reference
    pocket isn't penalized for the part it doesn't reach.

    ref_ca is expected already restricted to the ~10 A pocket around the
    reference ligand -- true for free for anything built from
    get_ca_positions(ref_site), since extract_pockets.py only ever extracts
    that radius around the ligand in the first place (CANDIDATE_RADIUS/
    batch_motif_extraction.py's own convention). hit_ca is NOT pre-filtered by
    distance to anything; the nearest-neighbor matching does that implicitly
    (a hit CA far from every reference pocket CA just never becomes anyone's
    nearest match).

    Returns None if either side is empty, or if no ref pocket CA has a hit CA
    within match_cutoff (nothing to compute an RMSD over)."""
    if not ref_ca or not hit_ca:
        return None
    ref_xyz = np.array(list(ref_ca.values()))
    hit_xyz = np.array(list(hit_ca.values()))
    dists = np.sqrt(((ref_xyz[:, None, :] - hit_xyz[None, :, :]) ** 2).sum(axis=2))
    nearest = dists.min(axis=1)
    close = nearest[nearest < match_cutoff]
    if close.size == 0:
        return None
    return float(np.sqrt((close ** 2).mean()))


def canonicalize_ifp_for_alignment(ifp, protein_mapping=None):
    """Rewrites an IFP's residue keys for cross-structure comparison. Two changes:

    1. The ligand side is collapsed to a single placeholder ResidueId. Each site has
       a different ligand (different resname/resnum), so the DataFrame column key
       -- (ligand, protein, interaction) -- would never match across ref/hit even
       with identical protein positions unless the ligand identity is also unified.
    2. The protein side is remapped through `protein_mapping` (hit ResidueId -> ref
       ResidueId), if given; residues with no correspondence in `protein_mapping`
       are dropped, since they can't be compared position-wise. `protein_mapping=None`
       is a straight passthrough, used for the reference site itself.

    Collisions after remapping (two hit residues landing on the same ref residue) are
    merged by combining their per-interaction metadata tuples.
    """
    from prolif.residue import ResidueId
    canonical_lig = ResidueId("LIG", 1, None)

    remapped = {}
    for (lig_id, prot_id), data in ifp.items():
        if protein_mapping is None:
            ref_prot_id = prot_id
        else:
            ref_prot_id = protein_mapping.get(prot_id)
            if ref_prot_id is None:
                continue
        key = (canonical_lig, ref_prot_id)
        if key not in remapped:
            remapped[key] = dict(data)
        else:
            merged = dict(remapped[key])
            for name, metadata in data.items():
                merged[name] = merged.get(name, ()) + metadata
            remapped[key] = merged
    return remapped


def flatten_canon_bits(canon_ifp):
    """Flattens a canonicalize_ifp_for_alignment() result down to a plain set of
    (protein_residue, interaction_name) pairs -- the ligand side is already
    collapsed to a placeholder there, so this just discards the outer (ligand,
    protein) key structure. Shared by the Tanimoto intersection/union above and
    the PLIF-style recall score below, so both read from the identical bit set."""
    bits = set()
    for (lig, prot), data in canon_ifp.items():
        for name in data:
            bits.add((str(prot), name))
    return bits


def interaction_type_bitvector(ifp, interactions):
    """Collapses a site's residue-keyed IFP down to "does this interaction type
    occur anywhere in the pocket", discarding which residue made the contact.
    Residue-keyed bitvectors don't overlap across sites from different PDB
    structures (residue numbering never coincides), so this is the vector to use
    for cross-structure ref/hit Tanimoto comparisons."""
    present = {name for per_residue_pair in ifp.values() for name in per_residue_pair}
    bv = ExplicitBitVect(len(interactions))
    for i, name in enumerate(interactions):
        if name in present:
            bv.SetBit(i)
    return bv


def run_site_scoped(site, fp, pif):
    """Dispatches to the right scope per role, matching compare_with_homebrew_plif.py's
    convention exactly: reference sites are scored on their phosphate group(s) only,
    since that's the part of the reference ligand a phosphate isostere is actually
    meant to mimic; hit/mimic sites have no phosphorus by construction (they were
    selected as isosteres of a phosphate, not phosphates themselves) so they stay
    whole-ligand -- there's nothing to restrict them to.

    Returns (ifp, err, groups): ifp is the busiest phosphate group's IFP (by
    interaction count), used as this site's own hit-independent standalone
    fingerprint (prolif_bitvectors.csv/.pkl, prolif_interaction_type_vectors.csv).
    groups is the full {p_idx: ifp} dict for every phosphate group on a
    multi-phosphate reference ligand (e.g. ATP/GTP have 3; None for hits, or a
    single-entry dict for a reference with only one) -- the per-pair comparison
    loop in main() uses this to pick whichever specific group scores highest
    against each specific hit, since which phosphate a given mimic is actually
    replicating isn't necessarily the busiest one and can differ hit to hit."""
    if site["role"] != "reference":
        ifp, err = run_site(site, fp)
        return ifp, err, None

    groups, err = pif.phosphate_group_ifps(site, fp)
    if groups is None:
        return None, err, None
    if not groups:
        return None, "reference ligand has no phosphorus atom to anchor a phosphate group on", None
    best_p_idx = max(groups, key=lambda p: pif.count_interactions(groups[p]))
    return groups[best_p_idx], None, groups


# ── Parallel per-site fingerprinting (Phase 1 worker) ──────────────────────────
# Module-level, not nested in main(): ProcessPoolExecutor uses Windows's "spawn"
# start method here (fork isn't available), which pickles a submitted function
# by qualified name/import path, not by value -- a closure or function defined
# inside main() can't be pickled that way. Each spawned child re-imports this
# module fresh (harmless/cheap -- no expensive top-level work in this file), so
# _init_worker/_fingerprint_one_site are reachable as plain module attributes in
# every worker process.
_worker_fp = None
_worker_pif = None


def _init_worker():
    """ProcessPoolExecutor initializer -- runs once per worker process, not once
    per site. Builds this worker's OWN Fingerprint instance rather than having
    the parent's fp pickled across the process boundary: plf.Fingerprint has its
    own to_pickle()/from_pickle() (a custom scheme), not necessarily identical to
    what the standard `pickle` module ProcessPoolExecutor uses for IPC would
    produce, and building one fresh is cheap (no heavy state) -- so there's no
    reason to rely on that being equivalent when sidestepping it entirely is free.
    """
    global _worker_fp, _worker_pif
    from rdkit import RDLogger
    # RDKit's sanitizer logs "Explicit valence..."/"Kekulize..." warnings through
    # its OWN native C++ logger, straight to the process's stderr -- separate from
    # (and NOT silenced by) this module's warnings.filterwarnings("ignore") above,
    # which only touches Python-level warnings. With 8 worker processes all
    # writing to the SAME inherited stderr pipe concurrently, this was frequent
    # enough to deadlock the whole pool on Windows: confirmed directly -- a real
    # run hung solid at site 18/99 (0 progress for 60+ real seconds after 18 sites
    # completing in ~2s), and 8 orphaned worker processes were still alive after
    # killing the parent, each presumably blocked on a stderr write() the reader
    # side wasn't draining fast enough. Silencing RDKit's own logger per worker
    # removes the write volume that was causing the contention in the first place.
    RDLogger.DisableLog("rdApp.*")
    import prolif as plf
    import phosphate_ifp as pif  # deferred: phosphate_ifp imports this module at its own top level
    _worker_fp = plf.Fingerprint(
        INTERACTIONS, parameters={"VdWContact": {"preset": "rdkit"}}, use_segid=False, count=True,
    )
    _worker_pif = pif


def _fingerprint_one_site(site):
    """Runs in a worker process. Measures its own wall time internally (not by
    the parent timestamping around pool.submit()/future completion) so the
    reported per-site duration reflects actual compute time, not however long a
    site sat queued behind other workers -- that distinction only matters once
    sites run concurrently instead of strictly in submission order."""
    t0 = time.time()
    try:
        ifp, err, groups = run_site_scoped(site, _worker_fp, _worker_pif)
    except Exception as e:
        ifp, err, groups = None, f"{type(e).__name__}: {e}", None
    return site["site_id"], ifp, err, groups, time.time() - t0


def main():
    common.set_tmalign_json(_default_tmalign_json())
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # see _init_worker()'s comment -- same fix, parent process
    import prolif as plf
    from prolif.utils import to_bitvectors
    from prolif.utils import to_dataframe as ifp_pair_to_dataframe
    from rdkit.DataStructs import TanimotoSimilarity
    import phosphate_ifp as pif  # deferred: phosphate_ifp imports this module at its own top level

    with open(MANIFEST_PATH) as f:
        sites = list(csv.DictReader(f))

    # some site_ids repeat across rows (one reference site paired with several hits),
    # so only run each physical pocket through ProLIF once.
    unique_sites = list({s["site_id"]: s for s in sites}.values())

    # mdanalysis's default vdW radii table (ProLIF's own default preset) only covers 54
    # elements and is missing several metals this project cares about (Mn, W...) --
    # "not found" crashes the whole site rather than just skipping VdWContact for that
    # atom. rdkit's preset covers all 118 and includes every element seen in this
    # dataset so far; switching it out doesn't change how VdWContact itself is defined,
    # just which radius table it looks values up in.
    # count=True (not the default False): with count=False, ProLIF's Fingerprint
    # keeps only ONE atom-pair per (ligand_residue, protein_residue, interaction_type)
    # -- interaction.any(), which is whichever match its internal generator yields
    # FIRST, not the closest or "best" one (see prolif's Interaction.any()/.best()).
    # A genuinely bidentate contact (e.g. Arg's guanidinium salt-bridging BOTH
    # oxygens of a carboxylate) only ever gets one of those two oxygens recorded --
    # the other real contact is invisible to the fingerprint from generation time
    # onward, which under-marks isosteric atoms downstream (export_prolif_
    # datawarrior.py's matched_hit_atom_indices()). count=True switches to
    # interaction.all(), capturing every qualifying atom-pair instead.
    #
    # Verified this doesn't change any already-published score: flatten_canon_bits()
    # only reads the interaction-type KEYS present in the metadata dict, never the
    # length of the metadata tuple, so prolif_plif_score/Tanimoto are unaffected --
    # this only enriches the per-bit metadata used for atom-level marking.
    fp = plf.Fingerprint(INTERACTIONS, parameters={"VdWContact": {"preset": "rdkit"}}, use_segid=False, count=True)

    ifp_by_frame = {}
    ifp_by_site = {}
    site_id_by_frame = {}
    site_row_by_id = {}
    role_by_site = {}
    type_bv_by_site = {}
    ref_groups_by_site = {}  # site_id -> {p_idx: ifp}, only populated for multi-phosphate references
    n_fail = 0
    total_sites = len(unique_sites)
    print(f"\n=== Fingerprinting {total_sites} unique sites ({FINGERPRINT_WORKERS} workers) ===")
    t0 = time.time()
    site_by_id = {s["site_id"]: s for s in unique_sites}
    i = 0
    # Each site is fingerprinted completely independently of every other site (no
    # shared mutable state between them), so this is embarrassingly parallel --
    # ProcessPoolExecutor, not the sequential loop this used to be. Sites complete
    # in whatever order finishes first, not unique_sites' original order; that's
    # fine here since frame indices below are assigned from a running counter at
    # completion time, not from the original per-site index, so ifp_by_frame and
    # site_id_by_frame stay consistent with each other regardless of completion order.
    with ProcessPoolExecutor(max_workers=FINGERPRINT_WORKERS, initializer=_init_worker) as pool:
        futures = {pool.submit(_fingerprint_one_site, site): site for site in unique_sites}
        for future in as_completed(futures):
            site = futures[future]
            i += 1
            try:
                site_id, ifp, err, groups, site_dt = future.result()
            except Exception as e:
                site_id, ifp, err, groups, site_dt = site["site_id"], None, f"{type(e).__name__}: {e}", None, 0.0
            if ifp is None:
                n_fail += 1
            n_ok_so_far = i - n_fail
            elapsed = time.time() - t0
            eta = elapsed / i * (total_sites - i) if i else 0
            progress = f"[{i}/{total_sites} {100*i/total_sites:3.0f}%  ok={n_ok_so_far} fail={n_fail}]"
            if ifp is None:
                print(f"  {progress} {site_id}: FAILED {err} ({site_dt:.1f}s, ETA {fmt_eta(eta)})")
                continue
            frame = len(ifp_by_frame)
            ifp_by_frame[frame] = ifp
            ifp_by_site[site_id] = ifp
            site_id_by_frame[frame] = site_id
            site_row_by_id[site_id] = site_by_id[site_id]
            role_by_site[site_id] = site_by_id[site_id]["role"]
            type_bv_by_site[site_id] = interaction_type_bitvector(ifp, INTERACTIONS)
            if groups and len(groups) > 1:
                ref_groups_by_site[site_id] = groups
            n_bits = sum(len(interactions) for interactions in ifp.values())
            extra = f", {len(groups)} phosphate groups" if groups and len(groups) > 1 else ""
            print(f"  {progress} {site_id}: OK {len(ifp)} residue pairs, {n_bits} interactions"
                  f"{extra} ({site_dt:.1f}s, ETA {fmt_eta(eta)})")

    n_ok = len(ifp_by_frame)
    print(f"\nPhase done in {fmt_eta(time.time()-t0)}: {n_ok}/{total_sites} unique sites succeeded, {n_fail} failed.")
    if not ifp_by_frame:
        print("No sites succeeded; nothing to write.")
        sys.exit(1)

    # populate fp.ifp manually (normally set by fp.run() over trajectory frames) so the
    # native to_dataframe()/to_bitvectors() machinery treats each site as one "frame".
    fp.ifp = ifp_by_frame
    fp.site_ids = [site_id_by_frame[i] for i in range(n_ok)]

    # count=False explicitly: Fingerprint.to_dataframe()'s OWN default is count=None,
    # which falls back to self.count (True, now that the Fingerprint above is built
    # with count=True for atom-level metadata) -- that would silently turn this
    # dataframe's cells from bool presence into integer match-counts, corrupting the
    # meaning of every downstream "bitvectors" file below without changing anything
    # visible about how this function is called. Forcing count=False here keeps
    # these three files' semantics exactly as they were before the Fingerprint's own
    # count=True was added -- only the per-pair atom-tracing logic (which reads the
    # richer metadata directly off canon_hit_ifp, not off this dataframe) benefits
    # from the fuller metadata.
    df = fp.to_dataframe(count=False)
    df.index = fp.site_ids
    df.index.name = "site_id"

    fp_pickle = os.path.join(RESULTS_DIR, f"prolif_fingerprint{RUN_SUFFIX}.pkl")
    fp.to_pickle(fp_pickle)
    print(f"Wrote {fp_pickle} (full Fingerprint object, incl. per-interaction metadata/distances)")

    df_pickle = os.path.join(RESULTS_DIR, f"prolif_bitvectors{RUN_SUFFIX}.pkl")
    df.to_pickle(df_pickle)  # bool dtype, forced via to_dataframe(count=False) above
    df_csv = os.path.join(RESULTS_DIR, f"prolif_bitvectors{RUN_SUFFIX}.csv")
    df.astype(int).to_csv(df_csv)  # 0/1 instead of True/False, for readability
    print(f"Wrote {df_pickle} and {df_csv} ({df.shape[0]} sites x {df.shape[1]} bits)")

    type_csv = os.path.join(RESULTS_DIR, f"prolif_interaction_type_vectors{RUN_SUFFIX}.csv")
    with open(type_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "role", *INTERACTIONS])
        w.writeheader()
        for site_id, bv in type_bv_by_site.items():
            w.writerow({
                "site_id": site_id, "role": role_by_site[site_id],
                **{name: int(bv.GetBit(i)) for i, name in enumerate(INTERACTIONS)},
            })
    print(f"Wrote {type_csv} (site x interaction-type presence, collapsed over residues)")

    # pair up sites using the manifest's pdb_id <-> paired_with columns (paired_with
    # names a PDB ID, not a site_id, and a reference site_id can appear in several rows).
    pdbid_to_siteid = {s["pdb_id"]: s["site_id"] for s in sites}
    pairs, seen = [], set()
    for row in sites:
        other_site = pdbid_to_siteid.get(row["paired_with"])
        if other_site is None or other_site == row["site_id"]:
            continue
        key = frozenset((row["site_id"], other_site))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((row["site_id"], other_site))

    sim_records = []
    total_pairs = len(pairs)
    n_pair_ok = n_pair_fail = n_pair_skip = 0
    print(f"\n=== Scoring {total_pairs} pairs ===")
    t0_pairs = time.time()
    for i, (site_a, site_b) in enumerate(pairs, 1):

        def _pair_progress():
            elapsed = time.time() - t0_pairs
            eta = elapsed / i * (total_pairs - i) if i else 0
            tally = f"ok={n_pair_ok} fail={n_pair_fail} skip={n_pair_skip}"
            return f"[{i}/{total_pairs} {100*i/total_pairs:3.0f}%  {tally}]", fmt_eta(eta)

        # role_by_site/ifp_by_site are only populated for sites that succeeded Phase 1
        # (see the `if ifp is None: continue` guard above) -- indexing role_by_site
        # before checking that must not happen, since at full-manifest scale most
        # `pairs` entries reference at least one site whose pocket was never extracted
        # or failed fingerprinting. Uncaught KeyError here previously took down the
        # entire pair-scoring phase (and, with it, every already-scored pair, since
        # sim_records is only written to disk once at the very end).
        if site_a not in role_by_site or site_b not in role_by_site:
            missing = site_a if site_a not in role_by_site else site_b
            n_pair_skip += 1
            progress, eta_str = _pair_progress()
            print(f"  {progress} skip {site_a} vs {site_b}: {missing} has no fingerprint (failed earlier)"
                  f" (ETA {eta_str})")
            continue

        ref_id, hit_id = (site_a, site_b) if role_by_site[site_a] == "reference" else (site_b, site_a)
        ref_ifp, hit_ifp = ifp_by_site[ref_id], ifp_by_site[hit_id]

        try:
            # tanimoto_residue (raw ProLIF bitvector keyed by literal PDB residue number)
            # was dropped as a reported metric -- it was never anything but ~0 for a
            # cross-structure pair (different structures never share residue numbering).
            # interaction_type_bitvector/type_bv_by_site are still used for the
            # prolif_interaction_type_vectors output file, just no longer for picking
            # a multi-phosphate reference's representative group (see below).
            used_p_idx = None

            # aligned residue comparison: remap the hit's protein residues onto the ref's
            # numbering via the pipeline's saved TM-align superposition (nearest CA within
            # CA_MATCH_CUTOFF), so matching bit columns actually mean "same pocket position"
            # instead of "same PDB residue number" (which never coincides across structures).
            # Computed before group selection below since it doesn't depend on which
            # phosphate group ends up representing the reference -- only on the ref/hit
            # structures themselves -- and multi-phosphate selection now needs it too.
            mapping = build_residue_correspondence(site_row_by_id[ref_id], site_row_by_id[hit_id])
            canon_hit_ifp = None if mapping is None else canonicalize_ifp_for_alignment(hit_ifp, protein_mapping=mapping)
            hit_bits = None if canon_hit_ifp is None else flatten_canon_bits(canon_hit_ifp)

            if ref_id in ref_groups_by_site:
                # Multi-phosphate reference (e.g. ATP/GTP): the busiest group isn't
                # necessarily the one this specific hit is mimicking -- score every
                # candidate phosphate group against this specific hit's ALIGNED bits and
                # use whichever one has the most matched interactions (raw count, not a
                # fraction or Tanimoto) as the representative.
                #
                # Raw count over Tanimoto: a whole-ligand hit's contacts elsewhere in the
                # pocket -- unrelated to the phosphate-mimicking region -- inflate a
                # Tanimoto union and can drag down a group that's actually well-matched,
                # penalizing exactly the mimics this comparison cares about most (a big,
                # multi-functional hit that nails the phosphate position but also does
                # other chemistry elsewhere). Raw count isn't fooled by that, since it
                # only counts what actually matched.
                #
                # Raw count over fraction/recall: a near-empty group could otherwise get
                # its one interaction matched and "win" with a perfect-looking fraction
                # against a much better-matched, busier group -- this is exactly the
                # failure mode the old Tanimoto selection needed a separate
                # MIN_GROUP_INTERACTIONS_FOR_MATCH guard to patch (now removed: a bigger,
                # better-matched group can never lose to a smaller one on raw count the
                # way it could on fraction, so there's nothing left for that guard to do).
                if hit_bits is not None:
                    best_p_idx, best_ifp, best_matched = None, None, -1
                    for p_idx, cand_ifp in ref_groups_by_site[ref_id].items():
                        cand_bits = flatten_canon_bits(canonicalize_ifp_for_alignment(cand_ifp))
                        matched = len(cand_bits & hit_bits)
                        if matched > best_matched:
                            best_p_idx, best_ifp, best_matched = p_idx, cand_ifp, matched
                else:
                    # No TM-align transform on file -- no aligned bits to match against,
                    # so there's nothing hit-specific to base a choice on. Falls back to
                    # busiest-by-count so a p_idx is still reported (compare_with_homebrew_
                    # plif.py's same_phosphate_group flag needs one either way), even
                    # though no score can be computed for this pair regardless.
                    best_p_idx, best_ifp = max(
                        ref_groups_by_site[ref_id].items(),
                        key=lambda kv: pif.count_interactions(kv[1]),
                    )
                ref_ifp, used_p_idx = best_ifp, best_p_idx

            if mapping is None:
                aligned_sim, n_mapped, prolif_plif, prolif_ref_n = None, 0, None, None
                print(f"  [warn] {ref_id} vs {hit_id}: no TM-align transform on file, skipping aligned comparison")
            else:
                canon_ref_ifp = canonicalize_ifp_for_alignment(ref_ifp)
                # count=False explicit: this is prolif.utils.to_dataframe (a plain
                # function, aliased above), NOT Fingerprint.to_dataframe() -- its own
                # default is already count=False regardless of the Fingerprint's own
                # count=True, so tanimoto_aligned_residue is unaffected either way.
                # Spelled out explicitly so a future edit can't accidentally flip this
                # to count-aware (which WOULD double-count a bidentate interaction as
                # worth 2 bits instead of 1 in the Tanimoto below) without it being an
                # obvious, deliberate change right here.
                pair_df = ifp_pair_to_dataframe({0: canon_ref_ifp, 1: canon_hit_ifp}, INTERACTIONS, count=False)
                pair_bv = to_bitvectors(pair_df) if pair_df.shape[1] else []
                aligned_sim = TanimotoSimilarity(pair_bv[0], pair_bv[1]) if len(pair_bv) == 2 else 0.0
                n_mapped = len(mapping)

                # PLIF-style recall score: matches the homebrew scorer's own definition
                # exactly (matched_ref_interactions / total_ref_interactions -- see
                # analyze_plif.calculate_spatial_similarity's `len(matched_ref) /
                # len(ref_inters)`), computed from ProLIF's own aligned-residue bits
                # instead of the homebrew distance-based ones. Structurally comparable
                # to PLIF in a way Tanimoto isn't: both ask "what fraction of the
                # reference's own interactions did the mimic replicate", and neither
                # penalizes a mimic for making MORE contacts than the reference has --
                # unlike Tanimoto's intersection/union, where those extra hit-only
                # bits inflate the denominator and pull the score down regardless of
                # how well the reference's own interactions were replicated.
                ref_bits = flatten_canon_bits(canon_ref_ifp)
                prolif_plif = len(ref_bits & hit_bits) / len(ref_bits) if ref_bits else None
                prolif_ref_n = len(ref_bits)
        except Exception as e:
            # One bad pair (an unexpected shape mismatch, a corrupt correspondence, etc.)
            # must not take down the whole run -- especially at full-manifest scale
            # (thousands of pairs), where losing all prior work to one exception near
            # the end would be exactly the kind of crash this project already learned
            # from once tonight (the protonation batch). Log it and move on.
            n_pair_fail += 1
            progress, eta_str = _pair_progress()
            print(f"  {progress} FAIL {site_a} vs {site_b}: {type(e).__name__}: {e} (ETA {eta_str})")
            continue

        n_pair_ok += 1
        sim_records.append({
            "site_a": site_a, "role_a": role_by_site[site_a],
            "site_b": site_b, "role_b": role_by_site[site_b],
            "tanimoto_aligned_residue": aligned_sim,
            "prolif_plif_score": prolif_plif,
            "prolif_ref_n": prolif_ref_n,
            "n_ca_mapped": n_mapped,
            "ref_phosphate_p_idx": used_p_idx,
        })
        aligned_str = f"{aligned_sim:.3f}" if aligned_sim is not None else "n/a"
        plif_str = f"{prolif_plif:.3f}" if prolif_plif is not None else "n/a"
        progress, eta_str = _pair_progress()
        print(f"  {progress} {site_a} vs {site_b}: Tanimoto(aligned residue)={aligned_str}  "
              f"ProLIF_PLIF={plif_str} [{n_mapped} CA matched] (ETA {eta_str})")

    print(f"\nPhase done in {fmt_eta(time.time()-t0_pairs)}: {n_pair_ok}/{total_pairs} pairs scored, "
          f"{n_pair_fail} failed, {n_pair_skip} skipped.")

    sim_csv = os.path.join(RESULTS_DIR, f"prolif_pair_similarity{RUN_SUFFIX}.csv")
    fieldnames = ["site_a", "role_a", "site_b", "role_b",
                  "tanimoto_aligned_residue", "prolif_plif_score", "prolif_ref_n", "n_ca_mapped",
                  "ref_phosphate_p_idx"]
    with open(sim_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sim_records)
    print(f"Wrote {sim_csv} ({len(sim_records)} pairs)")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
