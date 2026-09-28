"""
Step 1 of the pocket_types subproject: turns each reference site's already-
computed ProLIF interaction fingerprint into a (residue_type, interaction_type,
backbone/sidechain) feature bag -- the composition-vector representation
discussed for clustering reference sites into recurring "phosphate binding
pocket types", independent of protein family/fold/sequence position.

Reuses ProLIF_v2's own cached fingerprint (results/prolif_fingerprint_full_
manifest.pkl, written by ProLIF_v2/scripts/run_prolif.py) rather than
re-running interaction detection: that pickle's fp.ifp already holds, for
every reference site, the busiest phosphate group's IFP (run_prolif.py's
run_site_scoped() dispatch) WITH atom-level metadata (fp was built with
metadata=True), so no ProLIF geometry work needs to happen again here.

Backbone vs side-chain provenance: each interaction's metadata carries
{"indices": {...}, "parent_indices": {"protein": (...), ...}, ...}.
"indices" are local to ProLIF's own per-residue Residue submol; "parent_indices"
are indices into the ORIGINAL whole-protein RDKit mol (the one
run_prolif.safe_molecule_from_mda() returns before residue splitting, tagged
via `atom.SetUnsignedProp("mapindex", atom.GetIdx())` in that function) --
confirmed directly against ref_5VKT_NAP_405: parent_indices (1600, 1601) came
back as ASN342's ND2/HD21 (side chain), while a backbone case (SER214's N/H)
came back as parent_indices (800, 806). So this rebuilds each site's protein
mol the same way run_prolif.py/phosphate_ifp.py do (pocket PDB -> protein_
selection -> safe_molecule_from_mda, WITHOUT re-running fp.generate -- that's
the expensive geometric step this script skips) purely to resolve
parent_indices -> PDB atom names via GetPDBResidueInfo().GetName().

VdWContact is excluded from the feature bag by default (--include-vdw to
keep it): same reasoning export_prolif_datawarrior.py's
ATOM_MARKING_EXCLUDED_INTERACTIONS already documents -- a pure atom-pair
distance check, any element, no directional/chemical role required, so it's
much weaker evidence of a genuine interaction "type" than HBDonor/HBAcceptor/
Anionic/Cationic/MetalAcceptor/etc. and would dominate the feature bag by
sheer promiscuity if left in.

Usage: python build_pocket_features.py [--manifest path] [--include-vdw]
                                        [--out-dir path]
"""
import argparse
import csv
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(POCKET_TYPES_ROOT)
PROLIF_V2_ROOT = os.path.join(PROJECT_ROOT, "ProLIF_v2")
PROLIF_SCRIPTS_DIR = os.path.join(PROLIF_V2_ROOT, "scripts")
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROLIF_SCRIPTS_DIR)
import run_prolif as rp  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from constants import METALS  # noqa: E402

import MDAnalysis as mda  # noqa: E402

# Canonical PDB-style backbone atom names, heavy atoms plus the hydrogens
# PDB2PQR attaches to them (confirmed present in this project's prepared
# pockets -- see module docstring's GLY45/SER214 examples). Everything else
# (side-chain heavy atoms and their hydrogens) is "sidechain". H1/H2/H3 cover
# the N-terminal ammonium's three hydrogens; HA1/HA2/HA3 cover glycine's two
# (PDB naming isn't fully consistent on HA2 vs HA3 vs HA1, so all three are
# listed) vs every other residue's single HA.
BACKBONE_ATOM_NAMES = {
    "N", "CA", "C", "O", "OXT",
    "H", "H1", "H2", "H3", "HN",
    "HA", "HA1", "HA2", "HA3",
}

DEFAULT_EXCLUDED_INTERACTIONS = {"VdWContact"}


def _manifest_reference_sites(manifest_path):
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {}
    for r in rows:
        if r["role"] == "reference":
            by_id[r["site_id"]] = r
    return by_id


def _load_ifp_by_site(fp_pickle):
    import prolif as plf
    fp = plf.Fingerprint.from_pickle(fp_pickle)
    return dict(zip(fp.site_ids, fp.ifp.values()))


def _protein_mol_for_site(site_id):
    pocket_path = os.path.join(rp.POCKETS_DIR, f"{site_id}.pdb")
    if not os.path.exists(pocket_path):
        return None, f"no extracted pocket at {pocket_path}"
    u = mda.Universe(pocket_path)
    protein = rp.protein_selection(u)
    if len(protein) == 0:
        return None, "protein selection matched 0 atoms"
    return rp.safe_molecule_from_mda(protein, use_segid=False), None


def _provenance(prot_mol, parent_idxs, resname):
    # Metal ions (Ca2+ etc.) aren't amino acids, so backbone/sidechain doesn't
    # apply -- and matters concretely here: PDB atom-naming convention gives a
    # bare Ca2+ ion the atom name "CA", identical to the alpha-carbon backbone
    # atom name, so without this check every calcium-mediated interaction
    # would be misclassified as "backbone" by pure name coincidence rather
    # than because it's an actual protein backbone atom (confirmed on
    # ref_1AWB_IPD_281 -- see METALS import above, same set run_prolif.py's
    # PROTEIN_SEL uses).
    if resname in METALS:
        return "metal"
    names = set()
    for idx in parent_idxs:
        info = prot_mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        names.add(info.GetName().strip())
    if names <= BACKBONE_ATOM_NAMES:
        return "backbone"
    if names.isdisjoint(BACKBONE_ATOM_NAMES):
        return "sidechain"
    return "mixed"


def site_feature_bag(site_id, ifp, prot_mol, excluded_interactions):
    """Returns a Counter of (resname, interaction_type, provenance) -> count
    for one reference site's phosphate-group IFP."""
    bag = Counter()
    for (lig_id, prot_id), interactions in ifp.items():
        resname = prot_id.name
        for interaction_name, metadata_tuple in interactions.items():
            if interaction_name in excluded_interactions:
                continue
            for md in metadata_tuple:
                parent_idxs = md["parent_indices"]["protein"]
                provenance = _provenance(prot_mol, parent_idxs, resname)
                bag[(resname, interaction_name, provenance)] += 1
    return bag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(PROLIF_V2_ROOT, "full_manifest.csv"))
    parser.add_argument("--fp-pickle", default=os.path.join(PROLIF_V2_ROOT, "results",
                                                              "prolif_fingerprint_full_manifest.pkl"))
    parser.add_argument("--include-vdw", action="store_true",
                         help="keep VdWContact bits in the feature bag (excluded by default -- see module docstring)")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    excluded = set() if args.include_vdw else set(DEFAULT_EXCLUDED_INTERACTIONS)

    ref_sites = _manifest_reference_sites(args.manifest)
    print(f"{len(ref_sites)} unique reference sites in {args.manifest}")

    print(f"Loading cached ProLIF fingerprint from {args.fp_pickle} ...")
    ifp_by_site = _load_ifp_by_site(args.fp_pickle)

    long_rows = []
    summary_rows = []
    n_ok = n_no_ifp = n_no_prot_mol = n_empty = 0
    total = len(ref_sites)
    for i, (site_id, site) in enumerate(sorted(ref_sites.items()), 1):
        ifp = ifp_by_site.get(site_id)
        if not ifp:
            n_no_ifp += 1
            continue
        prot_mol, err = _protein_mol_for_site(site_id)
        if prot_mol is None:
            n_no_prot_mol += 1
            print(f"  [{i}/{total}] {site_id}: SKIP ({err})")
            continue

        bag = site_feature_bag(site_id, ifp, prot_mol, excluded)
        if not bag:
            n_empty += 1
            continue
        n_ok += 1

        n_backbone = sum(c for (_, _, prov), c in bag.items() if prov == "backbone")
        n_sidechain = sum(c for (_, _, prov), c in bag.items() if prov == "sidechain")
        n_mixed = sum(c for (_, _, prov), c in bag.items() if prov == "mixed")
        n_metal = sum(c for (_, _, prov), c in bag.items() if prov == "metal")
        summary_rows.append({
            "site_id": site_id, "pdb_id": site["pdb_id"], "ref_ligand": site["lig_resname"],
            "n_distinct_bits": len(bag), "n_backbone_hits": n_backbone,
            "n_sidechain_hits": n_sidechain, "n_mixed_hits": n_mixed, "n_metal_hits": n_metal,
        })
        for (resname, interaction_type, provenance), count in sorted(bag.items()):
            long_rows.append({
                "site_id": site_id, "pdb_id": site["pdb_id"], "ref_ligand": site["lig_resname"],
                "resname": resname, "interaction_type": interaction_type,
                "provenance": provenance, "count": count,
            })
        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] {n_ok} ok, {n_no_ifp} no-ifp, {n_no_prot_mol} no-prot-mol, {n_empty} empty-bag")

    print(f"\nDone: {n_ok}/{total} reference sites produced a feature bag "
          f"({n_no_ifp} missing from cached fingerprint, {n_no_prot_mol} pocket/protein load failures, "
          f"{n_empty} had an IFP but zero non-excluded bits).")

    long_path = os.path.join(args.out_dir, "ref_site_pocket_bits_long.csv")
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "ref_ligand", "resname",
                                           "interaction_type", "provenance", "count"])
        w.writeheader()
        w.writerows(long_rows)
    print(f"Wrote {long_path} ({len(long_rows)} rows)")

    summary_path = os.path.join(args.out_dir, "ref_site_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "ref_ligand", "n_distinct_bits",
                                           "n_backbone_hits", "n_sidechain_hits", "n_mixed_hits", "n_metal_hits"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_path} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
