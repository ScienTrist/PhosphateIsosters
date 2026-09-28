"""
Standalone: for one ref/hit pair, prints which hit-ligand atoms are "isosteric"
(participated in a matched interaction -- see export_prolif_datawarrior.py's
matched_hit_atom_indices() for the full derivation) as JSON:
    {"score": float|null, "atom_names": [...], "fg_expansion_atom_names": [...], "error": str|null}

atom_names is the raw ProLIF-matched set (each one individually backed by a real
matched interaction bit). fg_expansion_atom_names is what expand_to_functional_
groups() added on top -- atoms with no direct ProLIF evidence of their own, pulled
in purely because they share an Ertl functional group (ifg.py) with a raw-matched
atom. Kept separate (not merged into one list) so callers can distinguish "this
atom matched an interaction" from "this atom completes the same functional group
as one that did" -- e.g. coloring them differently in a viewer.

Meant to be called via subprocess, not imported -- PyMOL's own bundled Python
doesn't have prolif/rdkit/MDAnalysis installed, same reason explain_pair.py is
invoked as a subprocess from pymol_browse_homebrew_vs_prolif.py rather than
imported directly.

Usage: python get_isosteric_atoms.py <ref_site_id> <hit_site_id> [manifest.csv]
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")
sys.path.insert(0, SCRIPT_DIR)

from export_prolif_datawarrior import (  # noqa: E402
    _manifest_rows, matched_hit_atom_indices, build_hit_ligand_mol, atom_names,
    expand_to_functional_groups,
)


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: get_isosteric_atoms.py <ref_site_id> <hit_site_id> [manifest.csv]"}))
        return

    ref_site, hit_site = sys.argv[1], sys.argv[2]
    manifest_name = sys.argv[3] if len(sys.argv) > 3 else "full_manifest.csv"
    manifest_tag = os.path.splitext(os.path.basename(manifest_name))[0]
    run_suffix = "" if manifest_tag == "sample_manifest" else f"_{manifest_tag}"
    fp_pickle = os.path.join(RESULTS_DIR, f"prolif_fingerprint{run_suffix}.pkl")

    manifest = _manifest_rows(os.path.join(PROLIF_V2_ROOT, manifest_name))
    if ref_site not in manifest or hit_site not in manifest:
        print(json.dumps({"error": f"{ref_site} or {hit_site} not in {manifest_name}"}))
        return
    ref_row, hit_row = manifest[ref_site], manifest[hit_site]

    if not os.path.exists(fp_pickle):
        print(json.dumps({"error": f"missing {fp_pickle}"}))
        return

    try:
        import prolif as plf
        fp = plf.Fingerprint.from_pickle(fp_pickle)
        ifp_by_site = dict(zip(fp.site_ids, fp.ifp.values()))
        hit_ifp = ifp_by_site.get(hit_site)
        if hit_ifp is None:
            print(json.dumps({"error": f"{hit_site} not in cached fingerprint"}))
            return

        atom_idxs, score, err = matched_hit_atom_indices(ref_row, hit_row, hit_ifp, fp, {})
        if err:
            print(json.dumps({"error": err}))
            return

        lig_mol, mol_err = build_hit_ligand_mol(hit_row)
        if lig_mol is None:
            print(json.dumps({"error": mol_err}))
            return

        expanded_idxs = expand_to_functional_groups(lig_mol, atom_idxs)
        added_idxs = expanded_idxs - atom_idxs
        names = atom_names(lig_mol, atom_idxs)
        added_names = atom_names(lig_mol, added_idxs)
        print(json.dumps({
            "score": score, "atom_names": names, "fg_expansion_atom_names": added_names, "error": None,
        }))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    main()
