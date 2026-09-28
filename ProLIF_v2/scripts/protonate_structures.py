"""
Step 3: adds hydrogens to each freshly downloaded FULL structure using
OpenBabel directly (no PLIP import, no plip.structure.preparation).

Uses OBMol.AddPolarHydrogens() rather than the more general AddHydrogens().
AddHydrogens() re-perceives bond orders/residues for the *whole* molecule from
geometry alone, and on real multi-thousand-atom structures this silently
misclassifies large chunks of real protein residues (confirmed on 7AWS: ~1/3
of true, correctly-numbered residues -- e.g. real SER A 10 -- got dropped and
replaced with generic renumbered "UNL"/"UNK" fragments on made-up chains, and
the actual ligand residue disappeared from the model entirely). That silently
breaks every downstream lookup by (chain, resname, resnum).
AddPolarHydrogens() is a narrower, non-destructive call -- verified to
preserve all 312 non-water residue identities on 7AWS exactly (same chain,
resnum, resname) while still emitting full explicit hydrogens. This happens
to be the same underlying OpenBabel call plip.structure.preparation uses
internally, but it's invoked here directly with no `import plip` and no
dependency on the plip package/pipeline. It has no pH argument (unlike
AddHydrogens), so protonation states are OpenBabel's residue-template
defaults, not pH-adjusted ionization.

Deliberately run on the *complete* deposited chain rather than a pre-cut 10 A
pocket: cutting first and protonating after would make bond-order perception
see artificial chain breaks at the cut boundary and misassign a free
N-terminus/C-terminus (extra NH3+/COO-) to residues that are actually
mid-chain in the real protein (confirmed on ref_10GW_A_6PG_401.cif: ARG at the
pocket edge got N1+). Protonating the intact chain first avoids this --
termini only show up at genuine chain ends -- and the pocket is sliced out of
the already-protonated structure afterward (extract_pockets.py), so no residue
is ever protonated in a truncated context.
"""
import os

from openbabel import pybel

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROLIF_V2_ROOT, "data", "raw")
PROTONATED_DIR = os.path.join(PROLIF_V2_ROOT, "data", "protonated")


def protonate(pdb_id):
    in_path = os.path.join(RAW_DIR, f"{pdb_id}.pdb")
    out_path = os.path.join(PROTONATED_DIR, f"{pdb_id}_protonated.pdb")
    if os.path.exists(out_path):
        print(f"  {pdb_id}: already protonated, skipping")
        return True
    try:
        mol = next(pybel.readfile("pdb", in_path))
        mol.OBMol.AddPolarHydrogens()
        mol.write("pdb", out_path, overwrite=True)
    except Exception as e:
        print(f"  {pdb_id}: FAILED ({e})")
        return False
    print(f"  {pdb_id}: protonated -> {os.path.basename(out_path)}")
    return True


def main():
    pdb_ids = sorted(f[:-4] for f in os.listdir(RAW_DIR) if f.endswith(".pdb"))
    print(f"Protonating {len(pdb_ids)} structures...")
    ok = 0
    for pdb_id in pdb_ids:
        if protonate(pdb_id):
            ok += 1
    print(f"\nDone. {ok}/{len(pdb_ids)} protonated.")


if __name__ == "__main__":
    main()
