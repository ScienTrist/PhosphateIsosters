"""
Step 5: merges each site's PDB2PQR-protonated protein (protein_protonated/)
with its separately, more-carefully protonated ligand (ligand_protonated/,
see protonate_ligand.py) -- dropping whatever PDB2PQR guessed for that one
ligand residue and splicing in the RDKit/template version instead -- then
slices a 10 A pocket around the ligand out of the merged, fully-protonated
structure. Same radius/inclusion rule as the main pipeline's
batch_motif_extraction.py (any atom of a residue within 10 A of any ligand
atom).

Pockets are written in their OWN native crystallographic frame -- both
reference and hit -- with no TM-align superposition baked in. An earlier
version of this script pre-superimposed each hit onto its paired reference's
frame before writing the pocket file, keyed only by site_id. That silently
broke for any hit paired with more than one reference (952 of 5,832 sites in
the full manifest, ~978 of the 5,499 scored pairs): a hit's correct rigid-body
transform is (ref_id, hit_id)-specific, but a site_id-keyed file can only
store one, so every reference sharing that hit beyond the first-processed one
got its own already-existing (and now wrongly-superimposed) pocket file
silently reused via the already-extracted-skip check. Confirmed directly on
hit_5XFW_MLI_405 (paired with 6 different references): its cached pocket sat
88.8 A from one paired reference's ligand and 113.6 A from another's, despite
both individual TM-align superpositions themselves being excellent (TM-score
0.92-0.98). Extraction is per-site_id and reference-independent again now, so
one file per site is correct regardless of how many references it's paired
with -- run_prolif.py's build_residue_correspondence applies the
(ref_id, hit_id)-specific transform itself, on demand, per pairing, instead
of trusting a pre-baked file (see that function's docstring).

Run after protein and ligand are protonated (any order); this is the only
step that reads both.

Preserves the ligand's CONECT-derived bond connectivity (see
_parse_ligand_bonds/_write_ligand_conect) through the merge+radius-slice, so
downstream ligand mol construction (run_prolif.py's safe_molecule_from_mda,
via build_hit_ligand_mol) doesn't have to fall back to MDAnalysis's
distance-only guess_bonds() -- confirmed directly on hit_3KE1_829_164 (ligand
829, "5'-deoxy-5'-[(pyridin-4-ylcarbonyl)amino]cytidine" per RCSB's chemical
component dictionary) that distance-guessed bonds produced a chemically
nonsensical structure (cumulated double bonds inside its pyridine ring,
stray formal charges) even though ligand_protonated/*_ligand.pdb -- and the
original data/raw/*.pdb this pipeline downloaded -- both have fully correct
CONECT records for it the whole time; extract_pocket() previously discarded
them (gemmi.Structure objects built via Residue.clone() never carry
connectivity, and out_st.write_pdb() defaults to conect_records=False even if
they had). gemmi doesn't parse PDB CONECT into Structure.connections on read
either (confirmed empirically: len(gemmi.read_structure(path).connections)
== 0 for a file with real CONECT records), so this parses/re-emits CONECT as
plain text rather than trying to round-trip it through gemmi's connection
API. Protein/water connectivity is NOT reconstructed here -- PDB2PQR-written
protein_protonated/*.pdb never had CONECT records to begin with (unlike the
separately, more carefully protonated ligand), and downstream protein mol
construction relies on residue-template bond assignment instead, which
doesn't need them.
"""
import csv
import json
import os
import sys
import time

import gemmi


PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from utils import fmt_eta  # noqa: E402
from protonate_protein import chainmap_path  # noqa: E402
# CLI override, e.g. `python extract_pockets.py full_manifest.csv` -- same pattern
# batch_prepare_structures_parallel.py already uses. Pocket files are named by
# site_id (stable/unique regardless of which manifest references them), so no
# output-naming changes needed here even at full-manifest scale -- pockets already
# extracted for the sample are just skipped (already-exists check) either way.
MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
PROTEIN_DIR = os.path.join(PROLIF_V2_ROOT, "data", "protein_protonated")
LIGAND_DIR = os.path.join(PROLIF_V2_ROOT, "data", "ligand_protonated")
POCKETS_DIR = os.path.join(PROLIF_V2_ROOT, "data", "pockets")

os.makedirs(POCKETS_DIR, exist_ok=True)
RADIUS = 10.0


def merge_ligand_into_protein(protein_st, ligand_st, chain_name, lig_resnum):
    """Returns a new gemmi.Structure with the target ligand residue removed
    from `chain_name` and replaced by the (already protonated) residue from
    ligand_st. Other residues/chains are carried over unchanged.

    Matches purely on (chain, resnum), not resname: legacy PDB format's
    3-character residue-name field silently truncates the newer 4-5
    character extended CCD codes (e.g. A1C67 -> A1C), and both gemmi's and
    RDKit's PDB writers do this on every write in this pipeline (protein
    and ligand alike) -- so comparing against the manifest's full-length
    resname here would never match for any of those codes. Resnum+chain is
    already what identifies this specific ligand instance (that's how
    site_id itself is built), so it's a sufficient key on its own."""
    ligand_residue = None
    for model in ligand_st:
        for chain in model:
            for res in chain:
                ligand_residue = res
        break
    if ligand_residue is None:
        return None

    out_st = gemmi.Structure()
    out_model = gemmi.Model("1")
    for model in protein_st:
        for chain in model:
            new_chain = gemmi.Chain(chain.name)
            for res in chain:
                if chain.name == chain_name and res.seqid.num == lig_resnum:
                    continue  # drop PDB2PQR's unreliable version of this residue
                new_chain.add_residue(res.clone(), -1)
            if chain.name == chain_name:
                new_chain.add_residue(ligand_residue.clone(), -1)
            out_model.add_chain(new_chain)
        break
    out_st.add_model(out_model)
    return out_st


def find_ligand(structure, chain_name, lig_resnum):
    # Matches on (chain, resnum) only -- see merge_ligand_into_protein's docstring
    # for why resname can't be trusted here (PDB format truncates long CCD codes).
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                if res.seqid.num == lig_resnum:
                    return res
        break
    return None


def _parse_ligand_bonds(ligand_path):
    """Reads CONECT records from a ligand PDB (see module docstring for why
    this can't just be gemmi.read_structure().connections) and resolves them
    from atom SERIAL to atom NAME, since serials are meaningless once the
    ligand is merged into a larger structure and gemmi renumbers everything
    on write. Returns {(name1, name2) sorted: bond_order}.

    Bond order comes from how many times a partner serial is repeated within
    ONE CONECT line (PDB's own convention, e.g. "CONECT 2 3 3 4" says atom 2
    is double-bonded to 3, single-bonded to 4) -- NOT from counting the same
    pair's line appearing under both atoms, because this pipeline's own
    ligand protonation writer doesn't always emit both directions (confirmed
    on hit_3KE1_829_164_ligand.pdb: e.g. O2's carbonyl double bond to C2 is
    only ever described from C2's own "CONECT 2 3 3 ..." line, O2 never gets
    its own CONECT line at all). Taking max() per unordered pair across
    however many directions ARE present handles both that case and the
    fully-symmetric case identically."""
    names = {}
    with open(ligand_path) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            names[int(line[6:11])] = line[12:16].strip()

    orders = {}
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        body = line.rstrip("\n")
        serials = [int(body[i:i + 5]) for i in range(6, len(body), 5) if body[i:i + 5].strip()]
        if len(serials) < 2:
            continue
        center = serials[0]
        if center not in names:
            continue
        for partner in set(serials[1:]):
            if partner not in names:
                continue
            order = serials[1:].count(partner)
            key = tuple(sorted((names[center], names[partner])))
            orders[key] = max(orders.get(key, 0), order)
    return orders


def _write_ligand_conect(out_path, bonds, chain_name, lig_resnum):
    """Appends CONECT records (before the final END line) for whichever of
    `bonds` (from _parse_ligand_bonds, keyed by atom NAME) match atoms
    actually present at (chain_name, lig_resnum) in the just-written
    out_path -- looked up fresh from that file's own serial numbers, since
    gemmi's write_pdb renumbers every atom from scratch (preserve_serial
    defaults to False) and this site's ligand isn't necessarily the only
    residue with that name in the pocket (e.g. hit_3KE1_829_164's own pocket
    contains a second, symmetry-related copy of ligand 829 in chain C --
    chain+resnum, not resname, is what's unique, same convention
    merge_ligand_into_protein/find_ligand already use elsewhere in this
    file)."""
    serial_by_name = {}
    with open(out_path) as f:
        pdb_lines = f.readlines()
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")) and line[21] == chain_name and int(line[22:26]) == lig_resnum:
            serial_by_name[line[12:16].strip()] = int(line[6:11])

    partners = {}
    for (n1, n2), order in bonds.items():
        if n1 not in serial_by_name or n2 not in serial_by_name:
            continue
        s1, s2 = serial_by_name[n1], serial_by_name[n2]
        partners.setdefault(s1, []).extend([s2] * order)
        partners.setdefault(s2, []).extend([s1] * order)

    conect_lines = []
    for serial in sorted(partners):
        entries = sorted(partners[serial])
        for i in range(0, len(entries), 4):  # PDB CONECT: max 4 partner slots/line
            chunk = entries[i:i + 4]
            conect_lines.append("CONECT" + f"{serial:>5}" + "".join(f"{p:>5}" for p in chunk) + "\n")

    if not conect_lines:
        return 0
    end_idx = next((i for i, line in enumerate(pdb_lines) if line.startswith("END")), len(pdb_lines))
    pdb_lines[end_idx:end_idx] = conect_lines
    with open(out_path, "w") as f:
        f.writelines(pdb_lines)
    return len(serial_by_name)


def extract_pocket(site):
    protein_path = os.path.join(PROTEIN_DIR, f"{site['pdb_id']}_protein.pdb")
    ligand_path = os.path.join(LIGAND_DIR, f"{site['site_id']}_ligand.pdb")
    out_path = os.path.join(POCKETS_DIR, f"{site['site_id']}.pdb")
    # Size check, not just existence -- guards against a 0-byte/truncated
    # out_path left by a killed run being trusted as "done" forever after.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  {site['site_id']}: already extracted, skipping")
        return True
    if not os.path.exists(protein_path) or not os.path.exists(ligand_path):
        print(f"  {site['site_id']}: FAILED (missing protonated protein or ligand)")
        return False

    protein_st = gemmi.read_structure(protein_path)
    ligand_st = gemmi.read_structure(ligand_path)
    chain, resname, resnum = site["chain"], site["lig_resname"], int(site["lig_resnum"])

    # If protonate_protein.py had to rename this structure's chains to fit
    # legacy PDB's 1-character limit, protein_protonated/*.pdb (written by
    # PDB2PQR --keep-chain from that renamed file) carries the RENAMED name,
    # not site["chain"]'s original mmCIF one -- translate before matching.
    map_path = chainmap_path(site["pdb_id"])
    if os.path.exists(map_path):
        with open(map_path) as f:
            chain = json.load(f).get(chain, chain)

    merged = merge_ligand_into_protein(protein_st, ligand_st, chain, resnum)
    if merged is None:
        print(f"  {site['site_id']}: FAILED (no ligand residue in {ligand_path})")
        return False

    # Both reference and hit sites are written in their own native crystallographic
    # frame -- no superposition here. A hit's correct rigid-body transform depends on
    # which reference it's being compared against, and a hit can be paired with
    # several different references (see module docstring), so a single site_id-keyed
    # file can never hold more than one pairing's alignment correctly. Superposition
    # is applied per-(ref,hit)-pairing, on demand, in run_prolif.py instead.

    ligand = find_ligand(merged, chain, resnum)
    if ligand is None:
        print(f"  {site['site_id']}: FAILED (merged ligand not found at {chain}/{resname}/{resnum})")
        return False
    lig_pos = [a.pos for a in ligand]

    out_st = gemmi.Structure()
    out_model = gemmi.Model("1")
    n_residues = 0
    for model in merged:
        for mchain in model:
            kept = [res.clone() for res in mchain
                    if any(any(a.pos.dist(lp) <= RADIUS for lp in lig_pos) for a in res)]
            if not kept:
                continue
            new_chain = gemmi.Chain(mchain.name)
            for res in kept:
                new_chain.add_residue(res, -1)
            out_model.add_chain(new_chain)
            n_residues += len(kept)
        break
    out_st.add_model(out_model)
    out_st.setup_entities()
    out_st.write_pdb(out_path)

    bonds = _parse_ligand_bonds(ligand_path)
    n_bonded_atoms = _write_ligand_conect(out_path, bonds, chain, resnum)
    conect_note = f", {n_bonded_atoms} ligand atoms CONECT'd" if n_bonded_atoms else ", NO ligand CONECT written"
    print(f"  {site['site_id']}: {n_residues} residues -> {os.path.basename(out_path)}{conect_note}")
    return True


def main():
    with open(MANIFEST_PATH) as f:
        sites = list(csv.DictReader(f))

    total = len(sites)
    print(f"Extracting {total} pockets (radius={RADIUS} A) from merged protein+ligand structures...")
    t0 = time.time()
    ok = fail = 0
    for i, site in enumerate(sites, 1):
        try:
            success = extract_pocket(site)
        except Exception as e:
            print(f"  {site['site_id']}: FAILED {type(e).__name__}: {e}")
            success = False
        ok += success
        fail += not success
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i) if i else 0
        print(f"    [{i}/{total} {100*i/total:3.0f}%  ok={ok} fail={fail}] ETA {fmt_eta(eta)}")
    print(f"\nDone. {ok}/{total} pockets extracted, {fail} failed.")


if __name__ == "__main__":
    main()
