"""
Protonates each manifest site's ligand following ProLIF's own documented
recipe (https://prolif.readthedocs.io/en/stable/notebooks/pdb.html#pdb,
"SMILES + PDB coordinates" option): combine the ligand's reference SMILES
(fetched by fetch_ligand_smiles.py) with its crystal coordinates via
rdkit.Chem.AllChem.AssignBondOrdersFromTemplate, so the ligand gets correct
bond orders/formal charges instead of guessed-from-distance ones, then add
explicit hydrogens with Chem.AddHs(addCoords=True).

Any pre-existing hydrogens in the deposited file are discarded and
regenerated from scratch (MolFromPDBFile's default removeHs=True) for
consistent treatment across all ligands, matching ProLIF's documented recipe
exactly rather than trusting depositor-supplied placements on some ligands
only.

Falls back to naive distance-based bond perception (no template) when the
template match fails -- this happens when the crystal structure is missing
an atom relative to the ideal chemical component (e.g. a disordered/
unresolved terminal group), which makes exact atom-count matching
impossible. The fallback still produces a usable protonated ligand, just
without the template's correct bond orders/charges.

Chem.AddHs() does not stamp PDBResidueInfo (chain/resname/resnum) onto the
new H atoms it creates, so _stamp_missing_residue_info fills that in
manually -- required so extract_pockets.py's residue-based radius search
groups these ligand hydrogens with the rest of the ligand residue.

Ligand ionization state: RCSB's reference SMILES represents every ligand in
its neutral, fully-protonated "textbook" form (confirmed directly: AMP/ADP/
ATP/GTP/TMP/ANP/FMN/DTP's phosphates, and every carboxylic/sulfonic acid
checked, all write -OH, never [O-]) -- physiologically wrong for any
ionizable group at pH 7.4, and ProLIF's own docs never address this (they
only cover getting bond orders/explicit-Hs onto a ligand, nothing about
formal charges). This used to be patched piecemeal, downstream, in-memory
only, one functional group at a time (phosphate_ifp.py's now-removed
_deprotonate_ionizable_oxygens covered reference phosphates only;
run_prolif.py's now-removed _deprotonate_ligand_ionizable_groups covered hit
carboxylic/sulfonic acids only) -- neither patch ever touched this file, so
PHIP's viewer (which loads pocket PDBs built from THIS script's output)
always showed the wrong, neutral form regardless of what the score was
computed on, and neither patch covered other acidic groups (tetrazole,
benzotriazole, ...) at all.

Replaced here with one general, pH-based ionizer, Dimorphite-DL
(https://github.com/durrantlab/dimorphite_dl), run on the reference SMILES
via _ionize_smiles() BEFORE AssignBondOrdersFromTemplate, at
min_ph=max_ph=IONIZE_PH -- covers every standard ionizable functional group
(phosphates/phosphonates, carboxylic/sulfonic acids, tetrazoles, imidazoles,
amines, guanidines, ...) uniformly, with no per-motif code, and needs no
change to ProLIF's own interaction detection: ProLIF's Anionic/Cationic
SMARTS key off any atom carrying an explicit formal charge, element-agnostic
-- confirmed directly against prolif.interactions.interactions's source, not
a per-functional-group pattern. Confirmed separately that
AssignBondOrdersFromTemplate's heavy-atom-topology match is indifferent to
the template's ionization state (only the affected atom's implicit-H
count/formal charge differs, not the heavy-atom graph, so template matching
behaves identically ionized or not), and that Chem.AddHs() correctly skips
adding a hydrogen back onto an atom the template already marked as charged.
Since the resulting charge/H-state is baked into THIS file's output PDB (not
just an in-memory fingerprint), this also fixes PHIP's display, not just the
score.

Dimorphite-DL's `precision` argument is a +/- std-dev window around its mean
pKa estimate, NOT an accuracy knob -- a HIGHER value enumerates MORE
candidate states as "plausible", not fewer. At the doc-example default
precision=1.0 it returns multiple states for exactly the groups this fix
exists for (confirmed directly: both mono- and di-anion for a phosphate's
second -OH, and both tautomers of tetrazole and benzotriazole) -- with no
principled way to pick one without re-adding the same kind of per-group
special-casing this change is meant to eliminate. precision=0.0 asks for
just the single mean-pKa-based call instead, which resolved every case
tested (phosphate, carboxylic/sulfonic acid, tetrazole, benzotriazole) to
exactly one state. _ionize_smiles still tolerates >1 result (deterministically
takes the most-ionized/lowest-net-charge one and prints a note) as a safety
net for a genuine tie, but at precision=0.0 this is not expected to fire in
ordinary use -- if it does, that's worth looking at directly rather than
trusting the tie-break silently.

Known residual limitation, found while testing this (not something either
precision setting fixes): Dimorphite-DL's mean-pKa call also deprotonates
the purine ring N-H on guanine-containing ligands (GMP/GDP/GTP and NAD/FAD's
guanine analogs use the same ring system) at pH 7.4, which reads as too
aggressive against guanine's textbook ~9.4 pKa for that proton -- a known
hard case for generic rule-based pKa tools on nucleobase tautomers, not
specific to this integration. Harmless for REFERENCE scoring specifically
(phosphate_group_ifps() only ever extracts atoms within 2.1 A of the
phosphorus, and the purine ring sits far outside that radius), but it DOES
change what PHIP's viewer draws for the ring, and could in principle affect
a HIT ligand's score if a hit happens to carry a similar N-heterocycle
(hits are scored whole-ligand, no radius restriction). Flagging this as a
disclosed, known limitation rather than silently patching around it: fixing
it specifically would mean reintroducing the same per-motif special-casing
this change exists to get away from, for a case that doesn't affect the
phosphate-region scoring this project is actually built around.
"""
import csv
import json
import os

import gemmi
from dimorphite_dl import protonate_smiles
from rdkit import Chem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
SMILES_CACHE_PATH = os.path.join(PROLIF_V2_ROOT, "ligand_smiles.json")
RAW_DIR = os.path.join(PROLIF_V2_ROOT, "data", "raw")
TMP_DIR = os.path.join(PROLIF_V2_ROOT, "data", "ligand_tmp")
OUT_DIR = os.path.join(PROLIF_V2_ROOT, "data", "ligand_protonated")
IONIZE_PH = 7.4

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

_ionized_smiles_cache = {}


def _ionize_smiles(resname, smiles):
    """Returns resname's reference SMILES re-ionized to its physiological
    (IONIZE_PH) state via Dimorphite-DL, covering every standard ionizable
    group generically (see module docstring). Cached per resname: ionization
    state depends only on 2D structure, never on a specific site instance's
    coordinates, and the same resname is looked up for every site that uses
    it (e.g. ~50+ AMP sites) -- no reason to recompute per-instance.

    precision=0.0 (mean-pKa point estimate, not a std-dev window -- see module
    docstring) resolves every group tested to a single state, so the
    more-than-one-result branch below is a safety net for a genuine tie, not
    the expected common case; if it fires, that's worth checking directly
    rather than trusting the tie-break silently."""
    if resname in _ionized_smiles_cache:
        return _ionized_smiles_cache[resname]
    variants = protonate_smiles(smiles, ph_min=IONIZE_PH, ph_max=IONIZE_PH, precision=0.0, max_variants=8)
    if not variants:
        result = smiles  # Dimorphite-DL found nothing ionizable -- use as-is
    else:
        def net_charge(smi):
            mol = Chem.MolFromSmiles(smi)
            return Chem.GetFormalCharge(mol) if mol is not None else 0
        result = min(variants, key=net_charge)
        if len(variants) > 1:
            print(f"  [ionization] {resname}: Dimorphite-DL returned {len(variants)} pH-{IONIZE_PH} "
                  f"states, using most-ionized: {result}")
    _ionized_smiles_cache[resname] = result
    return result


def extract_ligand_pdb(pdb_id, chain_name, lig_resname, lig_resnum, out_path, search_dirs=None):
    """Writes a single-residue PDB for this ligand, deduplicating altlocs by
    keeping only the highest-occupancy copy of each atom name (matches the
    dedup pdb2pqr already does for the protein, so both halves of the
    eventual merge are built from the same-occupancy conformer).

    Some raw structures only exist locally as .cif (no legacy .pdb on RCSB at
    all, e.g. 9RJH 404s on the .pdb URL). Reads the raw .cif directly with
    gemmi rather than going through protonate_protein._resolve_input()'s
    cif-to-pdb conversion: that conversion writes legacy PDB format, whose
    3-character residue-name field silently truncates the newer 5-character
    extended CCD ligand codes (e.g. A1JGW -> A1J on 9RJH, confirmed by
    comparing the converted .pdb against the source .cif) -- fine for
    PDB2PQR's protein-only protonation, which doesn't trust ligand identity
    anyway, but fatal for finding this ligand by its real resname. Reading
    the .cif directly keeps the full-length ligand code intact.

    search_dirs: directories to look for {pdb_id}.pdb/.cif in, tried in
    order. Defaults to [RAW_DIR] -- every existing caller keeps its old
    behavior unchanged."""
    search_dirs = search_dirs or [RAW_DIR]
    st = None
    for d in search_dirs:
        pdb_path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(pdb_path):
            st = gemmi.read_structure(pdb_path)
            break
        cif_path = os.path.join(d, f"{pdb_id}.cif")
        if os.path.exists(cif_path):
            st = gemmi.read_structure(cif_path)
            break
    if st is None:
        return False, f"no .pdb or .cif for {pdb_id} in {search_dirs}"
    residue = None
    for model in st:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                if res.name == lig_resname and res.seqid.num == lig_resnum:
                    residue = res
        break
    if residue is None:
        return False, f"{lig_resname} {lig_resnum} not found in chain {chain_name} of {pdb_id}"

    best_by_name = {}
    for atom in residue:
        prev = best_by_name.get(atom.name)
        if prev is None or atom.occ > prev.occ:
            best_by_name[atom.name] = atom

    out_st = gemmi.Structure()
    model = gemmi.Model("1")
    # Legacy PDB's chain-ID field holds exactly 1 character; mmCIF chain names
    # (chain_name here) can be longer, which gemmi's write_pdb refuses outright.
    # merge_ligand_into_protein() (extract_pockets.py) takes this file's sole
    # residue unconditionally without ever checking its chain name, so what we
    # call it here has no effect on correctness -- just truncate to fit.
    out_chain = gemmi.Chain(chain_name[:1] if chain_name else "L")
    new_res = gemmi.Residue()
    new_res.name = lig_resname
    new_res.seqid = gemmi.SeqId(lig_resnum, " ")
    for atom in best_by_name.values():
        a = atom.clone()
        a.altloc = "\0"
        new_res.add_atom(a)
    out_chain.add_residue(new_res)
    model.add_chain(out_chain)
    out_st.add_model(model)
    out_st.write_pdb(out_path)
    return True, None


def _stamp_missing_residue_info(mol, chain, resname, resnum):
    """Fills in PDBResidueInfo for atoms AddHs created (it copies coordinates
    but not residue metadata) so every atom -- original heavy atoms and the
    newly added hydrogens alike -- carries the same (chain, resname, resnum)."""
    counts = {}
    for atom in mol.GetAtoms():
        if atom.GetPDBResidueInfo() is not None:
            continue
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f" {symbol}{counts[symbol]}".ljust(4)[:4])
        info.SetResidueName(resname)
        info.SetResidueNumber(resnum)
        info.SetChainId(chain)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)


def protonate_ligand(site, smiles_cache, search_dirs=None):
    """Returns (outcome, msg): outcome is one of cached/template/fallback/failed;
    msg is the output filename on success or a short one-line failure reason
    (never None, always safe to print).

    search_dirs: see extract_ligand_pdb -- defaults to [RAW_DIR], unchanged
    behavior for existing callers."""
    tmp_path = os.path.join(TMP_DIR, f"{site['site_id']}_raw.pdb")
    out_path = os.path.join(OUT_DIR, f"{site['site_id']}_ligand.pdb")
    # Size check, not just existence -- guards against a 0-byte/truncated
    # out_path left by a killed run being trusted as "done" forever after.
    # Legitimate single-atom ligands (ions, CO, O2) are still ~80+ bytes,
    # so 0 is an unambiguous corruption signal, not a false positive.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "cached", "already protonated, skipping"

    chain, resname, resnum = site["chain"], site["lig_resname"], int(site["lig_resnum"])
    found, err = extract_ligand_pdb(site["pdb_id"], chain, resname, resnum, tmp_path, search_dirs)
    if not found:
        return "failed", err

    pdb_mol = Chem.MolFromPDBFile(tmp_path)  # removeHs=True default: strip any deposited H, rebuild from scratch
    if pdb_mol is None:
        return "failed", "RDKit could not parse extracted ligand"

    smiles = smiles_cache.get(resname)
    outcome = "template"
    mol = None
    if smiles:
        ionized_smiles = _ionize_smiles(resname, smiles)
        # Falls back to the neutral template if Dimorphite-DL's output SMILES
        # itself fails to parse (rare) -- still better than crashing, same
        # "degrade gracefully" spirit as the template/fallback split below.
        template = Chem.MolFromSmiles(ionized_smiles) or Chem.MolFromSmiles(smiles)
        try:
            mol = AssignBondOrdersFromTemplate(template, pdb_mol)
        except Exception:
            mol = None
    if mol is None:
        outcome = "fallback"
        mol = pdb_mol
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return "failed", f"fallback sanitization: {e}"

    mol_h = Chem.AddHs(mol, addCoords=True)
    _stamp_missing_residue_info(mol_h, chain, resname, resnum)
    Chem.MolToPDBFile(mol_h, out_path)
    return outcome, os.path.basename(out_path)


def main():
    with open(MANIFEST_PATH) as f:
        sites = list(csv.DictReader(f))
    smiles_cache = json.load(open(SMILES_CACHE_PATH))

    print(f"Protonating {len(sites)} ligands...")
    outcomes = {}
    for site in sites:
        outcome, msg = protonate_ligand(site, smiles_cache)
        print(f"  {site['site_id']}: {outcome} ({msg})")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    print(f"\nDone. {outcomes}")


if __name__ == "__main__":
    main()
