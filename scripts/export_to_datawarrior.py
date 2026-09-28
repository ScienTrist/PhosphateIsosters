import json
import os
import csv
import glob
try:
    import urllib.request as _urllib
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

CRYSTALLOGRAPHIC_SOLVENTS = {
    # Carboxylate buffers / precipitants
    "ACT", "FMT", "CIT", "TLA", "MLI", "GOL", "EDO", "PEG",
    # Sulfonate / MES-type buffers
    "MES", "EPE", "BES", "TRS", "PIP", "CAC",
    # Other common solvents
    "DMS", "DMF", "EOH", "MOH", "IPA", "MPD", "FLC",
    # Polyols
    "PGE", "P6G", "PE4", "PE8",
}

_rcsb_cache = {}

def _fetch_smiles_rcsb(lig_id):
    """Fetch canonical SMILES from RCSB for a CCD code not in the local file."""
    if not _HAS_URLLIB:
        return ""
    if lig_id in _rcsb_cache:
        return _rcsb_cache[lig_id]
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{lig_id}"
        with _urllib.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        smiles = ""
        for entry in data.get("pdbx_chem_comp_descriptor", []):
            if entry.get("type") == "SMILES_CANONICAL" and entry.get("program") == "OpenEye OEToolkits":
                smiles = entry.get("descriptor", "")
                break
        if not smiles:
            for entry in data.get("pdbx_chem_comp_descriptor", []):
                if "SMILES" in entry.get("type", ""):
                    smiles = entry.get("descriptor", "")
                    break
        _rcsb_cache[lig_id] = smiles
        return smiles
    except Exception:
        _rcsb_cache[lig_id] = ""
        return ""


def _load_confirmed_keys(confirmed_dir):
    """
    Read filenames from plif_confirmed/ and return a set of match keys.
    Key is (ref_id, hit_id, chain, ref_lig, ref_num) — the full site identity
    encoded in every hit filename: hit_{ref}_{hit}_{chain}_{lig}_{num}.cif
    """
    keys = set()
    for fpath in glob.glob(os.path.join(confirmed_dir, "hit_*.cif")) + \
                 glob.glob(os.path.join(confirmed_dir, "hit_*.pdb")):
        parts = os.path.basename(fpath).replace(".cif", "").replace(".pdb", "").split("_")
        if len(parts) >= 6:
            keys.add((parts[1].upper(), parts[2].upper(),
                      parts[3].upper(), parts[4].upper(), parts[5]))
    return keys


def export_for_datawarrior(results_path, smiles_path, output_path, confirmed_only=False, confirmed_dir=None):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    # 1. Load local SMILES mapping
    smiles_map = {}
    if os.path.exists(smiles_path):
        print("Loading SMILES data...")
        with open(smiles_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    smiles_map[parts[1].strip()] = parts[0].strip()

    # 2. Load PLIF results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # 3. Filter to confirmed hits only if requested
    if confirmed_only and confirmed_dir:
        if not os.path.exists(confirmed_dir):
            print(f"Warning: confirmed_dir {confirmed_dir} not found - exporting all results.")
        else:
            confirmed_keys = _load_confirmed_keys(confirmed_dir)
            has_site_keys  = results and "ref_chain" in results[0]

            before = len(results)
            if has_site_keys:
                results = [r for r in results
                           if (r["ref"].upper(), r["hit"].upper(),
                               r.get("ref_chain","").upper(),
                               r.get("ref_lig","").upper(),
                               str(r.get("ref_num",""))) in confirmed_keys]
            else:
                # Old JSON format: match on (ref, hit) pair only
                pair_keys = {(k[0], k[1]) for k in confirmed_keys}
                results   = [r for r in results
                             if (r["ref"].upper(), r["hit"].upper()) in pair_keys]

            print(f"Confirmed-only filter: {before} -> {len(results)} entries "
                  f"({len(confirmed_keys)} files in plif_confirmed/)")

    # 4. Fetch SMILES from RCSB for any mimic code not in the local file
    missing = list(dict.fromkeys(r['mimic'] for r in results if r['mimic'] not in smiles_map))
    if missing:
        print(f"Fetching SMILES from RCSB for {len(missing)} codes not in local file: {missing}")
        for lig_id in missing:
            smiles = _fetch_smiles_rcsb(lig_id)
            if smiles:
                smiles_map[lig_id] = smiles
                print(f"  {lig_id}: fetched OK")
            else:
                print(f"  {lig_id}: not found in RCSB")

    # 5. Write TSV for DataWarrior
    fieldnames = [
        'SMILES', 'ID', 'Ref_PDB', 'Hit_PDB', 'PLIF_Score', 'RMSD',
        'Ref_Interactions', 'Hit_Interactions', 'EC_Class', 'Crystallographic_Solvent',
        'Phosphate_Index', 'Ref_Phosphates', 'Phosphates_Mimicked',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for r in results:
            writer.writerow({
                'SMILES':                   smiles_map.get(r['mimic'], ''),
                'ID':                       r['mimic'],
                'Ref_PDB':                  r['ref'],
                'Hit_PDB':                  r['hit'],
                'PLIF_Score':               round(r['score'], 3),
                'RMSD':                     round(r['rmsd'], 3),
                'Ref_Interactions':         r.get('ref_inters', 0),
                'Hit_Interactions':         r.get('hit_inters', 0),
                'EC_Class':                 r.get('ec', ''),
                'Crystallographic_Solvent': 'Yes' if r['mimic'] in CRYSTALLOGRAPHIC_SOLVENTS else 'No',
                'Phosphate_Index':          r.get('ref_p_idx', ''),
                'Ref_Phosphates':           r.get('n_ref_phosphates', ''),
                'Phosphates_Mimicked':      r.get('n_phosphates_mimicked', ''),
            })

    print(f"Exported {len(results)} entries to {output_path}")


if __name__ == "__main__":
    import argparse
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    MOTIF_DIR    = os.path.join(PROJECT_ROOT, "results", "motif_analysis")

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Export all PLIF results, not just plif_confirmed hits")
    args = parser.parse_args()

    export_for_datawarrior(
        results_path   = os.path.join(MOTIF_DIR, "plif_results.json"),
        smiles_path    = os.path.join(PROJECT_ROOT, "data", "Components-smiles-stereo-cactvs.smi"),
        output_path    = os.path.join(MOTIF_DIR, "isostere_datawarrior.txt"),
        confirmed_only = not args.all,
        confirmed_dir  = os.path.join(MOTIF_DIR, "hits", "holo", "plif_confirmed"),
    )
