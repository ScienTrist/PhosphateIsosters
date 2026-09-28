"""
Protonates each freshly downloaded FULL structure's protein+water using
PDB2PQR (pKa-aware, via PROPKA) instead of OpenBabel -- unlike OpenBabel's
generic per-residue templates, PDB2PQR resolves each titratable residue
(His tautomer HID/HIE/HIP, Asp/Glu/Lys/Arg ionization) individually based on
its local hydrogen-bonding environment, which is exactly the caveat ProLIF's
own docs call out ("RDKit... may not recognize [histidine] correctly").

Ligand (HETATM) protonation is intentionally NOT trusted from this step --
PDB2PQR has no chemical knowledge of arbitrary ligands, it just passes
heavy atoms through and may add unreliable guessed hydrogens. The target
ligand of each site is separately protonated properly in protonate_ligand.py
and spliced in during pocket extraction, replacing whatever PDB2PQR did to
it.

Some raw structures only exist locally as .cif (a few aren't even available
as legacy .pdb from RCSB at all). PDB2PQR nominally reads CIF natively, but
its CIF parser has real bugs on real files -- confirmed on this project's
own data: 6/6 tested CIF-only structures crashed, in two distinct ways (a
missing ORIGX transform block, missing entity_src_gen metadata both raise
uncaught exceptions in pdb2pqr's cif.py). All were well under legacy PDB
format's size limits (62 chains / 99,999 atoms), so this isn't a "too big
for PDB" problem -- converting via gemmi first and handing PDB2PQR the
converted .pdb sidesteps the buggy parser entirely; validated 6/6 success
on the same structures that crashed via native CIF input.
"""
import json
import os
import subprocess
import sys

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROLIF_V2_ROOT, "data", "raw")
PROTEIN_DIR = os.path.join(PROLIF_V2_ROOT, "data", "protein_protonated")
CONVERTED_DIR = os.path.join(PROLIF_V2_ROOT, "data", "raw_converted")

os.makedirs(PROTEIN_DIR, exist_ok=True)

PH = 7.4


def chainmap_path(pdb_id):
    """Sidecar written next to a chain-renamed conversion (see
    _remap_long_chain_names). extract_pockets.py reads this back to translate
    a site's original (mmCIF) chain name to whatever this file renamed it to,
    since PDB2PQR's --keep-chain propagates the renamed chain straight through
    into protein_protonated/*.pdb."""
    return os.path.join(CONVERTED_DIR, f"{pdb_id}.chainmap.json")


def _remap_long_chain_names(st, pdb_id):
    """Legacy PDB's chain-ID field holds exactly 1 character; mmCIF chain
    names (auth_asym_id) can be longer (e.g. "AAA"), which gemmi's
    write_pdb refuses outright (RuntimeError: chain name too long) rather
    than silently truncating. Renames any offending chain in-place to a
    free single alphanumeric character (same greedy first-letter-then-
    fallback scheme as batch_motif_extraction.py's get_safe_chain_name) and
    persists the {original: renamed} mapping to chainmap_path(pdb_id) so
    every other pipeline step that still refers to the ORIGINAL chain name
    (site["chain"], sourced from the raw mmCIF -- unaffected by this
    rename) can translate it. No-op, no file written, if every chain
    already fits."""
    mapping = {}
    used_chars = set()
    for model in st:
        for chain in model:
            if len(chain.name) <= 1:
                used_chars.add(chain.name)
        break

    for model in st:
        for chain in model:
            if len(chain.name) <= 1:
                continue
            name = chain.name
            if name in mapping:
                chain.name = mapping[name]
                continue
            preferred = name[0]
            safe = None
            if preferred not in used_chars:
                safe = preferred
            else:
                for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz":
                    if char not in used_chars:
                        safe = char
                        break
            if safe is None:
                safe = "?"  # exhausted 62 letters/digits in one structure -- never observed
            mapping[name] = safe
            used_chars.add(safe)
            chain.name = safe
        break

    if mapping:
        os.makedirs(CONVERTED_DIR, exist_ok=True)
        with open(chainmap_path(pdb_id), "w") as f:
            json.dump(mapping, f)
    return mapping


def _resolve_input(pdb_id, search_dirs=None):
    """Returns a .pdb path PDB2PQR can safely read. If only a .cif is present
    locally, converts it to data/raw_converted/{pdb_id}.pdb via gemmi (cached
    -- skipped if already converted) rather than handing PDB2PQR the .cif
    directly. Returns None if neither format is present in any of search_dirs.

    search_dirs: directories to look in, tried in order. Defaults to [RAW_DIR]
    -- every existing caller keeps its old behavior unchanged. Passing the main
    pipeline's data/structures/{references,hits} dirs here lets main_prolif.py's
    discover_candidates.py protonate structures already downloaded by the main
    pipeline without first copying them into ProLIF_v2/data/raw."""
    search_dirs = search_dirs or [RAW_DIR]

    for d in search_dirs:
        pdb_path = os.path.join(d, f"{pdb_id}.pdb")
        if os.path.exists(pdb_path):
            return pdb_path

    cif_path = None
    for d in search_dirs:
        candidate = os.path.join(d, f"{pdb_id}.cif")
        if os.path.exists(candidate):
            cif_path = candidate
            break
    if cif_path is None:
        return None

    os.makedirs(CONVERTED_DIR, exist_ok=True)
    converted_path = os.path.join(CONVERTED_DIR, f"{pdb_id}.pdb")
    # Existence alone isn't enough to trust the cache: a run killed/crashed
    # mid-write (e.g. a forced process kill under this same 8-worker
    # ThreadPoolExecutor) leaves a 0-byte stub behind, which os.path.exists()
    # would then treat as "already converted" forever after -- confirmed on
    # this project's own data: 194 files in raw_converted/ were exactly 0
    # bytes, dated from two past interrupted runs (43 from one, 151 from
    # another), each permanently causing PDB2PQR to fail with the misleading
    # "Unable to find file" (PDB2PQR's own wording for "opened it, zero
    # records parsed" -- see pdb2pqr/io.py) on every subsequent run since.
    if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
        return converted_path

    import gemmi
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    _remap_long_chain_names(st, pdb_id)
    st.write_pdb(converted_path)
    return converted_path


def _last_error_line(stderr):
    """Pulls the single most useful line out of a pdb2pqr traceback for a
    one-line live progress display -- the last non-empty line is almost
    always the actual exception message (ValueError/RuntimeError/...); the
    full traceback is preserved separately in the .err.log file for anyone
    who needs to dig into a specific failure later."""
    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1] if lines else "pdb2pqr30 produced no output and no stderr"


def protonate(pdb_id, search_dirs=None, out_dir=PROTEIN_DIR):
    """Returns (ok, msg): msg is the output filename on success, or a short
    one-line failure reason on failure (never None, always safe to print).

    search_dirs: see _resolve_input. out_dir: where {pdb_id}_protein.pdb/.pqr
    get written -- defaults to ProLIF_v2/data/protein_protonated so existing
    callers are unaffected. Protonation output depends only on (pdb_id, force
    field, pH), never on which pipeline run asked for it, so pointing a
    different caller's out_dir at this same default directory is deliberate,
    free cross-run caching, not a coincidence."""
    os.makedirs(out_dir, exist_ok=True)
    out_pdb = os.path.join(out_dir, f"{pdb_id}_protein.pdb")
    out_pqr = os.path.join(out_dir, f"{pdb_id}.pqr")
    # Size check, not just existence -- a run killed mid-subprocess (e.g. a
    # forced Stop-Process during a long rerun) can leave a 0-byte/truncated
    # out_pdb behind, which would otherwise be trusted as "done" forever
    # after. Same bug class as _resolve_input's converted_path check above.
    if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
        return True, "already protonated, skipping"

    in_path = _resolve_input(pdb_id, search_dirs)
    if in_path is None:
        return False, f"no .pdb or .cif for {pdb_id} in {search_dirs or [RAW_DIR]}"

    cmd = [
        "pdb2pqr30", "--ff", "AMBER", "--with-ph", str(PH), "--keep-chain",
        "--pdb-output", out_pdb, in_path, out_pqr,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_pdb):
        err_log = os.path.join(out_dir, f"{pdb_id}.err.log")
        with open(err_log, "w") as f:
            f.write(result.stderr)
        return False, f"{_last_error_line(result.stderr)} (full: {os.path.basename(err_log)})"
    return True, os.path.basename(out_pdb)


def main():
    pdb_ids = sorted({f.rsplit(".", 1)[0] for f in os.listdir(RAW_DIR) if f.endswith((".pdb", ".cif"))})
    print(f"Protonating {len(pdb_ids)} structures with PDB2PQR (pH {PH})...")
    ok = 0
    for pdb_id in pdb_ids:
        success, msg = protonate(pdb_id)
        print(f"  {pdb_id}: {'OK' if success else 'FAILED'} ({msg})")
        ok += success
    print(f"\nDone. {ok}/{len(pdb_ids)} protonated.")
    if ok < len(pdb_ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
