import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

FULL_LIST_CSV = os.path.join(PROJECT_ROOT, "results", "motif_analysis", "isostere_full_list.csv")
# TMALIGN_JSON used to be a hardcoded path to tmalign_results_..._corrected.json.
# That file turned out to be neither a clean superset of the original run nor a
# complete run of the current pairing logic -- it silently dropped ~9,700 pairs
# that should have been present while still carrying ~17,900 pairs left over
# from an abandoned pairing scheme (see results/archived_tmalign_results/README.md).
# It has been archived and must never be the silent default again. Every entry
# point (resolve_pairs.py, run_prolif.py, main_prolif.py/discover_candidates.py)
# now calls set_tmalign_json(path) explicitly before the first get_transformation()
# call -- there is deliberately no working default.
TMALIGN_JSON = None
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "plip_isostere")

os.makedirs(OUT_DIR, exist_ok=True)


def find_pdb_path(pdb_id, category, allow_cif=False):
    """Returns the path to the raw structure file for pdb_id in the given category
    ("references" or "hits"), or None if nothing usable is available.

    By default only .pdb is considered, because PLIP's PDBParser only reads
    PDB-format text -- callers that feed this path into PLIP (score_pairs.py,
    generate_report_example.py, generate_prolif_example.py) must NOT pass
    allow_cif=True, or PLIP will fail on a .cif-only entry.

    allow_cif=True additionally falls back to the .cif copy when no .pdb is
    present -- safe for callers that only use gemmi.read_structure() (which
    reads both formats natively), e.g. resolve_pairs.py. Every .pdb-only
    caller's behavior is completely unchanged."""
    pdb_path = os.path.join(PROJECT_ROOT, "data", "structures", category, f"{pdb_id.upper()}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    if allow_cif:
        cif_path = os.path.join(PROJECT_ROOT, "data", "structures", category, f"{pdb_id.upper()}.cif")
        if os.path.exists(cif_path):
            return cif_path
    return None


_tmalign_cache = {}  # path -> loaded dict, so switching files mid-process is safe


def set_tmalign_json(path):
    """Selects which tmalign_results_*.json get_transformation() reads for the
    remainder of the process. Must be called once before the first
    get_transformation()/load_tmalign_index() call -- see TMALIGN_JSON's
    module-level comment for why there's no silent default anymore."""
    global TMALIGN_JSON
    TMALIGN_JSON = path


def load_tmalign_index(path=None):
    """Loads results/tmalign_results_*.json once (per distinct path) and
    returns the raw dict (ref_id -> list of {rmsd, tm_score_1, tm_score_2,
    transformation:{t,u}, hit})."""
    path = path or TMALIGN_JSON
    if path is None:
        raise RuntimeError(
            "common.TMALIGN_JSON is not configured. Call common.set_tmalign_json(path) "
            "once at process start -- see that function's docstring."
        )
    if path not in _tmalign_cache:
        with open(path, "r") as f:
            _tmalign_cache[path] = json.load(f)
    return _tmalign_cache[path]


def get_transformation(ref_id, hit_id, tmalign_json_path=None):
    """Returns (t, u) for the given (ref_id, hit_id) pair, or None if not found."""
    idx = load_tmalign_index(tmalign_json_path)
    for entry in idx.get(ref_id.upper(), []):
        if entry.get("hit", "").upper() == hit_id.upper():
            return entry["transformation"]["t"], entry["transformation"]["u"]
    return None


def apply_transform(t, u, xyz):
    """Maps a hit-frame coordinate into the reference frame: x_ref = u . x_hit + t
    (same convention batch_motif_extraction.py uses when superimposing hit onto ref)."""
    x, y, z = xyz
    return (
        t[0] + u[0][0] * x + u[0][1] * y + u[0][2] * z,
        t[1] + u[1][0] * x + u[1][1] * y + u[1][2] * z,
        t[2] + u[2][0] * x + u[2][1] * y + u[2][2] * z,
    )
