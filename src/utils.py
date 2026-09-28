"""Small shared helpers used across the pipeline scripts.

These were previously copy-pasted into individual scripts (ETA formatting, WSL
path conversion, structure-file lookup, the TM-align PDB transform, etc.). They
are collected here so there is a single source of truth. Import with the usual
`sys.path`-to-src convention the scripts already use, e.g.:

    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
    from utils import fmt_eta, to_wsl_path
"""

import hashlib
import os


def dist(a, b):
    """Euclidean distance between two 3D points (any indexable of length 3)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def fmt_eta(seconds):
    """Format a duration as a compact string: '1h05m', '3m20s', or '45s'."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def get_hash(seq, length=8):
    """Short, stable MD5 hex digest (first `length` chars) of a string."""
    return hashlib.md5(seq.encode()).hexdigest()[:length]


def to_wsl_path(win_path):
    """Convert a Windows path ('D:\\a\\b') to its WSL form ('/mnt/d/a/b').

    Falsy input is returned unchanged; a drive-less path just gets its
    separators normalized to forward slashes. Pass an absolute path
    (os.path.abspath) if the caller might hand in a relative one.
    """
    if not win_path:
        return win_path
    if ":" in win_path:
        drive, rest = win_path.split(":", 1)
        return f"/mnt/{drive.lower()}{rest.replace('\\', '/')}"
    return win_path.replace("\\", "/")


def find_structure_in_dir(pdb_id, directory):
    """Return the path to <PDBID>.pdb (preferred) or .cif in `directory`, else None."""
    for ext in (".pdb", ".cif"):
        path = os.path.join(directory, f"{pdb_id.upper()}{ext}")
        if os.path.exists(path):
            return path
    return None


def apply_transformation(input_path, matrix, chain_id):
    """Apply a TM-align matrix to every atom in a PDB file, returning the new lines.

    `matrix` is {'t': [3], 'u': [3][3]}; each ATOM/HETATM coordinate is mapped
    x' = t + u.x and the record's chain is forced to `chain_id`. Lines that
    can't be parsed as coordinates are passed through unchanged. Returns None
    for a .cif input -- callers must convert CIF->PDB atoms first.
    """
    if input_path.endswith(".cif"):
        return None
    t = matrix["t"]
    u = matrix["u"]
    transformed_lines = []
    with open(input_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    # PDB coord columns: 30-38, 38-46, 46-54
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    nx = t[0] + u[0][0] * x + u[0][1] * y + u[0][2] * z
                    ny = t[1] + u[1][0] * x + u[1][1] * y + u[1][2] * z
                    nz = t[2] + u[2][0] * x + u[2][1] * y + u[2][2] * z
                    new_line = list(line)
                    new_line[30:54] = list(f"{nx:8.3f}{ny:8.3f}{nz:8.3f}")
                    if len(new_line) > 21:
                        new_line[21] = chain_id
                    transformed_lines.append("".join(new_line))
                except (ValueError, IndexError):
                    transformed_lines.append(line)
            elif line.startswith("TER"):
                new_line = list(line)
                if len(new_line) > 21:
                    new_line[21] = chain_id
                transformed_lines.append("".join(new_line))
    return transformed_lines
