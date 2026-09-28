import os
import sqlite3

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "review.db")
POCKETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "pockets",
)
ALIGNED_HITS_DIR = os.path.join(DATA_DIR, "aligned_hits")
# Review-app-local, display-corrected copies of reference pockets: the
# canonical data/pockets/*.pdb (POCKETS_DIR above) shows reference phosphate
# groups in their neutral, fully-protonated textbook form -- correct for
# nothing but display (ProLIF's own scoring already deprotonates them
# in-memory via phosphate_ifp.py's _deprotonate_ionizable_oxygens, on a
# throwaway fragment, never touching this file). build_pool.py writes a
# physiologically-corrected copy of each unique reference site's pocket here
# (terminal phosphate hydroxyls stripped to their anionic form) so PyMOL/the
# webapp show the same charge state ProLIF actually scores against, without
# risking the canonical file the core pipeline depends on for scoring.
REF_POCKETS_DIR = os.path.join(DATA_DIR, "ref_pockets_display")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets readers (GET /api/pair, /api/structure/*, ...) proceed
    # without blocking on a concurrent writer, and busy_timeout makes a
    # writer that does collide (two reviewers submitting a rating in the
    # same instant) retry for up to 5s instead of failing immediately with
    # "database is locked" -- matters once multiple colleagues are actually
    # using this at once, not just during solo testing.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn
