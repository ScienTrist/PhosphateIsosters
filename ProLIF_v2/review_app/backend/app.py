"""
Isostere-validity review app backend.

Colleagues open the app, enter their name + the access code you give them,
pick how many pairs to rate, and are shown one ref/hit structure pair at a
time (ligands + pocket, hit already TM-aligned into the reference's frame
by build_pool.py) to score 1-5 with an optional comment. Every submission
is appended to `ratings` (never overwritten), keyed by reviewer name and
pair_id, so /api/export can join back to prolif_plif_score_no_vdw/
prolif_plif_score_with_vdw/ref_n_bits_novdw for the threshold analysis this
whole thing exists to support.

The reviewer never sees either prolif_plif_score or ref_n_bits_novdw --
showing them would anchor their judgment to the very numbers being
calibrated.

Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import random
import secrets
import sqlite3
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from db import get_conn, POCKETS_DIR, ALIGNED_HITS_DIR, REF_POCKETS_DIR

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "static")

ACCESS_CODE = os.environ.get("REVIEW_ACCESS_CODE")
ADMIN_TOKEN = os.environ.get("REVIEW_ADMIN_TOKEN")
SECONDS_PER_STRUCTURE = int(os.environ.get("REVIEW_SECONDS_PER_STRUCTURE", "15"))
COUNT_OPTIONS = [100, 150, 200]
MIN_COUNT, MAX_COUNT = 10, 10_000

if not ACCESS_CODE or not ADMIN_TOKEN:
    raise RuntimeError(
        "REVIEW_ACCESS_CODE and REVIEW_ADMIN_TOKEN must both be set via "
        "environment variables before starting this app."
    )

app = FastAPI(title="PHIP — Phosphate Isostere Hit Picker")


class StartRequest(BaseModel):
    reviewer_name: str = Field(min_length=1, max_length=100)
    access_code: str
    requested_count: int = Field(ge=MIN_COUNT, le=MAX_COUNT)


class RatingRequest(BaseModel):
    reviewer_name: str = Field(min_length=1, max_length=100)
    pair_id: str
    # Exactly one of score / wrong_reference_phosphate is set -- see
    # submit_rating(). wrong_reference_phosphate is its own vote, not a 6th
    # score value: it flags that the pipeline picked the wrong phosphate
    # group on the reference ligand to compare against (see the "wrong
    # reference phosphate" section of the user guide), which is a data
    # problem, not a judgment about how good a mimic the hit is.
    score: Optional[int] = Field(default=None, ge=1, le=5)
    wrong_reference_phosphate: bool = False
    comment: Optional[str] = Field(default=None, max_length=2000)


def _pool_size(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]


def _ensure_wrong_phosphate_column():
    """One-time migration for review.db files created before the wrong-
    reference-phosphate vote existed -- build_pool.py's own CREATE TABLE IF
    NOT EXISTS is a no-op against an already-existing ratings table, so a
    live DB needs this ALTER instead. No-op (and safe to call every startup)
    once the column is there."""
    conn = get_conn()
    try:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ratings'"
        ).fetchone():
            return  # fresh checkout, before the first build_pool.py run -- nothing to migrate yet
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(ratings)")}
        if "wrong_reference_phosphate" not in cols:
            conn.execute("ALTER TABLE ratings ADD COLUMN wrong_reference_phosphate INTEGER DEFAULT 0")
            conn.commit()
    finally:
        conn.close()


def _ensure_pocket_rmsd_column():
    """One-time migration for review.db files built before pocket_rmsd
    existed on `pairs` (i.e. anything built by build_pool.py rather than
    build_pool_from_discovery.py) -- same reasoning as
    _ensure_wrong_phosphate_column above."""
    conn = get_conn()
    try:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pairs'"
        ).fetchone():
            return  # fresh checkout, before the first pool build -- nothing to migrate yet
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(pairs)")}
        if "pocket_rmsd" not in cols:
            conn.execute("ALTER TABLE pairs ADD COLUMN pocket_rmsd REAL")
            conn.commit()
    finally:
        conn.close()


_ensure_wrong_phosphate_column()
_ensure_pocket_rmsd_column()


@app.get("/api/config")
def config():
    conn = get_conn()
    try:
        total = _pool_size(conn)
        shared = conn.execute("SELECT COUNT(*) FROM pairs WHERE is_shared=1").fetchone()[0]
    finally:
        conn.close()
    return {
        "seconds_per_structure": SECONDS_PER_STRUCTURE,
        "count_options": COUNT_OPTIONS,
        "pool_size": total,
        "shared_core_size": shared,
    }


def _shared_pairs(conn) -> list:
    return [r[0] for r in conn.execute("SELECT pair_id FROM pairs WHERE is_shared=1")]


@app.post("/api/start")
def start(req: StartRequest):
    if not secrets.compare_digest(req.access_code, ACCESS_CODE):
        raise HTTPException(status_code=403, detail="Wrong access code")

    reviewer_name = req.reviewer_name.strip()
    if not reviewer_name:
        raise HTTPException(status_code=400, detail="Name required")

    conn = get_conn()
    try:
        existing = conn.execute(
            "SELECT * FROM reviewers WHERE reviewer_name=?", (reviewer_name,)
        ).fetchone()

        # Every reviewer draws exclusively from the shared, weighted-sampled
        # core (is_shared=1, chosen by build_pool_from_discovery.py's own
        # Gaussian-weighted draw toward the pool's median score/ref_n_bits --
        # see its docstring) so all reviewers see the *same* structures,
        # capped at however many of those actually exist. No more
        # per-reviewer individualized top-up: that used to make each
        # reviewer's non-shared pairs differ from everyone else's.
        shared = _shared_pairs(conn)
        rng = random.Random(reviewer_name)
        rng.shuffle(shared)
        shared_core_size = len(shared)

        if existing is None:
            target = min(req.requested_count, shared_core_size)
            conn.execute(
                "INSERT INTO reviewers (reviewer_name, requested_count) VALUES (?,?)",
                (reviewer_name, target),
            )
            to_assign = shared[:target]
            conn.executemany(
                "INSERT INTO assignments (reviewer_name, pair_id, order_index) VALUES (?,?,?)",
                [(reviewer_name, pid, i) for i, pid in enumerate(to_assign)],
            )
        else:
            # Resuming, or asking for more than originally requested -- top up
            # from the same reviewer-shuffled order, skipping what's already
            # assigned (so a rebuilt/expanded shared core just adds more of
            # the same kind of pair rather than reshuffling what's already
            # been rated).
            current_total = conn.execute(
                "SELECT COUNT(*) FROM assignments WHERE reviewer_name=?", (reviewer_name,)
            ).fetchone()[0]
            target = min(max(req.requested_count, current_total), shared_core_size)
            already = {r[0] for r in conn.execute(
                "SELECT pair_id FROM assignments WHERE reviewer_name=?", (reviewer_name,)
            )}
            to_add = [pid for pid in shared if pid not in already][:max(0, target - current_total)]
            if to_add:
                conn.executemany(
                    "INSERT INTO assignments (reviewer_name, pair_id, order_index) VALUES (?,?,?)",
                    [(reviewer_name, pid, current_total + i) for i, pid in enumerate(to_add)],
                )
                conn.execute(
                    "UPDATE reviewers SET requested_count=? WHERE reviewer_name=?",
                    (target, reviewer_name),
                )

        conn.commit()
        # An assignment can outlive the pair it points at -- rebuilding the
        # pool (e.g. excluding zero-score pairs, as happened here) drops
        # some pair_ids from `pairs` while deliberately preserving
        # `assignments` so real ratings survive. Filter those dead
        # references out here rather than surfacing a 404 mid-review.
        assigned = [r[0] for r in conn.execute(
            "SELECT a.pair_id FROM assignments a JOIN pairs p ON p.pair_id = a.pair_id "
            "WHERE a.reviewer_name=? ORDER BY a.order_index",
            (reviewer_name,),
        )]
        rated = {r[0] for r in conn.execute(
            "SELECT DISTINCT pair_id FROM ratings WHERE reviewer_name=?", (reviewer_name,)
        )}
    finally:
        conn.close()

    return {
        "reviewer_name": reviewer_name,
        "pair_ids": assigned,
        "rated_pair_ids": list(rated),
        "total": len(assigned),
        "done": len(rated),
    }


@app.get("/api/pair/{pair_id}")
def get_pair(pair_id: str):
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM pairs WHERE pair_id=?", (pair_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown pair")
    return {
        "pair_id": row["pair_id"],
        "ref_site": row["ref_site"],
        "hit_site": row["hit_site"],
        "ref_pdb": row["ref_pdb"],
        "hit_pdb": row["hit_pdb"],
        "ref_chain": row["ref_chain"],
        "ref_resname": row["ref_resname"],
        "ref_resnum": row["ref_resnum"],
        "hit_chain": row["hit_chain"],
        "hit_resname": row["hit_resname"],
        "hit_resnum": row["hit_resnum"],
        "red_atom_names": json.loads(row["red_atom_names"]),
        "purple_atom_names": json.loads(row["purple_atom_names"]),
        "ref_binding_residues": json.loads(row["ref_binding_residues"] or "[]"),
        "hit_binding_residues": json.loads(row["hit_binding_residues"] or "[]"),
        "ref_phosphate_atom_names": json.loads(row["ref_phosphate_atom_names"] or "[]"),
        "ref_atom_links": json.loads(row["ref_atom_links"] or "[]"),
        "hit_atom_links": json.loads(row["hit_atom_links"] or "[]"),
        "pocket_rmsd": row["pocket_rmsd"],
    }


@app.get("/api/debug/pair/{pair_id}")
def debug_pair(pair_id: str, token: str = Query(...)):
    """Admin-only (same token as /api/export) lookup of the numbers
    deliberately hidden from reviewers in /api/pair -- for verifying the
    displayed residues/atoms actually match the score while testing, not
    for reviewers to see."""
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Bad admin token")
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT prolif_plif_score_no_vdw, prolif_plif_score_with_vdw, ref_n_bits_novdw, pocket_rmsd "
            "FROM pairs WHERE pair_id=?", (pair_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown pair")
    return {
        "prolif_plif_score_no_vdw": row["prolif_plif_score_no_vdw"],
        "prolif_plif_score_with_vdw": row["prolif_plif_score_with_vdw"],
        "ref_n_bits_novdw": row["ref_n_bits_novdw"],
        "pocket_rmsd": row["pocket_rmsd"],
    }


_ADMIN_METRIC_COLUMNS = {
    "no_vdw": "prolif_plif_score_no_vdw",
    "with_vdw": "prolif_plif_score_with_vdw",
}
_ADMIN_SORT_DIRECTIONS = {"asc": "ASC", "desc": "DESC"}


@app.get("/api/admin/pairs")
def admin_pairs(
    token: str = Query(...),
    metric: str = Query("no_vdw", pattern="^(no_vdw|with_vdw)$"),
    min_score: float = Query(0.0),
    max_score: float = Query(1.0),
    min_ref_bits: float = Query(0.0),
    max_ref_bits: float = Query(1e9),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=500),
):
    """Admin-only: list pairs whose score falls in [min_score, max_score]
    and whose ref_n_bits_novdw (number of reference interactions the no-VdW
    score is computed over) falls in [min_ref_bits, max_ref_bits] -- lets
    you exclude references with too few real interactions to be a
    convincing example either way, or references so interaction-rich that a
    partial match still scores high, for picking concrete good/bad examples
    by hand. Deliberately bypasses the reviewer assignment system entirely
    (no /api/start, no ratings recorded) since this is for browsing, not
    scoring."""
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Bad admin token")

    column = _ADMIN_METRIC_COLUMNS[metric]
    direction = _ADMIN_SORT_DIRECTIONS[sort]
    conn = get_conn()
    try:
        rows = conn.execute(f"""
            SELECT pair_id, ref_pdb, ref_resname, hit_pdb, hit_resname,
                   prolif_plif_score_no_vdw, prolif_plif_score_with_vdw, ref_n_bits_novdw,
                   pocket_rmsd
            FROM pairs
            WHERE {column} IS NOT NULL AND {column} BETWEEN ? AND ?
              AND ref_n_bits_novdw IS NOT NULL AND ref_n_bits_novdw BETWEEN ? AND ?
            ORDER BY {column} {direction}
            LIMIT ?
        """, (min_score, max_score, min_ref_bits, max_ref_bits, limit)).fetchall()
    finally:
        conn.close()
    return [
        {
            "pair_id": r["pair_id"],
            "ref_pdb": r["ref_pdb"], "ref_resname": r["ref_resname"],
            "hit_pdb": r["hit_pdb"], "hit_resname": r["hit_resname"],
            "score_no_vdw": r["prolif_plif_score_no_vdw"],
            "score_with_vdw": r["prolif_plif_score_with_vdw"],
            "ref_n_bits_novdw": r["ref_n_bits_novdw"],
            "pocket_rmsd": r["pocket_rmsd"],
        }
        for r in rows
    ]


@app.get("/api/admin/reviewers")
def admin_reviewers(token: str = Query(...)):
    """Admin-only: every reviewer who has ever started a session, with how
    much of their assignment they've rated so far -- feeds the "browse as
    this reviewer" picker below."""
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Bad admin token")
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT r.reviewer_name,
                   (SELECT COUNT(*) FROM assignments a WHERE a.reviewer_name = r.reviewer_name) AS total_assigned,
                   (SELECT COUNT(DISTINCT rt.pair_id) FROM ratings rt WHERE rt.reviewer_name = r.reviewer_name) AS rated_count
            FROM reviewers r
            ORDER BY r.reviewer_name COLLATE NOCASE
        """).fetchall()
    finally:
        conn.close()
    return [
        {
            "reviewer_name": row["reviewer_name"],
            "total_assigned": row["total_assigned"],
            "rated_count": row["rated_count"],
        }
        for row in rows
    ]


@app.get("/api/admin/reviewer/{reviewer_name}/pairs")
def admin_reviewer_pairs(reviewer_name: str, token: str = Query(...)):
    """Admin-only: the exact structures assigned to one reviewer, in their
    assignment order, joined with that reviewer's own rating if they've
    already given one -- lets you spectate exactly what a colleague sees
    (same pairs, same order) and step through them with the existing
    admin prev/next browser, without ever writing to `ratings` yourself.
    Dead assignments (pair rebuilt out of the pool since) are dropped the
    same way /api/start already does."""
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Bad admin token")
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT p.pair_id, p.ref_pdb, p.ref_resname, p.hit_pdb, p.hit_resname,
                   p.prolif_plif_score_no_vdw, p.prolif_plif_score_with_vdw, p.ref_n_bits_novdw,
                   p.pocket_rmsd, r.score AS reviewer_score,
                   r.wrong_reference_phosphate AS reviewer_wrong_reference_phosphate,
                   r.comment AS reviewer_comment
            FROM assignments a
            JOIN pairs p ON p.pair_id = a.pair_id
            LEFT JOIN ratings r ON r.reviewer_name = a.reviewer_name AND r.pair_id = a.pair_id
            WHERE a.reviewer_name = ?
            ORDER BY a.order_index
        """, (reviewer_name,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "pair_id": r["pair_id"],
            "ref_pdb": r["ref_pdb"], "ref_resname": r["ref_resname"],
            "hit_pdb": r["hit_pdb"], "hit_resname": r["hit_resname"],
            "score_no_vdw": r["prolif_plif_score_no_vdw"],
            "score_with_vdw": r["prolif_plif_score_with_vdw"],
            "ref_n_bits_novdw": r["ref_n_bits_novdw"],
            "pocket_rmsd": r["pocket_rmsd"],
            "reviewer_score": r["reviewer_score"],
            "reviewer_wrong_reference_phosphate": bool(r["reviewer_wrong_reference_phosphate"]),
            "reviewer_comment": r["reviewer_comment"],
        }
        for r in rows
    ]


@app.get("/api/structure/ref/{pair_id}")
def structure_ref(pair_id: str):
    conn = get_conn()
    try:
        row = conn.execute("SELECT ref_site FROM pairs WHERE pair_id=?", (pair_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown pair")
    # Prefer the display-corrected copy (physiologically-correct phosphate
    # protonation, see REF_POCKETS_DIR's docstring in db.py); fall back to
    # the canonical pocket file if build_pool.py hasn't generated one yet
    # (e.g. right after a code update but before the next rebuild) rather
    # than 404ing on an otherwise-valid pair.
    path = os.path.join(REF_POCKETS_DIR, f"{row['ref_site']}.pdb")
    if not os.path.isfile(path):
        path = os.path.join(POCKETS_DIR, f"{row['ref_site']}.pdb")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Structure file missing")
    return FileResponse(path, media_type="chemical/x-pdb")


@app.get("/api/structure/hit/{pair_id}")
def structure_hit(pair_id: str):
    conn = get_conn()
    try:
        row = conn.execute("SELECT pair_id FROM pairs WHERE pair_id=?", (pair_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown pair")
    path = os.path.join(ALIGNED_HITS_DIR, f"{pair_id}.pdb")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Aligned structure file missing")
    return FileResponse(path, media_type="chemical/x-pdb")


@app.post("/api/rating")
def submit_rating(req: RatingRequest):
    if req.wrong_reference_phosphate == (req.score is not None):
        raise HTTPException(
            status_code=400,
            detail="Provide either a 1-5 score or wrong_reference_phosphate, not both or neither",
        )

    conn = get_conn()
    try:
        assigned = conn.execute(
            "SELECT 1 FROM assignments WHERE reviewer_name=? AND pair_id=?",
            (req.reviewer_name, req.pair_id),
        ).fetchone()
        if assigned is None:
            raise HTTPException(status_code=400, detail="Pair not assigned to this reviewer")

        conn.execute(
            "INSERT INTO ratings (reviewer_name, pair_id, score, wrong_reference_phosphate, comment) "
            "VALUES (?,?,?,?,?)",
            (req.reviewer_name, req.pair_id, req.score, int(req.wrong_reference_phosphate), req.comment),
        )
        conn.commit()

        total = conn.execute(
            "SELECT COUNT(*) FROM assignments WHERE reviewer_name=?", (req.reviewer_name,)
        ).fetchone()[0]
        done = conn.execute(
            "SELECT COUNT(DISTINCT pair_id) FROM ratings WHERE reviewer_name=?",
            (req.reviewer_name,),
        ).fetchone()[0]
    finally:
        conn.close()
    return {"ok": True, "done": done, "total": total}


@app.get("/api/progress/{reviewer_name}")
def progress(reviewer_name: str):
    conn = get_conn()
    try:
        assigned = [r[0] for r in conn.execute(
            "SELECT a.pair_id FROM assignments a JOIN pairs p ON p.pair_id = a.pair_id "
            "WHERE a.reviewer_name=? ORDER BY a.order_index",
            (reviewer_name,),
        )]
        rated = {r[0] for r in conn.execute(
            "SELECT DISTINCT pair_id FROM ratings WHERE reviewer_name=?", (reviewer_name,)
        )}
    finally:
        conn.close()
    next_pair = next((p for p in assigned if p not in rated), None)
    return {"total": len(assigned), "done": len(rated), "next_pair_id": next_pair}


@app.get("/api/export")
def export(token: str = Query(...)):
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Bad admin token")

    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT r.reviewer_name, r.pair_id, r.score, r.wrong_reference_phosphate,
                   r.comment, r.submitted_at,
                   p.ref_site, p.hit_site, p.prolif_plif_score_no_vdw,
                   p.prolif_plif_score_with_vdw, p.ref_n_bits_novdw, p.pocket_rmsd, p.is_shared
            FROM ratings r JOIN pairs p ON p.pair_id = r.pair_id
            ORDER BY r.submitted_at
        """).fetchall()
    finally:
        conn.close()

    def gen():
        yield ("reviewer_name,pair_id,score,wrong_reference_phosphate,comment,submitted_at,"
               "ref_site,hit_site,prolif_plif_score_no_vdw,prolif_plif_score_with_vdw,"
               "ref_n_bits_novdw,pocket_rmsd,is_shared\n")
        for r in rows:
            comment = (r["comment"] or "").replace('"', '""')
            score = "" if r["score"] is None else r["score"]
            with_vdw = r["prolif_plif_score_with_vdw"]
            with_vdw = "" if with_vdw is None else with_vdw
            pocket_rmsd = "" if r["pocket_rmsd"] is None else r["pocket_rmsd"]
            yield (f'{r["reviewer_name"]},{r["pair_id"]},{score},{r["wrong_reference_phosphate"] or 0},'
                   f'"{comment}",{r["submitted_at"]},{r["ref_site"]},{r["hit_site"]},'
                   f'{r["prolif_plif_score_no_vdw"]},{with_vdw},{r["ref_n_bits_novdw"]},'
                   f'{pocket_rmsd},{r["is_shared"]}\n')

    return StreamingResponse(gen(), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=isostere_ratings_export.csv"
    })


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
