"""
Walks every stage of the full-manifest pipeline (protein protonation -> ligand
protonation -> pocket extraction -> ProLIF fingerprinting -> ProLIF pair
scoring -> homebrew-vs-ProLIF comparison) and reports, per stage, how many
unique items went in, how many came out, and -- for everything lost -- why,
grouped by failure reason.

Ground truth is unique site_id / pdb_id / (ref,hit) pair, not raw log line
counts: a site that appears in multiple manifest rows (e.g. a hit paired with
several references) is only counted once, and if a log shows an item failing
and then later succeeding (reprocessed on a resumed run), the last outcome
wins. This is why the counts here can differ from the raw "ok=/fail=" tallies
printed live during a run, which count manifest rows / attempts, not unique
items.

Run from anywhere; paths are resolved relative to this file. Reads only
existing logs/directories -- does not touch the pipeline itself.

    python scripts/pipeline_failure_report.py [full_manifest.csv]

Writes results/pipeline_failure_report.txt (human-readable funnel + reason
breakdown) and results/pipeline_failure_report.csv (one row per lost item:
stage, item_id, category, detail).
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")
DATA_DIR = os.path.join(PROLIF_V2_ROOT, "data")
MOTIF_DIR = os.path.join(PROJECT_ROOT, "results", "motif_analysis")

MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROLIF_V2_ROOT, "full_manifest.csv")
_tag = os.path.splitext(os.path.basename(MANIFEST_PATH))[0]
SUFFIX = "" if _tag == "sample_manifest" else f"_{_tag}"

PROTONATION_LOG = os.path.join(RESULTS_DIR, "full_protonation_run_v2.log")
EXTRACT_LOG = os.path.join(RESULTS_DIR, f"extract_pockets{SUFFIX}.log")
PROLIF_LOG = os.path.join(RESULTS_DIR, f"run_prolif{SUFFIX}.log")
COMPARE_LOG = os.path.join(RESULTS_DIR, f"compare{SUFFIX}.log")

PROTEIN_DIR = os.path.join(DATA_DIR, "protein_protonated")
LIGAND_DIR = os.path.join(DATA_DIR, "ligand_protonated")
RAW_DIR = os.path.join(DATA_DIR, "raw")

OUT_TXT = os.path.join(RESULTS_DIR, "pipeline_failure_report.txt")
OUT_CSV = os.path.join(RESULTS_DIR, "pipeline_failure_report.csv")


def classify(reason, rules, fallback_prefix="other"):
    """Returns a short category tag for a raw reason string: first matching
    rule (checked in order) wins; otherwise falls back to the leading
    `SomeError`/`SomeException` token if present, else a truncated catch-all
    so nothing silently disappears from the report."""
    for label, pattern in rules:
        if re.search(pattern, reason):
            return label
    m = re.match(r"^(\w+(?:Error|Exception))\b", reason)
    if m:
        return f"{fallback_prefix}:{m.group(1)}"
    return f"{fallback_prefix}:{reason[:60].strip()}"


class Stage:
    def __init__(self, name):
        self.name = name
        self.entered = set()
        self.outcome = {}   # item_id -> "ok" | reason string
        self.warnings = []  # (item_id, message) -- informational, not a loss
        self.totals_override = None  # (entered, ok) -- used when membership can't be fully enumerated

    def record(self, item_id, ok, reason=None):
        self.entered.add(item_id)
        self.outcome[item_id] = "ok" if ok else reason

    def losses(self):
        return {k: v for k, v in self.outcome.items() if v != "ok"}

    def counts(self):
        if self.totals_override:
            return self.totals_override
        lost = self.losses()
        return len(self.entered), len(self.outcome) - len(lost)

    def summary_lines(self, rules, fallback_prefix="other"):
        lost = self.losses()
        n_in, n_ok = self.counts()
        n_lost = n_in - n_ok
        lines = [f"{self.name}: {n_in} in -> {n_ok} ok, {n_lost} lost"]
        if len(lost) != n_lost:
            lines.append(f"  (reasons identified for {len(lost)}/{n_lost} of the lost items -- "
                          f"the rest never appear in the log at all)")
        if self.warnings:
            lines.append(f"  ({len(self.warnings)} additional warnings -- see below, not counted as losses)")
        by_cat = Counter()
        example = {}
        for item_id, reason in lost.items():
            cat = classify(reason, rules, fallback_prefix)
            by_cat[cat] += 1
            example.setdefault(cat, (item_id, reason))
        for cat, n in by_cat.most_common():
            ex_id, ex_reason = example[cat]
            lines.append(f"    {n:5d}  {cat:<28} e.g. {ex_id}: {ex_reason}")
        return lines, lost


def sniff_encoding(path):
    with open(path, "rb") as f:
        head = f.read(2)
    return "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8"


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding=sniff_encoding(path), errors="replace") as f:
        return f.readlines()


# ---------------------------------------------------------------------------
# Stage 1: protein protonation. Ground truth is on-disk state (data/raw vs
# data/protein_protonated), since the console log interleaves pdb2pqr's own
# multi-line stdout and isn't reliably line-per-item. Per-failure reason comes
# from data/protein_protonated/{pdb_id}.log when pdb2pqr ran, or from the
# console log's "[EXCEPTION] {id}: ..." line when the gemmi CIF->PDB
# conversion failed before pdb2pqr ever started (no .log written in that case).
# ---------------------------------------------------------------------------
def stage_protein_protonation():
    stage = Stage("1. Protein protonation")
    raw_ids = {f.rsplit(".", 1)[0] for f in os.listdir(RAW_DIR)} if os.path.isdir(RAW_DIR) else set()
    ok_ids = set()
    if os.path.isdir(PROTEIN_DIR):
        ok_ids = {f[: -len("_protein.pdb")] for f in os.listdir(PROTEIN_DIR) if f.endswith("_protein.pdb")}

    exception_reason = {}
    for line in read_lines(PROTONATION_LOG):
        m = re.match(r"^\s+\[EXCEPTION\] (\S+): (.+?)\s*$", line)
        if m and "_" not in m.group(1):  # bare pdb_id, not a ref_/hit_ site (that's phase 2)
            exception_reason[m.group(1)] = m.group(2)

    for pdb_id in sorted(raw_ids):
        if pdb_id in ok_ids:
            stage.record(pdb_id, True)
            continue
        log_path = os.path.join(PROTEIN_DIR, f"{pdb_id}.log")
        reason = None
        if os.path.exists(log_path):
            with open(log_path, encoding=sniff_encoding(log_path), errors="replace") as f:
                content = f.readlines()
            for line in reversed(content):
                line = line.strip()
                # strip pdb2pqr's leading "YYYY-MM-DD HH:MM:SS,fff " timestamp so
                # near-identical messages (differing only by timestamp) group together
                line = re.sub(r"^\d{4}-\d{2}-\d{2} [\d:,]+\s*", "", line)
                if line.endswith("main_driver:Giving up."):
                    continue  # generic wrapper message -- keep looking for the real cause above it
                if "CRITICAL:" in line or line.startswith("ERROR:"):
                    reason = line
                    break
            if reason is None:
                last = content[-1].strip() if content else "(empty log)"
                last = re.sub(r"^\d{4}-\d{2}-\d{2} [\d:,]+\s*", "", last)
                reason = f"pdb2pqr crashed with no error message -- log ends mid-run at: {last}"
        else:
            reason = exception_reason.get(pdb_id, "no pdb2pqr .log produced and no console traceback found")
        stage.record(pdb_id, False, reason)

    rules = [
        ("chain_name_too_long", r"chain name too long for the PDB format"),
        ("missing_backbone_atoms", r"[Tt]oo few atoms present to reconstruct|missing backbone atoms"),
        ("non_integer_charge", r"deviates.*from integral|non-integer charge"),
        ("silent_crash_mid_run", r"pdb2pqr crashed with no error message"),
    ]
    return stage, rules


# ---------------------------------------------------------------------------
# Stage 2: ligand protonation. Manifest-row level attempts collapse onto
# unique site_id (last outcome for that id in the log wins), cross-checked
# against data/ligand_protonated/*_ligand.pdb.
# ---------------------------------------------------------------------------
def stage_ligand_protonation(manifest_site_ids):
    stage = Stage("2. Ligand protonation")
    ok_ids = set()
    if os.path.isdir(LIGAND_DIR):
        ok_ids = {f[: -len("_ligand.pdb")] for f in os.listdir(LIGAND_DIR) if f.endswith("_ligand.pdb")}

    reason_by_id = {}
    exception_by_id = {}
    lines = read_lines(PROTONATION_LOG)
    in_phase2 = False
    for line in lines:
        if line.startswith("=== Phase 2"):
            in_phase2 = True
            continue
        if not in_phase2:
            continue
        m = re.match(r"^\s+\[EXCEPTION\] (\S+): (.+?)\s*$", line)
        if m:
            exception_by_id[m.group(1)] = m.group(2)
            continue
        m = re.match(r"^  (\S+): (cached|template|fallback|failed|FAILED) \((.+?)\)\s*$", line)
        if m:
            site_id, outcome, msg = m.groups()
            reason_by_id[site_id] = (outcome.lower(), msg)

    for site_id in sorted(manifest_site_ids):
        if site_id in ok_ids:
            stage.record(site_id, True)
            continue
        if site_id in exception_by_id:
            stage.record(site_id, False, exception_by_id[site_id])
        elif site_id in reason_by_id and reason_by_id[site_id][0] == "failed":
            stage.record(site_id, False, reason_by_id[site_id][1])
        elif site_id in reason_by_id:
            # logged as cached/template/fallback (success) but file missing on disk
            # (e.g. cleaned up after the fact) -- flag rather than silently drop.
            stage.record(site_id, False, f"logged as '{reason_by_id[site_id][0]}' but no output file on disk")
        else:
            stage.record(site_id, False, "no outcome found in protonation log for this site")

    rules = [
        ("chain_name_too_long", r"chain name too long for the PDB format"),
        ("rdkit_parse_failure", r"RDKit could not parse extracted ligand"),
        ("ligand_not_found_in_structure", r"not found in chain"),
        ("no_raw_structure_file", r"no \.pdb or \.cif for"),
        ("sanitization_failure", r"fallback sanitization"),
    ]
    return stage, rules


# ---------------------------------------------------------------------------
# Stage 3: pocket extraction (results/motif_analysis via extract_pockets.py).
# ---------------------------------------------------------------------------
def stage_extract_pockets(manifest_site_ids):
    stage = Stage("3. Pocket extraction")
    outcome = {}
    for line in read_lines(EXTRACT_LOG):
        m = re.match(r"^  (\S+): already extracted, skipping\s*$", line)
        if m:
            outcome[m.group(1)] = ("ok", None)
            continue
        m = re.match(r"^  (\S+): FAILED (.+?)\s*$", line)
        if m:
            outcome[m.group(1)] = ("failed", m.group(2))

    for site_id in sorted(manifest_site_ids):
        if site_id not in outcome:
            stage.record(site_id, False, "never attempted (not seen in extract_pockets log)")
            continue
        status, reason = outcome[site_id]
        stage.record(site_id, status == "ok", reason)

    rules = [
        ("missing_protonated_input", r"missing protonated protein or ligand"),
        ("no_ligand_residue_in_pocket", r"no ligand residue in"),
        ("merged_ligand_not_found", r"merged ligand not found at"),
        ("never_attempted", r"^never attempted"),
    ]
    return stage, rules


# ---------------------------------------------------------------------------
# Stage 4: ProLIF fingerprinting (unique sites) -- from run_prolif.py's first
# phase, inside run_prolif_full_manifest.log.
# ---------------------------------------------------------------------------
def stage_prolif_fingerprint(manifest_site_ids):
    stage = Stage("4. ProLIF fingerprinting")
    outcome = {}
    lines = read_lines(PROLIF_LOG)
    in_phase = False
    for line in lines:
        if "Fingerprinting" in line and "unique sites" in line:
            in_phase = True
            continue
        if "Phase done in" in line:
            in_phase = False
            continue
        if not in_phase:
            continue
        m = re.match(r"^  \[.*?\] (\S+): FAILED (.+?) \(", line)
        if m:
            outcome[m.group(1)] = ("failed", m.group(2))

    for site_id in sorted(manifest_site_ids):
        if site_id in outcome:
            stage.record(site_id, False, outcome[site_id][1])
        else:
            stage.record(site_id, True)

    rules = [
        ("empty_ligand_selection", r"ligand selection matched 0 atoms"),
        ("empty_protein_selection", r"protein selection matched 0 atoms"),
        ("no_phosphate_group", r"no phosphorus atom to anchor"),
    ]
    return stage, rules


# ---------------------------------------------------------------------------
# Stage 5: ProLIF pair scoring -- second phase of run_prolif_full_manifest.log.
# Distinct from stage 4: keyed by (ref_site, hit_site) pair, not a lone site.
# TM-align warnings are collected separately since they don't drop the pair,
# they just mean prolif_plif_score/tanimoto for that pair is unavailable.
# ---------------------------------------------------------------------------
def stage_prolif_pairs():
    # Successful pairs are never logged individually (only the running ok=
    # tally updates), so membership can't be fully enumerated from per-line
    # matches the way the other stages can. Losses (skip/FAIL) are still
    # logged per pair; the funnel total/ok counts instead come straight from
    # run_prolif.py's own end-of-phase summary line, which is authoritative.
    stage = Stage("5. ProLIF pair scoring")
    warnings = []
    lines = read_lines(PROLIF_LOG)
    in_phase = False
    for line in lines:
        if line.startswith("=== Scoring") and "pairs" in line:
            in_phase = True
            continue
        if not in_phase:
            continue
        m = re.match(r"^  \[warn\] (\S+) vs (\S+): (.+?)\s*$", line)
        if m:
            warnings.append((f"{m.group(1)} vs {m.group(2)}", m.group(3)))
            continue
        m = re.match(r"^  \[.*?\] skip (\S+) vs (\S+): (.+?) \(ETA", line)
        if m:
            stage.record(f"{m.group(1)} vs {m.group(2)}", False, m.group(3))
            continue
        m = re.match(r"^  \[.*?\] FAIL (\S+) vs (\S+): (.+?) \(ETA", line)
        if m:
            stage.record(f"{m.group(1)} vs {m.group(2)}", False, m.group(3))
            continue
        m = re.match(r"^Phase done in .+?: (\d+)/(\d+) pairs scored, (\d+) failed, (\d+) skipped\.", line)
        if m:
            n_ok, n_total = int(m.group(1)), int(m.group(2))
            stage.totals_override = (n_total, n_ok)
            in_phase = False
    stage.warnings = warnings

    rules = [
        ("upstream_fingerprint_missing", r"has no fingerprint \(failed earlier\)"),
    ]
    return stage, rules


# ---------------------------------------------------------------------------
# Stage 6: homebrew-vs-ProLIF comparison (compare_with_homebrew_plif.py).
# ---------------------------------------------------------------------------
def stage_compare():
    stage = Stage("6. Homebrew-vs-ProLIF comparison")
    for line in read_lines(COMPARE_LOG):
        m = re.match(r"^  \[.*?\] FAIL (\S+) vs (\S+): (.+?) \(ETA", line)
        if m:
            stage.record(f"{m.group(1)} vs {m.group(2)}", False, m.group(3))
            continue
        m = re.match(r"^(\d+)/(\d+) pairs scored, (\d+) failed", line)
        if m:
            n_ok, n_total = int(m.group(1)), int(m.group(2))
            stage.totals_override = (n_total, n_ok)

    rules = []
    return stage, rules


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    manifest_site_ids = {r["site_id"] for r in manifest_rows}

    report_lines = [
        f"Pipeline failure report for {os.path.basename(MANIFEST_PATH)}",
        f"({len(manifest_rows)} manifest rows, {len(manifest_site_ids)} unique sites)",
        "=" * 72,
        "",
    ]
    csv_rows = []

    s1, r1 = stage_protein_protonation()
    s2, r2 = stage_ligand_protonation(manifest_site_ids)
    s3, r3 = stage_extract_pockets(manifest_site_ids)
    s4, r4 = stage_prolif_fingerprint(manifest_site_ids)
    s5, r5 = stage_prolif_pairs()
    s6, r6 = stage_compare()

    for stage, rules in [(s1, r1), (s2, r2), (s3, r3), (s4, r4), (s5, r5), (s6, r6)]:
        lines, lost = stage.summary_lines(rules)
        report_lines.extend(lines)
        report_lines.append("")
        for item_id, reason in lost.items():
            csv_rows.append((stage.name, item_id, classify(reason, rules), reason))

    if s5.warnings:
        report_lines.append(f"Stage 5 warnings (pair scored, but no TM-align transform on file --")
        report_lines.append(f"prolif_plif_score/tanimoto unavailable for that pair): {len(s5.warnings)}")
        report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("Funnel (unique sites/pairs surviving each stage):")
    for stage in (s1, s2, s3, s4):
        n_in, n_ok = stage.counts()
        report_lines.append(f"  {stage.name:<32} {n_ok:6d} / {n_in:6d}")
    for stage in (s5, s6):
        n_in, n_ok = stage.counts()
        report_lines.append(f"  {stage.name:<32} {n_ok:6d} / {n_in:6d}  (pairs, not sites)")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "item_id", "category", "reason"])
        w.writerows(csv_rows)

    print("\n".join(report_lines))
    print(f"\nWrote {OUT_TXT}")
    print(f"Wrote {OUT_CSV} ({len(csv_rows)} lost items)")


if __name__ == "__main__":
    main()
