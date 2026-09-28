"""
Clusters build_ifg_prolif_dataset.py's per-pair IFG functional-group columns
(ifg_group_type_smiles / ifg_group_atoms_smiles) by exact canonical-SMILES
match -- i.e. groups together every occurrence of the SAME Ertl functional
group (ifg.py) across the whole dataset, regardless of which reference/hit
pair or PDB structure it came from -- and writes a frequency-ranked summary,
both as a CSV (for further analysis) and a human-readable text report.

A row's ifg_group_type_smiles/atoms_smiles cell can itself hold more than one
group ('; '-joined -- see build_ifg_prolif_dataset.matched_ifg_groups), e.g. a
malonate mimic where both of its two carboxylates independently caught a red
atom. Each individual group in that list is one "instance" here, so such a row
contributes 2 instances to the CC(=O)[O-] group's count, not 1 -- this counts
"how many times does this functional group show up as an isosteric hit
anywhere in the dataset", not "how many pairs feature it at all" (also
reported separately, as n_distinct_pairs, since the two numbers can diverge).

Default grouping key is ifg_group_type_smiles (the group PLUS its attached
unmarked carbons, ifg.py's own convention) rather than ifg_group_atoms_smiles
(the bare heteroatom cluster alone) -- e.g. distinguishes a carboxylate on a
plain alkyl chain from one conjugated to an aromatic ring, which
ifg_group_atoms_smiles alone (just "O=C[O-]" either way) can't. Use
--key atoms for the coarser grouping instead.

--key type_smarts / atoms_smarts group by a CONSISTENT SMARTS derived from
that same SMILES instead of the raw SMILES text itself. This matters because
plain per-instance Chem.MolToSmarts() (e.g. build_ifg_prolif_dataset.py's own
red_fragment_smarts column) is NOT canonical -- RDKit has no SMARTS
canonicalizer the way it does MolToSmiles(canonical=True), so two chemically
identical groups extracted from different source ligands (different internal
atom numbering) can legitimately serialize to different-looking SMARTS text,
which makes grouping directly on that column inconsistent. The fix used here
(smiles_to_consistent_smarts()) exploits the fact that ifg.py's SMILES output
IS already canonical/consistent (confirmed: chemically identical groups always
produce byte-identical SMILES, which is exactly why plain --key type/atoms
already groups correctly) -- reparsing that fixed string with
Chem.MolFromSmiles(sanitize=False) is a deterministic function of the string,
so the SAME input SMILES always yields the SAME SMARTS, and a partial
Chem.SanitizeMol() (skipping SANITIZE_KEKULIZE/SANITIZE_SETAROMATICITY, same
"degrade only the failing check" idiom used elsewhere in this pipeline, e.g.
run_prolif.py's _extract_submol) avoids crashing on fragments whose aromatic
flag no longer makes sense once cut out of their ring (a lone lowercase 'c'
with no ring left, e.g. "cO"). Side effect, confirmed on the real dataset: this
also drops the aromatic/non-aromatic distinction on the attachment atom itself
(elements only, e.g. both "CO" and "cO" -> "[#6]-[#8]"), which is what merges
e.g. an aromatic-attached carboxylate with a plain alkyl one into one bucket --
18 of 185 type_smiles groups merge this way in the full-manifest dataset, all
along that same C-vs-c axis, nothing else. --out/the report show which
original SMILES fed each SMARTS bucket (source_smiles / n_source_smiles_variants)
so that consolidation stays visible rather than silent.

Usage: python summarize_ifg_groups.py [dataset.csv]
                                       [--key type|atoms|type_smarts|atoms_smarts]
                                       [--out path] [--report path] [--top N]
"""
import argparse
import csv
import os
import sys

from rdkit import Chem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")

KEY_COLUMN = {
    "type": "ifg_group_type_smiles", "atoms": "ifg_group_atoms_smiles",
    "type_smarts": "ifg_group_type_smiles", "atoms_smarts": "ifg_group_atoms_smiles",
}
IS_SMARTS_KEY = {"type_smarts", "atoms_smarts"}


def smiles_to_consistent_smarts(smiles, cache):
    """See module docstring for why this round trip (not a plain per-instance
    Chem.MolToSmarts) is what makes SMARTS-based grouping consistent. Memoized
    in `cache` since the same group SMILES recurs hundreds of times across the
    dataset. Returns None (and caches that) for the rare string that can't
    even be parsed unsanitized -- callers should skip those instances."""
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            cache[smiles] = None
        else:
            try:
                ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY
                Chem.SanitizeMol(mol, sanitizeOps=ops)
                cache[smiles] = Chem.MolToSmarts(mol)
            except Exception:
                cache[smiles] = None
    return cache[smiles]


def _split(cell):
    """'; '-joined cell -> list of individual group SMILES, [] if empty."""
    return cell.split("; ") if cell else []


def _mean(values):
    return sum(values) / len(values) if values else None


def build_groups(rows, key_field, other_field, smarts_mode=False):
    """Returns {key: stats_dict}, plus (n_rows_with_group, n_instances,
    n_unparseable). If smarts_mode, `key` is smiles_to_consistent_smarts(raw
    SMILES) instead of the raw SMILES itself, and each group's
    "source_smiles_counts" tracks which original SMILES variant(s) fed it --
    the same slot plain-SMILES mode uses for the OTHER column's variants,
    since in SMARTS mode that's the more informative thing to show."""
    groups = {}
    n_rows_with_group = 0
    n_instances = 0
    n_unparseable = 0
    smarts_cache = {}

    for row in rows:
        keys = _split(row[key_field])
        others = _split(row[other_field])
        if not keys:
            continue
        n_rows_with_group += 1

        score = None
        if row.get("prolif_plif_score") not in ("", None):
            score = float(row["prolif_plif_score"])
        fraction = None
        if row.get("isosteric_fraction") not in ("", None):
            fraction = float(row["isosteric_fraction"])

        for raw_key, other_smiles in zip(keys, others):
            if smarts_mode:
                group_key = smiles_to_consistent_smarts(raw_key, smarts_cache)
                if group_key is None:
                    n_unparseable += 1
                    continue
            else:
                group_key = raw_key
            n_instances += 1
            g = groups.setdefault(group_key, {
                "count": 0, "other_smiles_counts": {}, "mimics": {},
                "pair_ids": set(), "ref_pdbs": set(), "hit_pdbs": set(),
                "scores": [], "fractions": [], "examples": [],
            })
            g["count"] += 1
            other_tally_key = raw_key if smarts_mode else other_smiles
            g["other_smiles_counts"][other_tally_key] = g["other_smiles_counts"].get(other_tally_key, 0) + 1
            g["mimics"][row["mimic"]] = g["mimics"].get(row["mimic"], 0) + 1
            g["pair_ids"].add((row["ref_site"], row["hit_site"]))
            g["ref_pdbs"].add(row["ref_pdb"])
            g["hit_pdbs"].add(row["hit_pdb"])
            if score is not None:
                g["scores"].append(score)
            if fraction is not None:
                g["fractions"].append(fraction)
            # Dedupe examples against (ref_site, hit_site), not just a length
            # cap -- a ligand with two identical groups (e.g. malonate's two
            # carboxylates) hits this loop body twice for the SAME row, and
            # showing the same ref/hit pair twice in a 6-line example list
            # reads as a bug even though the underlying instance count is
            # correct by design (see module docstring).
            pair_id = (row["ref_site"], row["hit_site"])
            if len(g["examples"]) < 6 and pair_id not in {(ex[3], ex[4]) for ex in g["examples"]}:
                g["examples"].append((row["mimic"], row["ref_pdb"], row["hit_pdb"],
                                       row["ref_site"], row["hit_site"]))

    return groups, n_rows_with_group, n_instances, n_unparseable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?",
                         default=os.path.join(RESULTS_DIR, "ifg_prolif_dataset_full_manifest.csv"))
    parser.add_argument("--key", choices=["type", "atoms", "type_smarts", "atoms_smarts"], default="type",
                         help="group by ifg_group_type_smiles (default, includes attached carbons), "
                              "ifg_group_atoms_smiles (bare heteroatom cluster only), or the _smarts "
                              "variants -- a consistent SMARTS derived from that same SMILES, see "
                              "module docstring for why this merges some SMILES-distinct groups "
                              "(e.g. aromatic- vs alkyl-attached carboxylate) into one bucket")
    parser.add_argument("--out", default=None, help="summary CSV path")
    parser.add_argument("--report", default=None, help="human-readable text report path")
    parser.add_argument("--top", type=int, default=0, help="limit the text report to the top N groups (0 = all)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} not found.")
        sys.exit(1)

    smarts_mode = args.key in IS_SMARTS_KEY
    base_key = args.key.removesuffix("_smarts")

    tag = os.path.splitext(os.path.basename(args.dataset))[0]
    out_path = args.out or os.path.join(RESULTS_DIR, f"{tag}_grouped_by_{args.key}.csv")
    report_path = args.report or os.path.join(RESULTS_DIR, f"{tag}_grouped_by_{args.key}.txt")

    key_field = KEY_COLUMN[args.key]
    other_field = KEY_COLUMN["atoms" if base_key == "type" else "type"]

    print(f"Reading {args.dataset} ...")
    with open(args.dataset, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} rows")

    groups, n_rows_with_group, n_instances, n_unparseable = build_groups(
        rows, key_field, other_field, smarts_mode=smarts_mode)
    ranked = sorted(groups.items(), key=lambda kv: kv[1]["count"], reverse=True)
    print(f"  {n_rows_with_group} rows carry at least one IFG group, {n_instances} group instances total, "
          f"{len(groups)} distinct groups (by {'consistent SMARTS derived from ' if smarts_mode else ''}{key_field})")
    if n_unparseable:
        print(f"  {n_unparseable} instances skipped -- SMILES didn't parse even unsanitized (rare)")

    # ── CSV summary ─────────────────────────────────────────────────────────
    group_col = "group_smarts" if smarts_mode else "group_smiles"
    other_col = "source_smiles" if smarts_mode else "representative_" + other_field
    fieldnames = [
        "rank", group_col, other_col,
        "n_instances", "pct_of_instances", "n_distinct_pairs",
        "n_distinct_mimics", "n_distinct_ref_pdbs", "n_distinct_hit_pdbs",
        "mean_prolif_plif_score", "mean_isosteric_fraction",
        "example_mimics", "example_pairs",
    ]
    if smarts_mode:
        fieldnames.insert(2, "n_source_smiles_variants")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (group_key, g) in enumerate(ranked, 1):
            top_mimics = sorted(g["mimics"].items(), key=lambda kv: kv[1], reverse=True)[:5]
            if smarts_mode:
                sorted_sources = sorted(g["other_smiles_counts"].items(), key=lambda kv: -kv[1])
                other_val = "; ".join(f"{s}({n})" for s, n in sorted_sources)
            else:
                other_val = max(g["other_smiles_counts"], key=g["other_smiles_counts"].get)
            row_out = {
                "rank": rank,
                group_col: group_key,
                other_col: other_val,
                "n_instances": g["count"],
                "pct_of_instances": round(100 * g["count"] / n_instances, 2) if n_instances else 0,
                "n_distinct_pairs": len(g["pair_ids"]),
                "n_distinct_mimics": len(g["mimics"]),
                "n_distinct_ref_pdbs": len(g["ref_pdbs"]),
                "n_distinct_hit_pdbs": len(g["hit_pdbs"]),
                "mean_prolif_plif_score": round(_mean(g["scores"]), 3) if g["scores"] else "",
                "mean_isosteric_fraction": round(_mean(g["fractions"]), 3) if g["fractions"] else "",
                "example_mimics": ", ".join(f"{m}({n})" for m, n in top_mimics),
                "example_pairs": "; ".join(f"{ref}->{hit}" for _, ref, hit, _, _ in g["examples"][:5]),
            }
            if smarts_mode:
                row_out["n_source_smiles_variants"] = len(g["other_smiles_counts"])
            writer.writerow(row_out)
    print(f"Wrote {out_path}")

    # ── Human-readable report ──────────────────────────────────────────────
    n_no_group = len(rows) - n_rows_with_group
    to_report = ranked[:args.top] if args.top else ranked
    lines = []
    lines.append("=" * 80)
    lines.append("IFG functional-group clustering summary")
    lines.append("=" * 80)
    lines.append(f"Source dataset : {args.dataset}")
    if smarts_mode:
        lines.append(f"Grouped by     : consistent SMARTS derived from {key_field} "
                     "(see module docstring -- this can merge SMILES-distinct groups, e.g. "
                     "aromatic- vs alkyl-attached, into one SMARTS bucket; merged sources are "
                     "listed per group below)")
    else:
        lines.append(f"Grouped by     : {key_field}"
                     + (" (functional group + attached carbon context)" if base_key == "type"
                        else " (bare heteroatom cluster only)"))
    lines.append(f"Total rows     : {len(rows)}  ({n_rows_with_group} with >=1 IFG group, "
                 f"{n_no_group} with none)")
    lines.append(f"Group instances: {n_instances}"
                 + (f"  ({n_unparseable} skipped, unparseable)" if n_unparseable else ""))
    lines.append(f"Distinct groups: {len(groups)}"
                 + (f"  (showing top {args.top})" if args.top else ""))
    lines.append("")

    for rank, (group_key, g) in enumerate(to_report, 1):
        pct = 100 * g["count"] / n_instances if n_instances else 0
        mean_score = _mean(g["scores"])
        mean_frac = _mean(g["fractions"])
        top_mimics = sorted(g["mimics"].items(), key=lambda kv: kv[1], reverse=True)[:8]
        mimics_str = ", ".join(f"{m} x{n}" if n > 1 else m for m, n in top_mimics)
        if len(g["mimics"]) > len(top_mimics):
            mimics_str += f", +{len(g['mimics']) - len(top_mimics)} more"

        lines.append("-" * 80)
        lines.append(f"#{rank}  {g['count']} instances ({pct:.1f}%)  |  "
                     f"{len(g['pair_ids'])} distinct ref/hit pairs  |  "
                     f"{len(g['mimics'])} distinct mimic ligands")
        if smarts_mode:
            lines.append(f"    {'smarts':14s}: {group_key}")
            sorted_sources = sorted(g["other_smiles_counts"].items(), key=lambda kv: -kv[1])
            sources_str = ", ".join(f"{s} x{n}" if n > 1 else s for s, n in sorted_sources)
            label = f"source SMILES ({len(sorted_sources)})"
            lines.append(f"    {label:14s}: {sources_str}")
        else:
            representative_other = max(g["other_smiles_counts"], key=g["other_smiles_counts"].get)
            lines.append(f"    {base_key + '_smiles':14s}: {group_key}")
            other_label = "atoms_smiles" if base_key == "type" else "type_smiles"
            lines.append(f"    {other_label:14s}: {representative_other}")
        score_str = f"{mean_score:.3f}" if mean_score is not None else "n/a"
        frac_str = f"{mean_frac:.3f}" if mean_frac is not None else "n/a"
        lines.append(f"    mean score    : {score_str}      mean isosteric fraction: {frac_str}")
        lines.append(f"    mimic ligands : {mimics_str}")
        lines.append("    examples      :")
        for mimic, ref_pdb, hit_pdb, ref_site, hit_site in g["examples"]:
            lines.append(f"        {mimic:8s} {ref_pdb} -> {hit_pdb}   ({ref_site} vs {hit_site})")
    lines.append("=" * 80)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
