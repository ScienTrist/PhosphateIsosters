"""
Adapter: converts classify_by_aa_type.py's output (aa_type_categories.csv,
one row per site with a category STRING like "ARG:1,LYS:1") into the schema
build_cluster_overlays.py expects (site_id, pdb_id, ref_ligand, cluster_id,
total_hits, with cluster_id a small int) -- so the existing overlay
pipeline (TM-align, atom-level display, per-residue-type coloring) can be
reused unchanged rather than rebuilt for this new classification.

By default EVERY category is included -- backbone-only ("1BB", "2BB", ...)
and the no-qualifying-interaction bucket ("NO_INT") too -- with --min-size
1, this covers literally all clusters classify_by_aa_type.py found, not just
the largest ones; pass --min-size N to raise the floor, or
--exclude-backbone-buckets to drop "NO_INT" and every "NBB" category, as
earlier runs excluded the single old "BACKBONE_ONLY" bucket those both used
to be lumped into. Selected categories are mapped to cluster_id 0, 1, 2, ...
largest first -- the mapping is written to aa_type_overlay_legend.txt
(human-readable) AND aa_type_overlay_labels.csv (cluster_id,category,slug --
machine-readable, for build_cluster_overlays.py's --labels-csv, which uses
slug instead of the bare cluster_id for output filenames/group names, e.g.
pymol_overlay_cluster_ARG1-LYS1.py instead of pymol_overlay_cluster_7.py).
slug is just the category string with ":" dropped and "," turned into "-"
(both illegal or awkward in filenames); collisions are not expected given
category strings are already unique and follow a consistent RESNAME:count
pattern, but would silently overwrite each other's output file if they ever
occurred, so a repeat-slug check aborts the run rather than guess.

total_hits comes from build_pocket_features.py's ref_site_summary.csv (n_
backbone+n_sidechain+n_mixed+n_metal hits) for build_cluster_overlays.py's
own anchor-selection convention (busiest/best-resolved site becomes the
anchor). That file has 658 rows; aa_type_categories.csv has 702 (classify_
by_aa_type.py succeeds on some sites build_pocket_features.py's own bag-
building logic didn't, e.g. backbone-only sites it counted as empty-bag
failures) -- sites with no matching summary row are skipped rather than
guessed at.

Usage: python prepare_aa_type_overlay_assignments.py [--categories-csv path]
                                                       [--summary-csv path]
                                                       [--min-size N] [--exclude-backbone-buckets]
                                                       [--out-dir path]
"""
import argparse
import csv
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")

NO_INTERACTION_KEY = "NO_INT"
BACKBONE_BUCKET_RE = re.compile(r"^\d+BB$")


def is_backbone_bucket(category):
    return category == NO_INTERACTION_KEY or bool(BACKBONE_BUCKET_RE.match(category))


def category_slug(category):
    """Filesystem-safe stand-in for a category string, e.g.
    "ARG:1,LYS:1+BB" -> "ARG1-LYS1+BB". ":" is illegal in Windows filenames;
    "," is replaced too, purely for readability (a bare comma in a filename
    is legal but reads poorly run together with the digits on either side)."""
    return category.replace(":", "").replace(",", "-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories-csv", default=os.path.join(RESULTS_DIR, "aa_type_categories.csv"))
    parser.add_argument("--summary-csv", default=os.path.join(RESULTS_DIR, "ref_site_summary.csv"))
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--exclude-backbone-buckets", action="store_true")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    summary_by_site = {r["site_id"]: r for r in csv.DictReader(open(args.summary_csv, newline=""))}

    by_category = defaultdict(list)
    with open(args.categories_csv, newline="") as f:
        for row in csv.DictReader(f):
            if args.exclude_backbone_buckets and is_backbone_bucket(row["category"]):
                continue
            by_category[row["category"]].append(row)

    selected = sorted(
        (cat for cat in by_category if len(by_category[cat]) >= args.min_size),
        key=lambda cat: -len(by_category[cat]),
    )
    print(f"{len(by_category)} categories total "
          f"({'NO_INT/NBB excluded' if args.exclude_backbone_buckets else 'NO_INT/NBB included'}), "
          f"{len(selected)} have >= {args.min_size} sites (visualizing those)")

    slug_owner = {}
    out_rows = []
    legend = []
    for cluster_id, category in enumerate(selected):
        slug = category_slug(category)
        if slug in slug_owner:
            raise SystemExit(f"Slug collision: '{category}' and '{slug_owner[slug]}' both slug to "
                              f"'{slug}' -- category_slug() needs a tie-breaker, aborting rather than "
                              f"silently overwriting one cluster's output file with the other's.")
        slug_owner[slug] = category

        members = by_category[category]
        n_missing_summary = 0
        n_included = 0
        for r in members:
            s = summary_by_site.get(r["site_id"])
            if s is None:
                n_missing_summary += 1
                continue
            total_hits = (int(s["n_backbone_hits"]) + int(s["n_sidechain_hits"])
                          + int(s["n_mixed_hits"]) + int(s["n_metal_hits"]))
            out_rows.append({"site_id": r["site_id"], "pdb_id": r["pdb_id"], "ref_ligand": r["ref_ligand"],
                              "cluster_id": cluster_id, "total_hits": total_hits})
            n_included += 1
        legend.append((cluster_id, category, slug, len(members), n_included, n_missing_summary))
        print(f"  cluster_{cluster_id} = {category}  ({n_included}/{len(members)} sites usable"
              f"{f', {n_missing_summary} missing summary row' if n_missing_summary else ''})")

    assignments_path = os.path.join(args.out_dir, "aa_type_overlay_assignments.csv")
    with open(assignments_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "ref_ligand", "cluster_id", "total_hits"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {assignments_path}")

    legend_path = os.path.join(args.out_dir, "aa_type_overlay_legend.txt")
    with open(legend_path, "w") as f:
        for cluster_id, category, slug, n_total, n_included, n_missing in legend:
            f.write(f"cluster_{cluster_id} = {category}  (slug: {slug}, {n_included} sites"
                    f"{f', {n_missing} skipped -- no summary row' if n_missing else ''})\n")
    print(f"Wrote {legend_path}")

    labels_path = os.path.join(args.out_dir, "aa_type_overlay_labels.csv")
    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "category", "slug"])
        w.writeheader()
        for cluster_id, category, slug, n_total, n_included, n_missing in legend:
            w.writerow({"cluster_id": cluster_id, "category": category, "slug": slug})
    print(f"Wrote {labels_path}")


if __name__ == "__main__":
    main()
