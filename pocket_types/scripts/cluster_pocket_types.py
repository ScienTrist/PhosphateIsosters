"""
Step 2 of the pocket_types subproject: clusters reference sites into recurring
phosphate-binding "pocket types" from the (resname, interaction_type,
provenance) composition bags build_pocket_features.py wrote to
results/ref_site_pocket_bits_long.csv.

Feature representation: each site's raw interaction counts are L1-normalized
into a composition profile (fractions summing to 1) BEFORE clustering, not
clustered on raw counts directly -- the whole point of this representation is
"what chemical strategy does this pocket use", which shouldn't depend on how
many total contacts a given pocket happens to have (a big, contact-rich
pocket and a small, sparse one using the same recognition chemistry should
land in the same cluster). The composition matrix is then column-standardized
(StandardScaler) before KMeans, since some features (e.g. sidechain Anionic
contacts from Arg/Lys, the single most common phosphate-recognition contact
in this dataset) are far more prevalent than others and would otherwise
dominate Euclidean distance purely by scale.

k is chosen by silhouette score over a range (2..min(15, n_sites // 10)),
not fixed -- the score curve is written out alongside the final clustering so
the choice is inspectable, not a black box. Re-run with --k to override.

Roughly a fifth of sites (127/658 per build_pocket_features.py's own output)
have only 1-2 total interactions, so their composition vector is a single
100%-weighted feature -- not wrong, just thin evidence. These stay IN the
clustering (dropping them would throw away real sites, and "minimal-contact
pocket" may itself be a meaningful type), but each site's total hit count is
carried through to the assignments CSV so thin-evidence rows are identifiable
rather than silently trusted the same as a 20-contact site.

Usage: python cluster_pocket_types.py [--long-csv path] [--summary-csv path]
                                       [--k N] [--out-dir path]
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")

RANDOM_STATE = 42
N_INIT = 10


def _feature_name(resname, interaction_type, provenance):
    return f"{provenance}:{resname}:{interaction_type}"


def load_matrix(long_csv):
    """Returns (site_ids: list[str], feature_names: list[str], raw: ndarray
    [n_sites, n_features]) from the long-format bits CSV."""
    counts = defaultdict(lambda: defaultdict(int))
    feature_set = set()
    site_order = []
    seen_sites = set()
    with open(long_csv, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["site_id"]
            if sid not in seen_sites:
                seen_sites.add(sid)
                site_order.append(sid)
            feat = _feature_name(row["resname"], row["interaction_type"], row["provenance"])
            feature_set.add(feat)
            counts[sid][feat] += int(row["count"])

    feature_names = sorted(feature_set)
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    raw = np.zeros((len(site_order), len(feature_names)), dtype=float)
    for i, sid in enumerate(site_order):
        for feat, c in counts[sid].items():
            raw[i, feat_idx[feat]] = c
    return site_order, feature_names, raw


def load_site_meta(summary_csv):
    with open(summary_csv, newline="") as f:
        return {r["site_id"]: r for r in csv.DictReader(f)}


def choose_k(X, k_min, k_max):
    """Returns (best_k, [(k, silhouette_score), ...]) -- best_k maximizes
    silhouette over [k_min, k_max]."""
    scores = []
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append((k, score))
    best_k = max(scores, key=lambda ks: ks[1])[0]
    return best_k, scores


def cluster_signature(cluster_frac_rows, feature_names, top_n=8):
    """Mean composition fraction per feature across a cluster's sites, top_n
    highest, excluding all-zero features."""
    mean_frac = cluster_frac_rows.mean(axis=0)
    order = np.argsort(-mean_frac)
    sig = [(feature_names[i], mean_frac[i]) for i in order[:top_n] if mean_frac[i] > 0]
    return sig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-csv", default=os.path.join(RESULTS_DIR, "ref_site_pocket_bits_long.csv"))
    parser.add_argument("--summary-csv", default=os.path.join(RESULTS_DIR, "ref_site_summary.csv"))
    parser.add_argument("--k", type=int, default=None, help="fixed cluster count; default: pick by silhouette score")
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    site_ids, feature_names, raw = load_matrix(args.long_csv)
    meta = load_site_meta(args.summary_csv)
    n_sites, n_features = raw.shape
    print(f"{n_sites} sites x {n_features} features loaded from {args.long_csv}")

    row_totals = raw.sum(axis=1, keepdims=True)
    frac = raw / row_totals  # L1-normalized composition profile per site
    thin_evidence = int((row_totals.flatten() <= 2).sum())
    print(f"{thin_evidence}/{n_sites} sites have <=2 total interaction hits (thin composition evidence, "
          f"kept in clustering -- see module docstring)")

    scaler = StandardScaler()
    X = scaler.fit_transform(frac)

    k_max = min(args.k_max, n_sites // 10)
    if args.k is not None:
        best_k, k_scores = args.k, []
    else:
        print(f"Selecting k by silhouette score over [2, {k_max}] ...")
        best_k, k_scores = choose_k(X, 2, k_max)
        for k, score in k_scores:
            marker = " <- chosen" if k == best_k else ""
            print(f"  k={k:2d}  silhouette={score:.4f}{marker}")

    print(f"\nFitting final KMeans with k={best_k} ...")
    km = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    # ── assignments CSV ─────────────────────────────────────────────────────
    assignments_path = os.path.join(args.out_dir, "pocket_cluster_assignments.csv")
    with open(assignments_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "ref_ligand", "cluster_id", "total_hits"])
        w.writeheader()
        for sid, label, total in zip(site_ids, labels, row_totals.flatten()):
            m = meta.get(sid, {})
            w.writerow({"site_id": sid, "pdb_id": m.get("pdb_id", ""), "ref_ligand": m.get("ref_ligand", ""),
                        "cluster_id": int(label), "total_hits": int(total)})
    print(f"Wrote {assignments_path}")

    # ── k-selection curve (only meaningful when k wasn't fixed via --k) ────
    if k_scores:
        k_curve_path = os.path.join(args.out_dir, "pocket_cluster_k_selection.csv")
        with open(k_curve_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "silhouette_score"])
            w.writerows(k_scores)
        print(f"Wrote {k_curve_path}")

    # ── human-readable per-cluster summary ──────────────────────────────────
    summary_path = os.path.join(args.out_dir, "pocket_cluster_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Phosphate-binding pocket types (KMeans on L1-normalized, standardized\n")
        f.write("(resname, interaction_type, backbone/sidechain/metal) composition bags)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{n_sites} reference sites, k={best_k} clusters, {thin_evidence} sites with <=2 total hits\n\n")

        for cluster_id in sorted(set(labels)):
            idx = [i for i, lab in enumerate(labels) if lab == cluster_id]
            cluster_sites = [site_ids[i] for i in idx]
            cluster_frac = frac[idx]
            cluster_totals = row_totals[idx].flatten()
            sig = cluster_signature(cluster_frac, feature_names)

            f.write("-" * 80 + "\n")
            f.write(f"Cluster {cluster_id}  ({len(idx)} sites, mean total hits {cluster_totals.mean():.1f})\n")
            f.write("  Signature (mean fraction of site's interactions, top features):\n")
            for feat, val in sig:
                f.write(f"    {val:5.1%}  {feat}\n")
            examples = sorted(zip(cluster_sites, cluster_totals), key=lambda st: -st[1])[:6]
            f.write("  Examples (highest total hits first):\n")
            for sid, total in examples:
                m = meta.get(sid, {})
                f.write(f"    {sid:28s} {m.get('pdb_id',''):5s} {m.get('ref_ligand',''):5s} "
                         f"({int(total)} hits)\n")
            f.write("\n")

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
