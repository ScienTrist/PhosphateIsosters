"""
Computes the MMseqs2 reference-cluster size distribution (how many chains
collapsed onto each Set-Cover representative) from the last production
homology-enrichment run's raw cluster.tsv, bucketed into the same six
order-of-magnitude size ranges used for the thesis report figure (see the
pgfplots code built from this data), and writes a matching matplotlib
PDF/PNG as a fallback/preview.

Source: cache/subset_clustered_<run_id>_cluster.tsv (rep_id \t member_id per
line, one line per chain -- run_homology_enrichment()'s clustering step, see
src/blast_enrichment.py).

Usage: python analysis/cluster_size_distribution.py
"""
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"
CLUSTER_TSV = os.path.join(PROJECT_ROOT, "cache", f"subset_clustered_{RUN_ID}_cluster.tsv")
OUT_DIR = os.path.join(PROJECT_ROOT, "figures")

BUCKETS = [
    ("1", lambda n: n == 1),
    ("2-5", lambda n: 2 <= n <= 5),
    ("6-10", lambda n: 6 <= n <= 10),
    ("11-25", lambda n: 11 <= n <= 25),
    ("26-50", lambda n: 26 <= n <= 50),
    (">50", lambda n: n > 50),
]


def load_cluster_sizes(tsv_path):
    clusters = collections.defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            clusters[rep].append(member)
    return [len(members) for members in clusters.values()]


def main():
    sizes = load_cluster_sizes(CLUSTER_TSV)
    print(f"{len(sizes)} clusters, {sum(sizes)} chains total")

    bucket_counts = [(label, sum(1 for n in sizes if pred(n))) for label, pred in BUCKETS]
    print("\nBucket counts (same six ranges as the report figure):")
    for label, count in bucket_counts:
        print(f"  {label:>6}: {count}")

    pgf_coords = " ".join(f"({label},{count})" for label, count in bucket_counts)
    print("\npgfplots coordinates (symbolic x coords, plain linear y axis):")
    print(pgf_coords)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    labels = [b[0] for b in bucket_counts]
    values = [b[1] for b in bucket_counts]
    bars = ax.bar(labels, values, width=0.6, color="#2a5f8f")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Cluster size (chains per cluster)")
    ax.set_ylabel("Number of clusters")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, which="major", axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "cluster_size_distribution.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "cluster_size_distribution.png"), dpi=220)
    print(f"\nWrote {OUT_DIR}/cluster_size_distribution.{{pdf,png}}")


if __name__ == "__main__":
    main()
