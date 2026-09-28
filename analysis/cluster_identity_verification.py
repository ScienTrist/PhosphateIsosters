"""
Independent sanity check on the MMseqs2 reference-cluster Set-Cover
clustering: pulls real sequences for a sample of clusters (the 20 largest +
150 random) and re-verifies every non-representative member against its
representative with a proper full-length global (Needleman-Wunsch)
alignment, rather than trusting the 95%-identity/80%-coverage threshold
label at face value. Written to answer "how certain are we that this only
collapsed truly-identical structures onto each other" -- see the worked
example in the analysis writeup (7T47_A vs 9GTK_A: an N-terminal His-tag +
TEV cleavage site drags a naive global-identity score down to ~77% even
though the catalytic domain itself is >95% identical, which is exactly what
MMseqs2's LOCAL identity/coverage calculation is supposed to see past).

Source: cache/subset_clustered_<run_id>_cluster.tsv (cluster membership) and
cache/subset_queries_<run_id>.fasta (the actual chain sequences).

Usage: python analysis/cluster_identity_verification.py
"""
import collections
import difflib
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"
CLUSTER_TSV = os.path.join(PROJECT_ROOT, "cache", f"subset_clustered_{RUN_ID}_cluster.tsv")
QUERY_FASTA = os.path.join(PROJECT_ROOT, "cache", f"subset_queries_{RUN_ID}.fasta")

N_LARGEST_CLUSTERS = 20
N_RANDOM_CLUSTERS = 150
RANDOM_SEED = 0


def load_fasta(path):
    seqs = {}
    with open(path) as f:
        cur_id, cur_seq = None, []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id:
            seqs[cur_id] = "".join(cur_seq)
    return seqs


def load_clusters(path):
    clusters = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            clusters[rep].append(member)
    return clusters


def nw_identity(a, b):
    """Global alignment percent identity. Falls back to difflib's ratio for
    pairs too large for the O(n*m) DP table to be worth it (kept small here
    since these are single protein chains, not whole proteomes)."""
    n, m = len(a), len(b)
    if n * m > 4_000_000:
        return difflib.SequenceMatcher(None, a, b).ratio(), "approx(difflib)"
    NEG = -1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * NEG
    for j in range(m + 1):
        dp[0][j] = j * NEG
    for i in range(1, n + 1):
        ai, row, prow = a[i - 1], dp[i], dp[i - 1]
        for j in range(1, m + 1):
            match = prow[j - 1] + (1 if ai == b[j - 1] else NEG)
            row[j] = max(match, prow[j] + NEG, row[j - 1] + NEG)
    i, j, matches, aln_len = n, m, 0, 0
    while i > 0 and j > 0:
        score = dp[i][j]
        if score == dp[i - 1][j - 1] + (1 if a[i - 1] == b[j - 1] else NEG):
            if a[i - 1] == b[j - 1]:
                matches += 1
            i, j = i - 1, j - 1
        elif score == dp[i - 1][j] + NEG:
            i -= 1
        else:
            j -= 1
        aln_len += 1
    aln_len += max(i, j)
    return (matches / aln_len if aln_len else 0.0), "NW"


def main():
    seqs = load_fasta(QUERY_FASTA)
    clusters = load_clusters(CLUSTER_TSV)
    print(f"Loaded {len(seqs)} sequences, {len(clusters)} clusters")

    multi = [(r, m) for r, m in clusters.items() if len(m) > 1]
    top = sorted(multi, key=lambda kv: -len(kv[1]))[:N_LARGEST_CLUSTERS]
    top_reps = {r for r, _ in top}
    random.seed(RANDOM_SEED)
    rest = random.sample([c for c in multi if c[0] not in top_reps],
                          min(N_RANDOM_CLUSTERS, len(multi) - len(top)))
    sample = top + rest

    n_exact = 0
    idents = []
    for rep, members in sample:
        rep_seq = seqs.get(rep)
        if rep_seq is None:
            continue
        for mem in members:
            if mem == rep:
                continue
            mem_seq = seqs.get(mem)
            if mem_seq is None:
                continue
            if mem_seq == rep_seq:
                n_exact += 1
                idents.append((rep, mem, 1.0))
            else:
                ident, _ = nw_identity(rep_seq, mem_seq)
                idents.append((rep, mem, ident))

    print(f"\nSampled {len(sample)} clusters, {len(idents)} member-vs-representative pairs")
    print(f"Exact string match: {n_exact} ({100 * n_exact / len(idents):.1f}%)")
    vals = sorted(x[2] for x in idents)
    print(f"min={vals[0]*100:.1f}%  median={vals[len(vals)//2]*100:.1f}%  "
          f"mean={100*sum(vals)/len(vals):.1f}%  max={vals[-1]*100:.1f}%")
    for thresh in (0.99, 0.95, 0.90):
        n_below = sum(1 for v in vals if v < thresh)
        print(f"  below {thresh*100:.0f}%: {n_below} ({100*n_below/len(vals):.1f}%)")

    print("\nLowest-identity pairs (inspect these first if re-verifying):")
    for rep, mem, ident in sorted(idents, key=lambda x: x[2])[:10]:
        print(f"  {rep} vs {mem}: {ident*100:.1f}%")


if __name__ == "__main__":
    main()
