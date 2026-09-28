"""
Counts how many distinct PDB structures the MMseqs2 clustering-input chains
(53,214 of them) actually come from, and their per-structure chain-count
distribution (single-chain monomers vs. large multi-chain assemblies).

Source: cache/subset_queries_<run_id>.fasta (one FASTA entry per chain,
header format "<pdbid>_<chain> ...").

Usage: python analysis/chain_to_structure_counts.py
"""
import collections
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"
QUERY_FASTA = os.path.join(PROJECT_ROOT, "cache", f"subset_queries_{RUN_ID}.fasta")


def main():
    chains = []
    with open(QUERY_FASTA) as f:
        for line in f:
            if line.startswith(">"):
                chains.append(line[1:].split()[0])

    pdbs = {c.split("_")[0].upper() for c in chains}
    per_pdb = collections.Counter(c.split("_")[0].upper() for c in chains)

    print(f"Total chain entries: {len(chains)}")
    print(f"Distinct PDB structures: {len(pdbs)}")
    print(f"Average chains per structure: {len(chains) / len(pdbs):.2f}")

    dist = collections.Counter(per_pdb.values())
    print("\nChains-per-structure distribution:")
    for n_chains in sorted(dist):
        print(f"  {n_chains} chain(s): {dist[n_chains]} structures")


if __name__ == "__main__":
    main()
