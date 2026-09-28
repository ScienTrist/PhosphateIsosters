import os

root = "Master_thesis/Phosphate-binding-site"
keys_path = os.path.join(root, "data/sample_500_keys.txt")
rep_fasta = os.path.join(root, "results/clustered_mmseqs/pdb_clustered_rep_seq.fasta")
out_fasta = os.path.join(root, "data/sample_500.fasta")

with open(keys_path, 'r') as f:
    sample_set = set(line.strip() for line in f if line.strip())

found_count = 0
with open(rep_fasta, 'r') as f_in, open(out_fasta, 'w') as f_out:
    write_current = False
    for line in f_in:
        if line.startswith(">"):
            # Header line: >ID ...
            header_id = line[1:].split()[0]
            if header_id in sample_set:
                write_current = True
                f_out.write(line)
                found_count += 1
            else:
                write_current = False
        else:
            if write_current:
                f_out.write(line)

print(f"Extracted {found_count} sequences to {out_fasta}")
