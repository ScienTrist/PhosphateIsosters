import os
import json
import gemmi
import argparse
import glob
import sys
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from constants import STANDARD_AA, EC_NAMES, IGNORE_LIGANDS
from interaction_utils import load_ec_map, is_metal_coordinated


def calculate_pocket_rmsd(st_ref, st_hit, ref_lig_pos, radius=10.0, match_cutoff=5.0):
    """Calculates pocket RMSD by matching each reference pocket Ca to its nearest
    Ca in the hit structure by 3D distance, rather than by identical
    (chain, residue_number). Both structures are already superposed via TM-align,
    so spatial proximity is a more reliable correspondence than residue numbering:
    homologs are frequently numbered differently (indels, different constructs)
    even when the fold occupies the same position post-superposition."""
    ref_pocket_cas = []
    for model in st_ref:
        for chain in model:
            for res in chain:
                if any(any(a.pos.dist(lp) <= radius for lp in ref_lig_pos) for a in res):
                    if "CA" in res:
                        p = res["CA"][0].pos
                        ref_pocket_cas.append((p.x, p.y, p.z))

    if not ref_pocket_cas:
        return None

    hit_cas = []
    for model in st_hit:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    p = res["CA"][0].pos
                    hit_cas.append((p.x, p.y, p.z))

    if not hit_cas:
        return None

    ref_arr = np.array(ref_pocket_cas)
    hit_arr = np.array(hit_cas)
    dists = np.sqrt(((ref_arr[:, None, :] - hit_arr[None, :, :]) ** 2).sum(axis=2))
    nearest = dists.min(axis=1)

    close = nearest[nearest < match_cutoff]
    if close.size == 0:
        return None
    return float(np.sqrt((close ** 2).mean()))


def analyze_isosteres_simple(motif_dir, include_modified=False):
    """
    Identifies ligands in hit structures near the reference phosphate site.
    Categorizes results by: 1. Apo/Holo, 2. EC Class, 3. Metal/Non-Metal.
    """
    if not os.path.exists(motif_dir):
        print(f"Error: Directory {motif_dir} does not exist.")
        return

    os.makedirs(motif_dir, exist_ok=True)

    results = {
        "APO": defaultdict(lambda: defaultdict(list)),
        "HOLO": defaultdict(lambda: defaultdict(list))
    }

    ec_map = load_ec_map(motif_dir)
    hit_dir = os.path.join(motif_dir, "hits")
    ref_dir = os.path.join(motif_dir, "references")
    hit_files = (
        glob.glob(os.path.join(hit_dir, "**", "hit_*.pdb"), recursive=True) +
        glob.glob(os.path.join(hit_dir, "**", "hit_*.cif"), recursive=True)
        if os.path.exists(hit_dir) else []
    )
    print(f"Scanning {len(hit_files)} hits in {motif_dir}...")
    if include_modified:
        print("  [Mode] Including modified residues as potential mimics.")

    for i, hit_path in enumerate(hit_files):
        if i % 500 == 0 or i == len(hit_files) - 1:
            print(f"  [{i+1}/{len(hit_files)}]...", end="\r", flush=True)
        basename = os.path.basename(hit_path)
        parts = basename.replace(".pdb", "").replace(".cif", "").split("_")
        if len(parts) < 6:
            continue

        ref_id, hit_id, ref_chain, ref_lig_name, ref_lig_num = parts[1:6]

        major_ec = ec_map.get(ref_id.upper(), "no_EC")
        ec_class = f"EC_{major_ec}_{EC_NAMES.get(major_ec, 'Unknown')}"

        ref_path = os.path.join(ref_dir, f"ref_{ref_id}_{ref_chain}_{ref_lig_name}_{ref_lig_num}.cif")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(ref_dir, f"ref_{ref_id}_{ref_chain}_{ref_lig_name}_{ref_lig_num}.pdb")
        if not os.path.exists(ref_path):
            continue

        try:
            st_ref = gemmi.read_structure(ref_path)
            st_hit = gemmi.read_structure(hit_path)
        except Exception:
            continue

        ref_p_atoms = []
        ref_p_pos = []
        for model in st_ref:
            for chain in model:
                for res in chain:
                    if res.name == ref_lig_name:
                        ref_p_atoms.extend([a.pos for a in res])
                        ref_p_pos.extend([a.pos for a in res if a.element.name == "P"])
        if not ref_p_atoms:
            continue
        if not ref_p_pos:
            ref_p_pos = ref_p_atoms

        p_rmsd = calculate_pocket_rmsd(st_ref, st_hit, ref_p_atoms, radius=10.0)
        if p_rmsd is None:
            continue

        has_metal = is_metal_coordinated(st_ref, ref_lig_name, ref_lig_num)
        metal_status = "METAL" if has_metal else "NON_METAL"

        found_ligands = []
        for model in st_hit:
            for chain in model:
                if not any(c.isupper() for c in chain.name):
                    continue
                for res in chain:
                    if res.name in STANDARD_AA or res.name in IGNORE_LIGANDS:
                        continue
                    if any(a.element.name == "P" for a in res):
                        continue
                    if not include_modified:
                        if gemmi.find_tabulated_residue(res.name).is_amino_acid():
                            continue
                    min_dist = min(
                        (h_atom.pos.dist(p_pos) for h_atom in res
                         if h_atom.element.name not in ("C", "H")
                         for p_pos in ref_p_pos),
                        default=999.0
                    )
                    if min_dist < 3.0:
                        found_ligands.append((res.name, min_dist))

        hit_data = {
            "ref": ref_id, "hit": hit_id,
            "ref_lig": ref_lig_name, "ref_num": ref_lig_num,
            "p_rmsd": round(p_rmsd, 2),
        }

        if not found_ligands:
            results["APO"][ec_class][metal_status].append(hit_data)
        else:
            for lig_name, dist in found_ligands:
                item = hit_data.copy()
                item.update({"hit_lig": lig_name, "dist": round(dist, 2)})
                results["HOLO"][ec_class][metal_status].append(item)

    report_path = os.path.join(motif_dir, "isostere_hierarchical_report.txt")
    csv_path = os.path.join(motif_dir, "isostere_full_list.csv")

    with open(report_path, "w") as f, open(csv_path, "w") as csv_f:
        f.write("=== PHOSPHATE ISOSTERE HIERARCHICAL REPORT ===\n")
        if include_modified:
            f.write("(Mode: Including modified residues)\n\n")
        else:
            f.write("(Mode: Excluding modified residues)\n\n")

        csv_f.write("State,EC_Class,Metal_Status,Ref_ID,Hit_ID,Ref_Lig,Ref_Num,Hit_Ligand,Distance,Pocket_RMSD\n")

        for state in ["APO", "HOLO"]:
            f.write("=========================================\n")
            f.write(f"   STATE: {state}\n")
            f.write("=========================================\n\n")

            for ec in sorted(results[state].keys()):
                f.write(f"  --- EC CLASS: {ec} ---\n")

                for m_status in ["METAL", "NON_METAL"]:
                    hits = results[state][ec][m_status]
                    f.write(f"    [{m_status}] ({len(hits)} occurrences)\n")

                    if not hits:
                        f.write("      No hits found.\n")
                    elif state == "APO":
                        f.write(f"      {'Ref ID':<8} | {'Hit ID':<8} | {'P-RMSD'}\n")
                        f.write("      " + "-" * 28 + "\n")
                        for h in sorted(hits, key=lambda x: x["p_rmsd"]):
                            f.write(f"      {h['ref']:<8} | {h['hit']:<8} | {h['p_rmsd']:<7.2f}\n")
                            csv_f.write(
                                f"APO,{ec},{m_status},{h['ref']},{h['hit']},"
                                f"{h['ref_lig']},{h['ref_num']},None,None,{h['p_rmsd']}\n"
                            )
                    else:
                        lig_groups = defaultdict(list)
                        for h in hits:
                            lig_groups[h["hit_lig"]].append(h)
                            csv_f.write(
                                f"HOLO,{ec},{m_status},{h['ref']},{h['hit']},"
                                f"{h['ref_lig']},{h['ref_num']},{h['hit_lig']},{h['dist']},{h['p_rmsd']}\n"
                            )
                        f.write(f"      {'Ligand':<10} | {'Count':<5} | {'Best Dist':<10} | {'Example'}\n")
                        f.write("      " + "-" * 40 + "\n")
                        for lig in sorted(lig_groups.keys(), key=lambda k: len(lig_groups[k]), reverse=True):
                            l_hits = lig_groups[lig]
                            top = sorted(l_hits, key=lambda x: x["dist"])[0]
                            f.write(
                                f"      {lig:<10} | {len(l_hits):<5} | {top['dist']:<10.2f} | "
                                f"{top['ref']}/{top['hit']}\n"
                            )
                    f.write("\n")
                f.write("\n")

    num_apo = sum(len(results["APO"][ec][m]) for ec in results["APO"] for m in results["APO"][ec])
    num_metal = sum(len(results["HOLO"][ec]["METAL"]) for ec in results["HOLO"])
    num_non_metal = sum(len(results["HOLO"][ec]["NON_METAL"]) for ec in results["HOLO"])

    print(f"\nAnalysis complete!")
    print(f"Hierarchical Summary: {report_path}")
    print(f"Exhaustive Full List: {csv_path}")
    print(f"Summary: {num_apo} Apo, {num_metal} Metal-Holo, {num_non_metal} Non-Metal-Holo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze phosphate isosteres in motif environments.")
    parser.add_argument(
        "--motif_dir",
        default=os.path.join(PROJECT_ROOT, "results", "motif_analysis"),
        help="Path to the motif analysis directory"
    )
    parser.add_argument("--include-modified", action="store_true",
                        help="Include modified residues as mimics in hits")
    args = parser.parse_args()

    analyze_isosteres_simple(args.motif_dir, include_modified=args.include_modified)
