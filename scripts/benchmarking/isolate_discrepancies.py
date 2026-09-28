import os

ROOT = "D:/AI/Master_thesis/Phosphate-binding-site"
INPUT_FILE = os.path.join(ROOT, "results/benchmarking/full_hit_comparison_1000.txt")
OUTPUT_FILE = os.path.join(ROOT, "results/benchmarking/discrepancies_1000.txt")

def isolate_discrepancies():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    header = []
    footer = []
    sections = []
    
    # Simple state machine to capture sections
    current_section = []
    in_header = True
    in_footer = False
    
    for line in lines:
        if line.startswith("===") and "GLOBAL SUMMARY" in line:
            in_footer = True
        
        if in_footer:
            footer.append(line)
            continue
            
        if in_header:
            header.append(line)
            if line.startswith("===") and len(header) > 1:
                # Check if next line is query or more header
                in_header = False
            continue

        if line.startswith("QUERY:"):
            if current_section:
                sections.append(current_section)
            current_section = [line]
        elif line.startswith("--------------------------------------------------------------------------------"):
            current_section.append(line)
            sections.append(current_section)
            current_section = []
        else:
            if current_section is not None:
                current_section.append(line)

    # Filter sections for discrepancies
    discrepancy_sections = []
    for section in sections:
        has_discrepancy = False
        for line in section:
            if "MISSING" in line or "EXTRA (NEW)" in line:
                has_discrepancy = True
                break
            if "Recall=" in line:
                # Check if recall is not 100.0% and not N/A
                try:
                    parts = line.split("Recall=")
                    val = parts[1].split("%")[0]
                    if val != "100.0" and val != "N/A":
                        # Be careful with N/A cases where REF HITS is 0
                        pass 
                except:
                    pass
        
        if has_discrepancy:
            discrepancy_sections.append(section)

    with open(OUTPUT_FILE, 'w') as out:
        for line in header:
            out.write(line)
        out.write(f"\nISOLATED DISCREPANCIES ({len(discrepancy_sections)} cases)\n\n")
        for section in discrepancy_sections:
            for line in section:
                out.write(line)
        out.write("\n")
        for line in footer:
            out.write(line)

    print(f"Isolated {len(discrepancy_sections)} discrepancy cases to {OUTPUT_FILE}")

if __name__ == "__main__":
    isolate_discrepancies()
