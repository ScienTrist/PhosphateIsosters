# uniprot_mapping.py
# This module groups PDB IDs by EC numbers and hierarchical classes.

import collections
import re
import os
from rcsbapi.data import DataQuery

def extract_ec_from_annotations(annotations):
    """
    Attempts to find an EC number or activity that implies one from annotations.
    """
    if not annotations:
        return None
    
    # 1. Look for explicit EC type
    for ann in annotations:
        if ann.get("type") == "EC":
            return ann.get("annotation_id")
    
    # 2. Look for GO terms that mention EC numbers or specific activities
    # (e.g., "protein kinase activity" often maps to EC 2.7.x.x)
    for ann in annotations:
        if ann.get("type") == "GO":
            name = ann.get("name", "").lower()
            if "kinase activity" in name:
                return "2.7.-.-" # Generic Kinase class
            if "phosphatase activity" in name:
                return "3.1.3.-" # Generic Phosphatase class
                
    return None

def group_pdbs_by_uniprot_id(high_quality_phosphate_pdb_codes):
    """
    Groups PDB IDs hierarchically: Major Class (1, 2) -> Sub Class (1.1, 1.2) -> UniProt -> PDBs.
    """
    if not high_quality_phosphate_pdb_codes:
        return {}

    all_entries = []
    print(f"Fetching functional metadata for {len(high_quality_phosphate_pdb_codes)} PDB entries...")
    
    attributes = [
        "rcsb_id",
        "polymer_entities.rcsb_polymer_entity.pdbx_ec",
        "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
        "polymer_entities.rcsb_polymer_entity_annotation"
    ]
    
    try:
        dq = DataQuery(input_type="entry", input_ids=high_quality_phosphate_pdb_codes, return_data_list=attributes)
        query_result = dq.exec()
        if query_result and "data" in query_result:
            all_entries = query_result["data"].get("entries", [])
    except Exception as e:
        print(f"Error fetching PDB entries: {e}")

    uniprot_to_ec = {}
    uniprot_to_pdbs = collections.defaultdict(set)
    mapped_pdbs = set()

    for entry in all_entries:
        if not entry: continue
        pdb_id = entry.get("rcsb_id")
        polymer_entities = entry.get("polymer_entities") or []
        
        for entity in polymer_entities:
            # 1. Get UniProt IDs
            container = entity.get("rcsb_polymer_entity_container_identifiers", {})
            ref_ids = container.get("reference_sequence_identifiers") or []
            
            uids = []
            for ref in ref_ids:
                acc = ref.get("database_accession")
                db_name = ref.get("database_name")
                if acc and db_name == "UniProt":
                    uids.append(acc)
            
            if not uids:
                continue

            mapped_pdbs.add(pdb_id)

            # 2. Get EC number (Try pdbx_ec first, then annotations)
            ec = entity.get("rcsb_polymer_entity", {}).get("pdbx_ec")
            if not ec:
                ec = extract_ec_from_annotations(entity.get("rcsb_polymer_entity_annotation"))
            
            for uid in uids:
                uniprot_to_pdbs[uid].add(pdb_id)
                if ec and uid not in uniprot_to_ec:
                    uniprot_to_ec[uid] = ec

    unmapped_pdbs = set(high_quality_phosphate_pdb_codes) - mapped_pdbs

    # Hierarchical Grouping
    # First pass: Group by Major Class -> Sub Class -> UniProt
    temp_hierarchy = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(set)))
    
    for uid, pdbs in uniprot_to_pdbs.items():
        ec_full = uniprot_to_ec.get(uid)
        
        if ec_full:
            # Clean EC string
            ec_clean = ec_full.split(",")[0].strip()
            parts = ec_clean.split(".")
            major_class = parts[0]
            sub_class = ".".join(parts[:2]) if len(parts) >= 2 else f"{major_class}.?"
        else:
            major_class = "no_EC"
            sub_class = "no_EC"
            
        temp_hierarchy[major_class][sub_class][uid] = pdbs

    # Second pass: Consolidate UniProt IDs within each Sub Class that share the same PDB set
    hierarchy = collections.defaultdict(lambda: collections.defaultdict(dict))
    
    for m in temp_hierarchy:
        for s in temp_hierarchy[m]:
            # Map frozenset(pdbs) -> list of UniProt IDs
            pdb_set_to_uniprots = collections.defaultdict(list)
            for uid, pdbs in temp_hierarchy[m][s].items():
                pdb_set_to_uniprots[frozenset(pdbs)].append(uid)
            
            # Re-group: key is comma-separated UniProts, value is dictionary with "pdbs" list
            for pdb_frozenset, uids in pdb_set_to_uniprots.items():
                combined_uid = ", ".join(sorted(uids))
                hierarchy[m][s][combined_uid] = {"pdbs": sorted(list(pdb_frozenset))}

    # Add unmapped PDBs to the hierarchy so they can be recovered/printed
    if unmapped_pdbs:
        hierarchy["UNMAPPED"]["UNMAPPED"]["no_uniprot_id"] = {"pdbs": sorted(list(unmapped_pdbs))}

    # Sort results
    def sort_key(s):
        if s == "no_EC" or s == "UNMAPPED":
            # (1, []) ensures "no_EC" and "UNMAPPED" come after (0, [...])
            return (1, [])
            
        parts = []
        for x in s.split("."):
            if x.isdigit():
                # (0, int(x)) ensures it sorts numerically
                parts.append((0, int(x)))
            else:
                # (1, x) handles non-digit placeholders like '-'
                parts.append((1, str(x)))
        # (0, parts) ensures valid ECs come first
        return (0, parts)

    sorted_hierarchy = collections.OrderedDict()
    major_keys = sorted(hierarchy.keys(), key=sort_key)
    
    for m in major_keys:
        sorted_subs = collections.OrderedDict()
        sub_keys = sorted(hierarchy[m].keys(), key=sort_key)
        for s in sub_keys:
            sorted_subs[s] = collections.OrderedDict(
                sorted(hierarchy[m][s].items(), key=lambda x: (-len(x[1]["pdbs"]), x[0]))
            )
        sorted_hierarchy[m] = sorted_subs

    print(f"Grouped into {len(uniprot_to_pdbs)} UniProt IDs across {len(sorted_hierarchy)} major classes.")
    return sorted_hierarchy
