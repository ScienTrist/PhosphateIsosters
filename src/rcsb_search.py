from rcsbapi.search import AttributeQuery

def query_rcsb(phosphate_ligands, min_res=2.7, max_r_free=0.25, method="X-RAY DIFFRACTION", mode="all"):
    """
    Queries the RCSB database for PDB entries matching experimental criteria.
    Modes:
    - "all": Any phosphate-containing ligand.
    - "only_po4": Contains simple phosphates (PO4, PI, or 2HP) and NO other phosphate ligands.
    - "no_po4": Contains other phosphate ligands (this effectively excludes structures that ONLY contain PO4, PI, or 2HP).
    """
    # Build quality filter
    quality_filters = []
    
    if method == "X-RAY DIFFRACTION":
        quality_filters.append(AttributeQuery(attribute="exptl.method", operator="exact_match", value="X-RAY DIFFRACTION"))
    
    if min_res is not None:
        quality_filters.append(AttributeQuery(attribute="rcsb_entry_info.resolution_combined", operator="less_or_equal", value=min_res))
    
    if max_r_free is not None and method == "X-RAY DIFFRACTION":
        quality_filters.append(AttributeQuery(attribute="refine.ls_R_factor_R_free", operator="less_or_equal", value=max_r_free))

    # Combine quality filters
    q_quality = quality_filters[0]
    for q in quality_filters[1:]:
        q_quality = (q_quality & q)

    # ... (rest of the ligand filtering logic remains the same)

    # Define simple phosphates that we might want to exclude/isolate
    simple_phosphates = ["PO4", "PI", "2HP"]
    other_phosphate_ligands = [l for l in phosphate_ligands if l not in simple_phosphates]

    if mode == "only_po4":
        # Check if at least one of the simple phosphates is in our initial list
        available_simple = [sp for sp in simple_phosphates if sp in phosphate_ligands]
        if not available_simple:
            return []
            
        # Match structures having at least one of the simple phosphates
        q_has_simple = AttributeQuery(
            attribute="rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
            operator="in",
            value=available_simple
        )
        
        # AND NOT having any other (more complex) phosphate ligands
        q_ligand = (
            q_has_simple &
            ~AttributeQuery(attribute="rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id", operator="in", value=other_phosphate_ligands)
        )
    elif mode == "no_po4":
        # Includes anything that has at least one of the OTHER phosphate ligands.
        # This effectively excludes structures that ONLY contain simple phosphates (PO4/PI).
        q_ligand = AttributeQuery(
            attribute="rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
            operator="in",
            value=other_phosphate_ligands
        )
    else: # Default: all phosphate ligands
        q_ligand = AttributeQuery(
            attribute="rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
            operator="in",
            value=list(phosphate_ligands)
        )

    # Combining quality filters with ligand criteria
    query = (q_quality & q_ligand)

    # Execute the query and return the result
    return list(query())

def filter_by_quality(high_quality_phosphate_pdb_codes, min_res=2.7, max_r_free=0.25, method="X-RAY DIFFRACTION"):
    """
    Filters a list of PDB IDs to keep only those that meet the quality criteria.
    Batched into groups of 500 for API stability and progress tracking.
    """
    if not high_quality_phosphate_pdb_codes:
        return []
    
    unique_ids = list(set(high_quality_phosphate_pdb_codes))
    total = len(unique_ids)
    batch_size = 500
    filtered_results = []

    print(f"  Validating quality for {total} candidate structures...")

    for i in range(0, total, batch_size):
        batch = unique_ids[i:i + batch_size]
        
        quality_filters = []
        if method == "X-RAY DIFFRACTION":
            quality_filters.append(AttributeQuery(attribute="exptl.method", operator="exact_match", value="X-RAY DIFFRACTION"))
        
        if min_res is not None:
            quality_filters.append(AttributeQuery(attribute="rcsb_entry_info.resolution_combined", operator="less_or_equal", value=min_res))
        
        if max_r_free is not None and method == "X-RAY DIFFRACTION":
            quality_filters.append(AttributeQuery(attribute="refine.ls_R_factor_R_free", operator="less_or_equal", value=max_r_free))

        # Add the ID filter for this batch
        quality_filters.append(AttributeQuery(attribute="rcsb_entry_container_identifiers.entry_id", operator="in", value=batch))

        # Combine
        query = quality_filters[0]
        for q in quality_filters[1:]:
            query = (query & q)
        
        try:
            results = list(query())
            filtered_results.extend(results)
        except Exception as e:
            print(f"    Error filtering batch {i//batch_size + 1}: {e}")

        print(f"    [Filtered {min(i + batch_size, total)}/{total} structures... Found {len(filtered_results)} high-quality matches]")

    return filtered_results

