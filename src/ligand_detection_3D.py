import gemmi
from math import sqrt

def has_phosphate(block, max_po_distance=2.0):
    """
    Checks if a CIF block (ligand) contains at least one phosphorus atom
    bonded to exactly 4 Oxygen or Nitrogen atoms within a valid distance.
    This includes isosteres like GNP.
    """
    atoms = block.find_mmcif_category("_chem_comp_atom")
    bonds = block.find_mmcif_category("_chem_comp_bond")

    if not atoms or not bonds:
        return False

    # Building dictionaries for later processing
    atom_type = {}
    atom_coord = {}

    for row in atoms:
        atom_id = row["_chem_comp_atom.atom_id"]
        atom_type[atom_id] = row["_chem_comp_atom.type_symbol"]

        # Use ideal coordinates for detection
        x = row["_chem_comp_atom.pdbx_model_Cartn_x_ideal"]
        y = row["_chem_comp_atom.pdbx_model_Cartn_y_ideal"]
        z = row["_chem_comp_atom.pdbx_model_Cartn_z_ideal"]

        if x != "?" and y != "?" and z != "?":
            atom_coord[atom_id] = gemmi.Position(float(x), float(y), float(z))

    # Iterate through phosphorus atoms
    for atom_id, element in atom_type.items():
        if element != "P":
            continue

        # Find oxygen or nitrogen atoms bonded to phosphorus
        valid_neighbors = []
        for row in bonds:
            a1 = row["_chem_comp_bond.atom_id_1"]
            a2 = row["_chem_comp_bond.atom_id_2"]
            
            neighbor = None
            if a1 == atom_id:
                neighbor = a2
            elif a2 == atom_id:
                neighbor = a1
                
            if neighbor and atom_type.get(neighbor) in ["O", "N"]:
                valid_neighbors.append(neighbor)

        # Must have exactly 4 O/N neighbors to be a tetrahedral phosphate-like group
        if len(valid_neighbors) != 4:
            continue

        # Check coordinates and distances
        if atom_id not in atom_coord:
            continue # Try next P atom
            
        all_coords_present = True
        for n_id in valid_neighbors:
            if n_id not in atom_coord:
                all_coords_present = False
                break
        
        if not all_coords_present:
            continue

        # Check P–(O/N) bond distances
        p_pos = atom_coord[atom_id]
        geometry_valid = True
        for n_id in valid_neighbors:
            n_pos = atom_coord[n_id]
            dx = p_pos.x - n_pos.x
            dy = p_pos.y - n_pos.y
            dz = p_pos.z - n_pos.z
            dist = sqrt(dx*dx + dy*dy + dz*dz)

            if dist > max_po_distance:
                geometry_valid = False
                break

        if geometry_valid:
            return True  # Found at least one valid phosphate group

    return False  # No valid phosphate group found in this ligand

def get_phosphate_ligands_3d(cif_file, mode="all"):
    """
    Reads a CIF file and returns a set of component IDs for ligands
    that contain a phosphate group (or isostere) with valid geometry.
    
    Modes:
    - "all": Returns all phosphate-containing ligands.
    - "no_po4": Returns all phosphate-containing ligands except simple phosphates (PO4, PI, 2HP, PO3).
    """
    doc = gemmi.cif.read_file(cif_file)
    phosphate_ligands = set()
    simple_phosphates = {"PO4", "PI", "2HP", "PO3"}

    for block in doc:
        comp_id = block.name.strip()
        
        # Filtering if mode is no_po4
        if mode == "no_po4" and comp_id in simple_phosphates:
            continue
            
        if has_phosphate(block):
            phosphate_ligands.add(comp_id)
    
    return phosphate_ligands
