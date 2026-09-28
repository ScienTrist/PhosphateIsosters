from rdkit import Chem
from rdkit import rdBase

# Suppress RDKit console noise for unparseable organometallic SMILES
rdBase.DisableLog('rdApp.*')

# Pre-compiled SMARTS for Phosphorus bonded to 4 atoms that are each
# either Oxygen or Nitrogen (matches the 3D method's neighbor criterion --
# any combination of O/N, not a fixed ratio). Uses atomic-number queries
# (#7/#8) rather than element symbols (N/O): uppercase element symbols in
# SMARTS match ALIPHATIC atoms only, silently missing a phosphorus bonded
# to an AROMATIC ring nitrogen (e.g. an imidazole-type n) -- confirmed
# directly on A1CGJ/A1CWA/EQ1/etc., all real P-N bonds to an aromatic
# nitrogen that the element-symbol form missed. #7/#8 match both aromatic
# and aliphatic forms of N/O.
PHOSPHATE_SMARTS = Chem.MolFromSmarts('P(~[#7,#8])(~[#7,#8])(~[#7,#8])~[#7,#8]')

def has_phosphate_2d(mol):
    """
    Checks if a molecule contains a phosphate group in 2D using SMARTS.
    Matches Phosphorus bonded to 4 atoms that are each either Oxygen or
    Nitrogen (any combination of the two, matching the 3D method).
    """
    if mol is not None:
        return mol.HasSubstructMatch(PHOSPHATE_SMARTS)
    return False

def get_phosphate_ligands_2d(smi_file, mode="all"):
    """
    Reads a SMILES file and returns a set of component IDs for ligands
    that contain a phosphate group.
    
    Modes:
    - "all": Returns all phosphate-containing ligands.
    - "no_po4": Returns all phosphate-containing ligands except simple phosphates (PO4, PI, 2HP, PO3).

    The file is expected to be tab-separated: SMILES, ID, Name.
    """
    phosphate_ligands = set()
    simple_phosphates = {"PO4", "PI", "2HP", "PO3"}
    
    try:
        with open(smi_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                smiles = parts[0]
                comp_id = parts[1]

                # Filtering if mode is no_po4
                if mode == "no_po4" and comp_id in simple_phosphates:
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if has_phosphate_2d(mol):
                    phosphate_ligands.add(comp_id)
    except FileNotFoundError:
        print(f"Error: {smi_file} not found.")
    
    return phosphate_ligands
