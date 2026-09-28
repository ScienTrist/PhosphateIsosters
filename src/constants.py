STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}

EC_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
    "no_EC": "Unclassified"
}

DISTANCE_CUTOFFS = {
    "SALT_BRIDGE": 4.0,
    "H_BOND": 3.5,
    "METAL": 3.0,
    "LINKED_PHOSPHATE": 2.5
}

METALS = {"MG", "ZN", "MN", "CA", "FE", "CU", "CO", "NI"}

IGNORE_LIGANDS = {
    "HOH", "DOD", "O", "OH", "NH3", "F", "PO4", "PI", "2HP", "SO4", "SO3", "CO3", "CL", "NA", "BR", "IOD",  # water, hydroxide, inorganic ions
    "CO2",                                                    # dissolved/crystallization carbon dioxide
    "ZN", "MG", "MN", "CA", "FE", "CU", "CO", "NI", "CD",       # metal ions
    "EDO", "GOL", "ACT", "PEG", "CIT", "FLC", "BME", "MRD", "TRS", "1PE",  # crystallographic solvents/buffers
}
