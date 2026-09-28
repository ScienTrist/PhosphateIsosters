"""
Thin wrapper around the real PLIP library: load a full PDB structure, run its
interaction detection for a specific ligand, and canonicalize the mixed bag of
per-type namedtuples it returns into a flat list of dicts with a common shape:

    {"type": "HBOND", "resnr": 148, "restype": "GLN", "reschain": "A",
     "coords": (x, y, z), "quality": {...type-specific geometry fields...}}

`coords` is always the PROTEIN-side interaction anchor (donor/acceptor atom,
charge-group center, ring center, or metal-coordinating atom), in the
structure's own original crystal frame -- the caller is responsible for
transforming hit-structure coordinates into the reference frame before
comparing across structures. `lig_coords` is the corresponding ligand-side
anchor, same frame -- used to attribute a reference ligand's interactions to
one specific phosphate group (see phosphate_groups.py) instead of the whole
ligand.

Field mappings were confirmed empirically against live PLIP 3.0.0 output on
3KQR (hbond, waterbridge, metal_complex, hydroph_interaction) and 6PY9
(saltbridge) -- see conversation history. pistack/pication/halogenbond are
implemented directly from plip/structure/detection.py's construction of those
namedtuples (protein-side atom/group is unambiguous there: proteinring is
always the binding-site ring; pication/halogen pick the protein-side operand
via an explicit boolean flag) but weren't hit in the live probe since neither
test structure had one -- treat those two types as lower-confidence until
seen live.
"""
from plip.structure.preparation import PDBComplex


def load_complex(pdb_path):
    c = PDBComplex()
    c.load_pdb(pdb_path)
    c.analyze()
    return c


def _canonical_record(itype, resnr, restype, reschain, coords, quality, lig_coords=None):
    return {
        "type": itype,
        "resnr": resnr,
        "restype": restype,
        "reschain": reschain,
        "coords": (float(coords[0]), float(coords[1]), float(coords[2])),
        "quality": quality,
        # ligand-side anchor, in the same (structure-native) frame as `coords` --
        # used to restrict a reference ligand's interactions down to one specific
        # phosphate group (see phosphate_groups.py). None where PLIP doesn't
        # expose a single unambiguous ligand atom for the interaction.
        "lig_coords": (float(lig_coords[0]), float(lig_coords[1]), float(lig_coords[2])) if lig_coords is not None else None,
    }


def canonicalize(interaction_set):
    """Flattens interaction_set.all_itypes (a mix of 8 different namedtuple
    types) into the common record shape described in the module docstring."""
    records = []
    for tup in interaction_set.all_itypes:
        tname = type(tup).__name__

        if tname == "hydroph_interaction":
            records.append(_canonical_record(
                "HYDROPHOBIC", tup.resnr, tup.restype, tup.reschain,
                tup.bsatom.coords, {"distance": float(tup.distance)},
                lig_coords=tup.ligatom.coords))

        elif tname == "hbond":
            protein_atom = tup.d if tup.protisdon else tup.a
            ligand_atom = tup.a if tup.protisdon else tup.d
            records.append(_canonical_record(
                "HBOND", tup.resnr, tup.restype, tup.reschain,
                protein_atom.coords,
                {"distance": float(tup.distance_ad), "angle": float(tup.angle)},
                lig_coords=ligand_atom.coords))

        elif tname == "saltbridge":
            coords = tup.positive.center if tup.protispos else tup.negative.center
            lig_coords = tup.negative.center if tup.protispos else tup.positive.center
            records.append(_canonical_record(
                "SALTBRIDGE", tup.resnr, tup.restype, tup.reschain,
                coords, {"distance": float(tup.distance)}, lig_coords=lig_coords))

        elif tname == "pistack":
            records.append(_canonical_record(
                "PISTACK", tup.resnr, tup.restype, tup.reschain,
                tup.proteinring.center,
                {"distance": float(tup.distance), "angle": float(tup.angle),
                 "offset": float(tup.offset), "subtype": tup.type},
                lig_coords=tup.ligandring.center))

        elif tname == "pication":
            coords = tup.charge.center if tup.protcharged else tup.ring.center
            lig_coords = tup.ring.center if tup.protcharged else tup.charge.center
            records.append(_canonical_record(
                "PICATION", tup.resnr, tup.restype, tup.reschain,
                coords, {"distance": float(tup.distance), "offset": float(tup.offset)},
                lig_coords=lig_coords))

        elif tname == "halogenbond":
            # Protein/ligand side isn't disambiguated by a flag in this PLIP
            # version's halogen() -- resnr/restype are always derived from
            # acc.o, so acc is treated as protein-side and don (the halogen
            # atom) as ligand-side, matching that fixed convention.
            records.append(_canonical_record(
                "HALOGEN", tup.resnr, tup.restype, tup.reschain,
                tup.acc.o.coords,
                {"distance": float(tup.distance), "don_angle": float(tup.don_angle),
                 "acc_angle": float(tup.acc_angle)},
                lig_coords=tup.don.x.coords))

        elif tname == "waterbridge":
            protein_atom = tup.d if tup.protisdon else tup.a
            ligand_atom = tup.a if tup.protisdon else tup.d
            records.append(_canonical_record(
                "WATERBRIDGE", tup.resnr, tup.restype, tup.reschain,
                protein_atom.coords,
                {"distance_aw": float(tup.distance_aw), "distance_dw": float(tup.distance_dw),
                 "d_angle": float(tup.d_angle), "w_angle": float(tup.w_angle)},
                lig_coords=ligand_atom.coords))

        elif tname == "metal_complex":
            if tup.target.location != "protein":
                continue  # water- or ligand-coordinating contacts aren't protein fingerprint bits
            records.append(_canonical_record(
                "METAL", tup.resnr, tup.restype, tup.reschain,
                tup.target.atom.coords,
                {"distance": float(tup.distance), "rms": float(tup.rms),
                 "geometry": tup.geometry, "coordination_num": tup.coordination_num},
                lig_coords=tup.metal.coords))

    return records


def find_bsid(complex_obj, hetid, chain, resnum):
    """Finds the interaction_sets key for a specific ligand instance. Falls
    back to (1) matching on hetid+resnum only, since PLIP sometimes reports a
    different chain label than the raw PDB residue's chain, and (2) searching
    composite ligands' member lists -- PLIP merges covalently-linked HETATM
    groups into one ligand by default (e.g. an AMP that's covalently part of
    an acyl-adenylate intermediate gets absorbed into the other component's
    bsid), so the target hetid may not appear as its own bsid at all even
    though it's present and being analyzed as part of a larger ligand."""
    exact = f"{hetid}:{chain}:{resnum}"
    if exact in complex_obj.interaction_sets:
        return exact
    for bsid in complex_obj.interaction_sets:
        parts = bsid.split(":")
        if len(parts) == 3 and parts[0] == hetid and parts[2] == str(resnum):
            return bsid
    for ligand in complex_obj.ligands:
        # complex_obj.ligands holds the raw 'ligand' namedtuples from LigandFinder
        # (hetid/chain/position, not the wrapped Ligand class), so bsid is built
        # the same way PDBComplex.characterize_complex builds it internally.
        for member_hetid, member_chain, member_resnum in ligand.members:
            if member_hetid == hetid and member_resnum == resnum:
                return f"{ligand.hetid}:{ligand.chain}:{ligand.position}"
    return None


def extract_interactions(pdb_path, hetid, chain, resnum):
    """Returns (records, error). records is None on failure."""
    try:
        complex_obj = load_complex(pdb_path)
    except Exception as e:
        return None, f"PLIP failed to load/analyze {pdb_path}: {e}"

    bsid = find_bsid(complex_obj, hetid, chain, resnum)
    if bsid is None:
        available = list(complex_obj.interaction_sets.keys())
        return None, f"ligand {hetid}:{chain}:{resnum} not found; available bsids: {available[:15]}"

    return canonicalize(complex_obj.interaction_sets[bsid]), None
