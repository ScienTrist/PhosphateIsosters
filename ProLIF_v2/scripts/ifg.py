"""
Vendored (not imported) from RDKit's own Contrib/IFG/ifg.py -- copied rather than
imported via a Contrib path because Contrib isn't part of RDKit's normal package
namespace and isn't guaranteed to sit at a stable location across RDKit versions
or machines (same reasoning run_prolif.py's own _extract_submol/
safe_molecule_from_mda give for being copied from an earlier prototype instead of
imported). Unmodified except for this docstring.

Implementation of:
    An algorithm to identify functional groups in organic molecules
    Peter Ertl, J. Cheminform (2017) 9:36
    https://jcheminf.springeropen.com/articles/10.1186/s13321-017-0225-z

Original authors: Richard Hall (2017), Guillaume Godin (2017, output refinement).
BSD-licensed, part of RDKit (https://github.com/rdkit/rdkit).

Marks every heteroatom plus a few special carbon cases (non-aromatic C=X/C#X to a
heteroatom, acetal carbons, oxirane/aziridine/thiirane ring atoms), then merges
connected marked atoms into one functional group each -- so a partially-detected
oxyanion (e.g. only 2 of a sulfonate's 3 oxygens) expands to the whole group by
graph connectivity, not by matching a predefined per-chemotype SMARTS pattern.

Usage:
    fgs = identify_functional_groups(mol)  # -> [IFG(atomIds=(...), atoms=..., type=...), ...]
"""
from collections import namedtuple

from rdkit import Chem


def merge(mol, marked, aset):
  bset = set()
  for idx in aset:
    atom = mol.GetAtomWithIdx(idx)
    for nbr in atom.GetNeighbors():
      jdx = nbr.GetIdx()
      if jdx in marked:
        marked.remove(jdx)
        bset.add(jdx)
  if not bset:
    return
  merge(mol, marked, bset)
  aset.update(bset)


# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)


def identify_functional_groups(mol):
  marked = set()
  #mark all heteroatoms in a molecule, including halogens
  for atom in mol.GetAtoms():
    if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
      marked.add(atom.GetIdx())

#mark the four specific types of carbon atom
  for patt in PATT_TUPLE:
    for path in mol.GetSubstructMatches(patt):
      for atomindex in path:
        marked.add(atomindex)

#merge all connected marked atoms to a single FG
  groups = []
  while marked:
    grp = set([marked.pop()])
    merge(mol, marked, grp)
    groups.append(grp)


#extract also connected unmarked carbon atoms
  ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
  ifgs = []
  for g in groups:
    uca = set()
    for atomidx in g:
      for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
        if n.GetAtomicNum() == 6:
          uca.add(n.GetIdx())
    ifgs.append(
      ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
          type=Chem.MolFragmentToSmiles(mol, g.union(uca), canonical=True)))
  return ifgs
