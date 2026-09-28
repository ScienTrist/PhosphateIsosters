"""
Converts SMILES text to DataWarrior's native "idcode" structure encoding
using the REAL OpenChemLib classes bundled inside the local DataWarrior
install (datawarrior_all.jar) via JPype -- not a hand-rolled reimplementation
of the format, so the idcode written out is byte-for-byte what DataWarrior's
own SmilesParser/Canonizer would produce, i.e. exactly as if the user had run
the old "confirm as chemical structure column" import wizard themselves. A
column whose cells hold idcode text and whose <column properties> declare
columnProperty="specialType	idcode" renders as an inline picture the moment
DataWarrior opens the file -- no manual per-column conversion step needed
(see export_ifg_dataset_datawarrior.py's docstring for why that manual step
was needed before this).

This is a dev-machine-only dependency (needs DataWarrior actually installed
at DATAWARRIOR_DIR, for its bundled JRE + jar) -- not something the published
dataset itself needs; if DataWarrior isn't there, importing this module fails
loudly with a clear message rather than silently falling back to plain text.

IMPORTANT for anything that maps ATOM INDICES into an idcode-encoded molecule
(e.g. generate_highlighted_pngs.py's atom_color_cell(), which paints per-atom
highlight colors onto the SMILES column): idcode does NOT preserve
SmilesParser's parse order. Canonizer re-ranks atoms into ITS OWN canonical
order before encoding, and that canonical order -- not the original parse
order -- is what DataWarrior reconstructs when it later parses the idcode
back into a displayed structure. Verified empirically: for
"O=C(CC[C@@H](Cc1ccc(C(=O)[O-])cc1)C(=O)[O-])NO", the parse-order and
idcode-reparsed atomic-number sequences differ outright. graph_index(), from
Canonizer.getGraphIndexes(), is the correct original-parse-order ->
idcode-order mapping (graph_index[i] is where parse-order atom i ends up in
the idcode encoding) -- confirmed by round-tripping the idcode back through
IDCodeParser and checking atom-by-atom that idcode_atoms[graph_index[i]] ==
parse_order_atoms[i] for every i.

Usage as a library:
    conv = IdcodeConverter()
    idcode = conv.smiles_to_idcode("O=C([O-])CO")   # "" on parse failure
    graph_index = conv.graph_index("O=C([O-])CO")   # [] on parse failure

    # SMARTS query patterns (e.g. IFG_Group_SMARTS's "[#6]-[#6](=[#8])-[#8-]")
    # use "#"-atomic-number bracket syntax that plain SMILES mode rejects --
    # smarts_mode=True switches SmilesParser into its SMARTS_MODE_IS_SMARTS
    # mode instead, which idcode-encodes it as a query fragment (idcode itself
    # carries fragment/query state, so DataWarrior renders it distinctly from
    # a real molecule -- matching the rest of this codebase's own point that
    # these SMARTS are "a query pattern, not a molecule").
    smarts_conv = IdcodeConverter(smarts_mode=True)
    idcode = smarts_conv.smiles_to_idcode("[#6]-[#6](=[#8])-[#8-]")
"""
import os

DATAWARRIOR_DIR = r"C:\Program Files\DataWarrior"
JVM_PATH = os.path.join(DATAWARRIOR_DIR, "jre", "bin", "server", "jvm.dll")
JAR_PATH = os.path.join(DATAWARRIOR_DIR, "datawarrior_all.jar")

if not os.path.exists(JVM_PATH) or not os.path.exists(JAR_PATH):
    raise RuntimeError(
        f"DataWarrior install not found at {DATAWARRIOR_DIR} (need both {JVM_PATH} "
        f"and {JAR_PATH}) -- this module calls DataWarrior's own OpenChemLib classes "
        f"to generate idcode, it doesn't reimplement the format itself.")

import jpype  # noqa: E402

if not jpype.isJVMStarted():
    jpype.startJVM(jvmpath=JVM_PATH, classpath=[JAR_PATH])


class IdcodeConverter:
    def __init__(self, smarts_mode=False):
        SmilesParser = jpype.JClass("com.actelion.research.chem.SmilesParser")
        self._StereoMolecule = jpype.JClass("com.actelion.research.chem.StereoMolecule")
        self._Canonizer = jpype.JClass("com.actelion.research.chem.Canonizer")
        mode = SmilesParser.SMARTS_MODE_IS_SMARTS if smarts_mode else SmilesParser.SMARTS_MODE_IS_SMILES
        self._parser = SmilesParser(mode, True)
        self._cache = {}  # key -> (idcode, graph_index)
        self.n_failed = 0
        self.failures = []  # (original_cell_text, exception_str), capped -- see _convert

    def _convert(self, smiles):
        # "; "-joined multi-group cells (see export_ifg_dataset_datawarrior.py's
        # module docstring, IFG_Group_SMILES) aren't valid SMILES on their own --
        # "." is OpenChemLib's (and RDKit's) own disconnected-component separator,
        # so this depicts every joined group side by side in one picture instead
        # of failing to parse.
        key = smiles.replace("; ", ".")
        if key in self._cache:
            return self._cache[key]
        try:
            mol = self._StereoMolecule()
            self._parser.parse(mol, key)
            canonizer = self._Canonizer(mol)
            idcode = str(canonizer.getIDCode())
            graph_index = [int(x) for x in canonizer.getGraphIndexes()]
        except Exception as e:
            idcode, graph_index = "", []
            self.n_failed += 1
            if len(self.failures) < 20:
                self.failures.append((smiles, f"{type(e).__name__}: {e}"))
        self._cache[key] = (idcode, graph_index)
        return self._cache[key]

    def smiles_to_idcode(self, smiles):
        if not smiles:
            return ""
        return self._convert(smiles)[0]

    def graph_index(self, smiles):
        """graph_index[i] = position, in this SMILES's idcode encoding, of
        the atom that SmilesParser assigned index i to when parsing the SAME
        text (i.e. the atom-order RDKit's own MolToSmiles/_smilesAtomOutputOrder
        agrees with -- see generate_highlighted_pngs.py). Empty list if smiles
        is blank or fails to parse."""
        if not smiles:
            return []
        return self._convert(smiles)[1]
