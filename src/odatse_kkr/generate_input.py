"""
AkaiKKR input file helpers shared across ODAT-SE frontends.

The functions here were originally implemented inside ``odatse-specx``'s
``generate_input.py``.  They are now hosted in ``odatse-kkr`` so that the
conductivity and defect-energy objectives can reuse them without depending on
the full odatse-specx package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

__all__ = [
    "add_atom_type_definition",
    "count_atoms_by_type",
    "list_atomic_positions",
    "load_input_file",
    "modify_atom_type_definition",
    "parse_atomic_positions",
    "parse_atom_type_definitions",
    "replace_atom_types",
    "replace_atom_types_by_coordinates",
    "replace_atom_types_by_label",
    "write_input_file",
]


def parse_atomic_positions(
    input_lines: List[str],
) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    """
    Extract atomic position information from an AkaiKKR input file.
    """
    header_lines = []
    atomic_positions: List[Tuple[str, str, str, str]] = []
    in_atomic_section = False

    for line in input_lines:
        if "atmicx" in line.lower() or "atmtyp" in line.lower():
            in_atomic_section = True
            header_lines.append(line)
            continue

        if in_atomic_section:
            stripped = line.strip()
            if not stripped or stripped.startswith("c") or stripped.startswith("#"):
                if not stripped.startswith("end"):
                    header_lines.append(line)
                continue

            if "end" in stripped.lower():
                break

            parts = stripped.split()
            if len(parts) >= 4:
                x, y, z, atmtyp = parts[:4]
                atomic_positions.append((x, y, z, atmtyp))

    return header_lines, atomic_positions


def parse_atom_type_definitions(
    input_lines: List[str],
) -> Tuple[int, List[Dict], int]:
    """
    Extract atom type definition information from an AkaiKKR input file.
    """
    ntyp: Optional[int] = None
    atom_types: List[Dict] = []
    ntyp_start_idx = 0
    i = 0

    while i < len(input_lines):
        line = input_lines[i]
        stripped = line.strip()
        if "ntyp" in stripped.lower():
            ntyp_start_idx = i
            i += 1
            while i < len(input_lines):
                next_line = input_lines[i].strip()
                if next_line and not next_line.startswith("c"):
                    try:
                        ntyp = int(next_line.split()[0])
                    except (ValueError, IndexError):
                        pass
                    i += 1
                    break
                i += 1
            break
        i += 1

    while i < len(input_lines):
        stripped = input_lines[i].strip()
        if "typ" in stripped.lower() and "ncmp" in stripped.lower():
            i += 1
            break
        i += 1

    while i < len(input_lines) and len(atom_types) < (ntyp or float("inf")):
        stripped = input_lines[i].strip()
        if not stripped or stripped.startswith(("c", "#")):
            if "natm" in stripped.lower() or "atmicx" in stripped.lower():
                break
            i += 1
            continue

        if "natm" in stripped.lower() or "atmicx" in stripped.lower():
            break

        parts = stripped.split()
        if len(parts) >= 5:
            type_name = parts[0]
            try:
                ncmp = int(parts[1])
                rmt = float(parts[2])
                field = float(parts[3])
                mxl = int(parts[4])
            except (ValueError, IndexError):
                i += 1
                continue

            current_type = {
                "type": type_name,
                "ncmp": ncmp,
                "rmt": rmt,
                "field": field,
                "mxl": mxl,
                "atoms": [],
            }

            i += 1
            atom_count = 0
            while atom_count < ncmp and i < len(input_lines):
                atom_line_clean = input_lines[i].replace("\t", " ").strip()
                if atom_line_clean and not atom_line_clean.startswith("c"):
                    atom_parts = atom_line_clean.split()
                    if len(atom_parts) >= 2:
                        try:
                            anclr = int(atom_parts[0])
                            conc = float(atom_parts[1])
                            current_type["atoms"].append((anclr, conc))
                            atom_count += 1
                        except (ValueError, IndexError):
                            pass
                i += 1

            if len(current_type["atoms"]) == ncmp:
                atom_types.append(current_type)
        else:
            i += 1

    if ntyp is None:
        ntyp = len(atom_types)

    return ntyp, atom_types, ntyp_start_idx


def load_input_file(input_path: Union[str, Path]) -> Dict:
    """Load AkaiKKR input file and return structured data."""
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    ntyp, atom_type_definitions, ntyp_start_idx = parse_atom_type_definitions(lines)
    if ntyp is None:
        ntyp = len(atom_type_definitions)

    atomic_start_idx = None
    for idx, line in enumerate(lines):
        if "atmicx" in line.lower() or "atmtyp" in line.lower():
            atomic_start_idx = idx
            break

    if atomic_start_idx is None:
        raise ValueError("atmicx atmtyp section not found in input file")

    header = lines[:ntyp_start_idx]
    atomic_header, atomic_positions = parse_atomic_positions(lines[atomic_start_idx:])

    footer: List[str] = []
    end_found = False
    atomic_section_len = len(atomic_header) + len(atomic_positions)
    for idx in range(atomic_start_idx + len(atomic_header), len(lines)):
        line = lines[idx]
        if "end" in line.lower() and not line.strip().startswith("c"):
            footer.append(line)
            end_found = True
            break
        if end_found or idx >= atomic_start_idx + atomic_section_len:
            footer.append(line)

    return {
        "header": header,
        "ntyp": ntyp,
        "atom_type_definitions": atom_type_definitions,
        "atomic_header": atomic_header,
        "atomic_positions": atomic_positions,
        "footer": footer,
    }


def _copy_base(input_data: Dict) -> Dict:
    """Return a shallow copy of the structured data."""
    return {
        "header": input_data["header"][:],
        "ntyp": input_data.get("ntyp", 0),
        "atom_type_definitions": [
            defn.copy() for defn in input_data.get("atom_type_definitions", [])
        ],
        "atomic_header": input_data["atomic_header"][:],
        "atomic_positions": input_data["atomic_positions"][:],
        "footer": input_data["footer"][:],
    }


def replace_atom_types(input_data: Dict, atom_type_mapping: Dict[int, str]) -> Dict:
    """Replace atom types in structured data by index specification."""
    new_data = _copy_base(input_data)
    new_positions: List[Tuple[str, str, str, str]] = []
    for idx, (x, y, z, atmtyp) in enumerate(input_data["atomic_positions"]):
        new_positions.append((x, y, z, atom_type_mapping.get(idx, atmtyp)))
    new_data["atomic_positions"] = new_positions
    return new_data


def replace_atom_types_by_coordinates(
    input_data: Dict,
    coordinate_mapping: Dict[Tuple[str, str, str], str],
) -> Dict:
    """Replace atom types in structured data by coordinate specification."""
    new_data = _copy_base(input_data)
    new_positions: List[Tuple[str, str, str, str]] = []
    for x, y, z, atmtyp in input_data["atomic_positions"]:
        coord_key = (x, y, z)
        new_positions.append((x, y, z, coordinate_mapping.get(coord_key, atmtyp)))
    new_data["atomic_positions"] = new_positions
    return new_data


def replace_atom_types_by_label(
    input_data: Dict,
    label_mapping: Dict[str, str],
) -> Dict:
    """Replace all atoms with the same label (atmtyp) using a label mapping."""
    new_data = _copy_base(input_data)
    new_positions: List[Tuple[str, str, str, str]] = []
    for x, y, z, atmtyp in input_data["atomic_positions"]:
        new_positions.append((x, y, z, label_mapping.get(atmtyp, atmtyp)))
    new_data["atomic_positions"] = new_positions
    return new_data


def add_atom_type_definition(
    input_data: Dict,
    type_name: str,
    ncmp: int,
    rmt: float,
    field: float,
    mxl: int,
    atoms: List[Tuple[int, float]],
) -> Dict:
    """Add a new atom species definition."""
    new_data = _copy_base(input_data)
    ntyp_value = new_data.get("ntyp") or len(new_data.get("atom_type_definitions", []))
    new_data["ntyp"] = ntyp_value + 1
    new_type_def = {
        "type": type_name,
        "ncmp": ncmp,
        "rmt": rmt,
        "field": field,
        "mxl": mxl,
        "atoms": atoms[:],
    }
    new_data["atom_type_definitions"].append(new_type_def)
    return new_data


def modify_atom_type_definition(
    input_data: Dict,
    type_name: str,
    atoms: Optional[List[Tuple[int, float]]] = None,
    ncmp: Optional[int] = None,
    rmt: Optional[float] = None,
    field: Optional[float] = None,
    mxl: Optional[int] = None,
) -> Dict:
    """Modify an existing atom type definition."""
    new_data = _copy_base(input_data)
    new_defs: List[Dict] = []
    found = False

    for defn in input_data.get("atom_type_definitions", []):
        new_defn = {
            "type": defn["type"],
            "ncmp": defn["ncmp"],
            "rmt": defn["rmt"],
            "field": defn["field"],
            "mxl": defn["mxl"],
            "atoms": list(defn["atoms"]),
        }
        if defn["type"] == type_name:
            found = True
            if atoms is not None:
                new_defn["atoms"] = list(atoms)
                if ncmp is None:
                    new_defn["ncmp"] = len(atoms)
            if ncmp is not None:
                new_defn["ncmp"] = ncmp
            if rmt is not None:
                new_defn["rmt"] = rmt
            if field is not None:
                new_defn["field"] = field
            if mxl is not None:
                new_defn["mxl"] = mxl
        new_defs.append(new_defn)

    if not found:
        raise ValueError(f"Atom type '{type_name}' not found in input data.")

    new_data["atom_type_definitions"] = new_defs
    return new_data


def count_atoms_by_type(input_data: Dict, type_name: str) -> int:
    """Count the number of atoms with the specified type in atomic positions."""
    return sum(1 for _, _, _, atmtyp in input_data["atomic_positions"] if atmtyp == type_name)


def write_input_file(input_data: Dict, output_path: Union[str, Path]) -> None:
    """Write structured data as an AkaiKKR input file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.writelines(input_data["header"])

        if "ntyp" in input_data and "atom_type_definitions" in input_data:
            used_types = {atmtyp for _, _, _, atmtyp in input_data["atomic_positions"]}
            type_def_dict = {defn["type"]: defn for defn in input_data["atom_type_definitions"]}
            used_definitions: List[Dict] = []
            for _, _, _, atmtyp in input_data["atomic_positions"]:
                if atmtyp in type_def_dict and atmtyp not in [d["type"] for d in used_definitions]:
                    used_definitions.append(type_def_dict[atmtyp])

            fp.write("c------------------------------------------------------------\n")
            fp.write("c   ntyp\n")
            fp.write(f"    {len(used_definitions)}\n")
            fp.write("c------------------------------------------------------------\n")
            fp.write("c   typ ncmp rmt field mxl [anclr conc]\n")

            for type_def in used_definitions:
                fp.write(
                    f"    {type_def['type']}  {type_def['ncmp']}  "
                    f"{type_def['rmt']}  {type_def['field']}  {type_def['mxl']}\n"
                )
                for anclr, conc in type_def["atoms"]:
                    fp.write(f"                              {anclr}  {conc}\n")

        natm = len(input_data["atomic_positions"])
        fp.write("c------------------------------------------------------------\n")
        fp.write("c   natm\n")
        fp.write(f"    {natm}\n")
        fp.write("c------------------------------------------------------------\n")

        header_lines = [
            line
            for line in input_data["atomic_header"]
            if not line.strip().startswith("c---")
        ]
        fp.writelines(header_lines)

        for x, y, z, atmtyp in input_data["atomic_positions"]:
            fp.write(f"    {x}  {y}  {z}  {atmtyp}\n")

        fp.writelines(input_data["footer"])


def list_atomic_positions(input_data: Union[Dict, str, Path]) -> None:
    """List atomic positions in structured data or direct input file."""
    if isinstance(input_data, (str, Path)):
        input_data = load_input_file(input_data)

    print(f"Found {len(input_data['atomic_positions'])} atomic positions:")
    for idx, (x, y, z, atmtyp) in enumerate(input_data["atomic_positions"]):
        print(f"Index {idx}: ({x}, {y}, {z}) -> {atmtyp}")
