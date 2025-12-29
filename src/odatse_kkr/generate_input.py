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
    "apply_kkr_parameters_from_config",
    "count_atoms_by_type",
    "list_atomic_positions",
    "load_input_file",
    "modify_atom_type_definition",
    "modify_kkr_parameters",
    "parse_atomic_positions",
    "parse_atom_type_definitions",
    "parse_kkr_parameters",
    "replace_atom_types",
    "replace_atom_types_by_coordinates",
    "replace_atom_types_by_label",
    "write_input_file",
]


def parse_kkr_parameters(input_lines: List[str]) -> Dict:
    """
    Extract KKR calculation parameters from an AkaiKKR input file.

    This function parses the header section of an AkaiKKR input file and
    extracts lattice parameters, calculation parameters, and output settings.

    Parameters
    ----------
    input_lines : list of str
        Lines of the input file.

    Returns
    -------
    dict
        Dictionary containing:
        - lattice: dict with brvtyp, a, c_a, b_a, alpha, beta, gamma
        - calculation: dict with edelt, ewidth, reltyp, sdftyp, magtyp, record
        - output: dict with outtyp, bzqlty, maxitr, pmix
        - go: dict with command, pot_file (first line parameters)
        - line_indices: dict mapping parameter names to line indices
    """
    result: Dict = {
        "lattice": {},
        "calculation": {},
        "output": {},
        "go": {},
        "line_indices": {},
    }

    i = 0
    while i < len(input_lines):
        line = input_lines[i].strip()

        # Parse "go" line (first command line)
        if line.startswith("go"):
            parts = line.split()
            if len(parts) >= 2:
                result["go"]["command"] = parts[0]
                result["go"]["pot_file"] = parts[1]
            result["line_indices"]["go"] = i
            i += 1
            continue

        # Parse lattice parameters (brvtyp a c/a b/a alpha beta gamma)
        if "brvtyp" in line.lower() and line.startswith("c"):
            i += 1
            while i < len(input_lines):
                next_line = input_lines[i].strip()
                if next_line and not next_line.startswith("c"):
                    parts = next_line.split()
                    if len(parts) >= 7:
                        result["lattice"]["brvtyp"] = parts[0]
                        result["lattice"]["a"] = float(parts[1])
                        result["lattice"]["c_a"] = float(parts[2])
                        result["lattice"]["b_a"] = float(parts[3])
                        result["lattice"]["alpha"] = float(parts[4])
                        result["lattice"]["beta"] = float(parts[5])
                        result["lattice"]["gamma"] = float(parts[6])
                        result["line_indices"]["lattice"] = i
                    break
                i += 1
            i += 1
            continue

        # Parse calculation parameters (edelt ewidth reltyp sdftyp magtyp record)
        if "edelt" in line.lower() and line.startswith("c"):
            i += 1
            while i < len(input_lines):
                next_line = input_lines[i].strip()
                if next_line and not next_line.startswith("c"):
                    parts = next_line.split()
                    if len(parts) >= 6:
                        result["calculation"]["edelt"] = float(parts[0])
                        result["calculation"]["ewidth"] = float(parts[1])
                        result["calculation"]["reltyp"] = parts[2]
                        result["calculation"]["sdftyp"] = parts[3]
                        result["calculation"]["magtyp"] = parts[4]
                        result["calculation"]["record"] = parts[5]
                        result["line_indices"]["calculation"] = i
                    break
                i += 1
            i += 1
            continue

        # Parse output parameters (outtyp bzqlty maxitr pmix)
        if "outtyp" in line.lower() and line.startswith("c"):
            i += 1
            while i < len(input_lines):
                next_line = input_lines[i].strip()
                if next_line and not next_line.startswith("c"):
                    parts = next_line.split()
                    if len(parts) >= 4:
                        result["output"]["outtyp"] = parts[0]
                        result["output"]["bzqlty"] = int(parts[1])
                        result["output"]["maxitr"] = int(parts[2])
                        result["output"]["pmix"] = float(parts[3])
                        result["line_indices"]["output"] = i
                    break
                i += 1
            i += 1
            continue

        # Stop parsing when we reach ntyp section
        if "ntyp" in line.lower():
            break

        i += 1

    return result


def modify_kkr_parameters(
    input_data: Dict,
    lattice: Optional[Dict] = None,
    calculation: Optional[Dict] = None,
    output: Optional[Dict] = None,
    go: Optional[Dict] = None,
) -> Dict:
    """
    Modify KKR parameters in structured input data.

    Parameters
    ----------
    input_data : dict
        Structured data from load_input_file().
    lattice : dict, optional
        Lattice parameters to update. Keys: brvtyp, a, c_a, b_a, alpha, beta, gamma.
    calculation : dict, optional
        Calculation parameters to update. Keys: edelt, ewidth, reltyp, sdftyp,
        magtyp, record.
    output : dict, optional
        Output parameters to update. Keys: outtyp, bzqlty, maxitr, pmix.
    go : dict, optional
        Go command parameters. Keys: command, pot_file.

    Returns
    -------
    dict
        New structured data with updated parameters.

    Examples
    --------
    >>> data = load_input_file("template.in")
    >>> new_data = modify_kkr_parameters(
    ...     data,
    ...     lattice={"a": 7.5, "c_a": 3.1},
    ...     calculation={"edelt": 0.0005, "maxitr": 300}
    ... )
    >>> write_input_file(new_data, "modified.in")
    """
    new_data = _copy_base(input_data)

    # Copy KKR parameters if they exist
    if "kkr_parameters" in input_data:
        new_data["kkr_parameters"] = {
            "lattice": input_data["kkr_parameters"].get("lattice", {}).copy(),
            "calculation": input_data["kkr_parameters"].get("calculation", {}).copy(),
            "output": input_data["kkr_parameters"].get("output", {}).copy(),
            "go": input_data["kkr_parameters"].get("go", {}).copy(),
            "line_indices": input_data["kkr_parameters"].get("line_indices", {}).copy(),
        }
    else:
        new_data["kkr_parameters"] = {
            "lattice": {},
            "calculation": {},
            "output": {},
            "go": {},
            "line_indices": {},
        }

    # Update lattice parameters
    if lattice:
        for key, value in lattice.items():
            new_data["kkr_parameters"]["lattice"][key] = value

    # Update calculation parameters
    if calculation:
        for key, value in calculation.items():
            new_data["kkr_parameters"]["calculation"][key] = value

    # Update output parameters
    if output:
        for key, value in output.items():
            new_data["kkr_parameters"]["output"][key] = value

    # Update go parameters
    if go:
        for key, value in go.items():
            new_data["kkr_parameters"]["go"][key] = value

    # Update header lines with new parameters
    new_header = _rebuild_header_with_kkr_params(
        input_data["header"],
        new_data["kkr_parameters"],
    )
    new_data["header"] = new_header

    return new_data


def _rebuild_header_with_kkr_params(
    header_lines: List[str],
    kkr_params: Dict,
) -> List[str]:
    """
    Rebuild header lines with updated KKR parameters.

    Parameters
    ----------
    header_lines : list of str
        Original header lines.
    kkr_params : dict
        KKR parameters containing lattice, calculation, output, go sections.

    Returns
    -------
    list of str
        Updated header lines.
    """
    new_lines = header_lines[:]
    line_indices = kkr_params.get("line_indices", {})

    # Update go line
    if "go" in line_indices and kkr_params.get("go"):
        idx = line_indices["go"]
        if idx < len(new_lines):
            go_params = kkr_params["go"]
            command = go_params.get("command", "go")
            pot_file = go_params.get("pot_file", "pot.dat")
            new_lines[idx] = f"    {command}  {pot_file}\n"

    # Update lattice line
    if "lattice" in line_indices and kkr_params.get("lattice"):
        idx = line_indices["lattice"]
        if idx < len(new_lines):
            lat = kkr_params["lattice"]
            brvtyp = lat.get("brvtyp", "so")
            a = lat.get("a", 1.0)
            c_a = lat.get("c_a", 1.0)
            b_a = lat.get("b_a", 1.0)
            alpha = lat.get("alpha", 90.0)
            beta = lat.get("beta", 90.0)
            gamma = lat.get("gamma", 90.0)
            new_lines[idx] = f"    {brvtyp}  {a}  {c_a}  {b_a}  {alpha}  {beta}  {gamma}\n"

    # Update calculation line
    if "calculation" in line_indices and kkr_params.get("calculation"):
        idx = line_indices["calculation"]
        if idx < len(new_lines):
            calc = kkr_params["calculation"]
            edelt = calc.get("edelt", 0.001)
            ewidth = calc.get("ewidth", 2.0)
            reltyp = calc.get("reltyp", "sra")
            sdftyp = calc.get("sdftyp", "mjw")
            magtyp = calc.get("magtyp", "mag")
            record = calc.get("record", "init")
            new_lines[idx] = f"    {edelt}  {ewidth}  {reltyp}  {sdftyp}  {magtyp}  {record}\n"

    # Update output line
    if "output" in line_indices and kkr_params.get("output"):
        idx = line_indices["output"]
        if idx < len(new_lines):
            out = kkr_params["output"]
            outtyp = out.get("outtyp", "update")
            bzqlty = out.get("bzqlty", 6)
            maxitr = out.get("maxitr", 200)
            pmix = out.get("pmix", 0.02)
            new_lines[idx] = f"    {outtyp}  {bzqlty}  {maxitr}  {pmix}\n"

    return new_lines


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
    """
    Load AkaiKKR input file and return structured data.

    Parameters
    ----------
    input_path : str or Path
        Path to the AkaiKKR input file.

    Returns
    -------
    dict
        Structured data containing:
        - header: list of header lines
        - ntyp: number of atom types
        - atom_type_definitions: list of atom type definitions
        - atomic_header: header lines for atomic positions section
        - atomic_positions: list of (x, y, z, atmtyp) tuples
        - footer: list of footer lines
        - kkr_parameters: dict with lattice, calculation, output, go parameters
    """
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    # Parse KKR parameters from header
    kkr_parameters = parse_kkr_parameters(lines)

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
        "kkr_parameters": kkr_parameters,
    }


def _copy_base(input_data: Dict) -> Dict:
    """Return a shallow copy of the structured data."""
    result = {
        "header": input_data["header"][:],
        "ntyp": input_data.get("ntyp", 0),
        "atom_type_definitions": [
            defn.copy() for defn in input_data.get("atom_type_definitions", [])
        ],
        "atomic_header": input_data["atomic_header"][:],
        "atomic_positions": input_data["atomic_positions"][:],
        "footer": input_data["footer"][:],
    }
    # Copy KKR parameters if present
    if "kkr_parameters" in input_data:
        result["kkr_parameters"] = {
            "lattice": input_data["kkr_parameters"].get("lattice", {}).copy(),
            "calculation": input_data["kkr_parameters"].get("calculation", {}).copy(),
            "output": input_data["kkr_parameters"].get("output", {}).copy(),
            "go": input_data["kkr_parameters"].get("go", {}).copy(),
            "line_indices": input_data["kkr_parameters"].get("line_indices", {}).copy(),
        }
    return result


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


def apply_kkr_parameters_from_config(
    input_data: Dict,
    config: Dict,
) -> Dict:
    """
    Apply KKR parameters from a TOML configuration to structured input data.

    This function reads the [kkr] section from a configuration dictionary
    and applies the parameters to the input data.

    Parameters
    ----------
    input_data : dict
        Structured data from load_input_file().
    config : dict
        Configuration dictionary (typically from TOML file) containing
        an optional [kkr] section with lattice, calculation, output,
        and go subsections.

    Returns
    -------
    dict
        New structured data with applied KKR parameters.

    Examples
    --------
    TOML configuration example::

        [kkr]
        # Go command settings
        [kkr.go]
        command = "go"
        pot_file = "pot.dat"

        # Bravais lattice parameters
        [kkr.lattice]
        brvtyp = "so"
        a = 7.265372455718975
        c_a = 3.0753407056213957
        b_a = 1.0211940276767721
        alpha = 90.0
        beta = 90.0
        gamma = 90.0

        # Calculation parameters
        [kkr.calculation]
        edelt = 0.001
        ewidth = 2.0
        reltyp = "sra"
        sdftyp = "mjw"
        magtyp = "mag"
        record = "init"

        # Output parameters
        [kkr.output]
        outtyp = "update"
        bzqlty = 6
        maxitr = 200
        pmix = 0.02

    Usage::

        import tomllib
        from odatse_kkr import load_input_file, apply_kkr_parameters_from_config

        with open("config.toml", "rb") as f:
            config = tomllib.load(f)

        data = load_input_file("template.in")
        data = apply_kkr_parameters_from_config(data, config)
        write_input_file(data, "output.in")
    """
    kkr_config = config.get("kkr", {})

    if not kkr_config:
        # No KKR configuration, return unchanged
        return input_data

    lattice = kkr_config.get("lattice")
    calculation = kkr_config.get("calculation")
    output = kkr_config.get("output")
    go = kkr_config.get("go")

    return modify_kkr_parameters(
        input_data,
        lattice=lattice,
        calculation=calculation,
        output=output,
        go=go,
    )
