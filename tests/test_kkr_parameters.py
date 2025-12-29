"""
Tests for KKR parameter parsing and modification functions.
"""

import tempfile
from pathlib import Path

import pytest

from odatse_kkr import (
    apply_kkr_parameters_from_config,
    load_input_file,
    modify_kkr_parameters,
    parse_kkr_parameters,
    write_input_file,
)


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEMPLATE_PATH = FIXTURES_DIR / "template.in"


class TestParseKkrParameters:
    """Tests for parse_kkr_parameters function."""

    def test_parse_lattice_parameters(self):
        """Test parsing of lattice parameters from input file."""
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        result = parse_kkr_parameters(lines)

        assert "lattice" in result
        lattice = result["lattice"]
        assert lattice["brvtyp"] == "so"
        assert lattice["a"] == pytest.approx(7.265372455718975)
        assert lattice["c_a"] == pytest.approx(3.0753407056213957)
        assert lattice["b_a"] == pytest.approx(1.0211940276767721)
        assert lattice["alpha"] == pytest.approx(90.0)
        assert lattice["beta"] == pytest.approx(90.0)
        assert lattice["gamma"] == pytest.approx(90.0)

    def test_parse_calculation_parameters(self):
        """Test parsing of calculation parameters from input file."""
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        result = parse_kkr_parameters(lines)

        assert "calculation" in result
        calc = result["calculation"]
        assert calc["edelt"] == pytest.approx(0.001)
        assert calc["ewidth"] == pytest.approx(2.0)
        assert calc["reltyp"] == "sra"
        assert calc["sdftyp"] == "mjw"
        assert calc["magtyp"] == "mag"
        assert calc["record"] == "init"

    def test_parse_output_parameters(self):
        """Test parsing of output parameters from input file."""
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        result = parse_kkr_parameters(lines)

        assert "output" in result
        output = result["output"]
        assert output["outtyp"] == "update"
        assert output["bzqlty"] == 6
        assert output["maxitr"] == 200
        assert output["pmix"] == pytest.approx(0.02)

    def test_parse_go_parameters(self):
        """Test parsing of go command parameters from input file."""
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        result = parse_kkr_parameters(lines)

        assert "go" in result
        go = result["go"]
        assert go["command"] == "go"
        assert go["pot_file"] == "pot.dat"

    def test_parse_line_indices(self):
        """Test that line indices are correctly recorded."""
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        result = parse_kkr_parameters(lines)

        assert "line_indices" in result
        indices = result["line_indices"]
        assert "go" in indices
        assert "lattice" in indices
        assert "calculation" in indices
        assert "output" in indices


class TestModifyKkrParameters:
    """Tests for modify_kkr_parameters function."""

    def test_modify_lattice_parameters(self):
        """Test modifying lattice parameters."""
        data = load_input_file(TEMPLATE_PATH)

        new_data = modify_kkr_parameters(
            data,
            lattice={"a": 8.0, "c_a": 3.5},
        )

        assert new_data["kkr_parameters"]["lattice"]["a"] == 8.0
        assert new_data["kkr_parameters"]["lattice"]["c_a"] == 3.5
        # Original values should be preserved for unmodified parameters
        assert new_data["kkr_parameters"]["lattice"]["brvtyp"] == "so"

    def test_modify_calculation_parameters(self):
        """Test modifying calculation parameters."""
        data = load_input_file(TEMPLATE_PATH)

        new_data = modify_kkr_parameters(
            data,
            calculation={"edelt": 0.0005, "maxitr": 500},
        )

        assert new_data["kkr_parameters"]["calculation"]["edelt"] == 0.0005
        # maxitr is in output section, not calculation
        assert new_data["kkr_parameters"]["calculation"]["reltyp"] == "sra"

    def test_modify_output_parameters(self):
        """Test modifying output parameters."""
        data = load_input_file(TEMPLATE_PATH)

        new_data = modify_kkr_parameters(
            data,
            output={"maxitr": 500, "pmix": 0.01},
        )

        assert new_data["kkr_parameters"]["output"]["maxitr"] == 500
        assert new_data["kkr_parameters"]["output"]["pmix"] == 0.01
        assert new_data["kkr_parameters"]["output"]["outtyp"] == "update"

    def test_modify_multiple_sections(self):
        """Test modifying multiple parameter sections at once."""
        data = load_input_file(TEMPLATE_PATH)

        new_data = modify_kkr_parameters(
            data,
            lattice={"a": 8.0},
            calculation={"edelt": 0.0005},
            output={"maxitr": 500},
        )

        assert new_data["kkr_parameters"]["lattice"]["a"] == 8.0
        assert new_data["kkr_parameters"]["calculation"]["edelt"] == 0.0005
        assert new_data["kkr_parameters"]["output"]["maxitr"] == 500

    def test_original_data_unchanged(self):
        """Test that original data is not modified."""
        data = load_input_file(TEMPLATE_PATH)
        original_a = data["kkr_parameters"]["lattice"]["a"]

        modify_kkr_parameters(data, lattice={"a": 8.0})

        assert data["kkr_parameters"]["lattice"]["a"] == original_a


class TestApplyKkrParametersFromConfig:
    """Tests for apply_kkr_parameters_from_config function."""

    def test_apply_from_toml_config(self):
        """Test applying KKR parameters from TOML-style config dict."""
        data = load_input_file(TEMPLATE_PATH)
        config = {
            "kkr": {
                "lattice": {"a": 8.5, "c_a": 3.2},
                "calculation": {"edelt": 0.002},
                "output": {"maxitr": 300, "pmix": 0.015},
            }
        }

        new_data = apply_kkr_parameters_from_config(data, config)

        assert new_data["kkr_parameters"]["lattice"]["a"] == 8.5
        assert new_data["kkr_parameters"]["lattice"]["c_a"] == 3.2
        assert new_data["kkr_parameters"]["calculation"]["edelt"] == 0.002
        assert new_data["kkr_parameters"]["output"]["maxitr"] == 300
        assert new_data["kkr_parameters"]["output"]["pmix"] == 0.015

    def test_apply_empty_config(self):
        """Test that empty config returns unchanged data."""
        data = load_input_file(TEMPLATE_PATH)
        original_a = data["kkr_parameters"]["lattice"]["a"]

        new_data = apply_kkr_parameters_from_config(data, {})

        assert new_data["kkr_parameters"]["lattice"]["a"] == original_a

    def test_apply_config_without_kkr_section(self):
        """Test config without [kkr] section returns unchanged data."""
        data = load_input_file(TEMPLATE_PATH)
        config = {
            "base": {"dimension": 1},
            "algorithm": {"name": "mapper"},
        }
        original_a = data["kkr_parameters"]["lattice"]["a"]

        new_data = apply_kkr_parameters_from_config(data, config)

        assert new_data["kkr_parameters"]["lattice"]["a"] == original_a

    def test_apply_partial_lattice_config(self):
        """Test applying only some lattice parameters."""
        data = load_input_file(TEMPLATE_PATH)
        config = {
            "kkr": {
                "lattice": {"a": 9.0},
            }
        }
        original_c_a = data["kkr_parameters"]["lattice"]["c_a"]

        new_data = apply_kkr_parameters_from_config(data, config)

        assert new_data["kkr_parameters"]["lattice"]["a"] == 9.0
        assert new_data["kkr_parameters"]["lattice"]["c_a"] == original_c_a


class TestWriteInputFileWithKkrParameters:
    """Tests for writing input files with modified KKR parameters."""

    def test_write_modified_lattice(self):
        """Test writing input file with modified lattice parameters."""
        data = load_input_file(TEMPLATE_PATH)
        data = modify_kkr_parameters(data, lattice={"a": 8.0})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            write_input_file(data, tmp_path)

            # Read back and verify
            new_data = load_input_file(tmp_path)
            assert new_data["kkr_parameters"]["lattice"]["a"] == pytest.approx(8.0)
        finally:
            tmp_path.unlink()

    def test_write_modified_calculation(self):
        """Test writing input file with modified calculation parameters."""
        data = load_input_file(TEMPLATE_PATH)
        data = modify_kkr_parameters(data, calculation={"edelt": 0.0005})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            write_input_file(data, tmp_path)

            # Read back and verify
            new_data = load_input_file(tmp_path)
            assert new_data["kkr_parameters"]["calculation"]["edelt"] == pytest.approx(
                0.0005
            )
        finally:
            tmp_path.unlink()

    def test_write_modified_output(self):
        """Test writing input file with modified output parameters."""
        data = load_input_file(TEMPLATE_PATH)
        data = modify_kkr_parameters(data, output={"maxitr": 500, "pmix": 0.01})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            write_input_file(data, tmp_path)

            # Read back and verify
            new_data = load_input_file(tmp_path)
            assert new_data["kkr_parameters"]["output"]["maxitr"] == 500
            assert new_data["kkr_parameters"]["output"]["pmix"] == pytest.approx(0.01)
        finally:
            tmp_path.unlink()

    def test_roundtrip_preserves_structure(self):
        """Test that load->modify->write->load preserves file structure."""
        data = load_input_file(TEMPLATE_PATH)

        # Modify several parameters
        data = modify_kkr_parameters(
            data,
            lattice={"a": 7.5},
            output={"maxitr": 400},
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            write_input_file(data, tmp_path)

            # Read back and verify structure is preserved
            new_data = load_input_file(tmp_path)

            # Check modified values
            assert new_data["kkr_parameters"]["lattice"]["a"] == pytest.approx(7.5)
            assert new_data["kkr_parameters"]["output"]["maxitr"] == 400

            # Check preserved values
            assert new_data["ntyp"] == data["ntyp"]
            assert len(new_data["atomic_positions"]) == len(data["atomic_positions"])
            assert len(new_data["atom_type_definitions"]) == len(
                data["atom_type_definitions"]
            )
        finally:
            tmp_path.unlink()


class TestLoadInputFileWithKkrParameters:
    """Tests for load_input_file including KKR parameters."""

    def test_load_includes_kkr_parameters(self):
        """Test that load_input_file includes kkr_parameters in result."""
        data = load_input_file(TEMPLATE_PATH)

        assert "kkr_parameters" in data
        assert "lattice" in data["kkr_parameters"]
        assert "calculation" in data["kkr_parameters"]
        assert "output" in data["kkr_parameters"]
        assert "go" in data["kkr_parameters"]

    def test_load_preserves_other_data(self):
        """Test that loading still includes header, atoms, etc."""
        data = load_input_file(TEMPLATE_PATH)

        assert "header" in data
        assert "ntyp" in data
        assert "atom_type_definitions" in data
        assert "atomic_positions" in data
        assert "footer" in data

