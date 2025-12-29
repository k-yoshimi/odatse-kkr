"""
Tests for retry functionality with convergence failure handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from odatse_kkr import (
    ConvergenceError,
    RetryConfig,
    check_convergence,
    run_with_retry,
)
from odatse_kkr.retry import create_retry_input


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEMPLATE_PATH = FIXTURES_DIR / "template.in"


class TestCheckConvergence:
    """Tests for check_convergence function."""

    def test_converged_output(self, tmp_path):
        """Test detection of converged output."""
        output_file = tmp_path / "converged.out"
        output_file.write_text(
            "Some calculation output\n"
            "total energy= -123.456\n"
            "Calculation completed successfully\n"
        )

        assert check_convergence(output_file) is True

    def test_no_convergence_output(self, tmp_path):
        """Test detection of 'no convergence' in output."""
        output_file = tmp_path / "not_converged.out"
        output_file.write_text(
            "Some calculation output\n"
            "no convergence\n"
            "total energy= -123.456\n"
        )

        assert check_convergence(output_file) is False

    def test_no_convergence_case_insensitive(self, tmp_path):
        """Test that 'No Convergence' detection is case-insensitive."""
        output_file = tmp_path / "not_converged.out"
        output_file.write_text("NO CONVERGENCE\n")

        assert check_convergence(output_file) is False

    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError for missing output file."""
        output_file = tmp_path / "nonexistent.out"

        with pytest.raises(FileNotFoundError):
            check_convergence(output_file)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_basic_initialization(self):
        """Test basic RetryConfig initialization."""
        config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])

        assert config.ewidth_list == [2.0, 2.5, 3.0]
        assert config.max_retries == 2  # len(ewidth_list) - 1
        assert config.change_record_on_retry is True

    def test_custom_max_retries(self):
        """Test RetryConfig with custom max_retries."""
        config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0, 3.5], max_retries=1)

        assert config.max_retries == 1

    def test_empty_ewidth_list_raises(self):
        """Test that empty ewidth_list raises ValueError."""
        with pytest.raises(ValueError, match="at least one value"):
            RetryConfig(ewidth_list=[])

    def test_from_config_with_retry_section(self):
        """Test creating RetryConfig from TOML-style config."""
        config = {
            "kkr": {
                "retry": {
                    "ewidth_list": [2.0, 2.5, 3.0],
                    "change_record_on_retry": False,
                }
            }
        }

        retry_config = RetryConfig.from_config(config)

        assert retry_config is not None
        assert retry_config.ewidth_list == [2.0, 2.5, 3.0]
        assert retry_config.change_record_on_retry is False

    def test_from_config_without_retry_section(self):
        """Test that from_config returns None without retry section."""
        config = {"kkr": {"output": {"bzqlty": 2}}}

        retry_config = RetryConfig.from_config(config)

        assert retry_config is None

    def test_from_config_empty(self):
        """Test that from_config returns None for empty config."""
        config = {}

        retry_config = RetryConfig.from_config(config)

        assert retry_config is None


class TestCreateRetryInput:
    """Tests for create_retry_input function."""

    def test_creates_modified_input(self, tmp_path):
        """Test that create_retry_input modifies ewidth and record."""
        # Copy template to temp location
        input_path = tmp_path / "test.in"
        output_path = tmp_path / "retry.in"

        with open(TEMPLATE_PATH) as f:
            input_path.write_text(f.read())

        create_retry_input(input_path, output_path, ewidth=3.0, change_record=True)

        # Read and verify output
        from odatse_kkr import load_input_file

        data = load_input_file(output_path)
        assert data["kkr_parameters"]["calculation"]["ewidth"] == pytest.approx(3.0)
        assert data["kkr_parameters"]["calculation"]["record"] == "2nd"

    def test_creates_modified_input_without_record_change(self, tmp_path):
        """Test create_retry_input without changing record."""
        input_path = tmp_path / "test.in"
        output_path = tmp_path / "retry.in"

        with open(TEMPLATE_PATH) as f:
            input_path.write_text(f.read())

        create_retry_input(input_path, output_path, ewidth=3.0, change_record=False)

        from odatse_kkr import load_input_file

        data = load_input_file(output_path)
        assert data["kkr_parameters"]["calculation"]["ewidth"] == pytest.approx(3.0)
        # record should remain unchanged
        assert data["kkr_parameters"]["calculation"]["record"] == "init"


class TestRunWithRetry:
    """Tests for run_with_retry function."""

    @patch("odatse_kkr.retry.run_command_template")
    @patch("odatse_kkr.retry.check_convergence")
    def test_success_on_first_try(
        self, mock_check_convergence, mock_run_command, tmp_path
    ):
        """Test successful convergence on first attempt."""
        mock_check_convergence.return_value = True

        input_path = tmp_path / "test.in"
        output_path = tmp_path / "test.out"
        input_path.touch()
        output_path.touch()

        retry_config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])
        attempts = run_with_retry(
            ["specx"],
            input_path=input_path,
            output_path=output_path,
            work_dir=tmp_path,
            retry_config=retry_config,
        )

        assert attempts == 1
        assert mock_run_command.call_count == 1

    @patch("odatse_kkr.retry.run_command_template")
    @patch("odatse_kkr.retry.check_convergence")
    @patch("odatse_kkr.retry.load_input_file")
    @patch("odatse_kkr.retry.write_input_file")
    def test_success_on_retry(
        self,
        mock_write,
        mock_load,
        mock_check_convergence,
        mock_run_command,
        tmp_path,
    ):
        """Test successful convergence after retry."""
        # First call fails, second succeeds
        mock_check_convergence.side_effect = [False, True]
        mock_load.return_value = {
            "header": [],
            "ntyp": 0,
            "atom_type_definitions": [],
            "atomic_header": [],
            "atomic_positions": [],
            "footer": [],
            "kkr_parameters": {
                "lattice": {},
                "calculation": {"ewidth": 2.0, "record": "init"},
                "output": {},
                "go": {},
                "line_indices": {},
            },
        }

        input_path = tmp_path / "test.in"
        output_path = tmp_path / "test.out"
        input_path.touch()
        output_path.touch()

        retry_config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])
        attempts = run_with_retry(
            ["specx"],
            input_path=input_path,
            output_path=output_path,
            work_dir=tmp_path,
            retry_config=retry_config,
        )

        assert attempts == 2
        assert mock_run_command.call_count == 2

    @patch("odatse_kkr.retry.run_command_template")
    @patch("odatse_kkr.retry.check_convergence")
    def test_raises_without_retry_config(
        self, mock_check_convergence, mock_run_command, tmp_path
    ):
        """Test that ConvergenceError is raised without retry config."""
        mock_check_convergence.return_value = False

        input_path = tmp_path / "test.in"
        output_path = tmp_path / "test.out"
        input_path.touch()
        output_path.touch()

        with pytest.raises(ConvergenceError):
            run_with_retry(
                ["specx"],
                input_path=input_path,
                output_path=output_path,
                work_dir=tmp_path,
                retry_config=None,
            )

    @patch("odatse_kkr.retry.run_command_template")
    @patch("odatse_kkr.retry.check_convergence")
    @patch("odatse_kkr.retry.load_input_file")
    @patch("odatse_kkr.retry.write_input_file")
    def test_raises_after_all_retries_exhausted(
        self,
        mock_write,
        mock_load,
        mock_check_convergence,
        mock_run_command,
        tmp_path,
    ):
        """Test ConvergenceError after all retries exhausted."""
        mock_check_convergence.return_value = False
        mock_load.return_value = {
            "header": [],
            "ntyp": 0,
            "atom_type_definitions": [],
            "atomic_header": [],
            "atomic_positions": [],
            "footer": [],
            "kkr_parameters": {
                "lattice": {},
                "calculation": {"ewidth": 2.0, "record": "init"},
                "output": {},
                "go": {},
                "line_indices": {},
            },
        }

        input_path = tmp_path / "test.in"
        output_path = tmp_path / "test.out"
        input_path.touch()
        output_path.touch()

        retry_config = RetryConfig(ewidth_list=[2.0, 2.5])

        with pytest.raises(ConvergenceError, match="after 2 attempts"):
            run_with_retry(
                ["specx"],
                input_path=input_path,
                output_path=output_path,
                work_dir=tmp_path,
                retry_config=retry_config,
            )

    @patch("odatse_kkr.retry.run_command_template")
    @patch("odatse_kkr.retry.check_convergence")
    @patch("odatse_kkr.retry.load_input_file")
    @patch("odatse_kkr.retry.write_input_file")
    def test_on_retry_callback(
        self,
        mock_write,
        mock_load,
        mock_check_convergence,
        mock_run_command,
        tmp_path,
    ):
        """Test that on_retry callback is called."""
        mock_check_convergence.side_effect = [False, True]
        mock_load.return_value = {
            "header": [],
            "ntyp": 0,
            "atom_type_definitions": [],
            "atomic_header": [],
            "atomic_positions": [],
            "footer": [],
            "kkr_parameters": {
                "lattice": {},
                "calculation": {"ewidth": 2.0, "record": "init"},
                "output": {},
                "go": {},
                "line_indices": {},
            },
        }

        input_path = tmp_path / "test.in"
        output_path = tmp_path / "test.out"
        input_path.touch()
        output_path.touch()

        callback = MagicMock()
        retry_config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])

        run_with_retry(
            ["specx"],
            input_path=input_path,
            output_path=output_path,
            work_dir=tmp_path,
            retry_config=retry_config,
            on_retry=callback,
        )

        callback.assert_called_once_with(1, 2.5)

