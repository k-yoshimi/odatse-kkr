"""
Retry utilities for AkaiKKR calculations with convergence failures.

This module provides functionality to automatically retry AkaiKKR calculations
when they fail to converge, by adjusting ewidth and record parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .commands import run_command_template
from .generate_input import (
    load_input_file,
    modify_kkr_parameters,
    write_input_file,
)
from .metrics import ConvergenceError, check_convergence

__all__ = [
    "RetryConfig",
    "run_with_retry",
]

logger = logging.getLogger(__name__)


class RetryConfig:
    """
    Configuration for retry behavior on convergence failure.

    Parameters
    ----------
    ewidth_list : list of float
        List of ewidth values to try sequentially. The first value is used
        for the initial calculation, subsequent values are used for retries.
    max_retries : int, optional
        Maximum number of retry attempts. Defaults to len(ewidth_list) - 1.
    change_record_on_retry : bool, optional
        If True, change record from "init" to "2nd" on retry. Default is True.

    Examples
    --------
    TOML configuration::

        [kkr.retry]
        ewidth_list = [2.0, 2.5, 3.0, 3.5]
        change_record_on_retry = true

    Usage::

        config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])
        run_with_retry(
            command_template=["specx", "<", "{input}", ">", "{output}"],
            input_path=input_path,
            output_path=output_path,
            work_dir=work_dir,
            retry_config=config,
        )
    """

    def __init__(
        self,
        ewidth_list: List[float],
        max_retries: Optional[int] = None,
        change_record_on_retry: bool = True,
    ):
        if not ewidth_list:
            raise ValueError("ewidth_list must contain at least one value")

        self.ewidth_list = ewidth_list
        self.max_retries = (
            max_retries if max_retries is not None else len(ewidth_list) - 1
        )
        self.change_record_on_retry = change_record_on_retry

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["RetryConfig"]:
        """
        Create RetryConfig from TOML configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary, typically from TOML file.
            Looks for [kkr.retry] section with ewidth_list.

        Returns
        -------
        RetryConfig or None
            RetryConfig if ewidth_list is specified, None otherwise.

        Examples
        --------
        >>> config = {"kkr": {"retry": {"ewidth_list": [2.0, 2.5, 3.0]}}}
        >>> retry_config = RetryConfig.from_config(config)
        """
        kkr_config = config.get("kkr", {})
        retry_config = kkr_config.get("retry", {})

        ewidth_list = retry_config.get("ewidth_list")
        if not ewidth_list:
            return None

        return cls(
            ewidth_list=ewidth_list,
            max_retries=retry_config.get("max_retries"),
            change_record_on_retry=retry_config.get("change_record_on_retry", True),
        )


def run_with_retry(
    command_template: Union[str, Sequence[str]],
    *,
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    retry_config: Optional[RetryConfig] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    on_retry: Optional[Callable[[int, float], None]] = None,
) -> int:
    """
    Run AkaiKKR command with automatic retry on convergence failure.

    When a calculation fails to converge (output contains "no convergence"),
    this function automatically modifies the input file with:
    - New ewidth value from the retry configuration
    - record changed from "init" to "2nd" (if change_record_on_retry is True)

    Parameters
    ----------
    command_template : str or Sequence[str]
        Command template for AkaiKKR execution.
    input_path : Path
        Path to the input file.
    output_path : Path
        Path to the output file.
    work_dir : Path
        Working directory for command execution.
    retry_config : RetryConfig, optional
        Configuration for retry behavior. If None, no retry is performed.
    env : Mapping[str, str], optional
        Environment variables for command execution.
    timeout : float, optional
        Timeout in seconds for each command execution.
    on_retry : callable, optional
        Callback function called before each retry with (attempt_number, new_ewidth).

    Returns
    -------
    int
        Number of attempts made (1 = success on first try, 2+ = retries needed).

    Raises
    ------
    ConvergenceError
        If calculation fails to converge after all retry attempts.

    Examples
    --------
    >>> retry_config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])
    >>> attempts = run_with_retry(
    ...     ["specx", "<", "{input}", ">", "{output}"],
    ...     input_path=Path("test.in"),
    ...     output_path=Path("test.out"),
    ...     work_dir=Path("."),
    ...     retry_config=retry_config,
    ... )
    >>> print(f"Converged after {attempts} attempt(s)")
    """
    # First attempt
    run_command_template(
        command_template,
        work_dir=work_dir,
        input_path=input_path,
        output_path=output_path,
        env=env,
        timeout=timeout,
    )

    if check_convergence(output_path):
        return 1

    # No retry config - raise error immediately
    if retry_config is None:
        raise ConvergenceError(
            f"Calculation did not converge. Output: {output_path}"
        )

    # Retry with different ewidth values
    max_attempts = min(retry_config.max_retries + 1, len(retry_config.ewidth_list))

    for attempt in range(1, max_attempts):
        new_ewidth = retry_config.ewidth_list[attempt]

        logger.info(
            f"Retry attempt {attempt}/{retry_config.max_retries}: "
            f"ewidth={new_ewidth}"
        )

        if on_retry:
            on_retry(attempt, new_ewidth)

        # Modify input file
        input_data = load_input_file(input_path)

        calc_params: Dict[str, Any] = {"ewidth": new_ewidth}
        if retry_config.change_record_on_retry:
            calc_params["record"] = "2nd"

        new_data = modify_kkr_parameters(input_data, calculation=calc_params)
        write_input_file(new_data, input_path)

        # Run calculation
        run_command_template(
            command_template,
            work_dir=work_dir,
            input_path=input_path,
            output_path=output_path,
            env=env,
            timeout=timeout,
        )

        if check_convergence(output_path):
            return attempt + 1

    # All retries exhausted
    raise ConvergenceError(
        f"Calculation did not converge after {max_attempts} attempts "
        f"with ewidth values: {retry_config.ewidth_list[:max_attempts]}. "
        f"Output: {output_path}"
    )


def create_retry_input(
    input_path: Path,
    output_path: Path,
    ewidth: float,
    change_record: bool = True,
) -> None:
    """
    Create a modified input file for retry calculation.

    This is a convenience function for creating retry input files
    without running the calculation.

    Parameters
    ----------
    input_path : Path
        Path to the original input file. Will be overwritten.
    output_path : Path
        Path where the modified file will be written.
    ewidth : float
        New ewidth value.
    change_record : bool, optional
        If True, change record from "init" to "2nd". Default is True.
    """
    input_data = load_input_file(input_path)

    calc_params: Dict[str, Any] = {"ewidth": ewidth}
    if change_record:
        calc_params["record"] = "2nd"

    new_data = modify_kkr_parameters(input_data, calculation=calc_params)
    write_input_file(new_data, output_path)

