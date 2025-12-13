"""
Temporary-directory helpers that keep MPI processes from clobbering each other.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


def ensure_tmp_subdir(
    base_dir: Path,
    *,
    name: str = "tmp",
    mpi_rank: Optional[int] = None,
    include_pid: bool = False,
) -> Path:
    """
    Create a temporary directory scoped to a specific MPI rank / process id.

    Examples
    --------
    >>> ensure_tmp_subdir(Path(\"runs/0\"))  # -> runs/0/tmp
    >>> ensure_tmp_subdir(Path(\"runs\"), mpi_rank=2, include_pid=True)
    ... # -> runs/tmp_rank2_pid12345
    """
    parts = [name]
    if mpi_rank is not None:
        parts.append(f"rank{mpi_rank}")
    if include_pid:
        parts.append(f"pid{os.getpid()}")

    dir_name = "_".join(parts)
    tmp_dir = base_dir / dir_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def apply_tmp_env(
    tmp_dir: Path,
    *,
    base_env: Optional[Mapping[str, str]] = None,
    include_fortran: bool = True,
    variables: Sequence[str] = ("TMPDIR", "TMP", "TEMP"),
) -> Dict[str, str]:
    """
    Return a copy of ``base_env`` with tmp-related variables pointing to ``tmp_dir``.
    """
    env: Dict[str, str] = dict(base_env or os.environ)
    tmp_dir_str = str(tmp_dir)
    for var in variables:
        env[var] = tmp_dir_str
    if include_fortran:
        env["FORTRAN_TMPDIR"] = tmp_dir_str
    return env


__all__ = ["apply_tmp_env", "ensure_tmp_subdir"]
