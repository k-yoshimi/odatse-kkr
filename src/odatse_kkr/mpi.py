"""
MPI helper utilities shared across ODAT-SE AkaiKKR frontends.

The implementations here consolidate the rank-detection logic that used to live
in ``odatse-cond``, ``odatse-specx``, and ``odatse-osc`` so that running under
different MPI launchers remains robust without duplicating code.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional dependency
    import odatse

_MPI_ENV_VARS: Sequence[str] = (
    "OMPI_COMM_WORLD_RANK",  # OpenMPI
    "PMI_RANK",  # Intel MPI / SLURM PMI
    "SLURM_PROCID",  # SLURM generic
    "PMIX_RANK",  # PMIx
    "MPIRUN_RANK",  # MPICH
    "MPI_RANKID",  # Spectrum MPI
)


def get_mpi_rank(
    info: "odatse.Info | None" = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    rank_keys: Iterable[str] = ("rank",),
    default: int = 0,
) -> int:
    """
    Detect the current MPI rank using ODAT-SE's Info object or environment vars.

    Parameters
    ----------
    info:
        Optional ``odatse.Info`` instance provided by ODAT-SE.
    env:
        Mapping of environment variables to inspect. Defaults to ``os.environ``.
    rank_keys:
        Keys that may appear inside ``info.base`` providing the MPI rank.
    default:
        Fallback rank when no information is available.
    """
    # 1) Info.base fields
    if info is not None:
        for key in rank_keys:
            value = info.base.get(key) if hasattr(info, "base") else None
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue

        # 2) Runner attribute used by some ODAT-SE runners
        runner = getattr(info, "runner", None)
        runner_rank = getattr(runner, "rank", None)
        if runner_rank is not None:
            try:
                return int(runner_rank)
            except (TypeError, ValueError):
                pass

    # 3) Environment variables exposed by different MPI implementations
    env_mapping: Mapping[str, str] = env or os.environ
    for var in _MPI_ENV_VARS:
        value = env_mapping.get(var)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue

    return default


__all__ = ["get_mpi_rank"]
