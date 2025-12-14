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
    "MV2_COMM_WORLD_RANK",  # MVAPICH2
)


def _try_mpi4py() -> Optional[int]:
    """Try to get rank from mpi4py if available."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return None
    except Exception:
        return None


def get_mpi_rank(
    info: "odatse.Info | None" = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    rank_keys: Iterable[str] = ("rank",),
    default: int = 0,
    verbose: bool = False,
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
    verbose:
        If True, print debug information about rank detection.
    """
    import logging
    logger = logging.getLogger(__name__)

    # 1) Info.base fields
    if info is not None:
        for key in rank_keys:
            value = info.base.get(key) if hasattr(info, "base") else None
            if value is not None:
                try:
                    rank = int(value)
                    if verbose:
                        logger.info(f"MPI rank from info.base['{key}']: {rank}")
                    return rank
                except (TypeError, ValueError):
                    continue

        # 2) Runner attribute used by some ODAT-SE runners
        runner = getattr(info, "runner", None)
        runner_rank = getattr(runner, "rank", None)
        if runner_rank is not None:
            try:
                rank = int(runner_rank)
                if verbose:
                    logger.info(f"MPI rank from info.runner.rank: {rank}")
                return rank
            except (TypeError, ValueError):
                pass

    # 3) Environment variables exposed by different MPI implementations
    env_mapping: Mapping[str, str] = env or os.environ
    for var in _MPI_ENV_VARS:
        value = env_mapping.get(var)
        if value is not None:
            try:
                rank = int(value)
                if verbose:
                    logger.info(f"MPI rank from env var {var}: {rank}")
                return rank
            except (TypeError, ValueError):
                continue

    # 4) Try mpi4py directly
    mpi4py_rank = _try_mpi4py()
    if mpi4py_rank is not None:
        if verbose:
            logger.info(f"MPI rank from mpi4py: {mpi4py_rank}")
        return mpi4py_rank

    if verbose:
        logger.info(f"MPI rank not detected, using default: {default}")
    return default


__all__ = ["get_mpi_rank"]
