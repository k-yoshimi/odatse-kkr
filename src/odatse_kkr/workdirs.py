"""
Work-directory helpers for ODAT-SE AkaiKKR workflows.

These helpers encapsulate common directory layout patterns that need to remain
consistent across multiple frontends (conductivity, HEA optimization, defect
energy).  Moving them into a shared package keeps the MPI-specific logic in one
place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover - optional dependency
    import odatse


def resolve_root_dir(
    info: "odatse.Info",
    *,
    config_dir: Optional[Path] = None,
    base_key: str = "root_dir",
) -> Path:
    """
    Resolve the root directory used by AkaiKKR calculations.

    ``odatse-cond`` allows relative ``root_dir`` paths inside configuration
    files.  This helper mirrors the behaviour so everybody can simply call the
    same function.
    """
    root_dir_str = info.base.get(base_key, ".") if hasattr(info, "base") else "."
    root_dir_path = Path(root_dir_str).expanduser()
    if not root_dir_path.is_absolute() and config_dir is not None:
        return (config_dir / root_dir_path).resolve()
    return root_dir_path.absolute()


def prepare_rank_work_dir(
    root_dir: Path,
    *,
    work_dir: Union[str, Path] = "runs",
    mpi_rank: int = 0,
    rank_dir_pattern: str = "{rank}",
    include_rank_zero: bool = True,
) -> Path:
    """
    Create/return the work directory for the current MPI rank.

    Parameters mirror the variations that existed in the individual projects:
    ``rank_dir_pattern`` selects between ``"0"``, ``"rank_0"``, etc.  Setting
    ``include_rank_zero=False`` recreates ``odatse-osc``'s behaviour where rank
    zero writes directly into ``runs/`` while other ranks use subdirectories.
    """
    base_dir = (root_dir / work_dir).absolute()
    base_dir.mkdir(parents=True, exist_ok=True)

    if mpi_rank == 0 and not include_rank_zero:
        return base_dir

    rank_name = rank_dir_pattern.format(rank=mpi_rank)
    rank_dir = base_dir / rank_name
    rank_dir.mkdir(parents=True, exist_ok=True)
    return rank_dir


def rank_aware_path(
    base_path: Path,
    target: Union[str, Path],
    *,
    mpi_rank: int,
    pattern: str = "{stem}_rank{rank}{suffix}",
) -> Path:
    """
    Return a path whose filename is suffixed with the MPI rank.

    Useful for error/energy logs that should not collide across ranks.
    """
    target_path = base_path / target
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if mpi_rank == 0:
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    ranked_name = pattern.format(stem=stem, suffix=suffix, rank=mpi_rank)
    ranked_path = target_path.with_name(ranked_name)
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    return ranked_path


@dataclass
class TrialDirectoryManager:
    """
    Helper that creates sequential ``trial_XXXXX`` directories.

    The manager keeps the counter in-memory (mirroring the existing behaviour)
    and ensures that directories are created lazily once requested.
    """

    work_dir: Path
    prefix: str = "trial_"
    digits: int = 5
    start: int = 0

    def __post_init__(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._counter = self.start

    def next(self) -> Path:
        """Create and return the next trial directory."""
        self._counter += 1
        name = f"{self.prefix}{self._counter:0{self.digits}d}"
        trial_dir = self.work_dir / name
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir


__all__ = [
    "TrialDirectoryManager",
    "prepare_rank_work_dir",
    "rank_aware_path",
    "resolve_root_dir",
]
