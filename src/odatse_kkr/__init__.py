"""
Reusable helpers for ODAT-SE AkaiKKR frontends.

Typical usage
-------------

```python
from pathlib import Path
from odatse_kkr import (
    TrialDirectoryManager,
    apply_tmp_env,
    ensure_tmp_subdir,
    get_mpi_rank,
    prepare_rank_work_dir,
    resolve_root_dir,
    run_command_template,
)

root_dir = resolve_root_dir(info, config_dir=config_dir)
rank = get_mpi_rank(info)
work_dir = prepare_rank_work_dir(root_dir, work_dir="runs", mpi_rank=rank)
trial_manager = TrialDirectoryManager(work_dir)
trial_dir = trial_manager.next()
input_path = trial_dir / "test.in"
output_path = trial_dir / "test.out"
tmp_dir = ensure_tmp_subdir(trial_dir, mpi_rank=rank)
env = apply_tmp_env(tmp_dir, base_env=os.environ)
run_command_template(config["akai_command"], work_dir=trial_dir,
                     input_path=input_path, output_path=output_path, env=env)
```
"""

from .commands import as_command_list, run_command_template
from .generate_input import (  # noqa: F401
    add_atom_type_definition,
    apply_kkr_parameters_from_config,
    count_atoms_by_type,
    list_atomic_positions,
    load_input_file,
    modify_atom_type_definition,
    modify_kkr_parameters,
    parse_kkr_parameters,
    replace_atom_types,
    replace_atom_types_by_coordinates,
    replace_atom_types_by_label,
    write_input_file,
)
from .metrics import (  # noqa: F401
    ConvergenceError,
    MetricExtractor,
    NaNError,
    check_convergence,
    check_nan_in_output,
)
from .retry import RetryConfig, run_with_retry  # noqa: F401
from .mpi import get_mpi_rank
from .tmpenv import apply_tmp_env, ensure_tmp_subdir
from .workdirs import (
    TrialDirectoryManager,
    prepare_rank_work_dir,
    rank_aware_path,
    resolve_root_dir,
)

__all__ = [
    "ConvergenceError",
    "MetricExtractor",
    "NaNError",
    "RetryConfig",
    "TrialDirectoryManager",
    "add_atom_type_definition",
    "apply_kkr_parameters_from_config",
    "apply_tmp_env",
    "as_command_list",
    "check_convergence",
    "check_nan_in_output",
    "count_atoms_by_type",
    "ensure_tmp_subdir",
    "get_mpi_rank",
    "list_atomic_positions",
    "load_input_file",
    "modify_atom_type_definition",
    "modify_kkr_parameters",
    "parse_kkr_parameters",
    "prepare_rank_work_dir",
    "rank_aware_path",
    "replace_atom_types",
    "replace_atom_types_by_coordinates",
    "replace_atom_types_by_label",
    "resolve_root_dir",
    "run_command_template",
    "run_with_retry",
    "write_input_file",
]
