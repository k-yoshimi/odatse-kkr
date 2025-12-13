# odatse-kkr

Shared helper modules for ODAT-SE based AkaiKKR workflows. The package collects
MPI-safe utilities originally implemented in `odatse-cond`, `odatse-specx`, and
`odatse-osc` so they can be reused from a single import location.

## Installation

```bash
git clone https://github.com/k-yoshimi/odatse-kkr.git
cd odatse-kkr
pip install -e .
```

## Usage

The package exports helpers for MPI rank detection, work directory management,
temporary directories, AkaiKKR input-file manipulation, and command execution.
Typical usage:

```python
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
work_dir = prepare_rank_work_dir(root_dir, mpi_rank=rank)
trial = TrialDirectoryManager(work_dir).next()
tmp_dir = ensure_tmp_subdir(trial, mpi_rank=rank)
env = apply_tmp_env(tmp_dir, base_env=os.environ)
run_command_template(command, work_dir=trial, input_path=input_in, output_path=output_out, env=env)
```

Additional utilities:

- `odatse_kkr.generate_input`: functions such as `load_input_file`, `add_atom_type_definition`,
  `replace_atom_types_by_label`, and `write_input_file` (previously hosted in `odatse-specx`).
- `odatse_kkr.MetricExtractor`: reusable parser for AkaiKKR metrics (total energy, band energy, etc.).

Projects currently consuming this package:

- [odatse-cond](https://github.com/k-yoshimi/odatse-cond)
- [odatse-specx](https://github.com/k-yoshimi/odatse-specx)
- [odatse-osc](https://github.com/k-yoshimi/odatse-osc)
*** End Patch***} to=functions.apply_patch codeinterpretable? error instructions? need plain patch. same as earlier but w ??? Actually tool expects plain string not JSON. Need re-send. 
