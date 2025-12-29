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

## KKR Parameters Configuration

The package supports configuring AkaiKKR calculation parameters via TOML files.
This allows you to define lattice parameters, calculation settings, and output
options in your configuration file instead of hardcoding them in the template.

### TOML Configuration

Add a `[kkr]` section to your TOML configuration file:

```toml
[kkr]
# Go command settings (optional)
[kkr.go]
command = "go"
pot_file = "pot.dat"

# Bravais lattice parameters (optional)
[kkr.lattice]
brvtyp = "so"          # Bravais lattice type
a = 7.265372455718975  # Lattice constant a
c_a = 3.0753407056213957  # c/a ratio
b_a = 1.0211940276767721  # b/a ratio
alpha = 90.0           # Angle alpha
beta = 90.0            # Angle beta
gamma = 90.0           # Angle gamma

# Calculation parameters (optional)
[kkr.calculation]
edelt = 0.001          # Energy mesh width
ewidth = 2.0           # Energy window width
reltyp = "sra"         # Relativistic type
sdftyp = "mjw"         # SDF type
magtyp = "mag"         # Magnetic type
record = "init"        # Record type

# Output parameters (optional)
[kkr.output]
outtyp = "update"      # Output type
bzqlty = 6             # BZ quality
maxitr = 200           # Maximum iterations
pmix = 0.02            # Mixing parameter
```

### Applying KKR Parameters

Use `apply_kkr_parameters_from_config()` to apply TOML settings:

```python
import tomllib  # Python 3.11+ (use tomli for older versions)
from odatse_kkr import (
    load_input_file,
    apply_kkr_parameters_from_config,
    write_input_file,
)

# Load TOML configuration
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Load template and apply KKR parameters
data = load_input_file("template.in")
data = apply_kkr_parameters_from_config(data, config)
write_input_file(data, "output.in")
```

### Partial Updates

You don't need to specify all parameters. Only the parameters you include
will be updated; others retain their original values from the template:

```toml
[kkr]
# Only update lattice constant and mixing parameter
[kkr.lattice]
a = 8.0

[kkr.output]
pmix = 0.01
```

### Direct Modification

For programmatic control, use `modify_kkr_parameters()` directly:

```python
from odatse_kkr import load_input_file, modify_kkr_parameters, write_input_file

data = load_input_file("template.in")
data = modify_kkr_parameters(
    data,
    lattice={"a": 8.0, "c_a": 3.2},
    calculation={"edelt": 0.0005},
    output={"maxitr": 500, "pmix": 0.01},
)
write_input_file(data, "output.in")
```

## Automatic Retry on Convergence Failure

When AkaiKKR outputs "no convergence", the package can automatically retry
the calculation with different `ewidth` values.

### TOML Configuration

Add a `[kkr.retry]` section to enable automatic retry:

```toml
[kkr.retry]
# List of ewidth values to try sequentially
# First value is used for initial calculation, subsequent values for retries
ewidth_list = [2.0, 2.5, 3.0, 3.5]

# Change record from "init" to "2nd" on retry (default: true)
change_record_on_retry = true
```

### Using run_with_retry

Use `run_with_retry()` instead of `run_command_template()` for automatic retry:

```python
from odatse_kkr import (
    RetryConfig,
    run_with_retry,
    ConvergenceError,
)

# Create retry configuration
retry_config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0, 3.5])

# Or load from TOML config
retry_config = RetryConfig.from_config(config)

try:
    attempts = run_with_retry(
        ["specx", "<", "{input}", ">", "{output}"],
        input_path=input_path,
        output_path=output_path,
        work_dir=work_dir,
        retry_config=retry_config,
    )
    print(f"Converged after {attempts} attempt(s)")
except ConvergenceError as e:
    print(f"Failed to converge: {e}")
```

### Retry Behavior

1. Initial calculation runs with original `ewidth` from template
2. If "no convergence" is detected in output:
   - `ewidth` is changed to the next value in `ewidth_list`
   - `record` is changed from `"init"` to `"2nd"` (if `change_record_on_retry=true`)
   - Calculation is retried
3. Process repeats until convergence or all values exhausted
4. `ConvergenceError` is raised if all attempts fail

### Checking Convergence

You can also check convergence manually:

```python
from odatse_kkr import check_convergence

if check_convergence(output_path):
    print("Calculation converged")
else:
    print("No convergence detected")
```

## Additional Utilities

- `odatse_kkr.generate_input`: functions such as `load_input_file`, `add_atom_type_definition`,
  `replace_atom_types_by_label`, `parse_kkr_parameters`, `modify_kkr_parameters`, and `write_input_file`.
- `odatse_kkr.MetricExtractor`: reusable parser for AkaiKKR metrics (total energy, band energy, etc.).
- `odatse_kkr.retry`: automatic retry functionality with `RetryConfig`, `run_with_retry`, and `check_convergence`.

## Projects Using This Package

- [odatse-cond](https://github.com/k-yoshimi/odatse-cond)
- [odatse-specx](https://github.com/k-yoshimi/odatse-specx)
- [odatse-osc](https://github.com/k-yoshimi/odatse-osc)

## License

See [LICENSE](LICENSE) for details.
