# Usage: .\firestarter.ps1 [config_file]
# Runs the firestarter simulation from the project root (fire directory).
# If no config file is provided, defaults to configs/config.toml

param(
    [string]$ConfigFile = "configs/config.toml"
)

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

python -m firestarter.main