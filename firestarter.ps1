# Usage: .\firestarter.ps1 [config_file]
# If no config file is provided, defaults to config.toml

param(
    [string[]]$Args
)

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

python -m firestarter.main $Args
