#!/bin/bash
# filepath: firestarter.sh

# Usage: ./firestarter.sh [config_file]
# Runs the firestarter simulation from the project root (fire directory).
# If no config file is provided, defaults to configs/config.toml

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export QT_QPA_PLATFORM=xcb
CONFIG_FILE=${1:-configs/config.toml}

python -m firestarter.main "$CONFIG_FILE"
