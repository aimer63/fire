#!/bin/bash
# filepath: firestarter.sh

# Usage: ./firestarter.sh [config_file]
# Runs the firestarter simulation from the project root (fire directory).
# If no config file is provided, defaults to configs/config.toml

CONFIG_FILE=${1:-configs/config.toml}

python -m firestarter.main "$CONFIG_FILE"