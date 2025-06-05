#!/bin/bash
# filepath: ignite.sh

# Usage: ./ignite.sh [config_file]
# Runs the ignite simulation from the project root (fire directory).
# If no config file is provided, defaults to configs/config.toml

CONFIG_FILE=${1:-configs/config.toml}

python -m ignite.main "$CONFIG_FILE"