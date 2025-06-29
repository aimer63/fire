#!/bin/bash
# Usage: ./update_spdx_year.sh [YEAR] [HOLDER]
# Example: ./update_spdx_year.sh 2025 NewHolder

YEAR="${1:-2025}"
HOLDER="${2:-aimer63}"

echo "Updating SPDX-FileCopyrightText to: $YEAR $HOLDER (Python files only)"

find . -type f -name "*.py" -print0 | \
  xargs -0 sed -i -E "s/SPDX-FileCopyrightText: [0-9 \-]+ [^ ]+/SPDX-FileCopyrightText: $YEAR $HOLDER/g"

echo "Done."