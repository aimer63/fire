# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Asset keys used throughout the simulation codebase
ASSET_KEYS = [
    "stocks",
    "bonds",
    "str",
    "fun",
    "real_estate",
]

# Withdrawal priority order (from most liquid to least)
WITHDRAWAL_PRIORITY = [
    "str",
    "bonds",
    "stocks",
    "fun",
]
