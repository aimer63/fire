from enum import Enum


class AssetKey(Enum):
    STOCKS = "stocks"
    BONDS = "bonds"
    STR = "str"
    FUN = "fun"
    REAL_ESTATE = "real_estate"


# List of all asset keys (in canonical order)
ASSET_KEYS = [
    AssetKey.STOCKS,
    AssetKey.BONDS,
    AssetKey.STR,
    AssetKey.FUN,
    AssetKey.REAL_ESTATE,
]

# Withdrawal priority order (from most liquid to least)
WITHDRAWAL_PRIORITY = [
    AssetKey.STR,
    AssetKey.BONDS,
    AssetKey.STOCKS,
    AssetKey.FUN,
]
