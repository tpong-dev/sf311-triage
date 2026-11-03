"""Shared loading, cleaning, labels and splits. Every model uses this so results are comparable.

    from sf311.dataset import load, photo_rows, split
    df = load()                     # all monthly files, cleaned
    photos = photo_rows(df)         # rows with a usable Cloudinary photo + `label`
    train, val, test = split(photos)
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Time-based split: never train on the future.
VAL_START = pd.Timestamp("2026-08-01")
TEST_START = pd.Timestamp("2026-09-01")

# Photo-routing classes: the service_names covering ~95% of photo requests (Nov 2025 – Oct 2026).
# Everything else becomes "Other".
LABELS = [
    "Street and Sidewalk Cleaning",
    "Parking Enforcement",
    "General Request",
    "Graffiti Public",
    "Blocked Street and Sidewalk",
    "Encampment",
    "Sidewalk and Curb",
    "Tree Maintenance",
    "RPD General",
    "Graffiti Private",
    "Litter Receptacle Maintenance",
    "Sewer",
]
OTHER = "Other"

# A request is "late" if it takes longer than this to close (7 days, matching the resolution model).
LATE_HOURS = 168


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(f for f in raw_dir.glob("sf311_*.parquet") if "_sample" not in f.name)
    if not files:
        raise FileNotFoundError(f"no monthly files in {raw_dir}; run `python -m sf311.fetch` first")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.drop_duplicates("service_request_id")
    df = df[df["source"] != "Test"].reset_index(drop=True)

    # Closures like "Duplicate" or "Case is Invalid" aren't real work; exclude them from time targets.
    notes = df["status_notes"].fillna("").str.lower().str.replace("_", " ").str.strip()
    df["is_admin_close"] = notes.str.contains(
        r"comment noted|invalid|duplicate|cancel|transferred|not a city issue", regex=True
    )

    hours = (df["closed_date"] - df["requested_datetime"]).dt.total_seconds() / 3600
    # Negative durations are data errors; closed_date missing means still open (right-censored).
    df["resolution_hours"] = hours.where(hours >= 0)
    df["is_closed"] = df["closed_date"].notna() & df["resolution_hours"].notna()
    df["is_late"] = np.where(df["is_closed"], df["resolution_hours"] > LATE_HOURS, np.nan)

    media = df["media_url"].fillna("")
    df["photo_url"] = media.where(media.str.contains("cloudinary.com"), None)
    df["label"] = df["service_name"].where(df["service_name"].isin(LABELS), OTHER)
    return df


def photo_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["photo_url"].notna()].reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = df["requested_datetime"]
    train = df[t < VAL_START]
    val = df[(t >= VAL_START) & (t < TEST_START)]
    test = df[t >= TEST_START]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
