"""Download San Francisco 311 cases from DataSF, one Parquet file per month.

Usage:
    python -m sf311.fetch --start 2025-01 --end 2025-12
    python -m sf311.fetch --start 2026-10 --end 2026-10 --max-rows 5000   # quick sample

Set SOCRATA_APP_TOKEN (free, from data.sf.gov) for higher rate limits.
Months that already have a file are skipped, so the script can be re-run safely.
"""

import argparse
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.sf.gov/resource/vw6y-z8j6.json"
PAGE_SIZE = 50_000
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Street address is left out on purpose: lat/long and neighbourhood are enough for modelling.
COLUMNS = [
    "service_request_id",
    "requested_datetime",
    "closed_date",
    "updated_datetime",
    "status_description",
    "status_notes",
    "agency_responsible",
    "service_name",
    "service_subtype",
    "service_details",
    "supervisor_district",
    "analysis_neighborhood",
    "police_district",
    "lat",
    "long",
    "source",
    "media_url",
]

DATETIME_COLUMNS = ["requested_datetime", "closed_date", "updated_datetime"]


def month_range(start: str, end: str) -> list[date]:
    first = date.fromisoformat(start + "-01")
    last = date.fromisoformat(end + "-01")
    months = []
    while first <= last:
        months.append(first)
        first = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    return months


def next_month(d: date) -> date:
    return date(d.year + d.month // 12, d.month % 12 + 1, 1)


def get_page(session: requests.Session, params: dict, retries: int = 5) -> list[dict]:
    for attempt in range(retries):
        try:
            resp = session.get(URL, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as err:
            wait = 2**attempt
            print(f"  request failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("DataSF request kept failing; try again later")


def fetch_month(session: requests.Session, month: date, max_rows: int | None) -> pd.DataFrame:
    where = (
        f"requested_datetime >= '{month.isoformat()}T00:00:00' "
        f"AND requested_datetime < '{next_month(month).isoformat()}T00:00:00'"
    )
    rows: list[dict] = []
    offset = 0
    while True:
        limit = PAGE_SIZE if max_rows is None else min(PAGE_SIZE, max_rows - len(rows))
        page = get_page(
            session,
            {
                "$select": ",".join(COLUMNS),
                "$where": where,
                "$order": "service_request_id",
                "$limit": limit,
                "$offset": offset,
            },
        )
        rows.extend(page)
        offset += len(page)
        print(f"  {month:%Y-%m}: {len(rows):,} rows")
        if len(page) < limit or (max_rows is not None and len(rows) >= max_rows):
            break

    df = pd.DataFrame(rows, columns=COLUMNS)
    # media_url arrives as {"url": ...}; keep the plain link only.
    df["media_url"] = df["media_url"].map(lambda m: m.get("url") if isinstance(m, dict) else None)
    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["lat", "long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="first month, YYYY-MM")
    parser.add_argument("--end", required=True, help="last month, YYYY-MM")
    parser.add_argument("--max-rows", type=int, help="cap rows per month (for quick samples)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if token := os.environ.get("SOCRATA_APP_TOKEN"):
        session.headers["X-App-Token"] = token

    for month in month_range(args.start, args.end):
        suffix = "_sample" if args.max_rows else ""
        out = OUT_DIR / f"sf311_{month:%Y-%m}{suffix}.parquet"
        if out.exists():
            print(f"skip {out.name} (exists)")
            continue
        df = fetch_month(session, month, args.max_rows)
        df.to_parquet(out, index=False)
        print(f"saved {out.name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
