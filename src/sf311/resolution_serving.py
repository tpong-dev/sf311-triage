"""Serving context for the resolution model.

The trained model uses backlog features (recent volume, open cases at filing time) that need the
full request history. At serving time we don't have that, so this builds a small JSON of
recent typical values per sub-category / category / agency, plus the most common category,
agency and details for each sub-category. Aggregates only: no case ids, addresses or photos.

    python -m sf311.resolution_serving      # writes models/resolution/serving_context.json
"""

import json
from pathlib import Path

import pandas as pd

from sf311.resolution import build_frame

OUT = Path(__file__).resolve().parents[2] / "models" / "resolution" / "serving_context.json"
RECENT_DAYS = 30

BACKLOG = {
    "service_subtype": ["n_subtype_prev_1d", "n_subtype_prev_7d", "open_subtype_at_t"],
    "service_name": ["n_name_prev_7d", "open_name_at_t"],
    "agency_responsible": ["open_agency_at_t"],
}


def build(out: Path = OUT) -> dict:
    df, snap = build_frame()
    recent = df[df["requested_datetime"] >= snap - pd.Timedelta(days=RECENT_DAYS)]

    ctx: dict = {
        "built_from": f"last {RECENT_DAYS} days before {snap:%Y-%m-%d}",
        "city": {"n_city_prev_1d": float(recent["n_city_prev_1d"].median())},
        "subtype_defaults": {},
    }
    for key, cols in BACKLOG.items():
        ctx[key] = {k: {c: float(g[c].median()) for c in cols} for k, g in recent.groupby(key) if len(g) >= 5}

    for subtype, g in df.groupby("service_subtype"):
        if len(g) < 20:
            continue
        ctx["subtype_defaults"][subtype] = {
            c: g[c].mode().iat[0] if g[c].notna().any() else None
            for c in ["service_name", "agency_responsible", "service_details"]
        }

    out.write_text(json.dumps(ctx, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(ctx['subtype_defaults'])} subtypes)")
    return ctx


if __name__ == "__main__":
    build()
