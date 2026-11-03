"""Streamlit demo: photo -> 311 category, agency hint and expected resolution time.

    streamlit run app/demo.py              (with PYTHONPATH=src;.)

Runs the predictors in-process (same code as the API). Uploaded photos are
held in memory for the current run only; nothing is written to disk.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `app.*` imports work under `streamlit run`

from app.predictors import SF_BOUNDS, PhotoRouter, ResolutionPredictor, in_sf, load_stub_stats  # noqa: E402

DEFAULT_LOC = (37.7793, -122.4193)  # Civic Center
GRID_STEP = 0.006  # degrees (~600 m) between clickable map points


@st.cache_resource
def get_models():
    return PhotoRouter.load(), ResolutionPredictor.load(), load_stub_stats()


@st.cache_data
def grid_points() -> list[dict]:
    (la0, la1), (lo0, lo1) = SF_BOUNDS["lat"], SF_BOUNDS["lon"]
    pts, lat = [], la0
    while lat <= la1:
        lon = lo0
        while lon <= lo1:
            pts.append({"lat": round(lat, 4), "lon": round(lon, 4)})
            lon += GRID_STEP
        lat += GRID_STEP
    return pts


def fmt_hours(h: float) -> str:
    return f"{h:.1f} hours" if h < 48 else f"{h / 24:.1f} days"


st.set_page_config(page_title="SF 311 Triage", layout="wide")
router, resolver, stats = get_models()
st.session_state.setdefault("lat", DEFAULT_LOC[0])
st.session_state.setdefault("lon", DEFAULT_LOC[1])

st.title("SF 311 Triage")
st.caption(
    "Upload a photo of a street issue, pick where it is, and get a suggested category and an expected resolution time."
)
if router.is_stub or resolver.is_stub:
    st.warning(
        f"Demo mode: router = `{router.name}`, resolution = `{resolver.name}`. "
        "Stub backends do not look at the photo / use only historical medians. See *About* below."
    )

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Photo")
    upload = st.file_uploader("Photo (JPEG, PNG or WebP)", type=["jpg", "jpeg", "png", "webp"])
    if upload is not None:
        st.image(upload, width=360)
    now = datetime.now()
    d1, d2 = st.columns(2)
    when = datetime.combine(d1.date_input("Date reported", now.date()), d2.time_input("Time", now.time()))
    source = st.selectbox("Reported via", ["Mobile", "Web", "Phone"])

with right:
    st.subheader("2. Location")
    st.caption("Click a dot on the map, or type coordinates.")
    layer_points = pdk.Layer(
        "ScatterplotLayer",
        id="grid",
        data=grid_points(),
        get_position="[lon, lat]",
        get_radius=90,
        get_fill_color=[30, 110, 200, 60],
        pickable=True,
    )
    layer_pin = pdk.Layer(
        "ScatterplotLayer",
        id="pin",
        data=[{"lat": st.session_state.lat, "lon": st.session_state.lon}],
        get_position="[lon, lat]",
        get_radius=160,
        get_fill_color=[220, 40, 40, 220],
    )
    deck = pdk.Deck(
        layers=[layer_points, layer_pin],
        initial_view_state=pdk.ViewState(latitude=37.758, longitude=-122.44, zoom=11.3),
        map_style=None,
    )
    event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", height=380)
    picked = (getattr(event, "selection", None) or {}).get("objects", {}).get("grid") if event else None
    if picked:
        pick = (picked[0]["lat"], picked[0]["lon"])
        if st.session_state.get("last_pick") != pick:  # apply each click once, so typed coordinates stick
            st.session_state.last_pick = pick
            st.session_state.lat, st.session_state.lon = pick
            st.rerun()
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", key="lat", format="%.5f", step=0.001)
    lon = c2.number_input("Longitude", key="lon", format="%.5f", step=0.001)
    if not in_sf(lat, lon):
        st.error("That point is outside San Francisco.")

st.divider()
st.subheader("3. Prediction")
if upload is None:
    st.info("Upload a photo to get a prediction.")
else:
    route = router.predict(upload.getvalue(), lat, lon, when)
    subtypes = stats.get("subtypes_by_label", {}).get(route.label, [])
    a, b, c = st.columns(3)
    a.metric("Category", route.label)
    a.caption(" | ".join(f"{lbl}: {p:.0%}" for lbl, p in route.top3))
    b.metric("Likely agency", route.agency_hint or "unknown")
    subtype = c.selectbox(
        "Sub-category", subtypes or ["(unknown)"], help="Most common sub-categories for this category"
    )
    res = resolver.predict(subtype, lat, lon, when, source, category=route.label)
    c.metric("Expected time to close", fmt_hours(res.median_hours))
    c.caption(f"P(open > {res.late_threshold_hours} h): {res.p_late:.0%}  ·  based on {res.basis}")
    for note in dict.fromkeys(route.notes + res.notes):
        st.caption(f"Note: {note}")

with st.expander("About / limitations", expanded=False):
    st.markdown(
        f"""
**What this is.** A portfolio prototype on public [DataSF 311 Cases](https://data.sf.gov/d/vw6y-z8j6) data.
It suggests a category; it does not file a request. Use the official SF311 app to report issues.

**Models in use.** Router: `{router.name}`{" (stub)" if router.is_stub else ""}.
Resolution: `{resolver.name}`{" (stub)" if resolver.is_stub else ""}.
Stub statistics: {stats.get("built_from", "not available")}.

**Limitations.**
- Labels are the 12 most common photo categories plus *Other*; rarer request types are lumped together.
- The agency shown is the most common one for that category in past data, not a routing decision.
- Resolution time is a historical median: real times vary widely, and "closed" in 311 data does not
  always mean fixed (duplicates and transfers are excluded from the statistics).
- "Late" means open longer than {stats.get("late_hours", 72)} hours, an analysis choice, not a city SLA.
- Trained on recent months only; patterns change with policy and staffing.

**Privacy.** Uploaded photos are processed in memory and not stored. This repo contains no citizen photos.
"""
    )
