"""Predictor interface used by the API and the Streamlit demo.

Two predictors, each with one ``predict`` method:

    router = PhotoRouter.load()
    router.predict(image, lat, lon, ts)                 -> RouteResult
    resolver = ResolutionPredictor.load()
    resolver.predict(subtype, lat, lon, ts, source)     -> ResolutionResult

``load()`` looks for trained artifacts under ``models/`` (override with the
``SF311_MODELS_DIR`` env var). If none are found it falls back to a clearly
labelled stub, so the app always starts. Every result carries ``model`` (which
backend answered) and ``is_stub``; the API and demo show both.

Expected artifact formats
-------------------------
Photo routing, ``models/image/``:

* ``model.pt``     TorchScript module. Input: float tensor ``(N, 3, H, W)``,
  RGB scaled to [0, 1] then normalised with ``mean``/``std``. Output: logits
  ``(N, len(labels))``.
* ``labels.json``  JSON list of class names, in logit order (normally
  ``sf311.dataset.LABELS + ["Other"]``).
* ``config.json``  optional: ``{"image_size": 224, "mean": [...], "std": [...]}``
  (defaults: 224 and ImageNet statistics).

Resolution time, ``models/resolution/``:

* ``model.joblib`` either
  (a) an object with ``predict_one(features: dict) -> (median_hours, p_late)``, or
  (b) a dict ``{"hours_model": reg, "late_model": clf, "feature_columns": [...],
  "target": "hours" | "log1p_hours"}`` where ``reg.predict(X)`` returns hours
  (or log1p hours) and ``clf.predict_proba(X)[:, 1]`` is P(late). ``X`` is a
  one-row DataFrame built by :func:`resolution_features`, restricted to
  ``feature_columns``; string columns are cast to ``category`` (LightGBM style).
  If training used different features, change :func:`resolution_features` only.

Stubs
-----
* Router stub ("prior-stub"): ignores the image and returns the class
  frequencies of photo requests in the training months. Optional zero-shot CLIP
  backend with ``SF311_ROUTER=clip`` (needs ``open_clip_torch`` and downloads
  weights on first use).
* Resolution stub ("subtype-median-stub"): historical median hours and late
  rate per sub-category (per sub-category and source when there is enough data).

Both stubs read ``app/stub_stats.json``: aggregate counts only, no case ids,
addresses or photos. Rebuild it from local data with ``python -m app.predictors``.
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("SF311_MODELS_DIR", ROOT / "models"))
STUB_STATS = Path(__file__).resolve().parent / "stub_stats.json"

LATE_HOURS = 168  # mirrors sf311.dataset.LATE_HOURS; kept here so the app has no pandas import at startup
OTHER = "Other"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# SF city bounds used to sanity-check coordinates.
SF_BOUNDS = {"lat": (37.6, 37.84), "lon": (-122.53, -122.35)}


# --------------------------------------------------------------------------- results


@dataclass
class RouteResult:
    label: str
    top3: list[tuple[str, float]]
    agency_hint: str | None
    model: str
    is_stub: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["top3"] = [{"label": lbl, "prob": round(p, 4)} for lbl, p in self.top3]
        return d


@dataclass
class ResolutionResult:
    median_hours: float
    p_late: float
    late_threshold_hours: int
    model: str
    is_stub: bool
    basis: str  # what the estimate is based on, e.g. "subtype+source (n=812)"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers


def load_stub_stats(path: Path = STUB_STATS) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def in_sf(lat: float, lon: float) -> bool:
    return SF_BOUNDS["lat"][0] <= lat <= SF_BOUNDS["lat"][1] and SF_BOUNDS["lon"][0] <= lon <= SF_BOUNDS["lon"][1]


def resolution_features(
    subtype: str, lat: float, lon: float, ts: datetime, source: str, category: str | None = None
) -> dict[str, Any]:
    """Feature row for the trained resolution model. Edit to match training."""
    return {
        "service_subtype": subtype,
        "service_name": category or "",
        "lat": lat,
        "long": lon,
        "source": source,
        "hour": ts.hour,
        "dayofweek": ts.weekday(),
        "month": ts.month,
        "is_weekend": int(ts.weekday() >= 5),
    }


def _context_notes(lat: float, lon: float) -> list[str]:
    return [] if in_sf(lat, lon) else ["location is outside San Francisco; predictions are unreliable"]


# --------------------------------------------------------------------------- photo router


class PhotoRouter:
    """Predict the 311 category (service_name) from a photo plus location/time."""

    name = "base"
    is_stub = True

    def __init__(self, stats: dict[str, Any] | None = None):
        self.stats = stats if stats is not None else load_stub_stats()
        self.agency_by_label: dict[str, str] = self.stats.get("agency_by_label", {})

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> PhotoRouter:
        mode = os.environ.get("SF311_ROUTER", "auto").lower()
        stats = load_stub_stats()
        if mode in ("auto", "model") and (models_dir / "image" / "model.pt").exists():
            return TorchScriptRouter(models_dir / "image", stats)
        if mode == "clip":
            try:
                return ClipZeroShotRouter(stats)
            except Exception as exc:  # missing package or weights: degrade, don't crash
                router = PriorRouter(stats)
                router.load_note = f"CLIP unavailable ({type(exc).__name__}); using prior stub"
                return router
        return PriorRouter(stats)

    def _probs(self, image, lat: float, lon: float, ts: datetime) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, image, lat: float, lon: float, ts: datetime | None = None) -> RouteResult:
        """``image``: PIL image or raw bytes. Returns label, top-3 probabilities and agency hint."""
        image = _to_pil(image)
        ts = ts or datetime.now()
        probs = self._probs(image, lat, lon, ts)
        top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        notes = _context_notes(lat, lon)
        if getattr(self, "load_note", None):
            notes.append(self.load_note)
        if self.is_stub:
            notes.append("stub router: no trained image model found in models/image/")
        return RouteResult(
            label=top[0][0],
            top3=[(k, float(v)) for k, v in top],
            agency_hint=self.agency_by_label.get(top[0][0]),
            model=self.name,
            is_stub=self.is_stub,
            notes=notes,
        )


def _to_pil(image):
    from PIL import Image

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    return image.convert("RGB")


class PriorRouter(PhotoRouter):
    """Stub: ignores the image and returns historical class frequencies."""

    name = "prior-stub"
    is_stub = True

    def _probs(self, image, lat, lon, ts):
        prior = self.stats.get("label_prior") or {OTHER: 1.0}
        total = sum(prior.values())
        return {k: v / total for k, v in prior.items()}


class TorchScriptRouter(PhotoRouter):
    name = "image-model"
    is_stub = False

    def __init__(self, model_dir: Path, stats: dict[str, Any]):
        import torch

        super().__init__(stats)
        self.model = torch.jit.load(str(model_dir / "model.pt"), map_location="cpu").eval()
        self.labels: list[str] = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        cfg_path = model_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.size = int(cfg.get("image_size", 224))
        self.mean = torch.tensor(cfg.get("mean", IMAGENET_MEAN)).view(3, 1, 1)
        self.std = torch.tensor(cfg.get("std", IMAGENET_STD)).view(3, 1, 1)
        self.name = cfg.get("name", self.name)

    def _probs(self, image, lat, lon, ts):
        import numpy as np
        import torch

        # Same as the training test transform: resize the short side, then centre-crop.
        w, h = image.size
        scale = self.size / min(w, h)
        img = image.resize((max(self.size, round(w * scale)), max(self.size, round(h * scale))))
        left, top = (img.width - self.size) // 2, (img.height - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - self.mean) / self.std).unsqueeze(0)
        with torch.inference_mode():
            p = torch.softmax(self.model(x), dim=1)[0].tolist()
        return dict(zip(self.labels, p, strict=True))


class ClipZeroShotRouter(PhotoRouter):
    """Opt-in stub: zero-shot CLIP with one text prompt per class."""

    name = "clip-zeroshot-stub"
    is_stub = True
    PROMPTS = {
        "Street and Sidewalk Cleaning": "a photo of trash, litter or dumped items on a city sidewalk",
        "Parking Enforcement": "a photo of a car parked illegally or blocking a driveway",
        "General Request": "a photo of a city street",
        "Graffiti Public": "a photo of graffiti on a public pole, sign or utility box",
        "Blocked Street and Sidewalk": "a photo of a sidewalk blocked by objects or construction",
        "Encampment": "a photo of a tent on a city sidewalk",
        "Sidewalk and Curb": "a photo of a broken, cracked sidewalk or curb",
        "Tree Maintenance": "a photo of a fallen or overgrown street tree",
        "RPD General": "a photo of a park, playground or public garden",
        "Graffiti Private": "a photo of graffiti on a building wall or garage door",
        "Litter Receptacle Maintenance": "a photo of an overflowing or damaged public trash can",
        "Sewer": "a photo of a clogged storm drain, sewer or flooding in the street",
        OTHER: "a photo of something in a city",
    }

    def __init__(self, stats: dict[str, Any]):
        import open_clip
        import torch

        super().__init__(stats)
        arch = os.environ.get("SF311_CLIP_ARCH", "ViT-B-32")
        pretrained = os.environ.get("SF311_CLIP_PRETRAINED", "laion2b_s34b_b79k")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval()
        tok = open_clip.get_tokenizer(arch)
        self.labels = list(self.PROMPTS)
        with torch.inference_mode():
            t = self.model.encode_text(tok(list(self.PROMPTS.values())))
            self.text = t / t.norm(dim=-1, keepdim=True)

    def _probs(self, image, lat, lon, ts):
        import torch

        with torch.inference_mode():
            f = self.model.encode_image(self.preprocess(image).unsqueeze(0))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).softmax(dim=-1)[0].tolist()
        return dict(zip(self.labels, p, strict=True))


# --------------------------------------------------------------------------- resolution time


class ResolutionPredictor:
    """Predict median hours to close and P(late > 7 days) for a request."""

    name = "base"
    is_stub = True

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> ResolutionPredictor:
        if os.environ.get("SF311_RESOLVER", "auto").lower() != "stub":
            res_dir = models_dir / "resolution"
            if (res_dir / "late7d_lgbm.txt").exists() and (res_dir / "serving_context.json").exists():
                return LightGBMResolutionPredictor(res_dir)
            if (res_dir / "model.joblib").exists():
                return JoblibResolutionPredictor(res_dir / "model.joblib")
        return MedianResolutionPredictor(load_stub_stats())

    def _estimate(self, subtype, lat, lon, ts, source, category) -> tuple[float, float, str]:
        raise NotImplementedError

    def predict(
        self,
        subtype: str,
        lat: float,
        lon: float,
        ts: datetime | None = None,
        source: str = "Mobile",
        category: str | None = None,
    ) -> ResolutionResult:
        """``category`` (service_name) is optional; the stub uses it when the subtype is unknown."""
        ts = ts or datetime.now()
        hours, p_late, basis = self._estimate(subtype, lat, lon, ts, source, category)
        notes = _context_notes(lat, lon)
        if self.is_stub:
            notes.append("stub estimate: historical medians, ignores location and time")
        return ResolutionResult(
            median_hours=round(float(hours), 2),
            p_late=round(min(max(float(p_late), 0.0), 1.0), 4),
            late_threshold_hours=LATE_HOURS,
            model=self.name,
            is_stub=self.is_stub,
            basis=basis,
            notes=notes,
        )


class MedianResolutionPredictor(ResolutionPredictor):
    name = "subtype-median-stub"
    is_stub = True
    MIN_N = 50  # use the subtype+source cell only with at least this many closed cases

    def __init__(self, stats: dict[str, Any]):
        self.stats = stats

    def _estimate(self, subtype, lat, lon, ts, source, category):
        s = self.stats
        cell = s.get("by_subtype_source", {}).get(f"{subtype}|{source}")
        if cell and cell["n"] >= self.MIN_N:
            return cell["median_hours"], cell["p_late"], f"subtype+source (n={cell['n']})"
        cell = s.get("by_subtype", {}).get(subtype)
        if cell:
            return cell["median_hours"], cell["p_late"], f"subtype (n={cell['n']})"
        cell = s.get("by_category", {}).get(category or "")
        if cell:
            return cell["median_hours"], cell["p_late"], f"category (n={cell['n']})"
        cell = s.get("overall")
        if cell:
            return cell["median_hours"], cell["p_late"], f"all requests (n={cell['n']})"
        return 24.0, 0.25, "built-in default (no stub_stats.json)"


class JoblibResolutionPredictor(ResolutionPredictor):
    name = "resolution-model"
    is_stub = False

    def __init__(self, path: Path):
        import joblib

        self.artifact = joblib.load(path)
        if isinstance(self.artifact, dict):
            self.name = self.artifact.get("name", self.name)

    def _estimate(self, subtype, lat, lon, ts, source, category):
        feats = resolution_features(subtype, lat, lon, ts, source, category)
        a = self.artifact
        if hasattr(a, "predict_one"):
            hours, p_late = a.predict_one(feats)
            return hours, p_late, "trained model"
        import pandas as pd

        X = pd.DataFrame([feats])
        cols = a.get("feature_columns") or list(X.columns)
        X = X[cols]
        for c in X.columns:
            if X[c].dtype == object:
                X[c] = X[c].astype("category")
        hours = float(a["hours_model"].predict(X)[0])
        if a.get("target") == "log1p_hours":
            hours = math.expm1(hours)
        p_late = float(a["late_model"].predict_proba(X)[0, 1])
        return max(hours, 0.0), p_late, "trained model"


class LightGBMResolutionPredictor(ResolutionPredictor):
    """The models saved by ``python -m sf311.resolution train``.

    Backlog features (recent volume, open cases) need the full request history, which the app
    doesn't have; they come from ``serving_context.json`` (recent medians, built by
    ``python -m sf311.resolution_serving``). Category, agency and details default to the most
    common values for the sub-category. Hours are on the 30-day-capped scale used in training.
    """

    name = "lightgbm-resolution"
    is_stub = False

    def __init__(self, res_dir: Path):
        import lightgbm as lgb

        self.meta = json.loads((res_dir / "meta.json").read_text(encoding="utf-8"))
        self.ctx = json.loads((res_dir / "serving_context.json").read_text(encoding="utf-8"))
        self.late_model = lgb.Booster(model_file=str(res_dir / "late7d_lgbm.txt"))
        self.hours_model = lgb.Booster(model_file=str(res_dir / "log_hours_cap30_lgbm.txt"))

    def _row(self, subtype, lat, lon, ts, source, category) -> dict[str, Any]:
        defaults = self.ctx["subtype_defaults"].get(subtype, {})
        name = category or defaults.get("service_name")
        agency = defaults.get("agency_responsible")
        row: dict[str, Any] = {
            "service_name": name,
            "service_subtype": subtype,
            "service_details": defaults.get("service_details"),
            "agency_responsible": agency,
            "source": source,
            "lat": lat,
            "long": lon,
            "hour": ts.hour,
            "weekday": ts.weekday(),
            "is_weekend": int(ts.weekday() >= 5),
            "has_photo": int(source == "Mobile"),
        }
        row.update(self.ctx["city"])
        for key, value in (("service_subtype", subtype), ("service_name", name), ("agency_responsible", agency)):
            row.update(self.ctx.get(key, {}).get(value, {}))
        return row

    def _estimate(self, subtype, lat, lon, ts, source, category):
        import pandas as pd

        row = self._row(subtype, lat, lon, ts, source, category)
        cats = self.meta["categories"]
        X = pd.DataFrame(index=[0])
        for c in self.meta["features"]:
            v = row.get(c)
            if c in cats:
                v = "NA" if v is None else v
                X[c] = pd.Categorical([v if v in cats[c] else None], categories=cats[c])
            else:
                X[c] = [float("nan") if v is None else float(v)]
        hours = math.expm1(float(self.hours_model.predict(X)[0]))
        p_late = float(self.late_model.predict(X)[0])
        known = subtype in self.ctx["subtype_defaults"]
        basis = "LightGBM, typical recent backlog" + ("" if known else " (unknown sub-category)")
        return max(hours, 0.0), p_late, basis


# --------------------------------------------------------------------------- stub stats builder


def build_stub_stats(out: Path = STUB_STATS) -> dict[str, Any]:
    """Aggregate training-period data into app/stub_stats.json (no row-level data)."""
    from sf311.dataset import LATE_HOURS as DS_LATE
    from sf311.dataset import load, photo_rows, split

    assert DS_LATE == LATE_HOURS, "LATE_HOURS drifted from sf311.dataset"
    df = load()
    train, _, _ = split(df)
    closed = train[train["is_closed"] & ~train["is_admin_close"]]

    def agg(g) -> dict[str, Any]:
        h = g["resolution_hours"]
        return {
            "n": int(len(g)),
            "median_hours": round(float(h.median()), 2),
            "p_late": round(float((h > LATE_HOURS).mean()), 4),
        }

    stats: dict[str, Any] = {
        "built_from": f"{train['requested_datetime'].min():%Y-%m-%d} to {train['requested_datetime'].max():%Y-%m-%d} (training split)",
        "late_hours": LATE_HOURS,
        "overall": agg(closed),
        "by_category": {k: agg(g) for k, g in closed.groupby("service_name") if len(g) >= 20},
        "by_subtype": {k: agg(g) for k, g in closed.groupby("service_subtype") if len(g) >= 20},
        "by_subtype_source": {
            f"{s}|{src}": agg(g) for (s, src), g in closed.groupby(["service_subtype", "source"]) if len(g) >= 50
        },
    }
    photos = photo_rows(train)
    stats["label_prior"] = {k: int(v) for k, v in photos["label"].value_counts().items()}
    # Most common responsible agency per photo label: a hint, since one category can map to many agencies.
    stats["agency_by_label"] = {
        lbl: g["agency_responsible"].mode().iat[0]
        for lbl, g in photos.groupby("label")
        if lbl != OTHER and g["agency_responsible"].notna().any()
    }
    # Sub-categories per label, most common first (lets the demo pick a subtype for the predicted label).
    stats["subtypes_by_label"] = {
        lbl: g["service_subtype"].value_counts().head(8).index.tolist() for lbl, g in photos.groupby("label")
    }
    out.write_text(json.dumps(stats, indent=1, sort_keys=True), encoding="utf-8")
    return stats


if __name__ == "__main__":
    s = build_stub_stats()
    print(f"wrote {STUB_STATS} ({len(s['by_subtype'])} subtypes, built from {s['built_from']})")
