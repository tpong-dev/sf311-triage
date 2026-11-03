"""Frontier vision-LLM baseline: classify a 311 photo into dataset.LABELS + "Other" with Claude.

    set PYTHONPATH=src
    python -m sf311.vision_llm --dry-run                         # prompt + cost estimate, NO API call
    python -m sf311.vision_llm --model claude-haiku-5-5 --n 500  # paid run (needs ANTHROPIC_API_KEY)
    python -m sf311.vision_llm --model claude-sonnet-5-5 --n 500 --report-only   # metrics from cache

Inputs: data/images/manifest.parquet (service_request_id, split, label, service_subtype, path, ok)
and the photos it points at (data/images/{split}/{id}.jpg). Only the test split is used.

Each photo is one Messages API request: a cached system prompt describing every label, the image
(downscaled in memory to --max-side px to control image tokens) and a structured-output JSON schema
returning {label, confidence, reason}. Results are appended to data/llm_preds/{model}.parquet keyed by
(service_request_id, prompt_hash), so reruns only pay for photos not yet classified with this prompt.

Privacy: photos never leave the machine except in the API request itself; nothing image-derived
(no bytes, no URLs) is written to the report or the predictions file - only labels, scores, a short
model-written reason, and token usage.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sf311.dataset import LABELS, OTHER

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "images" / "manifest.parquet"
PRED_DIR = ROOT / "data" / "llm_preds"

DEFAULT_MODEL = "claude-sonnet-5-5"
CHEAP_MODEL = "claude-haiku-5-5"
ALL_LABELS = LABELS + [OTHER]
SEED = 42

# USD per million tokens. Source: Anthropic published pricing as cached in the claude-api skill
# (model table dated 2026-10-06; https://platform.claude.com/docs/en/about-claude/pricing).
# Cache writes are 1.25x input (5-minute TTL); cache reads are 0.1x input.
# Haiku 5.5 rates apply to prompts <= 100K tokens (always true here). Re-check before budgeting.
PRICING = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
}
PRICING_SOURCE = "Anthropic pricing via claude-api skill model table, cached 2026-10-06"

# Models that accept the server-side refusal fallback ("default" form, Claude API only).
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# --- prompt -----------------------------------------------------------------------------------

LABEL_GUIDE = {
    "Street and Sidewalk Cleaning": (
        "Loose garbage or debris on a street or sidewalk that needs pickup: litter, bags of trash, "
        "dumped furniture, mattresses, appliances, boxes, construction debris piles, food waste, "
        "broken glass, needles, human or animal waste, spills. By far the most common class."
    ),
    "Parking Enforcement": (
        "A vehicle is the subject: car/truck parked on the sidewalk, blocking a driveway or garage, "
        "double-parked, in a bike lane, red/yellow zone or crosswalk, oversized vehicle, or an "
        "apparently abandoned vehicle (flat tyres, dusty, covered, no plates). Photos often show a "
        "parked car and a curb or driveway."
    ),
    "General Request": (
        "Catch-all request addressed to a department rather than a defined problem type. Most often "
        "a person in distress or someone needing outreach/welfare check without a tent setup, "
        "other public-works or transit (MTA) issues that fit no specific class. Visually varied: "
        "choose this only when the photo shows a real city issue that none of the specific classes covers "
        "and it is not one of the 'Other' examples."
    ),
    "Graffiti Public": (
        "Graffiti, tags, stickers or etching on PUBLIC property: utility poles, signal and electrical "
        "boxes, mailboxes, bus shelters, public trash cans, street signs, parking meters, sidewalks, "
        "retaining walls, bridges, city buildings, park structures."
    ),
    "Graffiti Private": (
        "Graffiti or tags on PRIVATE property: house or apartment walls, storefronts, roll-down shop "
        "gates, garage doors, private fences, business windows, private vehicles."
    ),
    "Blocked Street and Sidewalk": (
        "Objects (not loose trash, not parked cars) obstructing the sidewalk or street: construction "
        "materials, debris boxes/dumpsters, scaffolding, A-frame signs, tables and chairs, planters, "
        "stored items, barricades left behind, blocked parking strip."
    ),
    "Encampment": (
        "Homeless encampment: tents, tarps or makeshift shelters, sleeping bags/bedding set up on the "
        "sidewalk, shopping carts with belongings, a group of people living on the street."
    ),
    "Sidewalk and Curb": (
        "Physical damage to the pavement itself: cracked, raised or sunken sidewalk, trip hazards, "
        "broken curb or curb ramp, pavement defect, missing/damaged side-sewer vent cover or a "
        "manhole cover off, damaged public stairway."
    ),
    "Tree Maintenance": (
        "Street trees and landscaping: fallen or broken branches or trees, leaning or damaged tree, "
        "overgrown tree blocking sidewalk/signs/lights, roots damaging property, tree-well or "
        "median landscaping problems."
    ),
    "RPD General": (
        "Problem inside a city park, playground, sports field or recreation facility (Recreation and "
        "Parks Department): broken park structures or benches, park restrooms, lawns, park patrol "
        "issues, park trees. Choose this when the setting is clearly a park."
    ),
    "Litter Receptacle Maintenance": (
        "Public litter cans or bins themselves: overflowing, tipped over, damaged, missing liner or "
        "lid, broken door lock, residential toters/bins left out on the sidewalk, a bin that should "
        "be removed, cans at transit shelters/platforms."
    ),
    "Sewer": (
        "Water and sewer: clogged storm drain or catch basin, street flooding or ponding, water leak "
        "from the ground, sewage back-up or discharge, sewer odor source."
    ),
    OTHER: (
        "Anything else the city handles: streetlights out or damaged poles, traffic signals, street "
        "signs, potholes in the roadway, transit shelter damage, posters or flyers affixed to poles, "
        "noise complaints, and photos with no visible city issue (indoor, screenshots, documents, "
        "unclear or unrelated images)."
    ),
}

SYSTEM_PROMPT = (
    "You route San Francisco 311 service requests. A resident submitted the attached photo with a "
    "request; your job is to choose the single service category the city would file it under, "
    "using the photo only.\n\n"
    "Categories (use the exact name):\n"
    + "\n".join(f"- {name}: {LABEL_GUIDE[name]}" for name in ALL_LABELS)
    + "\n\nGuidance for hard cases:\n"
    "- Pick the category for the main problem the resident is most likely reporting, usually the "
    "thing in the centre or foreground of the photo.\n"
    "- Trash on the ground -> Street and Sidewalk Cleaning; problems with a trash CAN itself "
    "(overflowing, broken, bin left out) -> Litter Receptacle Maintenance.\n"
    "- A parked vehicle as the problem -> Parking Enforcement, even if it blocks the sidewalk. "
    "Non-vehicle objects blocking the way -> Blocked Street and Sidewalk.\n"
    "- Graffiti: decide public vs private by who owns the surface; city fixtures are public, "
    "buildings, shopfronts and garage doors are private. If you cannot tell, prefer Graffiti Public.\n"
    "- Tents, tarps or bedding set up as shelter -> Encampment; a person alone without a shelter "
    "setup -> General Request.\n"
    "- Damaged sidewalk or curb -> Sidewalk and Curb; potholes in the roadway -> Other.\n"
    "- Clearly inside a park -> RPD General, unless it is plainly graffiti, a tree on a street, "
    "or an encampment.\n"
    "- If the photo is unclear, unrelated or shows no city issue, answer Other with low confidence.\n\n"
    "Return JSON only: label (one category name exactly as written), confidence (0 to 1, your "
    "probability that the label is correct), reason (one short sentence naming what you saw; do not "
    "describe people's faces, identify anyone, or transcribe licence plates or addresses)."
)

USER_TEXT = "Classify this 311 photo."

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ALL_LABELS},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["label", "confidence", "reason"],
    "additionalProperties": False,
}


def prompt_hash() -> str:
    """Changes whenever anything the model sees (besides the image) changes; keys the result cache."""
    blob = json.dumps([SYSTEM_PROMPT, USER_TEXT, OUTPUT_SCHEMA], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --- images -----------------------------------------------------------------------------------


def encode_image(path: Path, max_side: int) -> tuple[str, str, tuple[int, int]]:
    """Return (base64 data, media type, (w, h) as sent). Downscales in memory only; never writes."""
    from PIL import Image

    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        fmt = (im.format or "JPEG").upper()
        w, h = im.size
        if max_side and max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg", im.size
    media = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
    if fmt not in media:  # e.g. MPO from phones: re-encode as JPEG
        with Image.open(io.BytesIO(raw)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=90)
            return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg", (w, h)
    return base64.standard_b64encode(raw).decode(), media[fmt], (w, h)


def image_tokens(w: int, h: int, max_edge: int = 2576) -> int:
    """Approximate image tokens (Anthropic's rule of thumb w*h/750; API downsizes beyond max_edge)."""
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        w, h = w * s, h * s
    return math.ceil(w * h / 750)


# --- request ----------------------------------------------------------------------------------


def build_request(model: str, image_b64: str, media_type: str, effort: str, fallback: bool) -> dict:
    req = {
        "model": model,
        "max_tokens": 1024,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                    {"type": "text", "text": USER_TEXT},
                ],
            }
        ],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }
    if fallback and model in FALLBACK_MODELS:
        req["betas"] = [FALLBACK_BETA]
        req["fallbacks"] = "default"
    return req


def cost_usd(model: str, usage: dict) -> float:
    p = PRICING.get(model)
    if p is None:
        return float("nan")
    return (
        usage.get("input_tokens", 0) * p["input"]
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
        + usage.get("output_tokens", 0) * p["output"]
    ) / 1e6


def parse_response(resp) -> dict:
    """Pull label/confidence/reason + usage out of a Messages response."""
    u = resp.usage
    usage = {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
    }
    out = {"stop_reason": resp.stop_reason, "model_served": getattr(resp, "model", None), **usage}
    if resp.stop_reason == "refusal":
        return {**out, "pred": OTHER, "confidence": 0.0, "reason": "refused", "error": "refusal"}
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if text is None:
        return {**out, "pred": None, "confidence": None, "reason": None, "error": f"no text ({resp.stop_reason})"}
    try:
        data = json.loads(text)
        label = data["label"] if data["label"] in ALL_LABELS else OTHER
        conf = float(min(max(data.get("confidence", 0.0), 0.0), 1.0))
        return {**out, "pred": label, "confidence": conf, "reason": str(data.get("reason", ""))[:300], "error": None}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return {**out, "pred": None, "confidence": None, "reason": None, "error": f"bad json: {e}"}


# --- sampling / cache -------------------------------------------------------------------------


def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    m = pd.read_parquet(path)
    m = m[(m["split"] == "test") & (m["ok"].astype(bool))].copy()
    m["service_request_id"] = m["service_request_id"].astype(str)
    return m.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED, min_per_class: int = 5) -> pd.DataFrame:
    """Proportional-by-label sample of n rows, with at least min_per_class per label when available.

    Deterministic for a given (df, n, seed), and the n=500 sample is not required to be a subset of
    n=1300; cached predictions are reused by id either way.
    """
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    counts = df["label"].value_counts()
    alloc = {lab: min(c, max(min_per_class, round(n * c / len(df)))) for lab, c in counts.items()}
    # trim/fill to hit n exactly, adjusting the largest classes first
    order = counts.index.tolist()
    i = 0
    while sum(alloc.values()) > n:
        lab = order[i % len(order)]
        if alloc[lab] > min(min_per_class, counts[lab]):
            alloc[lab] -= 1
        i += 1
    i = 0
    while sum(alloc.values()) < n:
        lab = order[i % len(order)]
        if alloc[lab] < counts[lab]:
            alloc[lab] += 1
        i += 1
    parts = [df[df["label"] == lab].sample(k, random_state=seed) for lab, k in alloc.items() if k > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def cache_path(model: str, pred_dir: Path = PRED_DIR) -> Path:
    return pred_dir / f"{model}.parquet"


def load_cache(model: str, pred_dir: Path = PRED_DIR) -> pd.DataFrame:
    p = cache_path(model, pred_dir)
    if p.exists():
        c = pd.read_parquet(p)
        c["service_request_id"] = c["service_request_id"].astype(str)
        return c
    return pd.DataFrame()


def save_cache(model: str, rows: list[dict], pred_dir: Path = PRED_DIR) -> None:
    if not rows:
        return
    pred_dir.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    old = load_cache(model, pred_dir)
    allc = pd.concat([old, new], ignore_index=True) if len(old) else new
    allc = allc.drop_duplicates(["service_request_id", "prompt_hash"], keep="last")
    tmp = cache_path(model, pred_dir).with_suffix(".tmp")
    allc.to_parquet(tmp, index=False)
    tmp.replace(cache_path(model, pred_dir))


# --- metrics ----------------------------------------------------------------------------------


def metrics(y_true: list[str], y_pred: list[str], labels: list[str] = ALL_LABELS) -> dict:
    """Accuracy, macro-F1 over labels present in y_true, and per-class recall/precision/F1/support."""
    pairs = list(zip(y_true, y_pred))
    n = len(pairs)
    per = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in pairs)
        fp = sum(t != lab and p == lab for t, p in pairs)
        fn = sum(t == lab and p != lab for t, p in pairs)
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if support and prec + rec else 0.0
        per[lab] = {"support": support, "recall": rec, "precision": prec, "f1": f1}
    present = [lab for lab in labels if per[lab]["support"] > 0]
    return {
        "n": n,
        "accuracy": sum(t == p for t, p in pairs) / n if n else float("nan"),
        "macro_f1": sum(per[lab]["f1"] for lab in present) / len(present) if present else float("nan"),
        "per_class": per,
    }


def summarize(preds: pd.DataFrame, model: str) -> str:
    ok = preds[preds["pred"].notna()]
    m = metrics(ok["label"].tolist(), ok["pred"].tolist())
    cost = ok["cost_usd"].sum()
    lines = [
        f"model={model}  n={m['n']}  errors={len(preds) - len(ok)}",
        f"accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}",
        f"cost=${cost:.4f}  cost_per_1000=${1000 * cost / max(len(ok), 1):.2f}  ({PRICING_SOURCE})",
        f"tokens/img: input={ok['input_tokens'].mean():.0f} cache_read={ok['cache_read_input_tokens'].mean():.0f} "
        f"cache_write={ok['cache_creation_input_tokens'].mean():.0f} output={ok['output_tokens'].mean():.0f}",
        "",
        f"{'label':32s} {'support':>7s} {'recall':>7s} {'prec':>6s} {'f1':>6s}",
    ]
    for lab, d in m["per_class"].items():
        if d["support"]:
            lines.append(f"{lab:32s} {d['support']:7d} {d['recall']:7.3f} {d['precision']:6.3f} {d['f1']:6.3f}")
    return "\n".join(lines)


# --- dry run ----------------------------------------------------------------------------------


def approx_text_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5)  # conservative chars-per-token heuristic; no API call


@dataclass
class Estimate:
    model: str
    n: int
    img_tokens: int
    system_tokens: int
    other_input_tokens: int
    output_tokens: int
    cost: float

    @property
    def per_1000(self) -> float:
        return 1000 * self.cost / self.n


def estimate(model: str, n: int, img_tokens: int, output_tokens: int = 250) -> Estimate:
    """First call writes the system-prompt cache; the rest read it (sequential warm-up, see run())."""
    p = PRICING[model]
    sys_t = approx_text_tokens(SYSTEM_PROMPT) + approx_text_tokens(json.dumps(OUTPUT_SCHEMA)) + 100
    other = img_tokens + approx_text_tokens(USER_TEXT) + 10
    cost = (
        sys_t * p["cache_write"] + (n - 1) * sys_t * p["cache_read"] + n * other * p["input"] + n * output_tokens * p["output"]
    ) / 1e6
    return Estimate(model, n, img_tokens, sys_t, other, output_tokens, cost)


def typical_image_tokens(paths: list[Path], max_side: int, k: int = 50) -> tuple[int, str]:
    """Mean image tokens over up to k local images (reads dimensions only), or a default."""
    from PIL import Image

    toks = []
    for p in paths[:k]:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue
        if max_side and max(w, h) > max_side:
            s = max_side / max(w, h)
            w, h = round(w * s), round(h * s)
        toks.append(image_tokens(w, h))
    if toks:
        return round(sum(toks) / len(toks)), f"measured on {len(toks)} local images"
    side = max_side or 2576
    return image_tokens(side, round(side * 0.75)), f"assumed 4:3 photo at {side}px long edge (no local images)"


def dry_run(args) -> None:
    print("=" * 80)
    print(f"DRY RUN - no API call. prompt_hash={prompt_hash()}")
    print("=" * 80)
    print("SYSTEM PROMPT (cached with cache_control):\n")
    print(SYSTEM_PROMPT)
    print("\nUSER: [image] +", repr(USER_TEXT))
    print("OUTPUT SCHEMA:", json.dumps(OUTPUT_SCHEMA))
    print(f"effort={args.effort}  max_side={args.max_side}  fallback={'on' if args.fallback else 'off'}")

    paths, have = [], 0
    if MANIFEST.exists():
        try:
            m = load_manifest()
            have = len(m)
            paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in m["path"]]
            print(f"\nmanifest: {have} usable test photos; label mix:\n{m['label'].value_counts().to_string()}")
        except Exception as e:  # manifest still being written by another job
            print(f"\nmanifest unreadable ({e}); using defaults")
    else:
        print(f"\nmanifest not found at {MANIFEST}; using default image size")
    img_t, how = typical_image_tokens(paths, args.max_side)
    print(f"image tokens/photo ~ {img_t} ({how})")

    models = [args.model] if args.model_given else [DEFAULT_MODEL, CHEAP_MODEL]
    ns = [args.n] if args.n_given else [500, 1300]
    print(f"\n{'model':20s} {'n':>5s} {'in tok/img':>10s} {'out tok/img':>11s} {'total $':>9s} {'$/1000':>8s}")
    for model in models:
        for n in ns:
            e = estimate(model, n, img_t)
            print(f"{model:20s} {n:5d} {e.system_tokens + e.other_input_tokens:10d} {e.output_tokens:11d} "
                  f"{e.cost:9.2f} {e.per_1000:8.2f}")
    print(f"\nPricing: {PRICING_SOURCE}. Output tokens assume ~250/img (JSON + low-effort thinking); "
          "actual cost is computed from usage tokens after a real run.")


# --- run --------------------------------------------------------------------------------------


async def classify_all(client, todo: pd.DataFrame, args, ph: str, on_result) -> None:
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row) -> None:
        path = Path(row.path) if Path(row.path).is_absolute() else ROOT / row.path
        try:
            data, media, _ = encode_image(path, args.max_side)
        except Exception as e:
            print(f"  skip {row.service_request_id}: cannot read image ({type(e).__name__})", file=sys.stderr)
            return
        req = build_request(args.model, data, media, args.effort, args.fallback)
        async with sem:
            for attempt in range(args.retries + 1):
                try:
                    resp = await client.beta.messages.create(**req)
                    res = parse_response(resp)
                    if res["error"] and res["error"].startswith(("bad json", "no text")) and attempt < args.retries:
                        continue
                    break
                except Exception as e:  # SDK already retried 429/5xx/connection max_retries times
                    import anthropic

                    retryable = isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError)) or (
                        isinstance(e, anthropic.APIStatusError) and e.status_code >= 500
                    )
                    if not retryable or attempt == args.retries:
                        print(f"  fail {row.service_request_id}: {type(e).__name__}", file=sys.stderr)
                        return
                    await asyncio.sleep(min(60, 2 ** (attempt + 2)))
        served = res.get("model_served") or args.model
        priced = served if served in PRICING else args.model
        on_result({
            "service_request_id": str(row.service_request_id),
            "split": row.split,
            "label": row.label,
            "service_subtype": getattr(row, "service_subtype", None),
            "model": args.model,
            "prompt_hash": ph,
            "effort": args.effort,
            "max_side": args.max_side,
            "cost_usd": cost_usd(priced, res),
            "ts": pd.Timestamp.now(tz="UTC"),
            **res,
        })

    rows = list(todo.itertuples(index=False))
    if not rows:
        return
    await one(rows[0])  # warm the prompt cache once before fanning out
    await asyncio.gather(*(one(r) for r in rows[1:]))


def run(args) -> None:
    ph = prompt_hash()
    sample = stratified_sample(load_manifest(), args.n)
    cache = load_cache(args.model)
    done = set()
    if len(cache):
        done = set(cache.loc[(cache["prompt_hash"] == ph) & cache["pred"].notna(), "service_request_id"])
    todo = sample[~sample["service_request_id"].isin(done)]
    print(f"sample={len(sample)}  cached={len(sample) - len(todo)}  to_classify={len(todo)}  prompt_hash={ph}")

    if len(todo) and not args.report_only:
        est = estimate(args.model, len(todo), typical_image_tokens(
            [ROOT / p for p in todo["path"]], args.max_side)[0])
        print(f"estimated cost for this run: ${est.cost:.2f}")
        if args.max_cost is not None and est.cost > args.max_cost:
            sys.exit(f"estimate ${est.cost:.2f} exceeds --max-cost {args.max_cost}; aborting")
        import anthropic

        client = anthropic.AsyncAnthropic(max_retries=args.sdk_retries)
        buf: list[dict] = []
        t0, n_done = time.time(), [0]

        def on_result(r: dict) -> None:
            buf.append(r)
            n_done[0] += 1
            if len(buf) >= 25:
                save_cache(args.model, buf)
                buf.clear()
                print(f"  {n_done[0]}/{len(todo)}  {time.time() - t0:.0f}s")

        try:
            asyncio.run(classify_all(client, todo, args, ph, on_result))
        finally:
            save_cache(args.model, buf)

    cache = load_cache(args.model)
    if not len(cache):
        print("no predictions yet")
        return
    preds = cache[(cache["prompt_hash"] == ph) & cache["service_request_id"].isin(sample["service_request_id"])]
    print()
    print(summarize(preds, args.model))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help=f"default {DEFAULT_MODEL}; cheap option {CHEAP_MODEL}")
    ap.add_argument("--n", type=int, default=None, help="stratified sample size from the test split (default 500)")
    ap.add_argument("--dry-run", action="store_true", help="print prompt + cost estimate; no API call")
    ap.add_argument("--report-only", action="store_true", help="compute metrics from cached predictions only")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2, help="app-level retries on top of the SDK's own")
    ap.add_argument("--sdk-retries", type=int, default=4, help="SDK max_retries (429/5xx/connection)")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-side", type=int, default=1024, help="downscale long edge to this many px (0 = original)")
    ap.add_argument("--no-fallback", dest="fallback", action="store_false",
                    help="disable server-side refusal fallback (Sonnet/Opus only)")
    ap.add_argument("--max-cost", type=float, default=None, help="abort if the estimate exceeds this many USD")
    args = ap.parse_args(argv)
    args.model_given, args.n_given = args.model is not None, args.n is not None
    args.model = args.model or DEFAULT_MODEL
    args.n = args.n or 500
    if args.model not in PRICING:
        print(f"warning: no pricing for {args.model}; cost will be NaN", file=sys.stderr)
    if args.dry_run:
        dry_run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
