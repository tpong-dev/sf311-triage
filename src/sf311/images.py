"""Local image cache for photo routing models.

Usage:
    python -m sf311.images download            # all splits
    python -m sf311.images download --splits test

Builds a label-balanced, fixed-seed subset of photo requests per split (cap per label:
train 1500, val 300, test 500; all rows if a label has fewer) and downloads each photo
as a small JPEG (max side 256) to data/images/{split}/{service_request_id}.jpg.

Bandwidth: Cloudinary resizes on its side when `w_256,h_256,c_limit/` is inserted right
after `/image/upload/` (~10x smaller files). If that URL fails, the original is fetched
and resized locally.

Privacy: photos can show people and licence plates. They stay under data/images
(gitignored) for local training only; never copy them or their URLs elsewhere.

Manifest: data/images/manifest.parquet with columns
    service_request_id, split, label, service_subtype, path, ok
`path` is relative to the project root (forward slashes). It is rewritten every
~1000 finished images so other jobs can start on the images that are already done;
rows not yet attempted are not in it. Re-running skips files that already exist.
"""

import argparse
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps

from sf311.dataset import load, photo_rows, split

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "images"
MANIFEST = IMG_DIR / "manifest.parquet"

CAPS = {"train": 1500, "val": 300, "test": 500}
SEED = 42
MAX_SIDE = 256
TRANSFORM = f"w_{MAX_SIDE},h_{MAX_SIDE},c_limit/"
MAX_THREADS = 8
TIMEOUT = (5, 20)
RETRIES = 3
DELAY = 0.05  # pause after each request per thread
FLUSH_EVERY = 1000
USER_AGENT = "sf311-triage-image-cache/0.1 (research; small thumbnails)"
COLUMNS = ["service_request_id", "split", "label", "service_subtype", "path", "ok"]


def balanced_subset() -> pd.DataFrame:
    """Fixed-seed, per-label capped sample of each split."""
    parts = []
    for name, df in zip(("train", "val", "test"), split(photo_rows(load()))):
        cap = CAPS[name]
        sub = pd.concat(
            g.sample(n=min(cap, len(g)), random_state=SEED) for _, g in df.groupby("label", sort=True)
        )
        parts.append(sub.assign(split=name))
    out = pd.concat(parts, ignore_index=True)
    out["service_request_id"] = out["service_request_id"].astype(str)
    out["path"] = "data/images/" + out["split"] + "/" + out["service_request_id"] + ".jpg"
    return out[["service_request_id", "split", "label", "service_subtype", "path", "photo_url"]]


def thumb_url(url: str) -> str:
    return url.replace("/image/upload/", "/image/upload/" + TRANSFORM, 1)


def _get(session: requests.Session, url: str) -> bytes | None:
    for attempt in range(RETRIES):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            time.sleep(DELAY)
            if resp.status_code == 200 and resp.content:
                return resp.content
            if resp.status_code in (400, 401, 403, 404, 410):
                return None  # permanent; don't hammer
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def _save(data: bytes, dest: Path) -> bool:
    """Decode (validates the bytes), fix orientation, shrink to MAX_SIDE, save JPEG."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        tmp = dest.with_suffix(".tmp")
        img.save(tmp, "JPEG", quality=90)
        tmp.replace(dest)
        return True
    except Exception:
        return False


def fetch_one(session: requests.Session, url: str, dest: Path) -> tuple[bool, str]:
    if dest.exists():
        return True, "cached"
    data = _get(session, thumb_url(url))
    if data and _save(data, dest):
        return True, "thumb"
    data = _get(session, url)  # fallback: original, resized locally
    if data and _save(data, dest):
        return True, "original"
    return False, "failed"


def write_manifest(rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS)
    tmp = MANIFEST.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(MANIFEST)


def download(splits: list[str]) -> None:
    t0 = time.time()
    subset = balanced_subset()
    subset = subset[subset["split"].isin(splits)].reset_index(drop=True)
    print(f"selected {len(subset):,} images")
    print(subset.groupby(["split", "label"]).size().unstack(0).to_string())
    for s in splits:
        (IMG_DIR / s).mkdir(parents=True, exist_ok=True)

    # Keep manifest rows from splits not being downloaded in this run.
    done_rows: list[dict] = []
    if MANIFEST.exists():
        old = pd.read_parquet(MANIFEST)
        done_rows = old[~old["split"].isin(splits)].to_dict("records")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_THREADS, pool_maxsize=MAX_THREADS)
    session.mount("https://", adapter)

    lock = threading.Lock()
    stats: dict[str, int] = {}
    # Small splits first so evaluation sets are ready early; train follows.
    order = {"val": 0, "test": 1, "train": 2}
    rows = subset.sort_values("split", key=lambda c: c.map(order), kind="stable").to_dict("records")

    def work(r: dict) -> dict:
        ok, how = fetch_one(session, r["photo_url"], ROOT / r["path"])
        out = {k: r[k] for k in COLUMNS if k != "ok"}
        out["ok"] = ok
        return out | {"_how": how}

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as pool:
        futures = [pool.submit(work, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            how = res.pop("_how")
            with lock:
                stats[how] = stats.get(how, 0) + 1
                done_rows.append(res)
                if i % FLUSH_EVERY == 0:
                    write_manifest(done_rows)
                    rate = i / (time.time() - t0)
                    print(f"  {i:,}/{len(rows):,}  {rate:.1f} img/s  {stats}", flush=True)

    write_manifest(done_rows)
    man = pd.DataFrame(done_rows)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min  {stats}")
    print(man.groupby("split")["ok"].agg(["sum", "count"]).to_string())
    print(f"wrote {MANIFEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="download the balanced image subset")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args = parser.parse_args()
    if args.command == "download":
        download(args.splits)


if __name__ == "__main__":
    main()
