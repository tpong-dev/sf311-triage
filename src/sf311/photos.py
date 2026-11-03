"""Check whether 311 photo links (media_url) actually work, using metadata only.

Usage:
    python -m sf311.photos check data/raw/sf311_2026-10_sample.parquet --n 300

Sends HEAD requests (falls back to a ranged GET that is closed without reading the
body) to a random, fixed-seed sample of links. Photos can show people and licence
plates, so no image bytes are read into a file or displayed; only status, content
type and size are recorded. Writes an aggregate summary to data/photo_check.json
(no URLs, since they contain case ids).
"""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "photo_check.json"
MAX_THREADS = 8
TIMEOUT = (5, 10)  # connect, read seconds
DELAY = 0.1  # pause after each request per thread, to stay polite
SEED = 42
USER_AGENT = "sf311-triage-link-check/0.1 (research; metadata only)"


def link_type(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("cloudinary.com"):
        return "cloudinary"
    if host.endswith("verintcloudservices.com"):
        return "verint"
    return "other"


def _size(resp: requests.Response) -> int | None:
    # Content-Range ("bytes 0-3/12345") gives the real total for ranged GETs.
    rng = resp.headers.get("Content-Range", "")
    if "/" in rng and rng.rsplit("/", 1)[1].isdigit():
        return int(rng.rsplit("/", 1)[1])
    length = resp.headers.get("Content-Length", "")
    return int(length) if length.isdigit() else None


def probe(session: requests.Session, url: str) -> dict:
    """Return status, content type, size, method used. Never reads the body."""
    result = {"status": None, "content_type": None, "size": None, "method": "HEAD", "error": None}
    try:
        resp = session.head(url, timeout=TIMEOUT, allow_redirects=True)
        if resp.status_code != 200:
            # Some servers reject HEAD; retry with a ranged GET of a few bytes.
            result["method"] = "GET"
            resp = session.get(
                url, headers={"Range": "bytes=0-3"}, timeout=TIMEOUT, allow_redirects=True, stream=True
            )
            resp.close()  # stream=True + close: body is never downloaded
        result["status"] = resp.status_code
        result["content_type"] = resp.headers.get("Content-Type", "").split(";")[0].strip().lower() or None
        result["size"] = _size(resp)
        result["final_host"] = urlparse(resp.url).hostname
    except requests.RequestException as err:
        result["method"] = "GET" if result["method"] == "GET" else "HEAD"
        result["error"] = type(err).__name__
    time.sleep(DELAY)
    return result


def median(values: list[int]) -> int | None:
    return int(statistics.median(values)) if values else None


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["status"] == 200]
    images = [r for r in ok if (r["content_type"] or "").startswith("image/")]
    sizes = sorted(r["size"] for r in ok if r["size"])
    return {
        "n": n,
        "share_200": round(len(ok) / n, 3) if n else None,
        "share_image_of_all": round(len(images) / n, 3) if n else None,
        "status_counts": dict(Counter(str(r["status"]) if r["status"] else r["error"] for r in rows)),
        "content_types_200": dict(Counter(r["content_type"] or "none" for r in ok).most_common()),
        "size_bytes_median": median(sizes),
        "size_bytes_p90": sizes[int(0.9 * (len(sizes) - 1))] if sizes else None,
        "n_with_size": len(sizes),
        "final_hosts": dict(Counter(r.get("final_host") for r in rows if r.get("final_host")).most_common(5)),
    }


def check(path: str, n: int) -> None:
    df = pd.read_parquet(path, columns=["service_name", "media_url"])
    df = df[df["media_url"].notna() & (df["media_url"] != "")]
    sample = df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"{len(df):,} rows with media_url; checking {len(sample)}")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as pool:
        results = list(pool.map(lambda u: probe(session, u), sample["media_url"]))

    # Plain dicts (not DataFrame columns) so missing values stay None instead of NaN.
    rows = [
        {**r, "type": link_type(u), "service_name": sv}
        for r, u, sv in zip(results, sample["media_url"], sample["service_name"])
    ]

    by_type = defaultdict(list)
    by_service = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
        if r["type"] == "cloudinary":
            by_service[r["service_name"] or "unknown"].append(r)

    summary = {
        "source": Path(path).name,
        "seed": SEED,
        "n_checked": len(rows),
        "by_link_type": {t: summarise(v) for t, v in sorted(by_type.items())},
        "cloudinary_by_service_name": {
            s: {k: v for k, v in summarise(r).items() if k in ("n", "share_200", "share_image_of_all", "size_bytes_median")}
            for s, r in sorted(by_service.items(), key=lambda kv: -len(kv[1]))
        },
        "head_fallback_to_get": sum(r["method"] == "GET" for r in rows),
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'type':<11}{'n':>5}{'200':>7}{'image/*':>9}{'median KB':>11}  content types / statuses")
    for t, s in summary["by_link_type"].items():
        kb = f"{s['size_bytes_median'] / 1024:.0f}" if s["size_bytes_median"] else "-"
        print(
            f"{t:<11}{s['n']:>5}{s['share_200']:>7.0%}{s['share_image_of_all']:>9.0%}{kb:>11}  "
            f"{s['content_types_200']} {s['status_counts']}"
        )
    print("\ncloudinary by service_name:")
    for s, v in summary["cloudinary_by_service_name"].items():
        print(f"  {s[:40]:<42}{v['n']:>4}  200: {v['share_200']:.0%}  image: {v['share_image_of_all']:.0%}")
    print(f"\nwrote {OUT_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="probe a random sample of media_url links")
    p.add_argument("parquet", help="Parquet file with a media_url column")
    p.add_argument("--n", type=int, default=300, help="sample size (default 300)")
    args = parser.parse_args()
    if args.command == "check":
        check(args.parquet, args.n)


if __name__ == "__main__":
    main()
