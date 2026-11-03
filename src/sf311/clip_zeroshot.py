"""Zero-shot CLIP baseline for photo routing (no training).

Usage:
    python -m sf311.clip_zeroshot embed            # image embeddings for cached images
    python -m sf311.clip_zeroshot eval             # zero-shot on test, writes report + figure

Reads data/images/manifest.parquet (from `python -m sf311.images download`).
Embeddings: data/embeddings/clip_vitb32_{split}.npy (float32, L2-normalised, one row per
image) with matching data/embeddings/clip_vitb32_{split}_ids.parquet
(service_request_id, label, service_subtype) in the same row order. Other models (linear
probe, fine-tuning) can reuse them.

Zero-shot: each label gets several hand-written prompts describing what such a photo
shows; prompt embeddings are averaged per class and each image goes to the most similar
class. 'General Request', 'RPD General' and 'Other' have no consistent visual content,
so their prompts are generic and weak by design.

Privacy: reports and figures contain aggregate numbers only, never images or URLs.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from sf311.dataset import LABELS, OTHER, load, photo_rows, split

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "images" / "manifest.parquet"
EMB_DIR = ROOT / "data" / "embeddings"
REPORT = ROOT / "reports" / "clip_zeroshot.md"
FIG_DIR = ROOT / "reports" / "figures"

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
TAG = "clip_vitb32"
BATCH = 64
CLASSES = LABELS + [OTHER]

PROMPTS: dict[str, list[str]] = {
    "Street and Sidewalk Cleaning": [
        "a photo of trash and litter on a city sidewalk",
        "a photo of garbage bags and debris dumped on the street",
        "a photo of an illegally dumped mattress and furniture on the sidewalk",
        "a photo of a dirty sidewalk with spilled waste",
        "a photo of a pile of junk left on the curb",
    ],
    "Parking Enforcement": [
        "a photo of a car parked illegally",
        "a photo of a vehicle blocking a driveway",
        "a photo of a car parked on the sidewalk or in a red zone",
        "a photo of a parked car on a city street",
        "a photo of a vehicle's licence plate",
    ],
    "General Request": [
        "a photo of a city street",
        "a photo of a building in San Francisco",
        "a photo of a public space",
        "a screenshot or document",
    ],
    "Graffiti Public": [
        "a photo of graffiti on a public pole, sign or utility box",
        "a photo of spray paint tags on a city trash can or bus shelter",
        "a photo of graffiti on a public sidewalk or street fixture",
        "a photo of stickers and tags on a street sign",
    ],
    "Graffiti Private": [
        "a photo of graffiti on the wall of a building",
        "a photo of spray paint tags on a storefront or garage door",
        "a photo of graffiti on a private house or fence",
        "a photo of a tagged building facade",
    ],
    "Blocked Street and Sidewalk": [
        "a photo of an object blocking the sidewalk",
        "a photo of construction materials or barriers blocking the street",
        "a photo of a blocked pedestrian path",
        "a photo of scooters or bins obstructing the sidewalk",
    ],
    "Encampment": [
        "a photo of a homeless encampment with tents on the sidewalk",
        "a photo of a tent on a city street",
        "a photo of tarps, shopping carts and belongings on the sidewalk",
        "a photo of people's belongings piled along a wall",
    ],
    "Sidewalk and Curb": [
        "a photo of a cracked and broken sidewalk",
        "a photo of a damaged curb",
        "a photo of an uneven sidewalk that is a tripping hazard",
        "a photo of a hole in the concrete sidewalk",
    ],
    "Tree Maintenance": [
        "a photo of a fallen tree branch",
        "a photo of a street tree that needs trimming",
        "a photo of an overgrown tree blocking the sidewalk",
        "a photo of a damaged or dead tree on a city street",
    ],
    "RPD General": [
        "a photo of a city park",
        "a photo of a playground",
        "a photo of a park bench, lawn or recreation facility",
        "a photo of grass and plants in a public park",
    ],
    "Litter Receptacle Maintenance": [
        "a photo of an overflowing public trash can",
        "a photo of a full city litter bin on the sidewalk",
        "a photo of a damaged public garbage can",
        "a photo of a street trash receptacle surrounded by litter",
    ],
    "Sewer": [
        "a photo of a clogged storm drain",
        "a photo of a sewer manhole cover",
        "a photo of flooding water in the street",
        "a photo of a blocked catch basin with leaves",
    ],
    OTHER: [
        "a photo of a street light or traffic signal",
        "a photo of a pothole in the road",
        "a photo of a street sign",
        "a photo of something in the city",
    ],
}

# Coarser classes that a photo can plausibly separate. Vague labels pooled together.
COARSE = {
    "Street and Sidewalk Cleaning": "Trash / litter",
    "Litter Receptacle Maintenance": "Trash / litter",
    "Parking Enforcement": "Vehicle / parking",
    "Graffiti Public": "Graffiti",
    "Graffiti Private": "Graffiti",
    "Blocked Street and Sidewalk": "Sidewalk (blocked / damaged)",
    "Sidewalk and Curb": "Sidewalk (blocked / damaged)",
    "Encampment": "Encampment",
    "Tree Maintenance": "Tree",
    "Sewer": "Sewer",
    "General Request": "Vague (General / RPD / Other)",
    "RPD General": "Vague (General / RPD / Other)",
    OTHER: "Vague (General / RPD / Other)",
}


def load_model():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(MODEL)


def embed(splits: list[str]) -> None:
    man = pd.read_parquet(MANIFEST)
    man = man[man["ok"]]
    model, preprocess, _ = load_model()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    timing = {}
    for s in splits:
        rows = man[man["split"] == s].sort_values("service_request_id").reset_index(drop=True)
        rows = rows[[(ROOT / p).exists() for p in rows["path"]]].reset_index(drop=True)
        if rows.empty:
            print(f"{s}: no images")
            continue
        out, t0 = [], time.time()
        with torch.no_grad():
            for i in range(0, len(rows), BATCH):
                paths = rows["path"].iloc[i : i + BATCH]
                x = torch.stack([preprocess(Image.open(ROOT / p).convert("RGB")) for p in paths])
                f = model.encode_image(x)
                out.append(torch.nn.functional.normalize(f, dim=-1).numpy().astype(np.float32))
                if (i // BATCH) % 20 == 0:
                    print(f"  {s} {i + len(paths):,}/{len(rows):,}", flush=True)
        secs = time.time() - t0
        np.save(EMB_DIR / f"{TAG}_{s}.npy", np.concatenate(out))
        rows[["service_request_id", "label", "service_subtype"]].to_parquet(
            EMB_DIR / f"{TAG}_{s}_ids.parquet", index=False
        )
        timing[s] = {"n": len(rows), "seconds": round(secs, 1), "ms_per_image": round(1000 * secs / len(rows), 1)}
        print(f"{s}: {len(rows):,} images in {secs:.0f}s ({timing[s]['ms_per_image']} ms/img)")
    timing_file = EMB_DIR / f"{TAG}_timing.json"
    old = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    timing_file.write_text(json.dumps(old | timing, indent=2))


def text_weights(model, tokenizer) -> np.ndarray:
    ws = []
    with torch.no_grad():
        for c in CLASSES:
            t = torch.nn.functional.normalize(model.encode_text(tokenizer(PROMPTS[c])), dim=-1).mean(0)
            ws.append(torch.nn.functional.normalize(t, dim=0).numpy())
    return np.stack(ws)


def natural_shares() -> pd.Series:
    """Label frequencies in the full photo test split (the sample is balanced; reality is not)."""
    test = split(photo_rows(load()))[2]
    return test["label"].value_counts(normalize=True).reindex(CLASSES).fillna(0)


def plot_confusion(cm: np.ndarray, names: list[str], title: str, path: Path) -> None:
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3.5, 0.55 * n + 2.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 0.05:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.5 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of true class (row-normalised)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metrics(y: np.ndarray, p: np.ndarray, names: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=names, average="macro", zero_division=0),
        "recall": dict(zip(names, recall_score(y, p, labels=names, average=None, zero_division=0))),
    }


def evaluate() -> None:
    X = np.load(EMB_DIR / f"{TAG}_test.npy")
    ids = pd.read_parquet(EMB_DIR / f"{TAG}_test_ids.parquet")
    y = ids["label"].to_numpy()

    model, _, tokenizer = load_model()
    t0 = time.time()
    W = text_weights(model, tokenizer)
    text_secs = time.time() - t0
    pred = np.array(CLASSES)[(X @ W.T).argmax(1)]

    fine = metrics(y, pred, CLASSES)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    plot_confusion(cm, CLASSES, f"CLIP {MODEL} zero-shot, test (n={len(y):,})", FIG_DIR / "clip_confusion.png")

    coarse_names = list(dict.fromkeys(COARSE.values()))
    yc, pc = np.vectorize(COARSE.get)(y), np.vectorize(COARSE.get)(pred)
    coarse = metrics(yc, pc, coarse_names)
    plot_confusion(confusion_matrix(yc, pc, labels=coarse_names), coarse_names,
                   f"CLIP {MODEL} zero-shot, coarse groups (n={len(y):,})", FIG_DIR / "clip_confusion_coarse.png")

    # Re-weight per-class recall by real class shares to estimate accuracy on unbalanced traffic.
    shares = natural_shares()
    nat_acc = float(sum(shares[c] * fine["recall"][c] for c in CLASSES))
    coarse_share = shares.groupby(shares.index.map(COARSE)).sum()
    nat_acc_coarse = float(sum(coarse_share[g] * coarse["recall"][g] for g in coarse_names))
    support = pd.Series(y).value_counts()
    pred_share = pd.Series(pred).value_counts(normalize=True)

    timing = json.loads((EMB_DIR / f"{TAG}_timing.json").read_text())
    tot_n = sum(v["n"] for v in timing.values())
    ms = sum(v["seconds"] for v in timing.values()) * 1000 / tot_n

    # Where each true class goes most often, other than itself.
    conf = {}
    for i, c in enumerate(CLASSES):
        row = pd.Series(cm[i], index=CLASSES).drop(c)
        conf[c] = (row.idxmax(), row.max() / max(cm[i].sum(), 1))

    write_report(fine, coarse, coarse_names, support, pred_share, shares, nat_acc, nat_acc_coarse,
                 ms, tot_n, timing, text_secs, len(y), conf)
    print(f"accuracy {fine['accuracy']:.3f}  macro-F1 {fine['macro_f1']:.3f}  "
          f"coarse acc {coarse['accuracy']:.3f}  natural-mix acc {nat_acc:.3f}")
    print(f"wrote {REPORT}")


def write_report(fine, coarse, coarse_names, support, pred_share, shares, nat_acc, nat_acc_coarse,
                 ms, tot_n, timing, text_secs, n_test, conf) -> None:
    man = pd.read_parquet(MANIFEST)
    dl = man.groupby("split")["ok"].agg(["sum", "count"]).reindex(["train", "val", "test"])
    majority = shares.max()
    L = []
    L.append("# CLIP zero-shot photo routing\n")
    L.append(f"Model: open_clip `{MODEL}` (`{PRETRAINED}` weights), CPU only, no training. "
             f"Each of the {len(CLASSES)} classes (12 `dataset.LABELS` + `Other`) gets 4-5 hand-written prompts "
             "describing what such a photo shows; prompt embeddings are averaged per class and each photo "
             "is assigned to the most similar class. Prompts are in `src/sf311/clip_zeroshot.py`. They were written "
             "before looking at results; one round of tweaks checked on **val** (more specific 'Blocked' prompts, "
             "extra 'Cleaning' prompts) lowered val accuracy from 0.469 to 0.448, so the originals were kept. "
             "Test was not used for any choice.\n")
    L.append("## Data\n")
    L.append("Label-balanced, fixed-seed sample of the time-based splits from `sf311.dataset` "
             "(cap per label: train 1,500, val 300, test 500; all rows when a label has fewer). "
             "Photos downloaded as 256px thumbnails via Cloudinary resizing (`python -m sf311.images download`).\n")
    L.append("| split | selected | downloaded OK |\n|---|---:|---:|")
    for s, r in dl.iterrows():
        L.append(f"| {s} | {int(r['count']):,} | {int(r['sum']):,} ({r['sum'] / r['count']:.1%}) |")
    L.append(f"\nZero-shot is evaluated on the **test** sample only (n = {n_test:,}, Sept-Oct 2026). "
             "Because the sample is balanced, plain accuracy is close to macro recall. The real test month mix "
             f"is dominated by Street and Sidewalk Cleaning ({shares['Street and Sidewalk Cleaning']:.0%}) and "
             f"Parking Enforcement ({shares['Parking Enforcement']:.0%}), so the table also gives a "
             "re-weighted estimate (per-class recall x real class share).\n")
    L.append("## Results\n")
    L.append("| view | accuracy (balanced test) | macro-F1 | est. accuracy on real mix |\n|---|---:|---:|---:|")
    L.append(f"| 13 classes | {fine['accuracy']:.3f} | {fine['macro_f1']:.3f} | {nat_acc:.3f} |")
    L.append(f"| {len(coarse_names)} coarse groups | {coarse['accuracy']:.3f} | {coarse['macro_f1']:.3f} | {nat_acc_coarse:.3f} |")
    L.append(f"\nReference points: random guessing on the balanced sample = {1 / len(CLASSES):.3f}; "
             f"always predicting the majority class on the real mix = {majority:.3f}.\n")
    L.append("### Per-class recall (13 classes)\n")
    L.append("| class | test n | recall | share of predictions | real share |\n|---|---:|---:|---:|---:|")
    for c in sorted(CLASSES, key=lambda c: -fine["recall"][c]):
        L.append(f"| {c} | {int(support.get(c, 0))} | {fine['recall'][c]:.2f} | "
                 f"{pred_share.get(c, 0):.1%} | {shares[c]:.1%} |")
    L.append("\n![confusion matrix](figures/clip_confusion.png)\n")
    L.append("### Coarse, visually distinguishable groups\n")
    L.append("Merges classes a photo cannot reliably separate: the two graffiti classes (public vs private "
             "depends on who owns the wall), street cleaning + litter bins, blocked + damaged sidewalk, and "
             "the vague classes (General Request, RPD General, Other).\n")
    L.append("| group | recall | real share |\n|---|---:|---:|")
    cs = shares.groupby(shares.index.map(COARSE)).sum()
    for g in coarse_names:
        L.append(f"| {g} | {coarse['recall'][g]:.2f} | {cs[g]:.1%} |")
    L.append("\n![coarse confusion matrix](figures/clip_confusion_coarse.png)\n")
    L.append("## Speed and cost\n")
    L.append(f"- Image embedding on CPU (batch {BATCH}, includes JPEG decode + preprocessing): "
             f"**{ms:.0f} ms per image** averaged over {tot_n:,} images "
             + ", ".join(f"{s}: {v['ms_per_image']} ms" for s, v in timing.items()) + ".")
    L.append(f"- Text side is computed once ({text_secs:.1f} s for all prompts); classification is one matrix product.")
    L.append(f"- Marginal cost is about **$0 per 1,000 requests** on hardware already owned "
             f"(~{ms:.0f} s of laptop CPU per 1,000 photos). No API calls; photos never leave the machine.\n")
    L.append("## Takeaways\n")
    r = fine["recall"]
    sc = "Street and Sidewalk Cleaning"
    L.append(f"- **Zero-shot barely beats the majority baseline on real traffic** ({nat_acc:.3f} vs {majority:.3f}). "
             f"The biggest class, {sc} ({shares[sc]:.0%} of real photo requests), has recall {r[sc]:.2f}: "
             f"{conf[sc][1]:.0%} of those photos go to {conf[sc][0]}. Dumped items on a sidewalk look like an "
             "obstruction, so prompts alone cannot draw the line the city draws.")
    L.append(f"- Classes with a distinct visual subject work: Parking Enforcement {r['Parking Enforcement']:.2f}, "
             f"Graffiti Private {r['Graffiti Private']:.2f}, Tree Maintenance {r['Tree Maintenance']:.2f}, "
             f"Encampment {r['Encampment']:.2f}.")
    L.append(f"- Sewer recall is {r['Sewer']:.2f}; {conf['Sewer'][1]:.0%} go to {conf['Sewer'][0]} "
             "(drains and manholes sit in the pavement).")
    L.append(f"- The vague classes are close to random ({1 / len(CLASSES):.3f}) (General Request {r['General Request']:.2f}, "
             f"RPD General {r['RPD General']:.2f}, Other {r['Other']:.2f}): the label describes the process, "
             f"not the picture. General Request photos mostly go to {conf['General Request'][0]} "
             f"({conf['General Request'][1]:.0%}).")
    L.append(f"- Merging look-alike classes lifts balanced accuracy from {fine['accuracy']:.3f} to "
             f"{coarse['accuracy']:.3f} and the real-mix estimate from {nat_acc:.3f} to {nat_acc_coarse:.3f}, "
             "so much of the error is between neighbouring classes, not random.")
    L.append("- Next step: a linear probe or fine-tune on the saved embeddings / images should learn the "
             "cleaning-vs-blocked boundary that prompts miss.")
    L.append("- Embeddings for train/val/test are saved in `data/embeddings/` for a cheap linear probe and "
             "for comparison with the fine-tuned model.\n")
    L.append("_Privacy: this report holds aggregate numbers only; no photos or photo URLs._\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("embed", help="compute image embeddings for cached images")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    sub.add_parser("eval", help="zero-shot evaluation on test")
    args = parser.parse_args()
    if args.command == "embed":
        embed(args.splits)
    else:
        evaluate()


if __name__ == "__main__":
    main()
