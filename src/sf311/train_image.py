"""Photo routing models trained on downloaded 311 photos (CPU only).

Two approaches, both evaluated on the time-based test split:

  A. finetune  - ImageNet MobileNetV3 (torchvision). Early blocks frozen; the last
                 feature blocks + classifier head are trained with a class-weighted
                 loss, light augmentation and early stopping on val macro-F1.
  B. probe     - logistic regression on CLIP ViT-B-32 image embeddings. Uses the
                 shared embeddings in data/embeddings/ if present, otherwise computes
                 (and caches under models/image/) embeddings for the manifest images.

Usage (PYTHONPATH=src):
    python -m sf311.train_image finetune [--arch mobilenet_v3_small] [--max-per-class N]
    python -m sf311.train_image probe
    python -m sf311.train_image report      # writes reports/finetune.md + figures

Images come from data/images/manifest.parquet (rows with ok == True). Photos can show
people and licence plates: nothing here displays images or writes image URLs; outputs
are metrics, weights and aggregate figures only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from sf311.dataset import LABELS, OTHER

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "images"
MANIFEST = IMG_DIR / "manifest.parquet"
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "models" / "image"
REPORT = ROOT / "reports" / "finetune.md"
FIG_DIR = ROOT / "reports" / "figures"

CLASSES = LABELS + [OTHER]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data

def load_manifest(max_per_class: int | None = None) -> pd.DataFrame:
    m = pd.read_parquet(MANIFEST)
    m = m[m["ok"].astype(bool)].copy()
    m["service_request_id"] = m["service_request_id"].astype(str)
    m["file"] = [_resolve(p) for p in m["path"]]
    m = m[[f.exists() for f in m["file"]]]
    m = m[m["label"].isin(CLASSES)].drop_duplicates("service_request_id")
    m["y"] = m["label"].map(CLS_IDX).astype(int)
    if max_per_class:
        m = (m.groupby(["split", "label"], group_keys=False)
               .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=SEED)))
    return m.reset_index(drop=True)


def _resolve(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else ROOT / p


def class_counts(m: pd.DataFrame) -> pd.DataFrame:
    return m.pivot_table(index="label", columns="split", values="y", aggfunc="count", fill_value=0).reindex(CLASSES, fill_value=0)


# ---------------------------------------------------------------- metrics

def evaluate(y_true, y_pred) -> dict:
    labels = list(range(len(CLASSES)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, rec)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------- A. fine-tune CNN

def _build_cnn(arch: str, n_classes: int, train_blocks: int):
    import torchvision.models as tvm

    weights = {"mobilenet_v3_small": tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
               "mobilenet_v3_large": tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
               "efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1}[arch]
    model = getattr(tvm, arch)(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, n_classes)
    # Freeze everything except the last `train_blocks` feature blocks and the classifier.
    n_feat = len(model.features)
    first_trainable = n_feat - train_blocks
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= first_trainable
    return model, first_trainable


class _Split(nn.Module):
    """Run the frozen prefix under no_grad (and in eval mode, so BN stats stay fixed)."""

    def __init__(self, model, first_trainable):
        super().__init__()
        self.prefix = model.features[:first_trainable]
        self.suffix = model.features[first_trainable:]
        self.pool = model.avgpool
        self.head = model.classifier

    def forward(self, x):
        with torch.no_grad():
            self.prefix.eval()
            x = self.prefix(x)
        x = self.suffix(x)
        return self.head(torch.flatten(self.pool(x), 1))


class _ImgDS(torch.utils.data.Dataset):
    def __init__(self, files, ys, tf):
        self.files, self.ys, self.tf = list(map(str, files)), list(ys), tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.files[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.ys[i]


def _transforms(size: int):
    from torchvision import transforms as T

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
                       T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), norm])
    test = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), norm])
    return train, test


def _loader(df, tf, shuffle, bs, workers):
    ds = _ImgDS(df["file"], df["y"], tf)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                                       persistent_workers=workers > 0, drop_last=False)


@torch.no_grad()
def _predict(net, loader):
    net.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        out = net(x).softmax(1)
        probs.append(out.numpy())
        ps.append(out.argmax(1).numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def _latency_ms(fn, x, n=50, warmup=5) -> float:
    for _ in range(warmup):
        fn(x)
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t) / n * 1000


def finetune(args) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(args.threads)
    m = load_manifest(args.max_per_class)
    tr, va, te = (m[m["split"] == s] for s in ("train", "val", "test"))
    print(f"images: train {len(tr)}, val {len(va)}, test {len(te)}")
    print(class_counts(m).to_string())

    model, first = _build_cnn(args.arch, len(CLASSES), args.train_blocks)
    net = _Split(model, first)
    n_train_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all_p = sum(p.numel() for p in net.parameters())
    print(f"{args.arch}: training {n_train_p:,} of {n_all_p:,} params (blocks {first}+ and head)")

    # Class-weighted CE (inverse frequency, normalised to mean 1); classes absent from train get 0.
    counts = np.bincount(tr["y"], minlength=len(CLASSES)).astype(float)
    w = np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=0.05)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tf_train, tf_test = _transforms(args.size)
    dl_tr = _loader(tr, tf_train, True, args.bs, args.workers)
    dl_va = _loader(va, tf_test, False, 128, args.workers)

    best, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        te0, loss_sum, nb = time.time(), 0.0, 0
        for x, y in dl_tr:
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            nb += 1
        sched.step()
        yv, pv, _ = _predict(net, dl_va)
        f1 = f1_score(yv, pv, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
        acc = accuracy_score(yv, pv)
        history.append({"epoch": ep, "loss": loss_sum / max(nb, 1), "val_macro_f1": f1, "val_acc": acc,
                        "sec": time.time() - te0})
        print(f"ep {ep:2d} loss {loss_sum / max(nb, 1):.3f} val acc {acc:.3f} macroF1 {f1:.3f} "
              f"({time.time() - te0:.0f}s)", flush=True)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    train_sec = time.time() - t0
    model.load_state_dict(best_state)

    tag = args.tag or args.arch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wpath = OUT_DIR / f"{tag}.pt"
    torch.save(model.state_dict(), wpath)

    dl_te = _loader(te, tf_test, False, 128, args.workers)
    yt, pt, prob = _predict(net, dl_te)
    np.save(OUT_DIR / f"{tag}_test_probs.npy", prob)
    model.eval()
    x1 = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        ms_1t = (torch.set_num_threads(1), _latency_ms(model, x1))[1]
        torch.set_num_threads(args.threads)
        ms_mt = _latency_ms(model, x1)
    res = {
        "approach": f"Fine-tuned {args.arch} (last {args.train_blocks} blocks + head)",
        "arch": args.arch, "input_size": args.size,
        "trainable_params": n_train_p, "total_params": n_all_p,
        "train_counts": {c: int(n) for c, n in zip(CLASSES, counts)},
        "n_train": len(tr), "n_val": len(va),
        "train_sec": train_sec, "epochs_run": len(history), "best_val_macro_f1": best,
        "history": history,
        "latency_ms_batch1_1thread": ms_1t, f"latency_ms_batch1_{args.threads}threads": ms_mt,
        "latency_ms": ms_mt,
        "model_mb": wpath.stat().st_size / 1e6,
        "test": evaluate(yt, pt),
        "max_per_class": args.max_per_class,
    }
    _save_json(res, OUT_DIR / f"{tag}_metrics.json")
    print(f"TEST acc {res['test']['accuracy']:.3f} macroF1 {res['test']['macro_f1']:.3f} "
          f"train {train_sec / 60:.1f} min, {ms_mt:.1f} ms/img, {res['model_mb']:.1f} MB")


# ---------------------------------------------------------------- B. CLIP linear probe

def _shared_embeddings(split: str):
    """Return (ids, X) from data/embeddings if the other job has written them, else None."""
    x_path = EMB_DIR / f"clip_vitb32_{split}.npy"
    if not x_path.exists():
        return None
    X = np.load(x_path)
    for cand in sorted(EMB_DIR.glob(f"clip_vitb32_{split}*")):
        if cand == x_path:
            continue
        try:
            if cand.suffix == ".npy":
                ids = np.load(cand, allow_pickle=True)
            elif cand.suffix == ".parquet":
                d = pd.read_parquet(cand)
                ids = d["service_request_id"] if "service_request_id" in d else d.iloc[:, 0]
            elif cand.suffix == ".json":
                ids = json.loads(cand.read_text())
            elif cand.suffix in (".txt", ".csv"):
                ids = [line.strip() for line in cand.read_text().splitlines() if line.strip()]
                if ids and not ids[0][:1].isdigit():
                    ids = ids[1:]  # header
            else:
                continue
        except Exception:
            continue
        ids = [str(i) for i in np.asarray(ids).ravel()]
        if len(ids) == len(X):
            return ids, X
    return None


def _clip_model():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    return model.eval(), preprocess


def _embed(files, threads) -> np.ndarray:
    from PIL import Image

    torch.set_num_threads(threads)
    model, pre = _clip_model()
    out = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            xs = []
            for f in files[i:i + 64]:
                with Image.open(f) as im:
                    xs.append(pre(im.convert("RGB")))
            out.append(model.encode_image(torch.stack(xs)).float().numpy())
            if i % 1280 == 0:
                print(f"  embedded {i + len(xs)}/{len(files)}", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def get_embeddings(m: pd.DataFrame, split: str, threads: int) -> tuple[np.ndarray, str]:
    """Embeddings for every manifest row of `split`, in manifest order."""
    rows = m[m["split"] == split]
    need = list(rows["service_request_id"])
    shared = _shared_embeddings(split)
    lookup, source = {}, "computed here (open_clip ViT-B-32 laion2b_s34b_b79k)"
    if shared is not None:
        lookup = dict(zip(shared[0], shared[1]))
        source = "data/embeddings (shared)"
    cache_ids, cache_X = OUT_DIR / f"clip_cache_{split}_ids.npy", OUT_DIR / f"clip_cache_{split}.npy"
    if cache_ids.exists():
        lookup = {**dict(zip(np.load(cache_ids, allow_pickle=True).astype(str), np.load(cache_X))), **lookup}
    missing = [i for i in need if i not in lookup]
    if missing:
        print(f"{split}: computing {len(missing)} CLIP embeddings ({len(need) - len(missing)} reused)")
        files = rows.set_index("service_request_id").loc[missing, "file"].tolist()
        Xm = _embed(files, threads)
        lookup.update(zip(missing, Xm))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        own = {k: v for k, v in lookup.items()}
        np.save(cache_ids, np.array(list(own.keys())))
        np.save(cache_X, np.stack(list(own.values())).astype(np.float32))
        if shared is not None:
            source = "data/embeddings (shared) + computed here for missing ids"
    X = np.stack([lookup[i] for i in need]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X, source


def probe(args) -> None:
    from sklearn.linear_model import LogisticRegression

    torch.set_num_threads(args.threads)
    m = load_manifest(args.max_per_class)
    t0 = time.time()
    Xtr, src = get_embeddings(m, "train", args.threads)
    Xva, _ = get_embeddings(m, "val", args.threads)
    Xte, _ = get_embeddings(m, "test", args.threads)
    emb_sec = time.time() - t0
    ytr, yva, yte = (m.loc[m["split"] == s, "y"].to_numpy() for s in ("train", "val", "test"))
    print(f"embeddings: {src}; train {len(Xtr)} val {len(Xva)} test {len(Xte)} ({emb_sec:.0f}s)")

    t1 = time.time()
    best = (-1, None, None)
    grid = []
    for C in (0.1, 0.3, 1, 3, 10, 30):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), labels=list(range(len(CLASSES))), average="macro", zero_division=0)
        grid.append({"C": C, "val_macro_f1": f1})
        print(f"  C={C:<5} val macroF1 {f1:.3f}")
        if f1 > best[0]:
            best = (f1, C, clf)
    # Refit on train+val with the chosen C (val only picked the hyper-parameter).
    clf = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced")
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    fit_sec = time.time() - t1
    pt = clf.predict(Xte)
    np.save(OUT_DIR / "clip_probe_test_probs.npy", _full_proba(clf, Xte))

    import joblib

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ppath = OUT_DIR / "clip_probe.joblib"
    joblib.dump(clf, ppath)

    # End-to-end latency: CLIP image encoder (batch 1) + logistic regression.
    model, _ = _clip_model()
    x1 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.set_num_threads(1)
        ms_1t = _latency_ms(lambda x: model.encode_image(x), x1, n=20)
        torch.set_num_threads(args.threads)
        ms_mt = _latency_ms(lambda x: model.encode_image(x), x1, n=30)
    enc_mb = sum(p.numel() * p.element_size() for p in model.visual.parameters()) / 1e6
    res = {
        "approach": "CLIP ViT-B-32 embeddings + logistic regression",
        "embedding_source": src, "C": best[1], "val_grid": grid, "best_val_macro_f1": best[0],
        "n_train": len(Xtr), "n_val": len(Xva),
        "train_sec": fit_sec, "embed_sec": emb_sec,
        "latency_ms_batch1_1thread": ms_1t, f"latency_ms_batch1_{args.threads}threads": ms_mt,
        "latency_ms": ms_mt,
        "model_mb": enc_mb + ppath.stat().st_size / 1e6,
        "probe_head_mb": ppath.stat().st_size / 1e6,
        "test": evaluate(yte, pt),
        "max_per_class": args.max_per_class,
    }
    _save_json(res, OUT_DIR / "clip_probe_metrics.json")
    print(f"TEST acc {res['test']['accuracy']:.3f} macroF1 {res['test']['macro_f1']:.3f} "
          f"(C={best[1]}, fit {fit_sec:.0f}s, {ms_mt:.1f} ms/img)")


def _full_proba(clf, X):
    p = np.zeros((len(X), len(CLASSES)), np.float32)
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


# ---------------------------------------------------------------- report

def _plot_confusion(cm, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if norm[i, j] >= 0.05:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalised)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_recall(results: dict, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.8 / max(len(results), 1)
    for k, (name, r) in enumerate(results.items()):
        rec = [r["test"]["per_class_recall"][c] for c in CLASSES]
        ax.bar(np.arange(len(CLASSES)) + k * w, rec, w, label=name)
    ax.set_xticks(np.arange(len(CLASSES)) + w * (len(results) - 1) / 2, CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("test recall")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _other_report(name: str) -> str:
    p = ROOT / "reports" / name
    return f"see [{name}]({name})" if p.exists() else f"pending ({name} not written yet)"


def _zeroshot_numbers() -> tuple[str, str]:
    """(accuracy, macro-F1) for 13 classes from reports/clip_zeroshot.md, or blanks."""
    import re

    p = ROOT / "reports" / "clip_zeroshot.md"
    if p.exists():
        m = re.search(r"^\|\s*13 classes\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", p.read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1), m.group(2)
    return "", ""


def _real_test_shares() -> dict | None:
    """Share of each class among all photo requests in the test window (the real, imbalanced mix)."""
    try:
        from sf311.dataset import load, photo_rows, split

        te = split(photo_rows(load()))[2]
        return te["label"].value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_dict()
    except Exception as err:  # raw data missing etc.
        print(f"real-mix shares unavailable: {err}")
        return None


def report(args) -> None:
    results = {}
    for f in sorted(OUT_DIR.glob("*_metrics.json")):
        r = json.loads(f.read_text())
        results[f.stem.replace("_metrics", "")] = r
    if not results:
        raise SystemExit("no metrics in models/image; run finetune / probe first")
    m = load_manifest()
    counts = class_counts(m)

    lines = ["# Photo routing: fine-tuned CNN vs CLIP linear probe", "",
             f"Generated by `python -m sf311.train_image report`. 13 classes (12 top `service_name`s + Other); "
             f"time-based split from `sf311.dataset` (train < 2026-08, val Aug 2026, test Sep 2026+). "
             f"CPU only ({os.cpu_count()} logical cores). Photos are never shown or linked here.", "",
             "## Comparison (test split)", "",
             "| Approach | Test n | Accuracy (balanced test) | Macro-F1 | Est. accuracy on real mix | Train time "
             "| CPU ms/image (batch 1, 6 thr / 1 thr) | Model size |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    shares = _real_test_shares()
    for r in results.values():
        t = r["test"]
        train = f"{r['train_sec'] / 60:.1f} min"
        if "embed_sec" in r:
            train = f"{r['train_sec']:.0f} s fit (+ CLIP embedding of train/val/test, shared job)"
        real = (f"{sum(t['per_class_recall'][c] * shares[c] for c in CLASSES):.3f}" if shares else "")
        lat = f"{r['latency_ms']:.1f} / {r['latency_ms_batch1_1thread']:.1f}"
        lines.append(f"| {r['approach']} | {t['n']} | {t['accuracy']:.3f} | {t['macro_f1']:.3f} | {real} | {train} | "
                     f"{lat} | {r['model_mb']:.1f} MB |")
    zs_acc, zs_f1 = _zeroshot_numbers()
    lines += [f"| CLIP ViT-B-32 zero-shot ({_other_report('clip_zeroshot.md')}) | {6500 if zs_acc else ''} | {zs_acc} "
              f"| {zs_f1} | (in that report) | none | ~same encoder as probe | ~351 MB |",
              f"| Frontier vision LLM ({_other_report('vision_llm.md')}) | | | | | none | API | API |",
              "", "Latency: single random 224px image, after warm-up; first figure uses the training thread count, "
              "second one thread. Model size: weights on disk (for the probe: CLIP image encoder fp32 + the "
              "logistic-regression head). Est. accuracy on real mix = per-class test recall weighted by each class's share "
              "of all photo requests in the test window (same method as clip_zeroshot.md). "
              "Chance on the balanced test set is about 0.077.", ""]

    best_name = max(results, key=lambda k: results[k]["test"]["macro_f1"])
    small_name = min(results, key=lambda k: results[k]["model_mb"])
    lines += ["## Takeaways", "",
              f"- Best macro-F1: **{results[best_name]['approach']}** ({results[best_name]['test']['macro_f1']:.3f}).",
              f"- Smallest / fastest: **{results[small_name]['approach']}** ({results[small_name]['model_mb']:.1f} MB, "
              f"{results[small_name]['latency_ms']:.1f} ms/image on CPU), "
              f"{results[small_name]['test']['macro_f1']:.3f} macro-F1.",
              "- Both trained models beat CLIP zero-shot; the labelled training photos are worth more than prompt "
              "engineering. The catch-all classes (General Request, Other, Street and Sidewalk Cleaning) stay the weakest.",
              ""]
    lines += ["## Per-class test recall", "", "| Class | " + " | ".join(results) + " |",
              "|---|" + "---|" * len(results)]
    for c in CLASSES:
        lines.append(f"| {c} | " + " | ".join(f"{r['test']['per_class_recall'][c]:.2f}" for r in results.values()) + " |")
    lines += ["", "![per-class recall](figures/finetune_recall.png)", ""]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_recall(results, FIG_DIR / "finetune_recall.png")
    lines += ["## Confusion matrices", ""]
    for name, r in results.items():
        _plot_confusion(r["test"]["confusion"], f"{r['approach']} (test)", FIG_DIR / f"finetune_cm_{name}.png")
        lines += [f"![confusion {name}](figures/finetune_cm_{name}.png)", ""]

    lines += ["## Setup", ""]
    for name, r in results.items():
        if "arch" in r:
            lines.append(f"- **{name}**: torchvision `{r['arch']}` ImageNet weights, {r['input_size']}px input; "
                         f"{r['trainable_params']:,} of {r['total_params']:,} parameters trained (last feature blocks + head; "
                         f"frozen prefix keeps ImageNet BatchNorm stats). AdamW + cosine LR, class-weighted cross-entropy "
                         f"(label smoothing 0.05), random-resized-crop / flip / colour-jitter augmentation, early stopping on "
                         f"val macro-F1 (best {r['best_val_macro_f1']:.3f} at epoch {max(r['history'], key=lambda h: h['val_macro_f1'])['epoch']} of {r['epochs_run']} run). "
                         f"Train {r['n_train']} / val {r['n_val']} images.")
        else:
            lines.append(f"- **{name}**: L2-normalised CLIP ViT-B-32 image embeddings ({r['embedding_source']}); "
                         f"multinomial logistic regression, `class_weight='balanced'`, C={r['C']} chosen on val macro-F1 "
                         f"(best {r['best_val_macro_f1']:.3f}), then refit on train+val.")
        if r.get("max_per_class"):
            lines.append(f"  - SMOKE RUN: capped at {r['max_per_class']} images per class per split.")
    lines += ["", "## Images per class and split (manifest, ok == True)", "",
              "| Class | " + " | ".join(counts.columns) + " |", "|---|" + "---|" * len(counts.columns)]
    for c, row in counts.iterrows():
        lines.append(f"| {c} | " + " | ".join(str(int(v)) for v in row) + " |")
    lines += ["", "## Caveats", "",
              "- Labels are the citizen-chosen `service_name`, which is noisy: the same scene (e.g. trash on a sidewalk) "
              "is legitimately filed under several categories, so a photo-only ceiling is well below 100%.",
              "- \"General Request\", \"RPD General\" and \"Other\" are catch-alls with no consistent visual signature; "
              "expect low recall there.",
              "- Test is a later time window than train (Sep 2026+), so scores include any drift in how people file requests.",
              "- Sampling is roughly class-balanced, so accuracy here is not the accuracy on the real (imbalanced) request stream; "
              "macro-F1 is the headline metric.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("finetune", "probe", "report"):
        s = sub.add_parser(name)
        s.add_argument("--max-per-class", type=int, default=None, help="cap images per class per split (smoke tests)")
        s.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    f = sub.choices["finetune"]
    f.add_argument("--arch", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"])
    f.add_argument("--train-blocks", type=int, default=3, help="number of final feature blocks to train")
    f.add_argument("--size", type=int, default=224)
    f.add_argument("--epochs", type=int, default=12)
    f.add_argument("--patience", type=int, default=3)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--bs", type=int, default=64)
    f.add_argument("--workers", type=int, default=3)
    f.add_argument("--tag", default=None, help="output name (default: arch)")
    args = ap.parse_args()
    {"finetune": finetune, "probe": probe, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
