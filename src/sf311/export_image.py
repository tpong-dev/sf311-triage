"""Export the fine-tuned image model to the format the app loads (TorchScript + labels + config).

    python -m sf311.export_image            # models/image/mobilenet_v3_small.pt -> models/image/model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

from sf311.train_image import CLASSES, IMAGENET_MEAN, IMAGENET_STD

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "image"


def export(arch: str = "mobilenet_v3_small", size: int = 224) -> Path:
    model = getattr(tvm, arch)(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_DIR / f"{arch}.pt", map_location="cpu"))
    model.eval()

    out = MODEL_DIR / "model.pt"
    traced = torch.jit.trace(model, torch.zeros(1, 3, size, size))
    traced.save(str(out))
    (MODEL_DIR / "labels.json").write_text(json.dumps(CLASSES, indent=1), encoding="utf-8")
    config = {"name": f"{arch}-finetuned", "image_size": size, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), labels.json, config.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="mobilenet_v3_small")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    export(args.arch, args.size)
