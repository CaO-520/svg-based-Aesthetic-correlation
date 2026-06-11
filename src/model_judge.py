from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    r"C:\model\shunk031\aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_JSON_PATH = METADATA_DIR / "model_judge.json"
HF_MODULES_CACHE = PROJECT_ROOT / ".cache" / "huggingface_modules"

CATEGORY_ORDER = ("human_design", "degraded", "model_generated")
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

os.environ["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE)

import transformers.utils.import_utils as transformers_import_utils

transformers_import_utils._sklearn_available = False
transformers_import_utils._scipy_available = False

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def iter_png_files(category: str) -> list[Path]:
    input_dir = PROCESSED_DIR / category / "png"
    return sorted(
        input_dir.glob("*.png"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)

    values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    values = values / 255.0

    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    values = (values - mean) / std
    return values


def load_model(device: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        "modeling_v2.AestheticsPredictorV2Linear",
        MODEL_DIR,
    )
    model = model_class.from_pretrained(
        MODEL_DIR,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    model.eval()
    return model


def score_batch(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: str,
) -> list[float]:
    pixel_values = torch.stack([preprocess_image(path) for path in image_paths]).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    scores = outputs.logits.squeeze(-1).detach().cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
    return [round(float(score), 2) for score in scores]


def score_category(
    model: torch.nn.Module,
    category: str,
    device: str,
    batch_size: int,
    start_index: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    image_paths = iter_png_files(category)
    if limit is None:
        image_paths = image_paths[start_index:]
    else:
        image_paths = image_paths[start_index : start_index + limit]

    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        scores = score_batch(model, batch_paths, device)
        for image_path, score in zip(batch_paths, scores):
            results.append({"filename": image_path.name, "score": score})

        print(
            f"{category}: scored {min(batch_start + batch_size, len(image_paths))}"
            f"/{len(image_paths)}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score processed PNG files with the local aesthetic predictor."
    )
    parser.add_argument(
        "--category",
        choices=CATEGORY_ORDER,
        help="Only score one category.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images to score per forward pass. Default: 16.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index inside each selected category. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to score inside each selected category.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. Default: auto.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = load_model(device)
    categories = (args.category,) if args.category else CATEGORY_ORDER

    output: dict[str, Any] = {}
    total = 0
    for category in categories:
        category_results = score_category(
            model=model,
            category=category,
            device=device,
            batch_size=args.batch_size,
            start_index=args.start_index,
            limit=args.limit,
        )
        output[category] = category_results
        total += len(category_results)

    output["total"] = total
    output["model_path"] = str(MODEL_DIR)
    output["score_precision"] = 2

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Scored {total} images.")
    print(f"Output: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
