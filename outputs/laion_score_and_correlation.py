from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
HUMAN_PATH = ROOT / "data" / "metadata" / "human_judge.json"
OUTPUT_SCORE_PATH = ROOT / "outputs" / "laion_model_judge.json"
OUTPUT_CORRELATION_PATH = ROOT / "outputs" / "laion_correlation.json"
SHUNK_MODEL_DIR = Path(
    r"C:\model\shunk031\aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
)
LAION_MODEL_PATH = Path(r"C:\model\laion\sa_0_4_vit_l_14_linear.pth")
HF_MODULES_CACHE = ROOT / ".cache" / "huggingface_modules"

CATEGORIES = ("human_design", "degraded", "model_generated")
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
BATCH_SIZE = 32

os.environ["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE)

import transformers.utils.import_utils as transformers_import_utils

transformers_import_utils._sklearn_available = False
transformers_import_utils._scipy_available = False

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def sort_key(filename: str) -> int | str:
    stem = Path(filename).stem
    return int(stem) if stem.isdigit() else stem


def iter_png_files(category: str) -> list[Path]:
    return sorted(
        (PROCESSED_DIR / category / "png").glob("*.png"),
        key=lambda path: sort_key(path.name),
    )


def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    values = values / 255.0
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (values - mean) / std


def load_clip_backbone(device: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(SHUNK_MODEL_DIR, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        "modeling_v2.AestheticsPredictorV2Linear",
        SHUNK_MODEL_DIR,
    )
    model = model_class.from_pretrained(
        SHUNK_MODEL_DIR,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    model.eval()
    return model


def load_laion_head(device: str) -> nn.Linear:
    head = nn.Linear(768, 1)
    state_dict = torch.load(LAION_MODEL_PATH, map_location="cpu")
    head.load_state_dict(state_dict)
    head.to(device)
    head.eval()
    return head


def score_batch(
    backbone: torch.nn.Module,
    head: nn.Linear,
    image_paths: list[Path],
    device: str,
) -> list[float]:
    pixel_values = torch.stack([preprocess_image(path) for path in image_paths]).to(device)
    with torch.no_grad():
        outputs = backbone(pixel_values=pixel_values)
        image_embeds = outputs.hidden_states
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = head(image_embeds).squeeze(-1).detach().cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
    return [round(float(score), 2) for score in scores]


def score_all() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_clip_backbone(device)
    head = load_laion_head(device)

    output: dict = {}
    total = 0
    for category in CATEGORIES:
        paths = iter_png_files(category)
        rows = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start : start + BATCH_SIZE]
            scores = score_batch(backbone, head, batch_paths, device)
            rows.extend(
                {"filename": path.name, "score": score}
                for path, score in zip(batch_paths, scores)
            )
            print(f"{category}: scored {min(start + BATCH_SIZE, len(paths))}/{len(paths)}")
        output[category] = rows
        total += len(rows)

    output["total"] = total
    output["model_path"] = str(LAION_MODEL_PATH)
    output["clip_backbone_path"] = str(SHUNK_MODEL_DIR)
    output["score_precision"] = 2
    OUTPUT_SCORE_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return numerator / (denominator_x * denominator_y)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end + 2) / 2
        for rank_index in range(index, end + 1):
            output[indexed[rank_index][0]] = average_rank
        index = end + 1
    return output


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def compute_correlation(laion_scores: dict) -> dict:
    human = json.load(open(HUMAN_PATH, encoding="utf-8-sig"))
    result = {}
    all_human = []
    all_laion = []

    for category in CATEGORIES:
        human_by_name = {item["filename"]: item["score"] for item in human[category]}
        laion_by_name = {item["filename"]: item["score"] for item in laion_scores[category]}
        filenames = sorted(set(human_by_name) & set(laion_by_name), key=sort_key)
        human_values = [float(human_by_name[name]) for name in filenames]
        laion_values = [float(laion_by_name[name]) for name in filenames]
        result[category] = {
            "n": len(filenames),
            "pearson": round(pearson(human_values, laion_values), 4),
            "spearman": round(spearman(human_values, laion_values), 4),
        }
        all_human.extend(human_values)
        all_laion.extend(laion_values)

    result["overall"] = {
        "n": len(all_human),
        "pearson": round(pearson(all_human, all_laion), 4),
        "spearman": round(spearman(all_human, all_laion), 4),
    }
    OUTPUT_CORRELATION_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    laion_scores = score_all()
    correlation = compute_correlation(laion_scores)
    print(json.dumps(correlation, ensure_ascii=False, indent=2))
    print(f"Scores: {OUTPUT_SCORE_PATH}")
    print(f"Correlation: {OUTPUT_CORRELATION_PATH}")


if __name__ == "__main__":
    main()
