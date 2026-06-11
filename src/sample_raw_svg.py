from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_DATASET_PATH = r"C:\dataset\starvector\svg-stack"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CATEGORY_ORDER = ("human_design", "degraded", "model_generated")
SVG_COLUMN_CANDIDATES = (
    "svg",
    "Svg",
    "SVG",
    "svg_code",
    "svg_content",
    "content",
    "text",
    "code",
)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def as_dataset(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset

    if "train" in dataset:
        return dataset["train"]

    splits = list(dataset.values())
    if not splits:
        raise ValueError("DatasetDict is empty.")
    return concatenate_datasets(splits)


def detect_svg_column(dataset: Dataset) -> str:
    for column in SVG_COLUMN_CANDIDATES:
        if column in dataset.column_names:
            return column

    preview_count = min(20, len(dataset))
    for column in dataset.column_names:
        for row in dataset.select(range(preview_count)):
            value = row.get(column)
            if isinstance(value, str) and "<svg" in value.lower():
                return column

    raise ValueError(
        "Could not find an SVG column. Available columns: "
        + ", ".join(dataset.column_names)
    )


def get_sampling_plan(config: dict[str, Any]) -> dict[str, int]:
    sampling = config.get("sampling", {})
    plan = {
        "human_design": int(sampling.get("human_design", 300)),
        "degraded": int(sampling.get("degraded", 300)),
        "model_generated": int(sampling.get("model_generated", 300)),
    }

    total_samples = sampling.get("total_samples")
    if total_samples is not None and int(total_samples) != sum(plan.values()):
        raise ValueError(
            f"sampling.total_samples={total_samples} does not match category total "
            f"{sum(plan.values())}."
        )

    return plan


def clean_svg_text(svg: Any) -> str:
    if not isinstance(svg, str):
        raise ValueError(f"Expected SVG value to be str, got {type(svg).__name__}.")

    svg = svg.strip()
    if not svg:
        raise ValueError("Encountered empty SVG content.")
    return svg


def write_samples(dataset: Dataset, svg_column: str, plan: dict[str, int]) -> None:
    file_id = 1
    for category in CATEGORY_ORDER:
        output_dir = RAW_DIR / category / "svg"
        output_dir.mkdir(parents=True, exist_ok=True)

        count = plan[category]
        for row in dataset.select(range(file_id - 1, file_id - 1 + count)):
            svg = clean_svg_text(row[svg_column])
            output_path = output_dir / f"{file_id}.svg"
            output_path.write_text(svg, encoding="utf-8")
            file_id += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample raw SVG files into data/raw/<category>/svg."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help="Local dataset path passed to datasets.load_dataset().",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sampling = config.get("sampling", {})
    random_seed = int(sampling.get("random_seed", 42))
    plan = get_sampling_plan(config)
    sample_count = sum(plan.values())

    dataset = as_dataset(load_dataset(args.dataset))
    if len(dataset) < sample_count:
        raise ValueError(
            f"Dataset has {len(dataset)} rows, but {sample_count} samples are required."
        )

    svg_column = detect_svg_column(dataset)
    indices = list(range(len(dataset)))
    random.Random(random_seed).shuffle(indices)
    sampled_dataset = dataset.select(indices[:sample_count])

    write_samples(sampled_dataset, svg_column, plan)

    print(f"Sampled {sample_count} SVG files from {args.dataset}.")
    print(f"SVG column: {svg_column}")
    print(f"Output root: {RAW_DIR}")


if __name__ == "__main__":
    main()
