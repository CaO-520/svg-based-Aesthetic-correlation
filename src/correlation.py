from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_PATH = PROJECT_ROOT / "data" / "metadata" / "human_judge.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "metadata" / "model_judge.json"
CATEGORY_ORDER = ("human_design", "degraded", "model_generated")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def sort_filename_key(filename: str) -> int | str:
    stem = Path(filename).stem
    return int(stem) if stem.isdigit() else stem


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Pearson inputs must have the same length.")
    if not xs:
        return float("nan")

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denominator_x == 0 or denominator_y == 0:
        return float("nan")
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


def collect_scores(
    human_data: dict,
    model_data: dict,
    category: str,
) -> tuple[list[str], list[float], list[float]]:
    human_scores = {
        item["filename"]: item["score"]
        for item in human_data.get(category, [])
        if item.get("score") is not None
    }
    model_scores = {
        item["filename"]: item["score"]
        for item in model_data.get(category, [])
        if item.get("score") is not None
    }
    filenames = sorted(
        set(human_scores) & set(model_scores),
        key=sort_filename_key,
    )
    return (
        filenames,
        [float(human_scores[name]) for name in filenames],
        [float(model_scores[name]) for name in filenames],
    )


def print_result(name: str, human_scores: list[float], model_scores: list[float]) -> None:
    print(
        f"{name}: n={len(human_scores)}, "
        f"pearson={pearson(human_scores, model_scores):.4f}, "
        f"spearman={spearman(human_scores, model_scores):.4f}"
    )


def selected_categories(category: str | None) -> Iterable[str]:
    if category:
        return (category,)
    return CATEGORY_ORDER


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute correlation between human and model aesthetic scores."
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=DEFAULT_HUMAN_PATH,
        help="Path to human_judge.json.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to model_judge.json.",
    )
    parser.add_argument(
        "--category",
        choices=CATEGORY_ORDER,
        help="Only compute one category.",
    )
    args = parser.parse_args()

    human_data = load_json(args.human)
    model_data = load_json(args.model)

    all_human_scores: list[float] = []
    all_model_scores: list[float] = []
    for category in selected_categories(args.category):
        _, human_scores, model_scores = collect_scores(human_data, model_data, category)
        print_result(category, human_scores, model_scores)
        all_human_scores.extend(human_scores)
        all_model_scores.extend(model_scores)

    if not args.category:
        print_result("overall", all_human_scores, all_model_scores)


if __name__ == "__main__":
    main()
