from __future__ import annotations

import argparse
from pathlib import Path

from augmentation import SVGTransforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
RAW_DEGRADED_SVG_DIR = PROJECT_ROOT / "data" / "raw" / "degraded" / "svg"
PROCESSED_DEGRADED_SVG_DIR = PROJECT_ROOT / "data" / "processed" / "degraded" / "svg"
LOG_DIR = PROJECT_ROOT / "logs"


def iter_svg_files(input_dir: Path) -> list[Path]:
    return sorted(
        input_dir.glob("*.svg"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def process_one(transforms: SVGTransforms, svg_path: Path, overwrite: bool) -> bool:
    output_path = PROCESSED_DEGRADED_SVG_DIR / svg_path.name

    if output_path.exists() and not overwrite:
        print(f"Skip existing SVG: {output_path}")
        return True

    try:
        augmented_svg, _ = transforms.augment(str(svg_path), render_image=False)
    except Exception as error:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"augmentation_error_{svg_path.stem}.txt"
        log_path.write_text(str(error), encoding="utf-8")
        print(f"Failed: {svg_path} ({error})")
        return False

    PROCESSED_DEGRADED_SVG_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(augmented_svg, encoding="utf-8")
    print(f"Wrote SVG: {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply SVG augmentation to raw degraded SVG files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--filename",
        help="Process one specific SVG filename, for example 301.svg.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of SVG files to process. Defaults to all files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in the sorted raw degraded SVG list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed degraded SVG outputs.",
    )
    args = parser.parse_args()

    transforms = SVGTransforms.from_config(args.config)

    if args.filename:
        svg_files = [RAW_DEGRADED_SVG_DIR / args.filename]
    else:
        svg_files = iter_svg_files(RAW_DEGRADED_SVG_DIR)
        if args.limit is None:
            svg_files = svg_files[args.start_index :]
        else:
            svg_files = svg_files[args.start_index : args.start_index + args.limit]

    success_count = 0
    for index, svg_path in enumerate(svg_files, start=1):
        print(f"[{index}/{len(svg_files)}] Processing {svg_path.name}")
        if process_one(transforms, svg_path, args.overwrite):
            success_count += 1

    print(f"Done. Generated {success_count}/{len(svg_files)} augmented SVG files.")
    print(f"Output root: {PROCESSED_DEGRADED_SVG_DIR}")


if __name__ == "__main__":
    main()
