from __future__ import annotations

import argparse
import base64
import html
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import yaml
from playwright.sync_api import Browser, Page, Playwright, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_ORDER = ("raw", "processed")
CATEGORY_ORDER = ("human_design", "degraded", "model_generated")
DEFAULT_CANVAS_SIZE = 512


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def iter_svg_files(input_dir: Path) -> list[Path]:
    return sorted(
        input_dir.glob("*.svg"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def parse_length(value: str | None) -> float | None:
    if not value:
        return None

    match = re.match(r"^\s*([0-9]*\.?[0-9]+)", value)
    if not match:
        return None
    return float(match.group(1))


def get_svg_size(svg_path: Path) -> tuple[int, int]:
    root = ET.parse(svg_path).getroot()
    width = parse_length(root.attrib.get("width"))
    height = parse_length(root.attrib.get("height"))

    if (not width or not height) and root.attrib.get("viewBox"):
        parts = root.attrib["viewBox"].replace(",", " ").split()
        if len(parts) == 4:
            width = width or float(parts[2])
            height = height or float(parts[3])

    if not width or not height:
        width = height = DEFAULT_CANVAS_SIZE

    return max(1, math.ceil(width)), max(1, math.ceil(height))


def get_output_size(rendering_config: dict[str, Any]) -> tuple[int, int]:
    width = int(rendering_config.get("output_width", DEFAULT_CANVAS_SIZE))
    height = int(rendering_config.get("output_height", DEFAULT_CANVAS_SIZE))

    if width <= 0 or height <= 0:
        raise ValueError("rendering.output_width and output_height must be positive.")

    return width, height


def normalize_background_color(color_name: str) -> str:
    color = color_name.strip()
    if color.lower() in {"white", "black", "transparent"}:
        return color.lower()

    if re.match(r"^#[0-9a-fA-F]{6}$", color):
        return color

    raise ValueError(
        f"Unsupported background_color={color_name!r}. "
        "Use white, black, transparent, or a #RRGGBB value."
    )


def launch_browser(preferred_channel: str | None = None) -> tuple[Playwright, Browser]:
    playwright = sync_playwright().start()
    candidates: Iterable[str | None]
    if preferred_channel:
        candidates = (preferred_channel,)
    else:
        candidates = ("msedge", "chrome", None)

    last_error: Exception | None = None
    for channel in candidates:
        try:
            browser = playwright.chromium.launch(channel=channel, headless=True)
            return playwright, browser
        except Exception as error:
            last_error = error

    playwright.stop()
    raise RuntimeError(
        "Could not launch a browser. Install one of these options:\n"
        "1. Microsoft Edge or Google Chrome, then rerun this script.\n"
        "2. Or run: python -m playwright install chromium"
    ) from last_error


def raster_svg_file(
    page: Page,
    svg_path: Path,
    png_path: Path,
    background_color: str,
    output_width: int,
    output_height: int,
    image_timeout_ms: int,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    page.set_viewport_size({"width": output_width, "height": output_height})

    svg_text = svg_path.read_text(encoding="utf-8")
    svg_data_url = (
        "data:image/svg+xml;base64,"
        + base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    )
    background = normalize_background_color(background_color)
    page.set_content(
        f"""
        <!doctype html>
        <html>
          <head>
            <style>
              html, body {{
                margin: 0;
                width: {output_width}px;
                height: {output_height}px;
                overflow: hidden;
                background: {html.escape(background)};
              }}
              #frame {{
                align-items: center;
                background: {html.escape(background)};
                display: flex;
                height: {output_height}px;
                justify-content: center;
                width: {output_width}px;
              }}
              #frame > img {{
                background: {html.escape(background)};
                display: block;
                height: 100%;
                max-height: 100%;
                max-width: 100%;
                object-fit: contain;
                width: 100%;
              }}
            </style>
          </head>
          <body>
            <div id="frame">
              <img src="{svg_data_url}" />
            </div>
          </body>
        </html>
        """,
        wait_until="load",
        timeout=image_timeout_ms,
    )
    page.wait_for_function(
        """
        () => {
          const image = document.querySelector('img');
          return image && image.complete && image.naturalWidth > 0;
        }
        """,
        timeout=image_timeout_ms,
    )
    page.locator("#frame").screenshot(path=str(png_path), scale="css")


def raster_directory(
    page: Page,
    split: str,
    category: str,
    background_color: str,
    output_width: int,
    output_height: int,
    skip_invalid_svg: bool,
    overwrite: bool,
    start_index: int,
    limit: int | None,
    image_timeout_ms: int,
) -> tuple[int, int, int]:
    input_dir = DATA_DIR / split / category / "svg"
    output_dir = DATA_DIR / split / category / "png"

    success_count = 0
    failed_count = 0
    skipped_count = 0

    if not input_dir.exists():
        print(f"Missing input directory: {input_dir}")
        return success_count, failed_count, skipped_count

    svg_files = iter_svg_files(input_dir)
    if limit is None:
        svg_files = svg_files[start_index:]
    else:
        svg_files = svg_files[start_index : start_index + limit]

    for svg_path in svg_files:
        png_path = output_dir / f"{svg_path.stem}.png"
        if png_path.exists() and not overwrite:
            skipped_count += 1
            continue

        try:
            raster_svg_file(
                page,
                svg_path,
                png_path,
                background_color,
                output_width,
                output_height,
                image_timeout_ms,
            )
            success_count += 1
        except Exception as error:
            failed_count += 1
            print(f"Skipped invalid SVG: {svg_path} ({error})")
            if not skip_invalid_svg:
                raise

    return success_count, failed_count, skipped_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize raw and processed SVG files into corresponding PNG files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--browser-channel",
        default=None,
        help="Browser channel to use, for example msedge or chrome.",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_ORDER,
        help="Only rasterize one split: raw or processed.",
    )
    parser.add_argument(
        "--category",
        choices=CATEGORY_ORDER,
        help="Only rasterize one category.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index inside each selected SVG directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of SVG files to process inside each selected directory.",
    )
    parser.add_argument(
        "--image-timeout",
        type=int,
        default=10000,
        help="Per-image browser load timeout in milliseconds.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rendering_config = config.get("rendering", {})
    background_color = str(rendering_config.get("background_color", "white"))
    output_width, output_height = get_output_size(rendering_config)
    skip_invalid_svg = bool(rendering_config.get("skip_invalid_svg", True))

    splits = (args.split,) if args.split else SPLIT_ORDER
    categories = (args.category,) if args.category else CATEGORY_ORDER

    playwright, browser = launch_browser(args.browser_channel)
    try:
        page = browser.new_page()
        total_success = 0
        total_failed = 0
        total_skipped = 0
        for split in splits:
            for category in categories:
                success_count, failed_count, skipped_count = raster_directory(
                    page=page,
                    split=split,
                    category=category,
                    background_color=background_color,
                    output_width=output_width,
                    output_height=output_height,
                    skip_invalid_svg=skip_invalid_svg,
                    overwrite=args.overwrite,
                    start_index=args.start_index,
                    limit=args.limit,
                    image_timeout_ms=args.image_timeout,
                )
                total_success += success_count
                total_failed += failed_count
                total_skipped += skipped_count
                print(
                    f"{split}/{category}: rasterized {success_count}, "
                    f"skipped existing {skipped_count}, invalid {failed_count}"
                )
    finally:
        browser.close()
        playwright.stop()

    print(
        f"Done. Rasterized {total_success}, skipped existing {total_skipped}, "
        f"invalid {total_failed}."
    )
    print(f"Output root: {DATA_DIR}")


if __name__ == "__main__":
    main()
