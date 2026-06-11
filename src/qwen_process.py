from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

import openai
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FILENAME_LIST_PATH = PROJECT_ROOT / "data" / "metadata" / "filename_list.json"
RAW_MODEL_PNG_DIR = PROJECT_ROOT / "data" / "raw" / "model_generated" / "png"
PROCESSED_MODEL_SVG_DIR = (
    PROJECT_ROOT / "data" / "processed" / "model_generated" / "svg"
)
LOG_DIR = PROJECT_ROOT / "logs"


def load_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_model_generated_filenames(filename_list_path: Path) -> list[str]:
    with filename_list_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    filenames: list[str] = []
    for item in data.get("model_generated", []):
        if isinstance(item, str):
            filenames.append(item)
        elif isinstance(item, dict) and item.get("filename"):
            filenames.append(str(item["filename"]))

    return sorted(
        filenames,
        key=lambda name: int(Path(name).stem)
        if Path(name).stem.isdigit()
        else Path(name).stem,
    )


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_svg(text: str) -> str | None:
    cleaned = text.strip()

    fenced_match = re.search(
        r"```(?:svg|xml)?\s*(<svg[\s\S]*?</svg>)\s*```",
        cleaned,
        flags=re.IGNORECASE,
    )
    if fenced_match:
        return normalize_svg(fenced_match.group(1))

    svg_match = re.search(r"<svg\b[\s\S]*?</svg>", cleaned, flags=re.IGNORECASE)
    if svg_match:
        return normalize_svg(svg_match.group(0))

    return None


def normalize_svg(svg: str) -> str:
    svg = svg.strip()
    svg = re.sub(r"^\s*```(?:svg|xml)?", "", svg, flags=re.IGNORECASE).strip()
    svg = re.sub(r"```\s*$", "", svg).strip()
    return svg


def get_model_config(config: dict[str, Any], provider: str) -> dict[str, Any]:
    model_generation_config = config.get("model_generation", {})
    model_config = model_generation_config.get(provider, model_generation_config)

    required_fields = ("qwen_model_id", "api_key", "base_url")
    missing_fields = [
        field for field in required_fields if not model_config.get(field)
    ]
    if missing_fields:
        raise ValueError(
            f"Missing model_generation.{provider} fields: "
            + ", ".join(missing_fields)
        )

    return model_config


def call_qwen_svg_generation(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    image_path: Path,
    enable_thinking: bool,
    thinking_budget: int,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        stream=True,
        extra_body={
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget,
        },
    )

    answer_content = ""
    for chunk in completion:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            answer_content += content

    return answer_content


def process_one(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    filename: str,
    overwrite: bool,
    enable_thinking: bool,
    thinking_budget: int,
) -> bool:
    image_path = RAW_MODEL_PNG_DIR / filename
    output_path = PROCESSED_MODEL_SVG_DIR / f"{Path(filename).stem}.svg"

    if not image_path.exists():
        print(f"Missing input image: {image_path}")
        return False

    if output_path.exists() and not overwrite:
        print(f"Skip existing SVG: {output_path}")
        return True

    raw_response = call_qwen_svg_generation(
        client,
        model,
        prompt,
        image_path,
        enable_thinking,
        thinking_budget,
    )
    svg = extract_svg(raw_response)

    if svg is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"qwen_response_{Path(filename).stem}.txt"
        log_path.write_text(raw_response, encoding="utf-8")
        print(f"No SVG found in response. Saved raw response: {log_path}")
        return False

    PROCESSED_MODEL_SVG_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    print(f"Wrote SVG: {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate processed SVG files from raw model_generated PNG files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--filename",
        help="Process one specific PNG filename, for example 601.png.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process. Defaults to all files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in metadata filename list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SVG outputs.",
    )
    parser.add_argument(
        "--provider",
        default="aliyuncs",
        choices=("aliyuncs", "aihubmix"),
        help="Model provider config under model_generation. Defaults to aliyuncs.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen thinking mode.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=81920,
        help="Maximum thinking token budget for Qwen.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="API request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of API retries.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_config = get_model_config(config, args.provider)
    prompt_config = config.get("model_prompt", {}).get("qwen_svg_generation", {})

    client = openai.OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    model = model_config["qwen_model_id"]
    prompt = prompt_config["template"]

    if args.filename:
        filenames = [args.filename]
    else:
        filenames = load_model_generated_filenames(FILENAME_LIST_PATH)
        if args.limit is None:
            filenames = filenames[args.start_index :]
        else:
            filenames = filenames[args.start_index : args.start_index + args.limit]

    success_count = 0
    for index, filename in enumerate(filenames, start=1):
        print(f"[{index}/{len(filenames)}] Processing {filename}")
        if process_one(
            client,
            model,
            prompt,
            filename,
            args.overwrite,
            not args.disable_thinking,
            args.thinking_budget,
        ):
            success_count += 1

    print(f"Done. Generated {success_count}/{len(filenames)} SVG files.")


if __name__ == "__main__":
    main()
