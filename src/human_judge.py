from __future__ import annotations

import argparse
import json
import mimetypes
import random
import secrets
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
DEFAULT_OUTPUT_PATH = METADATA_DIR / "human_judge.json"
CATEGORY_ORDER = ("human_design", "degraded", "model_generated")
SCORE_MIN = 1
SCORE_MAX = 10


def iter_png_files() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for category in CATEGORY_ORDER:
        image_dir = PROCESSED_DIR / category / "png"
        files = sorted(
            image_dir.glob("*.png"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )
        for path in files:
            items.append({"category": category, "filename": path.name})
    return items


def make_token() -> str:
    return secrets.token_urlsafe(12)


def make_initial_state(seed: int) -> dict[str, Any]:
    items = iter_png_files()
    rng = random.Random(seed)
    rng.shuffle(items)

    entries = []
    for item in items:
        entries.append(
            {
                "token": make_token(),
                "category": item["category"],
                "filename": item["filename"],
                "score": None,
                "scored_at": None,
            }
        )

    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": None,
            "random_seed": seed,
            "score_range": [SCORE_MIN, SCORE_MAX],
            "total": len(entries),
            "completed": 0,
        },
        "items": entries,
    }


def load_or_create_state(output_path: Path, seed: int, reset: bool) -> dict[str, Any]:
    if output_path.exists() and not reset:
        with output_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    state = make_initial_state(seed)
    save_state(output_path, state)
    return state


def save_state(output_path: Path, state: dict[str, Any]) -> None:
    completed = sum(1 for item in state["items"] if item.get("score") is not None)
    state["meta"]["completed"] = completed
    state["meta"]["total"] = len(state["items"])
    state["meta"]["updated_at"] = datetime.now().isoformat(timespec="seconds")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def image_path_for(item: dict[str, Any]) -> Path:
    return PROCESSED_DIR / item["category"] / "png" / item["filename"]


def find_current_item(state: dict[str, Any]) -> dict[str, Any] | None:
    for item in state["items"]:
        if item.get("score") is None:
            return item
    return None


def find_item_by_token(state: dict[str, Any], token: str) -> dict[str, Any] | None:
    for item in state["items"]:
        if item["token"] == token:
            return item
    return None


def json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Human Judge</title>
    <style>
      body {
        margin: 0;
        background: #f4f4f0;
        color: #202020;
        font-family: Arial, sans-serif;
      }
      main {
        box-sizing: border-box;
        display: grid;
        grid-template-rows: auto 1fr auto;
        gap: 16px;
        min-height: 100vh;
        padding: 18px;
      }
      header {
        align-items: center;
        display: flex;
        justify-content: space-between;
      }
      #progress {
        font-size: 14px;
      }
      #status {
        font-size: 14px;
        min-height: 20px;
      }
      #stage {
        align-items: center;
        background: white;
        border: 1px solid #d8d8d0;
        display: flex;
        justify-content: center;
        min-height: 0;
        overflow: hidden;
      }
      img {
        max-height: calc(100vh - 170px);
        max-width: 100%;
        object-fit: contain;
      }
      #scores {
        display: grid;
        gap: 8px;
        grid-template-columns: repeat(10, minmax(42px, 1fr));
      }
      button {
        background: #ffffff;
        border: 1px solid #b8b8b0;
        cursor: pointer;
        font-size: 18px;
        height: 44px;
      }
      button:hover {
        background: #eeeeea;
      }
      button:disabled {
        cursor: wait;
        opacity: 0.55;
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <strong>Human Judge</strong>
        <span id="progress"></span>
      </header>
      <section id="stage">
        <img id="image" alt="" />
      </section>
      <section>
        <div id="scores"></div>
        <div id="status"></div>
      </section>
    </main>
    <script>
      let current = null;
      let busy = false;

      function setButtonsDisabled(disabled) {
        document.querySelectorAll("button").forEach((button) => {
          button.disabled = disabled;
        });
      }

      async function loadCurrent() {
        const response = await fetch("/api/current");
        current = await response.json();

        const progress = document.getElementById("progress");
        const status = document.getElementById("status");
        const image = document.getElementById("image");

        progress.textContent = `${current.completed} / ${current.total}`;
        if (current.done) {
          image.removeAttribute("src");
          status.textContent = "All images have been scored.";
          setButtonsDisabled(true);
          return;
        }

        image.src = current.image_url + `?t=${Date.now()}`;
        status.textContent = "";
        setButtonsDisabled(false);
      }

      async function submitScore(score) {
        if (!current || current.done || busy) return;
        busy = true;
        setButtonsDisabled(true);
        document.getElementById("status").textContent = "Saving...";

        const response = await fetch("/api/score", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({token: current.token, score}),
        });

        if (!response.ok) {
          document.getElementById("status").textContent = "Save failed.";
          setButtonsDisabled(false);
          busy = false;
          return;
        }

        busy = false;
        await loadCurrent();
      }

      function buildButtons() {
        const scores = document.getElementById("scores");
        for (let score = 1; score <= 10; score += 1) {
          const button = document.createElement("button");
          button.textContent = String(score);
          button.addEventListener("click", () => submitScore(score));
          scores.appendChild(button);
        }
      }

      window.addEventListener("keydown", (event) => {
        if (event.key >= "1" && event.key <= "9") {
          submitScore(Number(event.key));
        } else if (event.key === "0") {
          submitScore(10);
        }
      });

      buildButtons();
      loadCurrent();
    </script>
  </body>
</html>
"""


def make_handler(state: dict[str, Any], output_path: Path) -> type[BaseHTTPRequestHandler]:
    class HumanJudgeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)

            if parsed.path == "/":
                text_response(self, HTML)
                return

            if parsed.path == "/api/current":
                current = find_current_item(state)
                completed = state["meta"]["completed"]
                total = state["meta"]["total"]
                if current is None:
                    json_response(
                        self,
                        {"done": True, "completed": completed, "total": total},
                    )
                    return

                json_response(
                    self,
                    {
                        "done": False,
                        "token": current["token"],
                        "image_url": f"/image/{current['token']}",
                        "completed": completed,
                        "total": total,
                    },
                )
                return

            if parsed.path.startswith("/image/"):
                token = parsed.path.rsplit("/", 1)[-1]
                item = find_item_by_token(state, token)
                if item is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                image_path = image_path_for(item)
                if not image_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                content = image_path.read_bytes()
                content_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/score":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
                token = str(payload["token"])
                score = int(payload["score"])
            except Exception:
                json_response(self, {"error": "Invalid payload."}, HTTPStatus.BAD_REQUEST)
                return

            if score < SCORE_MIN or score > SCORE_MAX:
                json_response(self, {"error": "Score must be 1-10."}, HTTPStatus.BAD_REQUEST)
                return

            item = find_item_by_token(state, token)
            if item is None:
                json_response(self, {"error": "Unknown token."}, HTTPStatus.NOT_FOUND)
                return

            item["score"] = score
            item["scored_at"] = datetime.now().isoformat(timespec="seconds")
            save_state(output_path, state)
            json_response(self, {"ok": True})

    return HumanJudgeHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local blind human scoring UI for processed PNG files."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the human score JSON file.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port. Default: 8765.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20020306,
        help="Random seed used when creating a new scoring order.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start a new random order and overwrite the existing output file.",
    )
    args = parser.parse_args()

    state = load_or_create_state(args.output, args.seed, args.reset)
    handler = make_handler(state, args.output)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    url = f"http://{args.host}:{args.port}"
    print(f"Human scoring UI: {url}")
    print(f"Output JSON: {args.output}")
    print(f"Progress: {state['meta']['completed']} / {state['meta']['total']}")
    print("Press Ctrl+C to stop. Restart the same command to resume.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
