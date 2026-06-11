from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    r"C:\model\shunk031\aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
)
IMAGE_PATH = PROJECT_ROOT / "outputs" / "test.png"
HF_MODULES_CACHE = PROJECT_ROOT / ".cache" / "huggingface_modules"
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

os.environ["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE)

import transformers.utils.import_utils as transformers_import_utils

transformers_import_utils._sklearn_available = False
transformers_import_utils._scipy_available = False

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)

    values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    values = values / 255.0

    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    values = (values - mean) / std
    return values.unsqueeze(0)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    pixel_values = preprocess_image(IMAGE_PATH).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    score = outputs.logits.squeeze().item()
    print(f"Image: {IMAGE_PATH}")
    print(f"Device: {device}")
    print(f"Aesthetic score: {score:.6f}")


if __name__ == "__main__":
    main()
