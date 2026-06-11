# SVG Experiment Pipeline

This project prepares SVG/PNG samples for human design, degraded SVG, and model-generated SVG experiments.

## Source Files

### `src/sample_raw_svg.py`

Randomly samples SVG code from the local dataset and writes it into:

```text
data/raw/<category>/svg/
```

Categories are `human_design`, `degraded`, and `model_generated`.

Example:

```powershell
python src/sample_raw_svg.py
```

Options:

- `--config`: path to the YAML config file. Default: `configs/config.yaml`.
- `--dataset`: local dataset path passed to `datasets.load_dataset()`. Default: `C:\dataset\starvector\svg-stack`.

### `src/raster_png.py`

Renders SVG files to PNG files with browser rendering. It can process both `raw` and `processed` data:

```text
data/raw/<category>/svg/       -> data/raw/<category>/png/
data/processed/<category>/svg/ -> data/processed/<category>/png/
```

Example:

```powershell
python src/raster_png.py --browser-channel msedge --overwrite
```

Options:

- `--config`: path to the YAML config file. Default: `configs/config.yaml`.
- `--browser-channel`: browser channel, for example `msedge` or `chrome`.
- `--split`: process only one split, either `raw` or `processed`.
- `--category`: process only one category: `human_design`, `degraded`, or `model_generated`.
- `--overwrite`: overwrite existing PNG files.
- `--start-index`: start index inside each selected SVG directory. Default: `0`.
- `--limit`: maximum number of SVG files to process inside each selected directory.
- `--image-timeout`: per-image browser load timeout in milliseconds. Default: `10000`.

### `src/augmentation.py`

Defines SVG augmentation utilities. It supports:

- rotation
- shift
- scale
- path noise
- color noise
- color replacement
- element drop by rate
- browser-based SVG rasterization

The main interface is:

```python
SVGTransforms.from_config("configs/config.yaml")
```

Augmentation strength is controlled by the `degradation` section in `configs/config.yaml`.

### `src/add_augmentation.py`

Applies augmentation to raw degraded SVG files and writes the augmented SVG files into processed degraded data:

```text
data/raw/degraded/svg/       -> data/processed/degraded/svg/
```

Example:

```powershell
python src/add_augmentation.py --overwrite
```

Options:

- `--config`: path to the YAML config file. Default: `configs/config.yaml`.
- `--filename`: process one specific SVG filename, for example `301.svg`.
- `--limit`: maximum number of SVG files to process. Defaults to all files.
- `--start-index`: start index in the sorted raw degraded SVG list. Default: `0`.
- `--overwrite`: overwrite existing processed degraded SVG outputs.

### `src/qwen_process.py`

Sends raw `model_generated` PNG files to Qwen VL and extracts SVG code from the model response:

```text
data/raw/model_generated/png/ -> data/processed/model_generated/svg/
```

Model settings and prompt are read from `configs/config.yaml`.

Example:

```powershell
python src/qwen_process.py --provider aliyuncs --timeout 300 --max-retries 1
```

Options:

- `--config`: path to the YAML config file. Default: `configs/config.yaml`.
- `--filename`: process one specific PNG filename, for example `601.png`.
- `--limit`: maximum number of images to process. Defaults to all files.
- `--start-index`: start index in the metadata filename list. Default: `0`.
- `--overwrite`: overwrite existing SVG outputs.
- `--provider`: model provider config under `model_generation`. Choices: `aliyuncs`, `aihubmix`. Default: `aliyuncs`.
- `--disable-thinking`: disable Qwen thinking mode.
- `--thinking-budget`: maximum thinking token budget for Qwen. Default: `81920`.
- `--timeout`: API request timeout in seconds. Default: `300`.
- `--max-retries`: maximum number of API retries. Default: `2`.

### `src/model_judge.py`

Scores processed PNG files with the local aesthetic predictor and writes model scores to:

```text
data/metadata/model_judge.json
```

Example:

```powershell
python src/model_judge.py --batch-size 32
```

Options:

- `--category`: score only one category: `human_design`, `degraded`, or `model_generated`.
- `--batch-size`: number of images per model forward pass. Default: `16`.
- `--start-index`: start index inside each selected category. Default: `0`.
- `--limit`: maximum number of images to score inside each selected category.
- `--device`: inference device. Choices: `auto`, `cuda`, `cpu`. Default: `auto`.

### `src/human_judge.py`

Runs a local blind human scoring UI for processed PNG files. Images are shown in randomized order, and the rater cannot see category, filename, or original number. Scores are saved immediately to:

```text
data/metadata/human_judge.json
```

Example:

```powershell
python src/human_judge.py
```

Open the printed local URL in a browser, then score each image from 1 to 10. Stop with `Ctrl+C`; restart the same command to resume from the first unscored image.

Options:

- `--output`: path to the human score JSON file. Default: `data/metadata/human_judge.json`.
- `--host`: server host. Default: `127.0.0.1`.
- `--port`: server port. Default: `8765`.
- `--seed`: random seed used when creating a new scoring order. Default: `20020306`.
- `--reset`: start a new random order and overwrite the existing output file.

### `src/test.py`

Scores `outputs/test.png` with the local aesthetic predictor.

Example:

```powershell
python src/test.py
```
