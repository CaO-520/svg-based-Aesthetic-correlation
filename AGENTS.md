# AGENTS.md

This repository is used from multiple machines, mainly a lab machine and a PC.
Keep GitHub synchronization in mind by default so the user does not have to
remind Codex every time.

## Default Git Workflow

- At the start of any work session, check the repository state with:
  - `git status --short --branch`
  - `git remote -v`
- If the working tree is clean, fetch the remote state before editing:
  - `git fetch origin`
  - compare the local branch with `origin/main`
- If the local branch is behind `origin/main` and there are no local changes,
  pull before making edits.
- If there are local uncommitted changes, do not pull, reset, stash, or overwrite
  them without explaining the situation to the user first.
- Never use destructive Git commands such as `git reset --hard` or
  `git checkout -- <file>` unless the user explicitly asks for them.

## End-of-Task Sync

- After completing file changes, run `git status --short --branch`.
- Summarize which files changed and whether the working tree is clean.
- For code or project changes, prepare to commit and push to GitHub unless the
  user says not to.
- Use concise commit messages that describe the actual change.
- Before pushing, make sure no secrets, credentials, local datasets, model
  weights, cache files, or large generated artifacts are staged.
- If GitHub/network permissions are required, request permission and continue
  once approved.

## Files That Should Stay Local

Do not commit local secrets or machine-specific configuration, including:

- `configs/API_KEY.txt`
- `configs/config.yaml`
- local dataset paths
- model checkpoints and weights
- cache directories
- temporary logs and generated experiment outputs unless the user explicitly
  asks to version them

The tracked example configuration should remain safe for GitHub:

- `configs/config.example.yaml`

## Project Notes

- This is an SVG experiment pipeline for sampling SVGs, rasterizing PNGs,
  applying augmentations, generating SVGs with Qwen VL, and collecting model or
  human aesthetic scores.
- Prefer existing scripts in `src/` and settings in `configs/config.yaml`.
- Use PowerShell-friendly commands in examples because the project is commonly
  used on Windows.
- Keep changes scoped and avoid unrelated refactors.

## Useful Commands

```powershell
python src/sample_raw_svg.py
python src/raster_png.py --browser-channel msedge --overwrite
python src/add_augmentation.py --overwrite
python src/qwen_process.py --provider aliyuncs --timeout 300 --max-retries 1
python src/model_judge.py --batch-size 32
python src/human_judge.py
```
