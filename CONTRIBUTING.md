## Contributing Guidelines (docs and experiments)

This project is currently in the documentation-and-experiments phase. We welcome improvements to docs, config, and experiment scripts. Implementation of the PID defense will come later.

### Environment
- Prefer the VS Code Dev Container in `.devcontainer/` (Python 3.11).
- Otherwise: `pip install -r requirements.txt`.

### Running
- Quick smoke test: `make dry-run` (writes `experiments/results/baseline.csv`).
- Plot from CSV: `make plots` (writes to `experiments/plots/`).
- When running scripts directly, use repo root with `PYTHONPATH=.`.

### Configuration conventions
- Read settings via `cfg.load_config.load()`; it resolves `data_dir`, `results_dir`, `plots_dir` to absolute paths.
- Defaults live in `cfg/project.yaml` (e.g., `clients`, `rounds`, `seed`).
- Ensure output dirs exist before writing: `os.makedirs(..., exist_ok=True)`.

### Adding experiments
- Use `experiments/dry_run/dry_run_baseline.py` as a template.
- Accept CLI args with defaults from `cfg/project.yaml` via `load()`.
- Write CSVs under `experiments/results/` and plots under `experiments/plots/`.

### Where to integrate PID defense
- Server-side aggregation layer in `framework/server.py` (score/filter client updates → weighted aggregation).
- Optional client-side signals in `framework/client.py` if needed.
- See `references/IntelliMAD_*` and `references/NeurIPS_PID_paper_*` for design details.

### Code style and commits
- Python style: PEP 8 (pragmatic). Type hints where helpful.
- Commit messages: imperative mood, concise summary, include scope when useful (e.g., `readme: add PID flow diagram`).
- Small focused PRs preferred (docs-only changes are welcome).

### Dependencies
- Add Python deps to `requirements.txt`. Avoid pinning unless necessary.
- If a script needs a new dep, update Makefile targets if relevant.

### File and path conventions
- Don’t construct relative paths manually; rely on `cfg.load_config.load()`.
- Keep result/plot outputs under `experiments/` subfolders.
- Preserve `PYTHONPATH=.` usage in Make targets and examples.

### Questions
- Background and rationale live in `references/`.
- For repo organization or conventions, see `.github/copilot-instructions.md` and `README.md`.

