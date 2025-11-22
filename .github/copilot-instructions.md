## Purpose

Short, actionable guidance for AI coding agents working in this repository. Focus on the project's architecture, developer workflows, config conventions, and concrete edit/run examples.

## Big-picture architecture (what to know first)

- Top-level layout: `framework/` holds the client–server orchestration (`server.py`, `client.py`); `model/` contains the PyTorch model (`cnn_model.py`); `experiments/` contains runnable experiment scripts and plotting utilities; `cfg/` holds `project.yaml` and `load_config.py` which centralize paths and defaults.
- Data and outputs: raw/processed data lives under `data/`. Results and plots are written to `experiments/results/` and `experiments/plots/` respectively (see `cfg/project.yaml`).
- Flow for a typical experiment: `experiments/*` scripts call `cfg.load_config.load()` to resolve paths → instantiate model from `model/cnn_model.py` → run client updates (via `framework/client.py`) → server aggregation in `framework/server.py` → write CSV results → plotting scripts under `experiments/dry_run/` produce images.

## Key integration points (where to edit)

- To add or change a defense/attack, modify `framework/server.py` (aggregation logic) and `framework/client.py` (local training/poison injection). The README notes the PID defense should be integrated as a layer in the aggregation pipeline.
- Byzantine fault tolerance strategies are documented in `references/Byzantine_Fault_Tolerance_Guide.md` with implementation patterns for Trust-based, Krum, Bulyan, PID-based, and RFA defenses. These integrate into `framework/server.py` as multi-layer defense mechanisms.
- Model changes go into `model/cnn_model.py`.
- Experiment parameters and paths are controlled by `cfg/project.yaml` and consumed by `cfg/load_config.py` (which resolves relative paths to absolute ones). Update `project.yaml` when adding new default values.
- Small scripts such as `experiments/dry_run/dry_run_baseline.py` are canonical examples — they parse args, call `load()`, ensure output dirs exist, and write CSVs (see `os.makedirs(...)` usage).

## How to run (concrete commands)

- Setup environment and dependencies:

  pip install -r requirements.txt

- Run the baseline dry-run (uses `Makefile` target):

  make dry-run

  Equivalent direct command (Makefile uses PYTHONPATH=.):

  PYTHONPATH=. python experiments/dry_run/dry_run_baseline.py --clients 10 --rounds 5 --out experiments/results/baseline.csv

- Generate plots:

  make plots

  Or:

  PYTHONPATH=. python experiments/dry_run/plot_metrics.py --in experiments/results/baseline.csv --out experiments/plots/

Notes: the repo includes a `.devcontainer/` to standardize Python (3.11) and deps — prefer opening the project in the devcontainer when available.

## Project-specific conventions and patterns

- Configuration: always use `cfg.load_config.load()` to read paths and defaults. `load()` converts `data_dir`, `results_dir`, `plots_dir` to absolute paths — rely on that rather than constructing relative paths manually.
- Outputs: experiments write CSVs under `experiments/results/` and plots under `experiments/plots/`. Tests and CI are not present; scripts are lightweight and intended for quick local runs.
- PYTHONPATH usage: Makefile runs scripts with `PYTHONPATH=.`, and many experiment scripts assume importable package roots; preserve this when running scripts directly.
- Minimal defensive coding: experiment scripts explicitly call `os.makedirs(os.path.dirname(out), exist_ok=True)` before writing; follow this pattern when adding new scripts.

## Concrete examples to copy from

- Adding a new experiment: copy `experiments/dry_run/dry_run_baseline.py`, update argparse defaults to read from `cfg/project.yaml` (via `cfg.load_config.load()`), ensure output dirs are created, and write results to `experiments/results/`.
- Hooking a new aggregation rule: implement the aggregation change in `framework/server.py` and add a small unit-like script under `experiments/` that drives a few rounds (use `--rounds` small values for quick iteration).

## Dependencies & environment

- Key packages: `torch`, `torchvision`, `flwr` (Flower / IntelliMAD usage), `numpy`, `matplotlib`. See `requirements.txt`.
- Use the included Dev Container for reproducible environment (Python 3.11). If not using the devcontainer, run `pip install -r requirements.txt`.

## Quick troubleshooting hints

- If imports fail when running experiment scripts, ensure you run them with `PYTHONPATH=.` (see Makefile) or run from repository root with the devcontainer.
- If paths to results/plots are incorrect, confirm `cfg/project.yaml` values and that `cfg/load_config.load()` is used to resolve them.

## Background references (for design choices)

- `references/IntelliMAD_*` and `references/IntelliMAD.pdf` — summaries and paper notes on IntelliMAD/Flower FL orchestration; useful when wiring server–client aggregation and scheduling.
- `references/NeurIPS_PID_paper_*` and `references/NeurIPS_PID_paper.pdf` — primary background for the PID defense; consult when defining the defense signals and where to place the PID layer in `framework/server.py`.
- `references/Byzantine_Fault_Tolerance_Guide.md` — comprehensive guide for integrating Trust-based, Krum, Bulyan, PID-based, and RFA Byzantine defenses into the aggregation pipeline.
- `references/Instructor_Research_FL_Project_Summary.md` — internal scope and objectives overview.
- `references/*Course_Project_Specification*` — course requirements/deliverables to keep scripts and outputs aligned.

## Files to reference when making changes

- `README.md` — high-level intents and make targets
- `cfg/project.yaml`, `cfg/load_config.py` — configuration and path resolution
- `framework/server.py`, `framework/client.py` — where to implement aggregation and client behavior
- `model/cnn_model.py` — model definition (PyTorch)
- `experiments/dry_run/dry_run_baseline.py` — canonical runnable example
- `Makefile` — convenient targets (`dry-run`, `plots`, `setup`, `clean`)

---

If any of these areas are unclear or you'd like the instructions to include more examples (for instance, sample code snippets to implement a PID aggregator), tell me which part to expand and I'll iterate.
