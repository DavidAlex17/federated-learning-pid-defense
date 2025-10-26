# Milestone 2 — Conventional Federated Learning (Baseline)

Status: current phase (in progress)

This milestone implements and evaluates a conventional FL baseline (no PID). Build a working FL loop, collect round-wise per-client metrics, and evaluate behavior with and without data poisoning.

## Objectives
- Implement baseline FL with a central server and ≥10 clients (simulation on one machine is fine).
- Local client training for a few steps per round; aggregate on server (FedAvg).
- Collect/save per-round metrics and generate plots.
- Run both no-attack and with data poisoning evaluations.

## Requirements (per course spec + summary)
- Framework: Flower + PyTorch recommended (supports per-client metrics and later custom aggregation).
- Clients: at least 10 per round (you may subsample each round if desired).
- Dataset: FEMNIST (recommended) or your chosen dataset; simulate realistic client partitions (non-IID preferred when feasible).
- Model: small CNN suitable for the dataset (see `model/cnn_model.py`).
- Runtime: single-machine simulation is acceptable.

## Configuration and outputs
- Read defaults from `cfg/project.yaml` via `cfg.load_config.load()`; it resolves `data_dir`, `results_dir`, `plots_dir` to absolute paths.
- Write CSV metrics to `experiments/results/`; save plots to `experiments/plots/`.
- Keep runs reproducible (e.g., `seed` in `cfg/project.yaml`).

## Metrics to collect each round (save to CSV)
- Per-client loss and average loss across clients.
- Per-client evaluation accuracy and average accuracy across clients.
- Include a `round` column and consistent headers for plotting.

## Evaluations and expected behavior
- Baseline (no attack): expect accuracy ↑ and loss ↓ over rounds.
- With data poisoning: poisoned clients’ accuracy should stagnate/degrade; global convergence should degrade vs baseline.
- If poisoned vs benign clients are not distinguishable in per-client plots:
  - Verify plotting logic and labeling.
  - Ensure poisoning is effective (consider creating poisoned data on disk rather than at runtime).
  - Confirm evaluation uses each client’s own local (possibly poisoned) subset.

## Plots to produce (8 total)
- No attack (4 plots): per-client loss; average loss; per-client accuracy; average accuracy.
- With attack (4 plots): per-client loss; average loss; per-client accuracy; average accuracy.

## Deliverables checklist
- [ ] Working FL baseline (≥10 clients) with server–client training loop and FedAvg aggregation.
- [ ] Round-wise CSV logs with per-client and average metrics under `experiments/results/`.
- [ ] 4 plots (no attack) + 4 plots (with attack) saved to `experiments/plots/`.
- [ ] Short discussion interpreting differences between no-attack and attack runs.

## Acceptance criteria
- The FL loop runs across multiple rounds and reliably logs metrics each round.
- Plots reflect expected trends (baseline improves; attack degrades).
- Results are reproducible given `cfg/project.yaml` and documented CLI args.
- The framework choice supports later custom aggregation (for PID in Milestone 3).

## Suggested implementation steps (non-binding)
1. Parameterize clients, rounds, and output paths; log metrics to CSV under `experiments/results/`.
2. Use Flower simulation APIs to run ≥10 clients per round and collect per-client metrics.
3. Implement a plotting script that reads CSVs and produces the 4 required plots for each scenario.
4. Add a data poisoning toggle and parameters; re-run and produce the 4 attack plots.

## References
- Course spec: `references/FISS_Fall_2025_Course_Project_Specification.txt` (Milestone 2 requirements and metrics/plots).
- Project summary: `references/Instructor_Research_FL_Project_Summary.md` (baseline requirements, metrics, plots, and evaluation guidance).