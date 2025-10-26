# Federated Learning PID Defense

This repository contains our research implementation for **Federated Learning (FL)** under **data poisoning attacks**, using a **PID-based defense mechanism** within the **IntelliMAD** framework.  
The experiments are run on the **FEMNIST dataset**, evaluating the defense performance compared to a baseline (FedAvg).

## Big-picture overview

- Orchestration: `framework/` will hold client–server logic (IntelliMAD/Flower). The PID defense integrates into `framework/server.py` aggregation.
- Model: `model/cnn_model.py` defines the CNN used in experiments.
- Experiments: `experiments/` contains runnable scripts and plotting utilities; results go to `experiments/results/`, plots to `experiments/plots/`.
- Configuration: `cfg/project.yaml` sets defaults (clients, rounds, paths). `cfg/load_config.py` loads YAML and resolves paths to absolute locations.
- References: `references/` contains summaries and PDFs for IntelliMAD and the NeurIPS PID paper guiding design.

## FL flow with PID aggregation

```mermaid
flowchart LR
      subgraph Clients
         A1[Client 1] -->|local train| U1[Update 1]
         A2[Client 2] -->|local train| U2[Update 2]
         A3[Client N] -->|local train| UN[Update N]
      end

      U1 & U2 & UN --> S[Server]
   S --> P[PID defense layer<br/>(score/filter updates)]
   P --> W[Weighted aggregation<br/>(e.g., FedAvg with PID weights)]
      W --> G[Global model]
      G -->|broadcast| A1
      G -->|broadcast| A2
      G -->|broadcast| A3
```

---

## Project Structure

```
federated-learning-pid-defense/
├── cfg/                    # Configuration files and loaders
│   ├── project.yaml        # Project-wide configuration (paths, params)
│   └── load_config.py      # Utility for loading YAML settings
│
├── data/                   # Data folder (preprocessed FEMNIST clients)
│   └── **init**.py
│
├── experiments/
│   ├── dry_run/            # Initial baseline (no-attack) experiments
│   │   ├── dry_run_baseline.py
│   │   └── plot_metrics.py
│   ├── plots/              # Generated plots (accuracy/loss)
│   └── results/            # Stored results (CSV logs)
│
├── framework/              # Client-server setup using IntelliMAD or Flower
│   ├── client.py
│   └── server.py
│
├── model/                  # CNN model definition
│   └── cnn_model.py
│
├── Makefile                # Convenient run commands (dry-run, plots, etc.)
├── requirements.txt        # Python dependencies
├── .devcontainer/          # Codespace configuration
├── .vscode/                # Editor settings
└── README.md

```

## Configuration conventions

- Always load settings with `cfg.load_config.load()`; it resolves `data_dir`, `results_dir`, and `plots_dir` to absolute paths.
- Prefer reading defaults (e.g., `clients`, `rounds`) from `cfg/project.yaml` rather than hardcoding.
- Before writing outputs, ensure the directory exists with `os.makedirs(..., exist_ok=True)`.

Example (pattern used by `experiments/dry_run/dry_run_baseline.py`):

```python
from cfg.load_config import load as load_cfg
import os, csv

CFG = load_cfg()  # {'data_dir': '/abs/...', 'results_dir': '/abs/...', ...}
out_path = os.path.join(CFG['results_dir'], 'baseline.csv')
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, 'w', newline='') as f:
   w = csv.writer(f)
   w.writerow(['round', 'avg_acc', 'avg_loss'])
   # write rows...
```

---

## Running Experiments

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Dry-Run Baseline

```bash
make dry-run
```

### 3. Generate Metrics Plots

```bash
make plots
```

**Results are saved under:**

* `experiments/results/baseline.csv`
* `experiments/plots/acc.png`
* `experiments/plots/loss.png`

---

## Using the Dev Container

For a consistent setup across all teammates, this repository includes a **VS Code Dev Container** configuration under `.devcontainer/`.

When opened in VS Code, the container automatically installs Python 3.11 and all dependencies from `requirements.txt`.

### Quick Start

1. **Open the repository in VS Code.**  
2. When prompted, click **“Reopen in Container.”**  
   - If not prompted automatically, open the Command Palette (**Ctrl + Shift + P** or **Cmd + Shift + P**) → select **“Dev Containers: Reopen in Container.”**
3. Wait for the container to build. It will:
   - Install Python 3.11  
   - Upgrade `pip`  
   - Auto-install everything in `requirements.txt`  

Once finished, the environment is fully ready — you can run:
```bash
make dry-run
make plots
```

---

## Team Responsibilities

| Member | Coding Focus                    | Writing Section           |
| ------ | ------------------------------- | ------------------------- |
| A      | Framework Setup (server–client) | Introduction & Background |
| B      | Dataset Preparation (FEMNIST)   | Methods – Dataset         |
| C      | Model & Baseline Run            | Results & Analysis        |

---

## Tech Stack

* **Language:** Python 3.10+
* **Framework:** IntelliMAD / Flower
* **Libraries:** PyTorch, NumPy, Matplotlib, YAML
* **Dataset:** FEMNIST

---

## Roadmap

* [ ] Implement PID defense layer in `framework/server.py` (scoring/thresholding + weighting)
* [ ] Add small FL driver + attack simulation (few clients, few rounds) for quick iteration
* [ ] Evaluate PID vs FedAvg baseline on FEMNIST subsets; log CSVs under `experiments/results/`
* [ ] Plot comparative metrics and save to `experiments/plots/`
* [ ] Document parameters in `cfg/project.yaml` and expose CLI overrides in experiment scripts
* [ ] Prepare report figures/tables (link to plots/CSVs)

---

## References that inform design

- IntelliMAD: `references/IntelliMAD_*`, `references/IntelliMAD.pdf`
- PID defense (NeurIPS): `references/NeurIPS_PID_paper_*`, `references/NeurIPS_PID_paper.pdf`
- Project scope/summary: `references/Instructor_Research_FL_Project_Summary.md`
- Course deliverables/spec: `references/*Course_Project_Specification*`

---

## 📄 Documents & SharePoint Links

All written deliverables are maintained in SharePoint instead of this repository.  
Use the links below to access the latest versions of each document:

- [Phase 0 – Proposal Draft](https://www.overleaf.com/project/68d1fc646c93a0c0a02abb8a)
- [Progress Report 1](https://www.overleaf.com/project/68ed8a6891b863897e14f920)
- [Progress Report 2]()

---


## License

For academic use — CSCI 4341 Foundations of Intelligent Security Systems (Fall 2025).

