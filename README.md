# Federated Learning PID Defense

This repository contains our research implementation for **Federated Learning (FL)** under **data poisoning attacks**, using a **PID-based defense mechanism** within the **IntelliMAD** framework.  
The experiments are run on the **FEMNIST dataset**, evaluating the defense performance compared to a baseline (FedAvg).

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
├── docs/progress_report_1/ # Draft + outline for Progress Report 1
│   └── draft_outline.md
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

````

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

## Next Steps

* [ ] Integrate PID defense layer
* [ ] Add attack simulation scripts
* [ ] Compare PID vs baseline (FedAvg)
* [ ] Expand report for Progress Report 2

---

## 📄 Documents & SharePoint Links

All written deliverables are maintained in SharePoint instead of this repository.  
Use the links below to access the latest versions of each document:

- [Phase 0 – Proposal Draft](https://utrgv-my.sharepoint.com/:w:/r/personal/david_sanchez15_utrgv_edu/_layouts/15/Doc.aspx?sourcedoc=%7B3A064FC5-FE71-444E-9A1A-8808A8653C06%7D&

file=CSCI4341%20Project%20Proposal%20Draft.docx&action=default&mobileredirect=true)
- [Progress Report 1](https://utrgv-my.sharepoint.com/:w:/g/personal/david_sanchez15_utrgv_edu/EY7EzboliIlMo1sxI2Nu-7MBzzuZG3rIZXGCj8G6Oijv8A?e=8NuHhH)

---


## License

For academic use — CSCI 4341 Foundations of Intelligent Security Systems (Fall 2025).

