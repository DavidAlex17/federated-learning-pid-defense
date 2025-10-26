# Summary: Federated Learning Research Project (Instructor-Based)

## Objective
Explore Federated Learning (FL) as a privacy-preserving ML paradigm by:
- Implementing a conventional FL setup (baseline) and evaluating robustness under data poisoning.
- Implementing an FL variant with a PID-based anomaly filter during aggregation.
- Comparing performance/robustness and discussing results.

## Topics to cover
- Literature review on FL vs centralized ML, aggregation methods (e.g., FedAvg), privacy/security, and open challenges.
- Hands-on implementation of FL baseline and FL+PID.
- Empirical evaluation with metrics, plots, and analysis.

## Learning outcomes
- Understand FL principles/architecture and key security/privacy aspects.
- Gain fluency in basic ML/FL techniques and experimental design.
- Design and implement an end-to-end research prototype with analysis and reporting.

## Project workflow (high level)
1. Learn FL principles and review literature/patents.
2. Select and prepare a dataset; simulate federated partitions if needed.
3. Build initial model and implement FL baseline (≥10 clients).
4. Evaluate without attack (baseline) and with data poisoning.
5. Implement and evaluate FL with PID-based client exclusion.
6. Save metrics, generate plots, and discuss results.

---

## Milestone 1: Research and Data Preparation

### A. Bibliography and patent review (pick ≥2 topics)
- FL features: aggregation methods (e.g., FedAvg), advances; FL vs centralized ML in privacy/security/scalability.
- Privacy/security in FL: differential privacy, secure MPC, homomorphic encryption; vulnerabilities (model/data poisoning, inference attacks) with examples.
- Challenges/future work: comms efficiency, heterogeneity, metacognition, labeling scarcity; future directions.

### B. Dataset options
- Option 1: Instructor-provided FEMNIST subset (IID across clients).
- Option 2: MedMNIST.
- Option 3: Another dataset (see Attachment 2 for common FL datasets).

#### Data preparation strategy (as applicable)
- Cleaning (deduplication, noise reduction).
- Normalization (e.g., [0,1]).
- Augmentation (rotations/translations/zoom/flip).
- Partitioning: simulate clients (e.g., by writer), maintain non-IID distribution.
- Labeling consistency (for Milestone 2).
- Data corruption: introduce label corruption on selected clients (for Milestone 3).

### C. Initial model
- Framework suggestion: Flower + PyTorch (also OK: TFF, PySyft, etc.).
- Options:
  - From scratch: define architecture, optionally pretrain centrally; otherwise random init.
  - Pre-trained: adapt head to task; optional fine-tuning.
- Distribute same initial model to all clients.
- Considerations: model complexity vs edge constraints, generalization to non-IID, communication cost.
- Example (FEMNIST): small CNN; authors provide a reference model (LEAF FEMNIST CNN).

---

## Milestone 2: Conventional FL (baseline)

### FL overview
- Server initializes global model; sends to selected clients.
- Clients train locally for a few steps; return updates.
- Server aggregates (typically FedAvg) and repeats until convergence.
- Advantages: privacy via keeping data local; scalable across institutions/devices.

### Threats considered
- Data poisoning (this project focuses on it).
- Model poisoning (not the main focus here).

### Requirements
- Use ≥10 clients.
- Recommended framework: Flower (collect per-client metrics each round; allow custom aggregation later for PID).
- Save metrics each round to CSV (or similar).

### Metrics to collect each round
- Loss per client and average loss across clients.
- Evaluation accuracy per client and average accuracy across clients.

### Evaluation and plots
- Without attack (baseline): expect accuracy ↑ and loss ↓ over rounds.
  - Plots: per-client loss; avg loss; per-client accuracy; avg accuracy.
- With data poisoning: poisoned clients’ accuracy should stagnate; convergence should degrade.
  - Same 4 plots; ensure you can distinguish benign vs poisoned clients.
  - If not distinguishable, check plotting, poisoning effectiveness (create poisoned data on disk), and evaluation correctness.

### Deliverables (Milestone 2)
- Working FL simulation, metrics CSVs, 4 plots (no attack) + 4 plots (with attack).

---

## Milestone 3: FL with PID-based exclusion

### PID concept (adapted)
Compute a per-client anomaly score based on distance of client weights to the centroid across rounds:

u(t) = Kp·D(t, μ) + Ki·Σ D(i, μ) + Kd·(D(t, μ) − D(t−1, μ))

- D(t): Euclidean distance of a client's weights to centroid μ at round t.
- Intuition: large/abnormal distances (current, cumulative, change) signal anomalous clients.

#### Suggested coefficients (starting point, tune empirically)
- Kp = 1, Ki = 0.05, Kd = 0.5 (FEMNIST subset).

### Algorithm
- Each round: receive client weights; compute PID scores; exclude clients whose PID exceeds the threshold; aggregate remaining (e.g., FedAvg); distribute model.
- Note: Exclusion condition should be “PID > threshold”.

### What to show
- Same 4 plots as baseline-with-attack, plus “client removal dynamics” (who got excluded when).
- Discussion should show improved robustness vs baseline (e.g., faster avg loss decrease after excluding malicious clients).

---

## Attachments (for background and datasets)

- Attachment 1: Background on FL, privacy risks (e.g., gradient leakage, membership inference), and defense directions (robust aggregation, DP, secure aggregation, etc.).
- Attachment 2: Common FL datasets (LEAF, Google FL datasets including FEMNIST/StackOverflow, medical datasets, federated CIFAR, Apple FLAIR), with typical use cases.

---

## Deliverables checklist
- Literature/patent mini-review covering ≥2 topics.
- Dataset choice and data-prep strategy (doc + code).
- Initial model choice and rationale.
- FL baseline implementation (≥10 clients) with:
  - Round-wise CSV metrics: per-client loss/accuracy and averages.
  - 4 baseline plots (no attack).
  - Data poisoning implementation and 4 plots (with attack).
- FL+PID implementation:
  - PID score computation, thresholding, client exclusion.
  - Same 4 plots (with attack) + client removal dynamics visualization.
- Comparative discussion: baseline vs PID (avg loss/accuracy over rounds on the same axes).
