# IntelliMAD: Domain-Agnostic Framework for Model Anomaly Detection (MAD) Benchmarking in FL — Summary

## What problem it tackles
Federated Learning (FL) is moving from theory to production, where noisy, imperfect, and shifting data are the norm. Existing FL toolchains focus on standard model training and curated datasets, making it hard to:
- Benchmark Model Anomaly Detection (MAD) algorithms under realistic, domain-specific conditions
- Configure and run batch experiments with runtime data quality changes
- Visualize results and derive robustness insights to tune deployments

## Core contribution
IntelliMAD is a domain-agnostic framework that extends existing FL stacks to benchmark and fine-tune MAD algorithms:
- Batch experiment configuration and scheduling via a central config
- Runtime data quality variation (e.g., data poisoning injected during runs)
- FL performance visualization (loss/accuracy histories) and MAD-specific metrics
- Composite robustness scoring to compare aggregation strategies and MAD settings
- Extensible to new aggregation strategies, modalities, and domains

A pilot usability study suggests IntelliMAD reduces setup time for FL experiments compared to using a vanilla framework directly (e.g., Flower).

## Key capabilities
- MAD-enabled aggregations alongside conventional aggregations
- Parameter search for MAD/aggregation to discover deployment-ready settings
- Per-client and aggregate histories (loss/accuracy); MAD score histories
- Robustness knowledge capture (what settings worked best under which conditions)

## Where it fits in your course project
- Quickly create reproducible baselines and “what-if” runs (clean vs. corrupted data)
- Compare conventional FL vs. FL+MAD settings across datasets
- Produce ready-to-use plots and composite robustness scores for your report

## Notable details from the paper
- Emphasizes bridging the gap between curated-dataset demos and real-world FL
- Highlights lack of unified tooling for MAD benchmarking with visualization and planning
- Shows a logical architecture with config-driven experiments and FL/MAD metrics dashboards

## Practical next steps
- Use IntelliMAD to: (1) define client datasets and runtime attacks; (2) enable MAD and tune its params; (3) export per-round metrics and composite robustness scores
- If you choose Flower, IntelIiMAD can layer on top to accelerate configuration, tracking, and visualization
