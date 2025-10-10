setup:
	pip install -r requirements.txt

dry-run:
	PYTHONPATH=. python experiments/dry_run/dry_run_baseline.py --clients 10 --rounds 5 --out experiments/results/baseline.csv

plots:
	PYTHONPATH=. python experiments/dry_run/plot_metrics.py --in experiments/results/baseline.csv --out experiments/plots/

clean:
	rm -f experiments/results/*.csv experiments/plots/*.{png,jpg,svg} 2>/dev/null || true
