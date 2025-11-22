setup:
	pip install -r requirements.txt

dry-run:
	PYTHONPATH=. python experiments/baseline_fl.py --clients 5 --rounds 3 --out experiments/results/baseline.csv
	PYTHONPATH=. python experiments/plot_metrics.py --in experiments/results/baseline.csv --out experiments/plots/

plots:
	PYTHONPATH=. python experiments/plot_metrics.py --in experiments/results/baseline.csv --out experiments/plots/

clean:
	rm -f experiments/results/*.csv experiments/plots/*.{png,jpg,svg} 2>/dev/null || true
