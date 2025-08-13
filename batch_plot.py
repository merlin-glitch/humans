
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

def load_runs_from_combined_csv(path):
    """
    Loads data from a combined CSV of all runs:
    run, day, blue_pop, red_pop, within_trust, between_trust
    Returns:
        - blue_runs[run_id]  = [(day, pop), ...]
        - red_runs[run_id]   = [(day, pop), ...]
        - within_runs[run_id] = [(day, trust), ...]
        - between_runs[run_id] = [(day, trust), ...]
    """
    blue_runs    = defaultdict(list)
    red_runs     = defaultdict(list)
    within_runs  = defaultdict(list)
    between_runs = defaultdict(list)

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = int(row["run"])
            day = int(row["day"])
            blue = float(row["blue_pop"])
            red  = float(row["red_pop"])
            wt   = float(row["within_trust"])
            bt   = float(row["between_trust"])

            blue_runs[run].append((day, blue))
            red_runs[run].append((day, red))
            within_runs[run].append((day, wt))
            between_runs[run].append((day, bt))

    # Sort values by day
    for runs in [blue_runs, red_runs, within_runs, between_runs]:
        for run in runs:
            runs[run].sort(key=lambda x: x[0])

    return blue_runs, red_runs, within_runs, between_runs


def plot_all_runs(path):
    blue_runs, red_runs, within_runs, between_runs = load_runs_from_combined_csv(path)

        # --- Population Plot ---
    plt.figure(figsize=(10, 4))
    for run, series in blue_runs.items():
        days, values = zip(*series)
        plt.plot(days, values, color='blue', alpha=0.2)
    for run, series in red_runs.items():
        days, values = zip(*series)
        plt.plot(days, values, color='red', alpha=0.2)

    # FIXED AVERAGING
    days_sorted = sorted({d for run in blue_runs for d, _ in blue_runs[run]})
    avg_blue = []
    avg_red  = []
    for d in days_sorted:
        blue_vals = [v for run in blue_runs for day, v in blue_runs[run] if day == d]
        red_vals  = [v for run in red_runs  for day, v in red_runs[run]  if day == d]
        avg_blue.append(sum(blue_vals) / len(blue_vals) if blue_vals else 0)
        avg_red.append(sum(red_vals) / len(red_vals) if red_vals else 0)


    plt.plot(days_sorted, avg_blue, label="Avg Blue", color="darkblue", linewidth=2)
    plt.plot(days_sorted, avg_red, label="Avg Red", color="darkred", linewidth=2)
    plt.title("Population per run + average")
    plt.xlabel("Day")
    plt.ylabel("Alive population")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Trust Plot ---
    plt.figure(figsize=(10, 4))
    for run, series in within_runs.items():
        days, values = zip(*series)
        plt.plot(days, values, color='green', alpha=0.2)
    for run, series in between_runs.items():
        days, values = zip(*series)
        plt.plot(days, values, color='orange', alpha=0.2)

    avg_within = []
    avg_between = []

    for d in days_sorted:
        w_vals = [v for run in within_runs for day, v in within_runs[run] if day == d]
        b_vals = [v for run in between_runs for day, v in between_runs[run] if day == d]
        
        avg_within.append(sum(w_vals) / len(w_vals) if w_vals else 0)
        avg_between.append(sum(b_vals) / len(b_vals) if b_vals else 0)


    plt.plot(days_sorted, avg_within, label="Avg Within-house", color="darkgreen", linewidth=2)
    plt.plot(days_sorted, avg_between, label="Avg Between-house", color="darkorange", linewidth=2)
    plt.title("Trust variation per run + average")
    plt.xlabel("Day")
    plt.ylabel("Trust level")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    path = "batch_results/all_runs_combined.csv"
    if not os.path.exists(path):
        raise RuntimeError(f"Missing combined CSV at: {path}")
    plot_all_runs(path)

if __name__ == "__main__":
    main()
