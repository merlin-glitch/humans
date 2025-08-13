

import os
import csv
import numpy as np
import pandas as pd
from tqdm import trange
from simulation import run_simulation

# Configuration
N_RUNS = 2
DAYS = 5
RESULTS_DIR = "batch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Combined CSV file
COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")

# Write header once
with open(COMBINED_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["run", "day", "blue_pop", "red_pop", "within_trust", "between_trust"])

print(f"Running {N_RUNS} headless simulations for {DAYS} days each...\n")

for run_idx in trange(N_RUNS, desc="Simulations", unit="run"):
    days, blue, red, within, between, food_spawned, food_picked = run_simulation(DAYS, seed=run_idx)
    # Save individual file (optional)
    df = pd.DataFrame({
        "day": days,
        "blue_pop": blue,
        "red_pop": red,
        "within_trust": within,
        "between_trust": between,
        "food_spawned": food_spawned,
        "food_picked": food_picked
    })

    df.to_csv(f"{RESULTS_DIR}/run_{run_idx:03}.csv", index=False)

    # Append to combined CSV
    with open(COMBINED_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        for d, bp, rp, wt, bt in zip(days, blue, red, within, between):
            writer.writerow([run_idx, d, bp, rp, wt, bt])

print(f"\n All {N_RUNS} simulations completed.")
print(f" Combined results written to: {COMBINED_CSV}")
