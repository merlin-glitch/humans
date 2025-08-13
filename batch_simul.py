
# import csv
# import os
# from tqdm import trange
# import matplotlib.pyplot as plt
# import pandas as pd

# from simulation import run_simulation  # your headless runner

# def main():
#     NUM_RUNS = 50
#     DAYS     = 10

#     combined_csv = "all_runs_summary.csv"
#     # --- 1) Write header and clear old file ---
#     with open(combined_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "run",
#             "day",
#             "blue_population",
#             "red_population",
#             "within_family_trust",
#             "between_family_trust"
#         ])

#     # --- 2) Run batch simulations ---
#     for run_id in trange(1, NUM_RUNS+1, desc="Batch runs"):
#         days, blue_pop, red_pop, within_trust, between_trust = run_simulation(DAYS)
#         with open(combined_csv, "a", newline="") as f:
#             writer = csv.writer(f)
#             for d, b, r, w, bt in zip(days, blue_pop, red_pop, within_trust, between_trust):
#                 writer.writerow([run_id, d, b, r, w, bt])

#     print(f"\nAll {NUM_RUNS} runs complete. Data in {combined_csv}")

#     # --- 3) Plot trust variations ---
#     plot_trust_variation(combined_csv)

#     # --- 4) Plot population variations ---
#     plot_population_variation(combined_csv)


# def plot_trust_variation(csv_path: str):
#     """
#     Produces three PNGs:
#       – within_trust_per_run.png
#       – between_trust_per_run.png
#       – average_trust.png
#     """
#     df = pd.read_csv(csv_path)

#     # A) Within‐family trust, one curve per run
#     plt.figure(figsize=(8, 4))
#     for run_id, grp in df.groupby("run"):
#         plt.plot(grp["day"], grp["within_family_trust"],
#                  color="blue", alpha=0.3)
#     plt.xlabel("Day")
#     plt.ylabel("Within‐family trust")
#     plt.title("Within‐family trust over time (per run)")
#     plt.tight_layout()
#     plt.savefig("within_trust_per_run.png")
#     plt.close()

#     # B) Between‐family trust, one curve per run
#     plt.figure(figsize=(8, 4))
#     for run_id, grp in df.groupby("run"):
#         plt.plot(grp["day"], grp["between_family_trust"],
#                  color="red", alpha=0.3)
#     plt.xlabel("Day")
#     plt.ylabel("Between‐family trust")
#     plt.title("Between‐family trust over time (per run)")
#     plt.tight_layout()
#     plt.savefig("between_trust_per_run.png")
#     plt.close()

#     # C) Average trust across runs
#     avg = df.groupby("day")[
#         ["within_family_trust", "between_family_trust"]
#     ].mean().reset_index()

#     plt.figure(figsize=(8, 4))
#     plt.plot(avg["day"], avg["within_family_trust"], label="Within", color="blue", linewidth=2)
#     plt.plot(avg["day"], avg["between_family_trust"], label="Between", color="red",  linewidth=2)
#     plt.xlabel("Day")
#     plt.ylabel("Average trust")
#     plt.title("Average trust over time")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("average_trust.png")
#     plt.close()


# def plot_population_variation(csv_path: str):
#     """
#     Produces two PNGs:
#       – blue_population_per_run.png
#       – red_population_per_run.png

#     Each shows one thin curve per run plus a thick average curve.
#     """
#     df = pd.read_csv(csv_path)

#     # --- Blue population ---
#     plt.figure(figsize=(8, 4))
#     # per-run
#     for run_id, grp in df.groupby("run"):
#         plt.plot(grp["day"], grp["blue_population"],
#                  color="blue", alpha=0.3, linewidth=1)
#     # average
#     avg_blue = df.groupby("day")["blue_population"].mean().reset_index()
#     plt.plot(avg_blue["day"], avg_blue["blue_population"],
#              color="blue", linewidth=3, label="Average")
#     plt.xlabel("Day")
#     plt.ylabel("Alive blue population")
#     plt.title("Blue population over time")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("blue_population_per_run.png")
#     plt.close()

#     # --- Red population ---
#     plt.figure(figsize=(8, 4))
#     for run_id, grp in df.groupby("run"):
#         plt.plot(grp["day"], grp["red_population"],
#                  color="red", alpha=0.3, linewidth=1)
#     avg_red = df.groupby("day")["red_population"].mean().reset_index()
#     plt.plot(avg_red["day"], avg_red["red_population"],
#              color="red", linewidth=3, label="Average")
#     plt.xlabel("Day")
#     plt.ylabel("Alive red population")
#     plt.title("Red population over time")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("red_population_per_run.png")
#     plt.close()


# if __name__ == "__main__":
#     main()
# batch_simul.py
import os
import csv
import random
import inspect
import numpy as np
import pandas as pd
from tqdm import trange
from simulation import run_simulation  # your headless runner

# Configuration
N_RUNS = 100
DAYS = 100
RESULTS_DIR = "batch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Combined CSV file
COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")

# Write header once
with open(COMBINED_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["run", "day", "blue_pop", "red_pop", "within_trust", "between_trust"])

print(f"Running {N_RUNS} headless simulations for {DAYS} days each...\n")

# Check if run_simulation supports a 'seed' kwarg
try:
    sig = inspect.signature(run_simulation)
    accepts_seed = "seed" in sig.parameters
except Exception:
    accepts_seed = False

for run_idx in trange(N_RUNS, desc="Simulations", unit="run"):
    # Reproducible seeding either via kwarg or by seeding RNGs directly
    if accepts_seed:
        result = run_simulation(DAYS, seed=run_idx)
    else:
        random.seed(run_idx)
        np.random.seed(run_idx)
        result = run_simulation(DAYS)

    # Unpack: support both 5-return and 7-return variants
    if len(result) == 5:
        days, blue, red, within, between = result
        food_spawned = food_picked = None
    elif len(result) == 7:
        days, blue, red, within, between, food_spawned, food_picked = result
    else:
        raise ValueError("run_simulation returned an unexpected number of values.")

    # Save per-run CSV
    per_run = {
        "day": days,
        "blue_pop": blue,
        "red_pop": red,
        "within_trust": within,
        "between_trust": between,
    }
    if food_spawned is not None and food_picked is not None:
        per_run["food_spawned"] = food_spawned
        per_run["food_picked"] = food_picked

    df = pd.DataFrame(per_run)
    df.to_csv(os.path.join(RESULTS_DIR, f"run_{run_idx:03}.csv"), index=False)

    # Append to combined CSV (5 core series only)
    with open(COMBINED_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        for d, bp, rp, wt, bt in zip(days, blue, red, within, between):
            writer.writerow([run_idx, d, bp, rp, wt, bt])

print(f"\nAll {N_RUNS} simulations completed.")
print(f"Combined results written to: {COMBINED_CSV}")
