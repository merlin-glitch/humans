# batch_simul.py
import os
import csv
import random
import inspect
import numpy as np
from tqdm import trange
from simulation import run_simulation 

# Config
N_RUNS = 1
DAYS = 400
RESULTS_DIR = "batch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")

# Extended header
HEADER = [
    "run", "day",
    "blue_pop", "red_pop", "within_trust", "between_trust",
    "within_blue_trust", "within_red_trust",
    "total_spawned", "total_consumed",
    "z0_spawn", "z1_spawn", "z2_spawn",
    "z0_cons",  "z1_cons",  "z2_cons",
    "z0_blue_cons", "z0_red_cons",
    "z1_blue_cons", "z1_red_cons",
    "z2_blue_cons", "z2_red_cons",
    "blue_dead", "red_dead",
    "blue_born", "red_born",
]

# Start fresh combined CSV
with open(COMBINED_CSV, "w", newline="") as f:
    csv.writer(f).writerow(HEADER)

print(f"Running {N_RUNS} headless simulations for {DAYS} days each...\n")

# Figure out which kwargs your run_simulation actually accepts
sig_params = set(inspect.signature(run_simulation).parameters)

def call_run_simulation(days, seed_val, want_zone=True):
    """Call run_simulation with only supported kwargs."""
    base_kwargs = {
        "num_days": days,
        "return_zone_series": want_zone,
        "progress": False,
        "seed": seed_val,
    }
    filtered = {k: v for k, v in base_kwargs.items() if k in sig_params}
    return run_simulation(**filtered)

for run_idx in trange(N_RUNS, desc="Simulations", unit="run"):
    # Make seeding reproducible whether or not run_simulation accepts 'seed'
    if "seed" in sig_params:
        result = call_run_simulation(DAYS, run_idx, want_zone=True)
    else:
        random.seed(run_idx); np.random.seed(run_idx)
        result = call_run_simulation(DAYS, None, want_zone=True)

    # Basic 5
    if len(result) < 5:
        raise ValueError("run_simulation returned too few values.")
    
    days, blue, red, within, between = result[:5]

    # Init holders
    total_spawned = []
    total_consumed = []
    zone_spawned_daily = []
    zone_consumed_daily = []
    zone_consumed_by_house = None  # [zone][house(0/1)][day]

    within_blue_list = []
    within_red_list  = []

    dead_blue_cum   = []
    dead_red_cum    = []
    born_blue_daily = []
    born_red_daily  = []

    # Next 4
    if len(result) >= 9:
        total_spawned, total_consumed, zone_spawned_daily, zone_consumed_daily = result[5:9]
    # Optional per-house zone consumption
    if len(result) >= 10:
        zone_consumed_by_house = result[9]
    # Within-house trust (blue, red)
    if len(result) >= 12:
        within_blue_list, within_red_list = result[10], result[11]
    # Deaths/Births series
    if len(result) >= 16:
        dead_blue_cum, dead_red_cum, born_blue_daily, born_red_daily = result[12:16]

    # Defaults (align lengths to days)
    n = len(days)
    def _zeros(n): return [0]*n
    def _zeros3(n): return [[0]*n for _ in range(3)]

    if not total_spawned: total_spawned = _zeros(n)
    if not total_consumed: total_consumed = _zeros(n)
    if not zone_spawned_daily: zone_spawned_daily = _zeros3(n)
    if not zone_consumed_daily: zone_consumed_daily = _zeros3(n)
    if not within_blue_list: within_blue_list = _zeros(n)
    if not within_red_list:  within_red_list  = _zeros(n)
    if not dead_blue_cum:    dead_blue_cum    = _zeros(n)
    if not dead_red_cum:     dead_red_cum     = _zeros(n)
    if not born_blue_daily:  born_blue_daily  = _zeros(n)
    if not born_red_daily:   born_red_daily   = _zeros(n)

    if zone_consumed_by_house is None:
        zone_consumed_by_house = [
            [_zeros(n), _zeros(n)],
            [_zeros(n), _zeros(n)],
            [_zeros(n), _zeros(n)],
        ]

    # Append one row per day into the combined CSV
    with open(COMBINED_CSV, "a", newline="") as f:
        w = csv.writer(f)
        for i, d in enumerate(days):
            row = [
                run_idx, d,
                blue[i], red[i], within[i], between[i],
                within_blue_list[i], within_red_list[i],
                total_spawned[i], total_consumed[i],
                zone_spawned_daily[0][i], zone_spawned_daily[1][i], zone_spawned_daily[2][i],
                zone_consumed_daily[0][i], zone_consumed_daily[1][i], zone_consumed_daily[2][i],
                zone_consumed_by_house[0][0][i], zone_consumed_by_house[0][1][i],
                zone_consumed_by_house[1][0][i], zone_consumed_by_house[1][1][i],
                zone_consumed_by_house[2][0][i], zone_consumed_by_house[2][1][i],
                dead_blue_cum[i], dead_red_cum[i],
                born_blue_daily[i], born_red_daily[i],
            ]
            w.writerow(row)

print(f"\nAll {N_RUNS} simulations completed.")
print(f"Combined results written to: {COMBINED_CSV}")
