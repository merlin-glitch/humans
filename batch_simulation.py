

"""
batch_simulation.py - Batch simulation runner for multi-run experiments

Part of the Human Society Simulation project.

Executes N independent simulation runs with different seeds and
aggregates results into a single CSV for statistical analysis.
"""
import os
import csv
from typing import List
from tqdm import tqdm

from headless_simulation import simulate_headless as simulate  # unified simulator

# ── Config ───────────────────────────────────────────────────────────────
N_RUNS = 10   # Reduced for faster testing
DAYS   = 200  # Reduced for faster testing

SEED_BASE = 244          # per-run seed = SEED_BASE + run_idx
SEEDS     = None         # or provide a list of explicit seeds (length >= N_RUNS)

RESULTS_DIR  = "batch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
COMBINED_CSV = os.path.join(RESULTS_DIR, "all_244_combined_sim_function.csv")

# Absolute path to the map image (fixes cv2.imread None error)
MAP_PATH = os.path.join(os.path.dirname(__file__), "images", "3_spots.png")

# ── Probe once to learn zone count & return shape ────────────────────────
probe = simulate(num_days=1, seed=0, map_path=MAP_PATH, min_size=1, tol=20)

# simulate(...) headless return (per sim_function.py) may be:
#  - 10-tuple: (days, blue, red, within_blue, within_red, between, zone_spawned_daily, zone_consumed_daily, zone_blue_consumed_daily, zone_red_consumed_daily)
#  - 8-tuple: (days, blue, red, within_blue, within_red, between, zone_spawned_daily, zone_consumed_daily)
#  - 6-tuple: (days, blue, red, within_blue, within_red, between)
if isinstance(probe, tuple) and len(probe) >= 8:
    if len(probe) == 10:
        _, _, _, _, _, _, probe_spawned, _, _, _ = probe
    else:
        _, _, _, _, _, _, probe_spawned, _ = probe
    N_ZONES = len(probe_spawned)
else:
    N_ZONES = 0

# ── CSV header ───────────────────────────────────────────────────────────
HEADER: List[str] = [
    "run", "day",
    "blue_pop", "red_pop",
    "within_blue_trust", "within_red_trust", "between_trust",
    "total_spawned", "total_consumed",
]
for z in range(N_ZONES):
    HEADER.append(f"z{z}_spawn")
for z in range(N_ZONES):
    HEADER.append(f"z{z}_cons")
for z in range(N_ZONES):
    HEADER.append(f"z{z}_blue_cons")
for z in range(N_ZONES):
    HEADER.append(f"z{z}_red_cons")

with open(COMBINED_CSV, "w", newline="") as f:
    csv.writer(f).writerow(HEADER)

print(f"Running {N_RUNS} headless simulations for {DAYS} days each...\n")

def get_seed_for_run(run_idx: int) -> int:
    if SEEDS is not None:
        if len(SEEDS) < N_RUNS:
            raise ValueError(f"SEEDS has length {len(SEEDS)} but N_RUNS is {N_RUNS}.")
        return int(SEEDS[run_idx])
    return int(SEED_BASE) + int(run_idx)

def zeros(n: int): 
    return [0] * n

def sums_by_day(per_zone_lists, n_days):
    if not per_zone_lists:
        return [0] * n_days
    out = []
    for d in range(n_days):
        s = 0
        for z in range(len(per_zone_lists)):
            lst = per_zone_lists[z]
            if d < len(lst):
                s += lst[d]
        out.append(s)
    return out


# ── Runs ─────────────────────────────────────────────────────────────────
pbar = tqdm(range(N_RUNS), desc="Simulations", unit="run")
for run_idx in pbar:
    seed_val = get_seed_for_run(run_idx)

    # Run one headless sim
    result = simulate(
        num_days=DAYS, 
        seed=seed_val, 
        map_path=MAP_PATH,
        min_size=1, 
        tol=20
    )

    # Unpack both shapes robustly
    if isinstance(result, tuple) and len(result) == 10:
        days, blue, red, within_blue, within_red, between, zone_spawned_daily, zone_consumed_daily, zone_blue_consumed_daily, zone_red_consumed_daily = result
    elif isinstance(result, tuple) and len(result) == 8:
        days, blue, red, within_blue, within_red, between, zone_spawned_daily, zone_consumed_daily = result
        zone_blue_consumed_daily, zone_red_consumed_daily = [], []
    else:
        days, blue, red, within_blue, within_red, between = result
        zone_spawned_daily, zone_consumed_daily = [], []
        zone_blue_consumed_daily, zone_red_consumed_daily = [], []

    n_days = len(days)

    # Totals derived from per-zone series (or zeros if no per-zone data)
    total_spawned  = sums_by_day(zone_spawned_daily,  n_days) if N_ZONES else zeros(n_days)
    total_consumed = sums_by_day(zone_consumed_daily, n_days) if N_ZONES else zeros(n_days)

    # Update progress bar
    pbar.set_postfix({
        "seed": seed_val,
        "final_blue": blue[-1] if blue else 0,
        "final_red":  red[-1] if red  else 0,
    })
    pbar.write(
        f"Run {run_idx+1}/{N_RUNS} finished "
        f"(Seed={seed_val}, Blue={blue[-1] if blue else 0}, "
        f"Red={red[-1] if red else 0}, LastDay={days[-1] if days else 'NA'})"
    )

    # Write rows (optimized: batch all rows for this run)
    rows_to_write = []
    for i, d in enumerate(days):
        row = [
            run_idx, d,
            blue[i], red[i],
            within_blue[i], within_red[i], between[i],
            total_spawned[i], total_consumed[i],
        ]
        # per-zone spawn/cons if present
        for z in range(N_ZONES):
            zs = zone_spawned_daily[z] if z < len(zone_spawned_daily) else []
            row.append(zs[i] if i < len(zs) else 0)
        for z in range(N_ZONES):
            zc = zone_consumed_daily[z] if z < len(zone_consumed_daily) else []
            row.append(zc[i] if i < len(zc) else 0)
        # per-family consumption per zone
        for z in range(N_ZONES):
            zbc = zone_blue_consumed_daily[z] if z < len(zone_blue_consumed_daily) else []
            row.append(zbc[i] if i < len(zbc) else 0)
        for z in range(N_ZONES):
            zrc = zone_red_consumed_daily[z] if z < len(zone_red_consumed_daily) else []
            row.append(zrc[i] if i < len(zrc) else 0)
        rows_to_write.append(row)
    
    # Single file write for all rows of this run
    with open(COMBINED_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows_to_write)

print(f"\nAll {N_RUNS} simulations completed.")
print(f"Combined results written to: {COMBINED_CSV}")
