

# # batch_simul.py
# import os
# import csv
# import random
# import numpy as np
# from tqdm import trange
# from simulation import run_simulation  # use directly

# # ── Config ───────────────────────────────────────────────────────────────
# N_RUNS = 10
# DAYS   = 100

# # Seeding:
# SEED_BASE = 42   # per-run seed = SEED_BASE + run_idx
# SEEDS     = None    # or e.g. [101, 202, 303] to set each run explicitly

# RESULTS_DIR  = "batch_results"
# os.makedirs(RESULTS_DIR, exist_ok=True)
# COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")

# # Extended header
# HEADER = [
#     "run", "day",
#     "blue_pop", "red_pop", "within_trust", "between_trust",
#     "within_blue_trust", "within_red_trust",
#     "total_spawned", "total_consumed",
#     "z0_spawn", "z1_spawn", "z2_spawn",
#     "z0_cons",  "z1_cons",  "z2_cons",
#     "z0_blue_cons", "z0_red_cons",
#     "z1_blue_cons", "z1_red_cons",
#     "z2_blue_cons", "z2_red_cons",
#     "blue_dead", "red_dead",
#     "blue_born", "red_born",
# ]

# # fresh csv
# with open(COMBINED_CSV, "w", newline="") as f:
#     csv.writer(f).writerow(HEADER)

# print(f"Running {N_RUNS} headless simulations for {DAYS} days each...\n")

# def get_seed_for_run(run_idx: int) -> int:
#     if SEEDS is not None:
#         if len(SEEDS) < N_RUNS:
#             raise ValueError(f"SEEDS has length {len(SEEDS)} but N_RUNS is {N_RUNS}.")
#         return int(SEEDS[run_idx])
#     if SEED_BASE is not None:
#         return int(SEED_BASE) + int(run_idx)
#     return int(run_idx)

# for run_idx in trange(N_RUNS, desc="Simulations", unit="run"):
#     seed_val = get_seed_for_run(run_idx)
#     # Make RNG deterministic for this run
#     random.seed(seed_val)
#     np.random.seed(seed_val)

#     # Call run_simulation directly
#     result = run_simulation(
#         num_days=DAYS,
#         return_zone_series=True,
#         progress=False,
#         seed=seed_val,
#     )

#     # Basic 5
#     if len(result) < 5:
#         raise ValueError("run_simulation returned too few values.")
#     days, blue, red, within, between = result[:5]

#     # Optional parts
#     total_spawned = []
#     total_consumed = []
#     zone_spawned_daily = []
#     zone_consumed_daily = []
#     zone_consumed_by_house = None  # [zone][house(0/1)][day]
#     within_blue_list = []
#     within_red_list  = []
#     dead_blue_cum   = []
#     dead_red_cum    = []
#     born_blue_daily = []
#     born_red_daily  = []

#     if len(result) >= 9:
#         total_spawned, total_consumed, zone_spawned_daily, zone_consumed_daily = result[5:9]
#     if len(result) >= 10:
#         zone_consumed_by_house = result[9]
#     if len(result) >= 12:
#         within_blue_list, within_red_list = result[10], result[11]
#     if len(result) >= 16:
#         dead_blue_cum, dead_red_cum, born_blue_daily, born_red_daily = result[12:16]

#     # Defaults to align with day count
#     n = len(days)
#     def _zeros(n):  return [0]*n
#     def _zeros3(n): return [[0]*n for _ in range(3)]

#     if not total_spawned: total_spawned = _zeros(n)
#     if not total_consumed: total_consumed = _zeros(n)
#     if not zone_spawned_daily: zone_spawned_daily = _zeros3(n)
#     if not zone_consumed_daily: zone_consumed_daily = _zeros3(n)
#     if not within_blue_list: within_blue_list = _zeros(n)
#     if not within_red_list:  within_red_list  = _zeros(n)
#     if not dead_blue_cum:    dead_blue_cum    = _zeros(n)
#     if not dead_red_cum:     dead_red_cum     = _zeros(n)
#     if not born_blue_daily:  born_blue_daily  = _zeros(n)
#     if not born_red_daily:   born_red_daily   = _zeros(n)

#     if zone_consumed_by_house is None:
#         zone_consumed_by_house = [
#             [_zeros(n), _zeros(n)],
#             [_zeros(n), _zeros(n)],
#             [_zeros(n), _zeros(n)],
#         ]

#     # Write one row/day
#     with open(COMBINED_CSV, "a", newline="") as f:
#         w = csv.writer(f)
#         for i, d in enumerate(days):
#             w.writerow([
#                 run_idx, d,
#                 blue[i], red[i], within[i], between[i],
#                 within_blue_list[i], within_red_list[i],
#                 total_spawned[i], total_consumed[i],
#                 zone_spawned_daily[0][i], zone_spawned_daily[1][i], zone_spawned_daily[2][i],
#                 zone_consumed_daily[0][i], zone_consumed_daily[1][i], zone_consumed_daily[2][i],
#                 zone_consumed_by_house[0][0][i], zone_consumed_by_house[0][1][i],
#                 zone_consumed_by_house[1][0][i], zone_consumed_by_house[1][1][i],
#                 zone_consumed_by_house[2][0][i], zone_consumed_by_house[2][1][i],
#                 dead_blue_cum[i], dead_red_cum[i],
#                 born_blue_daily[i], born_red_daily[i],
#             ])

# print(f"\nAll {N_RUNS} simulations completed.")
# print(f"Combined results written to: {COMBINED_CSV}")


# batch_simul.py
import os
import csv
import random
import numpy as np
from tqdm import trange
from simulation import run_simulation  # use directly

# ── Config ───────────────────────────────────────────────────────────────
N_RUNS = 5
DAYS   = 100

SEED_BASE = 42   # per-run seed = SEED_BASE + run_idx
SEEDS     = None

RESULTS_DIR  = "batch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")

# ── Run once to detect zone count ────────────────────────────────────────
probe_result = run_simulation(num_days=1, return_zone_series=True, progress=False, seed=0)
zone_spawned_daily = probe_result[7]
N_ZONES = len(zone_spawned_daily)

# ── Build dynamic header ────────────────────────────────────────────────
HEADER = [
    "run", "day",
    "blue_pop", "red_pop", "within_trust", "between_trust",
    "within_blue_trust", "within_red_trust",
    "total_spawned", "total_consumed",
]

# Per-zone spawns/cons
for z in range(N_ZONES):
    HEADER.append(f"z{z}_spawn")
for z in range(N_ZONES):
    HEADER.append(f"z{z}_cons")

# Per-zone, per-house consumption
for z in range(N_ZONES):
    HEADER.append(f"z{z}_blue_cons")
    HEADER.append(f"z{z}_red_cons")

# Birth/death
HEADER += ["blue_dead", "red_dead", "blue_born", "red_born"]

# Fresh CSV
with open(COMBINED_CSV, "w", newline="") as f:
    csv.writer(f).writerow(HEADER)

print(f"Running {N_RUNS} headless simulations for {DAYS} days each...\n")

# ── Helpers ─────────────────────────────────────────────────────────────
def get_seed_for_run(run_idx: int) -> int:
    if SEEDS is not None:
        if len(SEEDS) < N_RUNS:
            raise ValueError(f"SEEDS has length {len(SEEDS)} but N_RUNS is {N_RUNS}.")
        return int(SEEDS[run_idx])
    if SEED_BASE is not None:
        return int(SEED_BASE) + int(run_idx)
    return int(run_idx)

def zeros(n): return [0]*n
def zeros_zones(n): return [[0]*n for _ in range(N_ZONES)]
def zeros_zones_houses(n): return [[zeros(n), zeros(n)] for _ in range(N_ZONES)]

# ── Main loop ───────────────────────────────────────────────────────────
for run_idx in trange(N_RUNS, desc="Simulations", unit="run"):
    seed_val = get_seed_for_run(run_idx)
    random.seed(seed_val)
    np.random.seed(seed_val)

    result = run_simulation(
        num_days=DAYS,
        return_zone_series=True,
        progress=False,
        seed=seed_val,
    )

    # Unpack basic
    days, blue, red, within, between = result[:5]

    # Defaults
    n = len(days)
    total_spawned = zeros(n)
    total_consumed = zeros(n)
    zone_spawned_daily = zeros_zones(n)
    zone_consumed_daily = zeros_zones(n)
    zone_consumed_by_house = zeros_zones_houses(n)
    within_blue_list = zeros(n)
    within_red_list  = zeros(n)
    dead_blue_cum    = zeros(n)
    dead_red_cum     = zeros(n)
    born_blue_daily  = zeros(n)
    born_red_daily   = zeros(n)

    # Fill from result if available
    if len(result) >= 9:
        total_spawned, total_consumed, zone_spawned_daily, zone_consumed_daily = result[5:9]
    if len(result) >= 10:
        zone_consumed_by_house = result[9]
    if len(result) >= 12:
        within_blue_list, within_red_list = result[10], result[11]
    if len(result) >= 16:
        dead_blue_cum, dead_red_cum, born_blue_daily, born_red_daily = result[12:16]

    # Write rows
    with open(COMBINED_CSV, "a", newline="") as f:
        w = csv.writer(f)
        for i, d in enumerate(days):
            row = [
                run_idx, d,
                blue[i], red[i], within[i], between[i],
                within_blue_list[i], within_red_list[i],
                total_spawned[i], total_consumed[i],
            ]
            # zone-level spawns/cons
            for z in range(N_ZONES):
                row.append(zone_spawned_daily[z][i])
            for z in range(N_ZONES):
                row.append(zone_consumed_daily[z][i])
            # per-house consumption
            for z in range(N_ZONES):
                row.append(zone_consumed_by_house[z][0][i])  # blue
                row.append(zone_consumed_by_house[z][1][i])  # red
            # births/deaths
            row += [
                dead_blue_cum[i], dead_red_cum[i],
                born_blue_daily[i], born_red_daily[i],
            ]
            w.writerow(row)

print(f"\nAll {N_RUNS} simulations completed.")
print(f"Combined results written to: {COMBINED_CSV}")
