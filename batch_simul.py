# # # #!/usr/bin/env python3
# # # import os
# # # import csv
# # # from tqdm import trange

# # # # # Import your fast simulation runner and the CSV loader
# # # # from simulation import run_simulation, load_population_series

# # # # def main():
# # # #     NUM_RUNS = 5
# # # #     DAYS     = 60

# # # #     # Combined output CSV
# # # #     combined_csv = "population_all_runs.csv"
# # # #     # Overwrite any existing file and write header
# # # #     with open(combined_csv, "w", newline="") as f:
# # # #         writer = csv.writer(f)
# # # #         writer.writerow(["run", "day", "blue_population", "red_population"])

# # # #     # Batch loop with progress bar
# # # #     for run_id in trange(1, NUM_RUNS + 1, desc="Batch runs"):
# # # #         # Run the simulation for the given number of days
# # # #         run_simulation(DAYS)

# # # #         # After the run, read back the two CSVs it produced
# # # #         days_b, blue_series = load_population_series("blue_population.csv")
# # # #         days_r, red_series  = load_population_series("red_population.csv")

# # # #         # Sanity check: the day indices must match
# # # #         if days_b != days_r:
# # # #             raise RuntimeError(f"Day index mismatch in run {run_id}")

# # # #         # Append each day's data to the combined CSV
# # # #         with open(combined_csv, "a", newline="") as f:
# # # #             writer = csv.writer(f)
# # # #             for day, b_pop, r_pop in zip(days_b, blue_series, red_series):
# # # #                 writer.writerow([run_id, day, b_pop, r_pop])

# # # #     print(f"All {NUM_RUNS} runs complete. Results saved to {combined_csv}")

# # # # if __name__ == "__main__":
# # # #     main()


# # # #!/usr/bin/env python3
# # # import csv
# # # from tqdm import trange

# # # # tweak these imports to wherever you put your fast sim & common helpers
# # # from simulation import run_simulation  
# # # from common import average_trust_within_vs_between, log_population_by_house

# # # def main():
# # #     NUM_RUNS = 10
# # #     DAYS     = 10

# # #     pop_csv   = "population_all_runs.csv"
# # #     trust_csv = "trust_all_runs.csv"

# # #     # overwrite+header for populations
# # #     with open(pop_csv,   "w", newline="") as f:
# # #         w = csv.writer(f)
# # #         w.writerow(["run","day","blue_population","red_population"])

# # #     # overwrite+header for trusts
# # #     with open(trust_csv, "w", newline="") as f:
# # #         w = csv.writer(f)
# # #         w.writerow(["run","day","avg_within_trust","avg_between_trust"])

# # #     for run_id in trange(1, NUM_RUNS+1, desc="Batch runs"):
# # #         # run_simulation_fast must now return:
# # #         # days, blue_series, red_series, within_trust_series, between_trust_series
# # #         days, blue_s, red_s, w_t, b_t = run_simulation(DAYS)

# # #         # sanity check
# # #         if not (len(days)==len(blue_s)==len(red_s)==len(w_t)==len(b_t)):
# # #             raise RuntimeError(f"Length mismatch in run {run_id}")

# # #         # append populations
# # #         with open(pop_csv, "a", newline="") as f:
# # #             w = csv.writer(f)
# # #             for d, b, r in zip(days, blue_s, red_s):
# # #                 w.writerow([run_id, d, b, r])

# # #         # append trusts
# # #         with open(trust_csv, "a", newline="") as f:
# # #             w = csv.writer(f)
# # #             for d, wi, bt in zip(days, w_t, b_t):
# # #                 w.writerow([run_id, d, wi, bt])

# # #     print(f"All runs done. Populations → {pop_csv}, Trusts → {trust_csv}")

# # # if __name__=="__main__":
# # #     main()

# # #!/usr/bin/env python3
# # import csv
# # from tqdm import trange
# # from simulation import run_simulation

# # def main():
# #     NUM_RUNS = 100
# #     DAYS     = 50

# #     # Prepare a CSV to collect everything:
# #     # columns: run, day, blue_pop, red_pop, within_trust, between_trust
# #     combined = "all_runs_summary.csv"
# #     with open(combined, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow([
# #             "run",
# #             "day",
# #             "blue_population",
# #             "red_population",
# #             "within_family_trust",
# #             "between_family_trust"
# #         ])

# #     # Loop over runs
# #     for run_id in trange(1, NUM_RUNS+1, desc="Batch runs"):
# #         days, blue, red, w_tr, b_tr = run_simulation(DAYS)

# #         # For each day, append a row
# #         with open(combined, "a", newline="") as f:
# #             writer = csv.writer(f)
# #             for d, bp, rp, wt, bt in zip(days, blue, red, w_tr, b_tr):
# #                 writer.writerow([run_id, d, bp, rp, wt, bt])

# #     print(f"Done! Data in {combined}")

# # if __name__ == "__main__":
# #     main()
# #!/usr/bin/env python3
# import csv
# from tqdm import trange
# import matplotlib.pyplot as plt
# import pandas as pd

# from simulation import run_simulation  # your headless runner

# def main():
#     NUM_RUNS = 5
#     DAYS     = 10

#     combined_csv = "all_runs_summary.csv"
#     # 1) Write header
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

#     # 2) Run batch
#     for run_id in trange(1, NUM_RUNS+1, desc="Batch runs"):
#         days, blue_pop, red_pop, within_trust, between_trust = run_simulation(DAYS)

#         with open(combined_csv, "a", newline="") as f:
#             writer = csv.writer(f)
#             for d, b, r, w, bt in zip(days, blue_pop, red_pop, within_trust, between_trust):
#                 writer.writerow([run_id, d, b, r, w, bt])

#     print(f"\nAll {NUM_RUNS} runs complete. Data in {combined_csv}")

#     # 3) Plot the trust‐variation
#     plot_trust_variation(combined_csv)

# def plot_trust_variation(csv_path: str):
#     """
#     Reads the combined CSV and produces three PNGs:
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
#     plt.plot(avg["day"], avg["within_family_trust"], label="Within", color="blue")
#     plt.plot(avg["day"], avg["between_family_trust"], label="Between", color="red")
#     plt.xlabel("Day")
#     plt.ylabel("Average trust")
#     plt.title("Average trust over time")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("average_trust.png")
#     plt.show()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import csv
import os
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd

from simulation import run_simulation  # your headless runner

def main():
    NUM_RUNS = 50
    DAYS     = 10

    combined_csv = "all_runs_summary.csv"
    # --- 1) Write header and clear old file ---
    with open(combined_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run",
            "day",
            "blue_population",
            "red_population",
            "within_family_trust",
            "between_family_trust"
        ])

    # --- 2) Run batch simulations ---
    for run_id in trange(1, NUM_RUNS+1, desc="Batch runs"):
        days, blue_pop, red_pop, within_trust, between_trust = run_simulation(DAYS)
        with open(combined_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for d, b, r, w, bt in zip(days, blue_pop, red_pop, within_trust, between_trust):
                writer.writerow([run_id, d, b, r, w, bt])

    print(f"\nAll {NUM_RUNS} runs complete. Data in {combined_csv}")

    # --- 3) Plot trust variations ---
    plot_trust_variation(combined_csv)

    # --- 4) Plot population variations ---
    plot_population_variation(combined_csv)


def plot_trust_variation(csv_path: str):
    """
    Produces three PNGs:
      – within_trust_per_run.png
      – between_trust_per_run.png
      – average_trust.png
    """
    df = pd.read_csv(csv_path)

    # A) Within‐family trust, one curve per run
    plt.figure(figsize=(8, 4))
    for run_id, grp in df.groupby("run"):
        plt.plot(grp["day"], grp["within_family_trust"],
                 color="blue", alpha=0.3)
    plt.xlabel("Day")
    plt.ylabel("Within‐family trust")
    plt.title("Within‐family trust over time (per run)")
    plt.tight_layout()
    plt.savefig("within_trust_per_run.png")
    plt.close()

    # B) Between‐family trust, one curve per run
    plt.figure(figsize=(8, 4))
    for run_id, grp in df.groupby("run"):
        plt.plot(grp["day"], grp["between_family_trust"],
                 color="red", alpha=0.3)
    plt.xlabel("Day")
    plt.ylabel("Between‐family trust")
    plt.title("Between‐family trust over time (per run)")
    plt.tight_layout()
    plt.savefig("between_trust_per_run.png")
    plt.close()

    # C) Average trust across runs
    avg = df.groupby("day")[
        ["within_family_trust", "between_family_trust"]
    ].mean().reset_index()

    plt.figure(figsize=(8, 4))
    plt.plot(avg["day"], avg["within_family_trust"], label="Within", color="blue", linewidth=2)
    plt.plot(avg["day"], avg["between_family_trust"], label="Between", color="red",  linewidth=2)
    plt.xlabel("Day")
    plt.ylabel("Average trust")
    plt.title("Average trust over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("average_trust.png")
    plt.close()


def plot_population_variation(csv_path: str):
    """
    Produces two PNGs:
      – blue_population_per_run.png
      – red_population_per_run.png

    Each shows one thin curve per run plus a thick average curve.
    """
    df = pd.read_csv(csv_path)

    # --- Blue population ---
    plt.figure(figsize=(8, 4))
    # per-run
    for run_id, grp in df.groupby("run"):
        plt.plot(grp["day"], grp["blue_population"],
                 color="blue", alpha=0.3, linewidth=1)
    # average
    avg_blue = df.groupby("day")["blue_population"].mean().reset_index()
    plt.plot(avg_blue["day"], avg_blue["blue_population"],
             color="blue", linewidth=3, label="Average")
    plt.xlabel("Day")
    plt.ylabel("Alive blue population")
    plt.title("Blue population over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("blue_population_per_run.png")
    plt.close()

    # --- Red population ---
    plt.figure(figsize=(8, 4))
    for run_id, grp in df.groupby("run"):
        plt.plot(grp["day"], grp["red_population"],
                 color="red", alpha=0.3, linewidth=1)
    avg_red = df.groupby("day")["red_population"].mean().reset_index()
    plt.plot(avg_red["day"], avg_red["red_population"],
             color="red", linewidth=3, label="Average")
    plt.xlabel("Day")
    plt.ylabel("Alive red population")
    plt.title("Red population over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("red_population_per_run.png")
    plt.close()


if __name__ == "__main__":
    main()
