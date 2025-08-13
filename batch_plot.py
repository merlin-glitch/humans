
# #!/usr/bin/env python3
# import os, csv
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def load_all(path, value_col):
#     """
#     Reads CSV with columns ["run","day",..., value_col]
#     and returns dict run_id→ list of (day,value)
#     """
#     data = defaultdict(list)
#     with open(path, newline="") as f:
#         r = csv.DictReader(f)
#         for row in r:
#             run = int(row["run"])
#             day = int(row["day"])
#             val = float(row[value_col])
#             data[run].append((day, val))
#     # ensure sorted
#     for run in data:
#         data[run].sort(key=lambda x: x[0])
#     return data

# def plot_all_runs(pop_csv, trust_csv):
#     # load populations
#     blue_runs = load_all(pop_csv, "blue_population")
#     red_runs  = load_all(pop_csv, "red_population")

#     # load trusts
#     w_runs = load_all(trust_csv, "avg_within_trust")
#     b_runs = load_all(trust_csv, "avg_between_trust")

#     # 1) Plot each run individually (populations)
#     plt.figure(figsize=(10,4))
#     for run, series in sorted(blue_runs.items()):
#         days, pops = zip(*series)
#         plt.plot(days, pops, alpha=0.2, color="blue")
#     for run, series in sorted(red_runs.items()):
#         days, pops = zip(*series)
#         plt.plot(days, pops, alpha=0.2, color="red")
#     # now overlay the averages
#     avg_blue = []
#     avg_red  = []
#     days0    = sorted({d for run in blue_runs for d,_ in blue_runs[run]})
#     for d in days0:
#         avg_blue.append( sum(v for _,v in blue_runs[run] if _==d) / len(blue_runs) )
#         avg_red.append(  sum(v for _,v in red_runs[run]  if _==d) / len(red_runs) )
#     plt.plot(days0, avg_blue, label="Avg Blue", linewidth=2, color="darkblue")
#     plt.plot(days0, avg_red,  label="Avg Red",  linewidth=2, color="darkred")
#     plt.title("Population per run, plus overall average")
#     plt.xlabel("Day")
#     plt.ylabel("Alive population")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # 2) Plot trust variation
#     plt.figure(figsize=(10,4))
#     for run, series in sorted(w_runs.items()):
#         days, vals = zip(*series)
#         plt.plot(days, vals, alpha=0.2, color="green")
#     for run, series in sorted(b_runs.items()):
#         days, vals = zip(*series)
#         plt.plot(days, vals, alpha=0.2, color="orange")
#     # averages
#     avg_w = []
#     avg_b = []
#     days1 = sorted({d for run in w_runs for d,_ in w_runs[run]})
#     for d in days1:
#         avg_w.append(sum(v for _,v in w_runs[run] if _==d) / len(w_runs))
#         avg_b.append(sum(v for _,v in b_runs[run] if _==d) / len(b_runs))
#     plt.plot(days1, avg_w, label="Avg Within‑house", linewidth=2, color="darkgreen")
#     plt.plot(days1, avg_b, label="Avg Between‑house", linewidth=2, color="darkorange")
#     plt.title("Trust variation per run, plus overall average")
#     plt.xlabel("Day")
#     plt.ylabel("Trust score")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def main():
#     pop_csv   = "population_all_runs.csv"
#     trust_csv = "trust_all_runs.csv"
#     if not os.path.exists(pop_csv) or not os.path.exists(trust_csv):
#         raise RuntimeError("Missing one of the combined CSVs!")
#     plot_all_runs(pop_csv, trust_csv)

# if __name__=="__main__":
#     main()


# plot_batch_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR  = "batch_results"
COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")

def plot_trust_variation(csv_path: str = COMBINED_CSV, out_dir: str = RESULTS_DIR):
    """
    Produces:
      – within_trust_per_run.png
      – between_trust_per_run.png
      – average_trust.png
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # A) Within-family trust, one curve per run
    plt.figure(figsize=(8, 4))
    for run_id, grp in df.groupby("run"):
        grp = grp.sort_values("day")
        plt.plot(grp["day"], grp["within_trust"], alpha=0.3)
    plt.xlabel("Day")
    plt.ylabel("Within-family trust")
    plt.title("Within-family trust over time (per run)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "within_trust_per_run.png"))
    plt.close()

    # B) Between-family trust, one curve per run
    plt.figure(figsize=(8, 4))
    for run_id, grp in df.groupby("run"):
        grp = grp.sort_values("day")
        plt.plot(grp["day"], grp["between_trust"], alpha=0.3)
    plt.xlabel("Day")
    plt.ylabel("Between-family trust")
    plt.title("Between-family trust over time (per run)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "between_trust_per_run.png"))
    plt.close()

    # C) Average trust across runs
    avg = df.groupby("day")[["within_trust", "between_trust"]].mean().reset_index()

    plt.figure(figsize=(8, 4))
    plt.plot(avg["day"], avg["within_trust"], label="Within", linewidth=2)
    plt.plot(avg["day"], avg["between_trust"], label="Between", linewidth=2)
    plt.xlabel("Day")
    plt.ylabel("Average trust")
    plt.title("Average trust over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "average_trust.png"))
    plt.close()


def plot_population_variation(csv_path: str = COMBINED_CSV, out_dir: str = RESULTS_DIR):
    """
    Produces:
      – blue_population_per_run.png
      – red_population_per_run.png

    Each shows one thin curve per run plus a thick average curve.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- Blue population ---
    plt.figure(figsize=(8, 4))
    for run_id, grp in df.groupby("run"):
        grp = grp.sort_values("day")
        plt.plot(grp["day"], grp["blue_pop"], alpha=0.3, linewidth=1)
    avg_blue = df.groupby("day")["blue_pop"].mean().reset_index()
    plt.plot(avg_blue["day"], avg_blue["blue_pop"], linewidth=3, label="Average")
    plt.xlabel("Day")
    plt.ylabel("Alive blue population")
    plt.title("Blue population over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "blue_population_per_run.png"))
    plt.close()

    # --- Red population ---
    plt.figure(figsize=(8, 4))
    for run_id, grp in df.groupby("run"):
        grp = grp.sort_values("day")
        plt.plot(grp["day"], grp["red_pop"], alpha=0.3, linewidth=1)
    avg_red = df.groupby("day")["red_pop"].mean().reset_index()
    plt.plot(avg_red["day"], avg_red["red_pop"], linewidth=3, label="Average")
    plt.xlabel("Day")
    plt.ylabel("Alive red population")
    plt.title("Red population over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "red_population_per_run.png"))
    plt.close()


if __name__ == "__main__":
    plot_trust_variation()
    plot_population_variation()
    print(f"Figures saved to: {RESULTS_DIR}")
