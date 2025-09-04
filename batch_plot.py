


# # batch_plots.py
# import os
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.lines import Line2D
# from matplotlib.ticker import MaxNLocator

# ZONES = [0, 1, 2,3,4]


# def _agg_across_runs(d: pd.DataFrame, value_col: str, smooth):
#     """Return x, mean, p10, p90 for a metric across runs per day."""
#     v = pd.to_numeric(d[value_col], errors="coerce")
#     piv = d.assign(v=v).pivot_table(index="day", columns="run", values="v", aggfunc="mean")
#     x   = piv.index.to_numpy()
#     mean = piv.mean(axis=1)
#     p10  = pd.Series(np.nanpercentile(piv.to_numpy(), 10, axis=1), index=piv.index)
#     p90  = pd.Series(np.nanpercentile(piv.to_numpy(), 90, axis=1), index=piv.index)

#     if smooth:
#         mean = smooth(mean)
#         p10  = smooth(p10)
#         p90  = smooth(p90)

#     # ensure plain float arrays (Matplotlib can’t handle pd.NA)
#     mean = pd.to_numeric(mean, errors="coerce").astype(float).to_numpy()
#     p10  = pd.to_numeric(p10,  errors="coerce").astype(float).to_numpy()
#     p90  = pd.to_numeric(p90,  errors="coerce").astype(float).to_numpy()
#     return x, mean, p10, p90

# def _runs(df: pd.DataFrame) -> list[int]:
#     return sorted(df["run"].dropna().unique().tolist())

# def make_smoother(enabled: bool, window: int):
#     suffix = f"_sm{window}" if enabled and window > 1 else ""
#     title_note = f" (rolling {window})" if enabled and window > 1 else ""
#     def S(s: pd.Series) -> pd.Series:
#         if enabled and window > 1 and len(s) > 1:
#             return s.rolling(window, min_periods=1).mean()
#         return s
#     return S, suffix, title_note

# def plot_population(df: pd.DataFrame, S, outdir, suf, note,
#                     show_band: bool = True, alpha_runs: float = 0.10):
#     """
#     One figure for ALL runs:
#       - thin translucent line per run (blue/red)
#       - bold mean line per family (blue/red)
#       - optional 10–90% band
#     """
#     os.makedirs(outdir, exist_ok=True)

#     # Sort for safety
#     df = df.sort_values(["run", "day"])

#     plt.figure(figsize=(10, 5))

#     # 1) Overlay each run lightly
#     for r in sorted(df["run"].dropna().unique()):
#         d = df[df.run == r]
#         # Cast to float so matplotlib doesn't choke on pd.NA
#         x  = d.day.to_numpy()
#         yb = pd.to_numeric(S(d.blue_pop), errors="coerce").astype(float).to_numpy()
#         yr = pd.to_numeric(S(d.red_pop),  errors="coerce").astype(float).to_numpy()

#         plt.plot(x, yb, color="blue", alpha=alpha_runs, linewidth=1, label="_nolegend_")
#         plt.plot(x, yr, color="red",  alpha=alpha_runs, linewidth=1, label="_nolegend_")

#     # 2) Mean & bands across runs (pivot -> aggregate by day)
#     blue_piv = df.pivot_table(index="day", columns="run", values="blue_pop", aggfunc="mean")
#     red_piv  = df.pivot_table(index="day", columns="run", values="red_pop",  aggfunc="mean")

#     # stats per day
#     blue_mean = blue_piv.mean(axis=1)
#     red_mean  = red_piv.mean(axis=1)

#     if show_band:
#         blue_lo = blue_piv.quantile(0.10, axis=1)
#         blue_hi = blue_piv.quantile(0.90, axis=1)
#         red_lo  = red_piv.quantile(0.10, axis=1)
#         red_hi  = red_piv.quantile(0.90, axis=1)
#         # smoothing
#         blue_lo, blue_hi = S(blue_lo), S(blue_hi)
#         red_lo,  red_hi  = S(red_lo),  S(red_hi)

#     # smooth means
#     blue_mean = S(blue_mean)
#     red_mean  = S(red_mean)

#     # Ensure numeric arrays (NaNs are fine)
#     x = blue_mean.index.to_numpy()
#     plt.plot(x, blue_mean.astype(float).to_numpy(), color="blue", linewidth=2.5, label="Blue mean")
#     plt.plot(x, red_mean.astype(float).to_numpy(),  color="red",  linewidth=2.5, label="Red mean")

#     if show_band:
#         plt.fill_between(
#             x,
#             blue_lo.astype(float).to_numpy(),
#             blue_hi.astype(float).to_numpy(),
#             color="blue", alpha=0.12, label="Blue 10–90%"
#         )
#         plt.fill_between(
#             x,
#             red_lo.astype(float).to_numpy(),
#             red_hi.astype(float).to_numpy(),
#             color="red", alpha=0.12, label="Red 10–90%"
#         )

#     plt.xlabel("Day")
#     plt.ylabel("Population")
#     plt.title(f"Population — all runs{note}")
#     plt.grid(True, alpha=.25)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, f"pop_ALL_runs{suf}.png"))
#     plt.close()

# def plot_population_per_run(df: pd.DataFrame, S, outdir, suf, note):
#     """
#     One PNG per run:
#       - Blue vs Red population for that run only
#       - No 10–90% band
#     """
#     os.makedirs(outdir, exist_ok=True)

#     for r in sorted(df["run"].dropna().unique()):
#         d = df[df.run == r].sort_values("day")

#         x  = d["day"].to_numpy()
#         yb = pd.to_numeric(S(d["blue_pop"]), errors="coerce").astype(float).to_numpy()
#         yr = pd.to_numeric(S(d["red_pop"]),  errors="coerce").astype(float).to_numpy()

#         plt.figure(figsize=(9, 4))
#         plt.plot(x, yb, color="blue", linewidth=2, label="Blue")
#         plt.plot(x, yr, color="red",  linewidth=2, label="Red")
#         plt.xlabel("Day"); plt.ylabel("Population")
#         plt.title(f"Population — run {r}{note}")
#         plt.grid(True, alpha=.25); plt.legend(); plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"pop_run{r}{suf}.png"))
#         plt.close()

# def plot_births_deaths_all_runs(df: pd.DataFrame, S, outdir, suf, note):
#     """
#     ONE figure (no per-run files):
#       • Solid lines: mean births/day (blue/red) on left axis
#       • Dashed lines: mean cumulative deaths (blue/red) on right axis
#       • Means only (no bands)
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df.sort_values(["run", "day"]).copy()

#     # Means across runs (ignore percentiles)
#     x, blue_b_mean, *_ = _agg_across_runs(d, "blue_born", S)
#     _,  red_b_mean,  *_ = _agg_across_runs(d, "red_born",  S)
#     _, blue_d_mean,  *_ = _agg_across_runs(d, "blue_dead", S)
#     _,  red_d_mean,   *_ = _agg_across_runs(d, "red_dead",  S)

#     fig, ax1 = plt.subplots(figsize=(10, 5))

#     # births/day (left axis) — means only
#     l1, = ax1.plot(x, blue_b_mean, color="blue", linewidth=2.2, label="Blue births/day (mean)")
#     l2, = ax1.plot(x, red_b_mean,  color="red",  linewidth=2.2, label="Red births/day (mean)")
#     ax1.set_xlabel("Day")
#     ax1.set_ylabel("Births per day")
#     ax1.grid(True, alpha=.25)

#     # cumulative deaths (right axis) — means only
#     ax2 = ax1.twinx()
#     l3, = ax2.plot(x, blue_d_mean, color="blue", linestyle="--", linewidth=2.2, label="Blue deaths (cum, mean)")
#     l4, = ax2.plot(x, red_d_mean,  color="red",  linestyle="--", linewidth=2.2, label="Red deaths (cum, mean)")
#     ax2.set_ylabel("Cumulative deaths")

#     # single legend
#     lines = [l1, l2, l3, l4]
#     labels = [ln.get_label() for ln in lines]
#     ax1.legend(lines, labels, loc="upper left")

#     plt.title(f"Births & deaths — all runs (means only){note}")
#     fig.tight_layout()
#     plt.savefig(os.path.join(outdir, f"births_deaths_ALL_runs{suf}.png"))
#     plt.close(fig)


# def plot_trust_agg(df: pd.DataFrame, S, outdir, suf, note):
#     """
#     TWO figures total (aggregated across runs):
#       1) Within vs Between (mean + 10–90% band for each)
#       2) Within Blue vs Within Red (mean + 10–90% band for each)
#     Colors: Blue=blue, Red=red; Within(overall)=black, Between=purple.
#     """
#     import numpy as np
#     from matplotlib.patches import Patch

#     def _agg_col(col: str):
#         # mean / p10 / p90 by day across runs (safe numerics)
#         s = pd.to_numeric(df[col], errors="coerce")
#         g = s.groupby(df["day"])
#         m   = g.mean().sort_index()
#         p10 = g.quantile(0.10).sort_index()
#         p90 = g.quantile(0.90).sort_index()
#         # optional smoothing (keeps index)
#         return S(m), S(p10), S(p90)

#     # ---------- Figure A: Within vs Between ----------
#     w_m, w_p10, w_p90 = _agg_col("within_trust")
#     b_m, b_p10, b_p90 = _agg_col("between_trust")

#     # unify x domain
#     x = sorted(set(w_m.index) | set(b_m.index))
#     wM, wL, wU = w_m.reindex(x), w_p10.reindex(x), w_p90.reindex(x)
#     bM, bL, bU = b_m.reindex(x), b_p10.reindex(x), b_p90.reindex(x)

#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.fill_between(x, wL, wU, color="gray", alpha=0.18)
#     ax.plot(x, wM, color="green", linewidth=2, label="Within mean")
#     ax.fill_between(x, bL, bU, color="mediumorchid", alpha=0.18)
#     ax.plot(x, bM, color="purple", linewidth=2, label="Between mean")

#     band_legend = [
#         Patch(facecolor="gray", alpha=0.18, label="Within 10–90%"),
#         Patch(facecolor="mediumorchid", alpha=0.18, label="Between 10–90%"),
#     ]
#     ax.legend(handles=[*ax.get_legend_handles_labels()[0], *band_legend], loc="upper left")
#     ax.set_xlabel("Day"); ax.set_ylabel("Trust")
#     ax.set_title(f"Trust — Within vs Between (all runs){note}")
#     ax.grid(True, alpha=.25); fig.tight_layout()
#     fig.savefig(os.path.join(outdir, f"trust_within_between_all_runs{suf}.png"))
#     plt.close(fig)

#     # ---------- Figure B: Within Blue vs Within Red ----------
#     bl_m, bl_p10, bl_p90 = _agg_col("within_blue_trust")
#     rd_m, rd_p10, rd_p90 = _agg_col("within_red_trust")

#     x = sorted(set(bl_m.index) | set(rd_m.index))
#     blM, blL, blU = bl_m.reindex(x), bl_p10.reindex(x), bl_p90.reindex(x)
#     rdM, rdL, rdU = rd_m.reindex(x), rd_p10.reindex(x), rd_p90.reindex(x)

#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.fill_between(x, blL, blU, color="blue", alpha=0.12)
#     ax.plot(x, blM, color="blue", linewidth=2, label="Within Blue mean")
#     ax.fill_between(x, rdL, rdU, color="red", alpha=0.12)
#     ax.plot(x, rdM, color="red", linewidth=2, label="Within Red mean")

#     band_legend = [
#         Patch(facecolor="blue", alpha=0.12, label="Blue 10–90%"),
#         Patch(facecolor="red",  alpha=0.12, label="Red 10–90%"),
#     ]
#     ax.legend(handles=[*ax.get_legend_handles_labels()[0], *band_legend], loc="upper left")
#     ax.set_xlabel("Day"); ax.set_ylabel("Trust")
#     ax.set_title(f"Trust — Within Blue vs Within Red (all runs){note}")
#     ax.grid(True, alpha=.25); fig.tight_layout()
#     fig.savefig(os.path.join(outdir, f"trust_by_house_all_runs{suf}.png"))
#     plt.close(fig)


# def plot_total_spawn_vs_consumption(df: pd.DataFrame, S, outdir, suf, note):
#     """
#     ONE figure (no per-run files):
#       • Solid lines = mean across runs
#       • Shaded = 10–90% across runs
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df.sort_values(["run", "day"]).copy()
#     d = _ensure_numeric(d, ["total_spawned", "total_consumed"])

#     x, sp_mean, sp_p10, sp_p90 = _agg_across_runs(d, "total_spawned", S)
#     _,  co_mean, co_p10, co_p90 = _agg_across_runs(d, "total_consumed", S)

#     fig, ax = plt.subplots(figsize=(10, 4.5))

#     # bands
#     ax.fill_between(x, sp_p10, sp_p90, color="#2ca02c", alpha=0.18, label="Spawned 10–90%")
#     ax.fill_between(x, co_p10, co_p90, color="#ff7f0e", alpha=0.18, label="Consumed 10–90%")

#     # means
#     ax.plot(x, sp_mean, color="#2ca02c", linewidth=2.2, label="Spawned (mean)")
#     ax.plot(x, co_mean, color="#ff7f0e", linewidth=2.2, label="Consumed (mean)")

#     ax.set_xlabel("Day")
#     ax.set_ylabel("Units / day")
#     ax.set_title(f"Global spawn vs consumption — all runs (mean ± 10–90%){note}")
#     ax.grid(True, alpha=.25)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, f"global_spawn_cons_ALL_runs{suf}.png"))
#     plt.close(fig)


# def plot_zone_spawn_vs_consumption(df: pd.DataFrame, S, outdir, suf, note):
#     """
#     ONE PNG with 5 subplots (Zone 0/1/2):
#       • Solid lines = mean across runs (spawned & consumed)
#       • Shaded = 10–90% across runs (spawned & consumed)
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df.sort_values(["run", "day"]).copy()

#     # make sure everything we need is numeric
#     cols = []
#     for z in ZONES:
#         cols += [f"z{z}_spawn", f"z{z}_cons"]
#     d = _ensure_numeric(d, cols)

#     fig, axes = plt.subplots(1, 5, figsize=(15, 4.5), sharex=True, sharey=True)

#     for z, ax in zip(ZONES, axes):
#         # aggregate across runs for this zone
#         x, sp_mean, sp_p10, sp_p90 = _agg_across_runs(d, f"z{z}_spawn", S)
#         _,  co_mean, co_p10, co_p90 = _agg_across_runs(d, f"z{z}_cons",  S)

#         # 10–90% bands
#         ax.fill_between(x, sp_p10, sp_p90, color="#2ca02c", alpha=0.18, label="Spawned 10–90%")
#         ax.fill_between(x, co_p10, co_p90, color="#ff7f0e", alpha=0.18, label="Consumed 10–90%")

#         # mean lines
#         ax.plot(x, sp_mean, color="#2ca02c", linewidth=2.2, label="Spawned (mean)")
#         ax.plot(x, co_mean, color="#ff7f0e", linewidth=2.2, label="Consumed (mean)")

#         ax.set_title(f"Zone {z}")
#         ax.set_xlabel("Day")
#         if z == ZONES[0]:
#             ax.set_ylabel("Units / day")
#         ax.grid(True, alpha=.25)

#     # one legend for all subplots
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))

#     fig.suptitle(f"Spawn vs consumption — mean ± 10–90% across runs{note}", y=1.12)
#     fig.tight_layout(rect=[0, 0, 1, 0.98])
#     fig.savefig(os.path.join(outdir, f"zone_spawn_cons_ALL_runs{suf}.png"))
#     plt.close(fig)


# def plot_zone_consumption_by_house(df: pd.DataFrame, S, outdir, suf, note):
#     """
#     One figure, 5 subplots (zones). For each zone we plot, across runs:
#       - Blue median (solid blue) with 10–90% band (light blue)
#       - Red  median (solid red)  with 10–90% band (light red)
#     """
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     # ensure numeric columns (ignore missing if any)
#     cols = []
#     for z in ZONES:
#         cols += [f"z{z}_blue_cons", f"z{z}_red_cons"]
#     d = df.copy()
#     for c in cols:
#         if c in d.columns:
#             d[c] = pd.to_numeric(d[c], errors="coerce")

#     # we aggregate ACROSS runs at each day
#     fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharex=True, sharey=True)

#     for z, ax in zip(ZONES, axes):
#         bcol = f"z{z}_blue_cons"
#         rcol = f"z{z}_red_cons"

#         tmp = d[["day", bcol, rcol]].dropna(subset=["day"]).copy()
#         tmp["day"] = pd.to_numeric(tmp["day"], errors="coerce")
#         tmp = tmp.dropna(subset=["day"]).sort_values("day")

#         g = tmp.groupby("day", dropna=True)

#         # quantiles/medians across runs per day
#         b_q10 = g[bcol].quantile(0.10)
#         b_med = g[bcol].median()
#         b_q90 = g[bcol].quantile(0.90)

#         r_q10 = g[rcol].quantile(0.10)
#         r_med = g[rcol].median()
#         r_q90 = g[rcol].quantile(0.90)

#         # Apply optional smoothing (S returns a Series)
#         b_q10, b_med, b_q90 = S(b_q10), S(b_med), S(b_q90)
#         r_q10, r_med, r_q90 = S(r_q10), S(r_med), S(r_q90)

#         x = b_med.index.to_numpy()

#         # Blue band + median
#         ax.fill_between(x, b_q10.to_numpy(), b_q90.to_numpy(), color="blue", alpha=0.15, label="Blue 10–90%")
#         ax.plot(x, b_med.to_numpy(), color="blue", linewidth=2, label="Blue median")

#         # Red band + median
#         ax.fill_between(x, r_q10.to_numpy(), r_q90.to_numpy(), color="red", alpha=0.15, label="Red 10–90%")
#         ax.plot(x, r_med.to_numpy(), color="red", linewidth=2, label="Red median")

#         ax.set_title(f"Zone {z} — per-house consumption{note}")
#         ax.set_xlabel("Day")
#         ax.set_ylabel("Units / day")
#         ax.grid(True, alpha=0.25)

#     # one shared legend (avoid duplicates)
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2)
#     fig.suptitle("Per-house consumption per zone — medians & 10–90% bands (across runs)")
#     fig.tight_layout(rect=[0, 0, 1, 0.94])
#     fig.savefig(os.path.join(outdir, f"zone_house_cons_all_runs{suf}.png"))
#     plt.close(fig)


# def plot_consumption_share(df: pd.DataFrame, outdir: str, suf: str, note: str):
#     """
#     ONE PNG:
#       • x = day
#       • each bar = 100% of that day's total consumption (all runs combined)
#       • segments = share from Zone 0 / Zone 1 / Zone 2
#     """
#     os.makedirs(outdir, exist_ok=True)

#     cols = ["run","day","z0_cons","z1_cons","z2_cons","z3_cons","z4_cons"]
#     d = _ensure_numeric(df[cols], cols).sort_values(["run","day"])

#     # Sum across runs for each day, then convert to % per day
#     g = d.groupby("day")[["z0_cons","z1_cons","z2_cons","z3_cons","z4_cons"]].sum(min_count=1)
#     denom = g.sum(axis=1).replace(0, np.nan)
#     share = (g.div(denom, axis=0) * 100).fillna(0.0)

#     # Stacked 100% bars
#     x = share.index.to_numpy()
#     z0 = share["z0_cons"].to_numpy()
#     z1 = share["z1_cons"].to_numpy()
#     z2 = share["z2_cons"].to_numpy()
#     z3 = share["z3_cons"].to_numpy()
#     z4 = share["z4_cons"].to_numpy()

#     bottom1 = z0
#     bottom2 = z0 + z1
#     bottom3 = z0 + z1 + z2
#     bottom4 = z0 + z1 + z2 + z3

#     fig, ax = plt.subplots(figsize=(14, 4.5))
#     ax.bar(x, z0, color="#1f77b4", label="Zone 0")  # blue-ish
#     ax.bar(x, z1, bottom=bottom1, color="#d62728", label="Zone 1")  # red-ish
#     ax.bar(x, z2, bottom=bottom2, color="#2ca02c", label="Zone 2")  # green
#     ax.bar(x, z3, bottom=bottom3, color="#ff7f0e", label="Zone 3")  # orange
#     ax.bar(x, z4, bottom=bottom4, color="#9467bd", label="Zone 4")  # purple


#     ax.set_ylim(0, 100)
#     ax.yaxis.set_major_locator(MaxNLocator(6, integer=True))
#     ax.xaxis.set_major_locator(MaxNLocator(12, integer=True))
#     ax.set_ylabel("Share of daily consumption (%)")
#     ax.set_xlabel("Day")
#     ax.set_title(f"Daily consumption mix by zone — 100% stacked (all runs){note}")
#     ax.grid(True, axis="y", alpha=0.25)
#     ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, f"zone_mix_daily_pct_ALL_runs{suf}.png"), dpi=120)
#     plt.close(fig)

# def plot_per_capita_all_runs(df: pd.DataFrame, S, outdir, suf, note):
#     """
#     ONE PNG: per-capita consumption (blue/red) aggregated across runs.
#     Solid lines = mean; translucent bands = 10–90% across runs.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df.sort_values(["run", "day"]).copy()

#     # Make sure needed columns are numeric
#     cols = [
#         "blue_pop", "red_pop",
#         "z0_blue_cons", "z1_blue_cons", "z2_blue_cons", "z3_blue_cons", "z4_blue_cons",
#         "z0_red_cons",  "z1_red_cons",  "z2_red_cons", "z3_red_cons",  "z4_red_cons",
        
#     ]
#     for c in cols:
#         d[c] = pd.to_numeric(d[c], errors="coerce")

#     # Total consumption per house/day
#     blue_cons = d["z0_blue_cons"] + d["z1_blue_cons"] + d["z2_blue_cons"]+ d["z3_blue_cons"] + d["z4_blue_cons"]
#     red_cons  = d["z0_red_cons"]  + d["z1_red_cons"]  + d["z2_red_cons"]+ d["z3_red_cons"] + d["z4_red_cons"]

#     # Avoid division by zero -> NaN
#     blue_pop = d["blue_pop"].astype(float).where(d["blue_pop"] > 0, np.nan)
#     red_pop  = d["red_pop"].astype(float).where(d["red_pop"]  > 0, np.nan)

#     # Per-capita time series per run/day
#     d["blue_pc"] = (blue_cons / blue_pop).astype(float)
#     d["red_pc"]  = (red_cons  / red_pop ).astype(float)

#     # Aggregate across runs per day
#     blue_stats = (
#         d.groupby("day")["blue_pc"]
#          .agg(mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
#     )
#     red_stats = (
#         d.groupby("day")["red_pc"]
#          .agg(mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
#     )

#     # Align days in case of gaps
#     all_days = pd.Index(sorted(set(blue_stats.index) | set(red_stats.index)))
#     blue_stats = blue_stats.reindex(all_days)
#     red_stats  = red_stats.reindex(all_days)

#     # Optional smoothing (S) then convert to numpy
#     x          = all_days.to_numpy()
#     blue_mean  = pd.Series(S(blue_stats["mean"])).astype(float).to_numpy()
#     blue_p10   = pd.Series(S(blue_stats["p10"])).astype(float).to_numpy()
#     blue_p90   = pd.Series(S(blue_stats["p90"])).astype(float).to_numpy()
#     red_mean   = pd.Series(S(red_stats["mean"])).astype(float).to_numpy()
#     red_p10    = pd.Series(S(red_stats["p10"])).astype(float).to_numpy()
#     red_p90    = pd.Series(S(red_stats["p90"])).astype(float).to_numpy()

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.fill_between(x, blue_p10, blue_p90, color="blue", alpha=0.15, label="Blue 10–90%")
#     ax.fill_between(x, red_p10,  red_p90,  color="red",  alpha=0.12, label="Red 10–90%")
#     ax.plot(x, blue_mean, color="blue", linewidth=2.2, label="Blue mean")
#     ax.plot(x, red_mean,  color="red",  linewidth=2.2, label="Red mean")

#     ax.set_xlabel("Day")
#     ax.set_ylabel("Per-capita consumption (units/person/day)")
#     ax.set_title(f"Per-capita consumption — all runs (mean ± 10–90%){note}")
#     ax.grid(True, alpha=0.25)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, f"per_capita_ALL_runs{suf}.png"))
#     plt.close(fig)


# def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
#     d = df.copy()
#     for c in cols:
#         d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
#     return d

# def plot_overall_zone_exploitation_totals(df: pd.DataFrame, outdir: str):
#     """
#     Stacked bars: per zone, total consumption by Blue and Red
#     aggregated across ALL runs and ALL days, with % labels.
#     """
#     cols = [
#         "z0_blue_cons","z1_blue_cons","z2_blue_cons","z3_blue_cons","z4_blue_cons",
#         "z0_red_cons", "z1_red_cons", "z2_red_cons","z3_red_cons", "z4_red_cons",
#     ]
#     d = _ensure_numeric(df, cols)

#     totals_blue = np.array([d[f"z{z}_blue_cons"].sum() for z in ZONES], dtype=float)
#     totals_red  = np.array([d[f"z{z}_red_cons"].sum()  for z in ZONES], dtype=float)
#     zone_totals = totals_blue + totals_red

#     x = np.arange(len(ZONES))
#     fig, ax = plt.subplots(figsize=(8, 4))
#     b = ax.bar(x, totals_blue, width=0.6, label="Blue", color="blue")
#     r = ax.bar(x, totals_red,  width=0.6, bottom=totals_blue, label="Red", color="red")

#     ax.set_xticks(x)
#     ax.set_xticklabels([f"Zone {z}" for z in ZONES])
#     ax.set_ylabel("Total units consumed (all runs & days)")
#     ax.set_title("Overall exploitation by zone and family")
#     ax.grid(True, axis="y", alpha=.25)
#     ax.legend()

#     # ── annotate % shares inside bars + total above each bar
#     y_max = zone_totals.max() if len(zone_totals) else 0.0
#     pad   = 0.03 * y_max  # space above bar for the total label

#     for i, total in enumerate(zone_totals):
#         if total <= 0:
#             continue
#         p_blue = 100.0 * totals_blue[i] / total
#         p_red  = 100.0 * totals_red[i]  / total

#         # Blue % centered in blue segment
#         ax.text(
#             x[i], totals_blue[i] * 0.5,
#             f"{p_blue:.1f}%",
#             ha="center", va="center", color="white", fontsize=10, fontweight="bold"
#         )
#         # Red % centered in red segment
#         ax.text(
#             x[i], totals_blue[i] + totals_red[i] * 0.5,
#             f"{p_red:.1f}%",
#             ha="center", va="center", color="white", fontsize=10, fontweight="bold"
#         )
       
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "overall_zone_exploitation_stacked.png"))
#     plt.close(fig)


# def plot_zone_dominance_by_run(df: pd.DataFrame, outdir: str):
#     """
#     For each zone, count how many runs Blue consumed the majority vs Red.
#     Robust to dtypes; we sum within run first, then compare.
#     Adds % labels on top of bars.
#     """
#     import os
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     os.makedirs(outdir, exist_ok=True)

#     cols = [
#         "z0_blue_cons", "z1_blue_cons", "z2_blue_cons","z3_blue_cons", "z4_blue_cons",
#         "z0_red_cons",  "z1_red_cons",  "z2_red_cons","z3_red_cons",  "z4_red_cons",
#     ]

#     d = df.copy()
#     # ensure numeric, keep run
#     d["run"] = pd.to_numeric(d["run"], errors="coerce")
#     for c in cols:
#         d[c] = pd.to_numeric(d[c], errors="coerce")

#     runs = np.sort(d["run"].dropna().unique())

#     blue_wins, red_wins, ties, no_data = [], [], [], []

#     # optional: quick sanity print
#     # print("Total per-zone (all runs):", d[cols].sum(numeric_only=True))

#     for z in ZONES:
#         shares = []
#         for r in runs:
#             sub = d[d["run"] == r]
#             b_sum = sub[f"z{z}_blue_cons"].sum(skipna=True)
#             r_sum = sub[f"z{z}_red_cons"].sum(skipna=True)
#             tot = b_sum + r_sum
#             if tot <= 0:
#                 shares.append(np.nan)        # no data for this run/zone
#             else:
#                 shares.append(b_sum / tot)   # blue share for run r in zone z

#         s = pd.Series(shares, index=runs)
#         blue_wins.append(int((s > 0.5).sum()))
#         red_wins .append(int((s < 0.5).sum()))
#         ties     .append(int((s == 0.5).sum()))
#         no_data  .append(int(s.isna().sum()))

#     # ---- plot grouped bars with % labels ----
#     x = np.arange(len(ZONES))
#     w = 0.22

#     fig, ax = plt.subplots(figsize=(10, 4.5))
#     b = ax.bar(x - w, blue_wins, width=w, color='blue',  label="Blue wins (runs)")
#     r = ax.bar(x,       red_wins,  width=w, color='red',   label="Red wins (runs)")
#     t = ax.bar(x + w,   ties,      width=w, color='grey',  label="Ties")

#     # show “no data” as a thin hatched bar behind (if any)
#     nd = ax.bar(x, no_data, width=0.65, color='none', edgecolor='#666',
#                 linewidth=1, hatch='////', label="No data")

#     ax.set_xticks(x)
#     ax.set_xticklabels([f"Zone {z}" for z in ZONES])
#     ax.set_ylabel("# of runs")
#     ax.set_title("Which family exploited each zone most (by run)")
#     ax.grid(True, axis="y", alpha=.25)
#     ax.legend(loc="upper left")

#     # annotate counts + percentages on each colored bar
#     totals = np.array(blue_wins) + np.array(red_wins) + np.array(ties) + np.array(no_data)
#     def _annotate(bar_container, counts):
#         for rect, c, tot in zip(bar_container, counts, totals):
#             if tot > 0 and c > 0:
#                 pct = 100.0 * c / tot
#                 ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.05,
#                         f"{c} ({pct:.0f}%)", ha='center', va='bottom', fontsize=9)

#     _annotate(b, blue_wins)
#     _annotate(r, red_wins)
#     _annotate(t, ties)

#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "zone_dominance_by_run.png"))
#     plt.close(fig)


# def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
#     d = df.copy()
#     for c in cols:
#         if c in d.columns:
#             d[c] = pd.to_numeric(d[c], errors="coerce")
#     return d


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv",    default="batch_results/all_runs_combined.csv")
#     ap.add_argument("--out",    default="batch_results/plots")
#     ap.add_argument("--smooth", action="store_true", help="Enable rolling-average smoothing")
#     ap.add_argument("--window", type=int, default=7, help="Rolling window (days)")
#     args = ap.parse_args()

#     os.makedirs(args.out, exist_ok=True)
#     df = pd.read_csv(args.csv).sort_values(["run","day"])

#     S, suf, note = make_smoother(args.smooth, args.window)

#     plot_population(df, S, args.out, suf, note)
#     plot_population_per_run(df, S, args.out, suf, note)
#     plot_births_deaths_all_runs(df, S, args.out, suf, note)
#     plot_trust_agg(df, S, args.out, suf, note)    
#     plot_total_spawn_vs_consumption(df, S, args.out, suf, note)
#     plot_zone_spawn_vs_consumption(df, S, args.out, suf, note)
#     plot_zone_consumption_by_house(df, S, args.out, suf, note)
#     plot_consumption_share(df, args.out, suf, note)
#     plot_per_capita_all_runs(df, S, args.out, suf, note)
#     plot_overall_zone_exploitation_totals(df, args.out)
#     plot_zone_dominance_by_run(df, args.out)
#            # ---- new diagnostics (5 figs) ----
#     plot_survival_curves(df, args.out)
#     plot_extinction_histograms(df, args.out)
#     plot_spawn_vs_cons_scatter(df, args.out, note)
#     plot_lag_ccf_spawn_cons(df, args.out, max_lag=10, note=note)
#     plot_cons_spawn_ratio_by_zone(df, S, args.out, suf, note)
#         # ---- nouvelles figures ----
#     # ---- nouvelles figures par run ----
#     plot_population_consumption_trust_per_run(df, S, args.out, suf, note)
#     plot_spawn_vs_cons_by_family_per_run(df, S, args.out, suf, note)






#     print(f"Saved plots to: {args.out}")

# def plot_cons_spawn_ratio_by_zone(df: pd.DataFrame, S, outdir: str, suf: str, note: str):
#     """
#     For each zone: ratio = consumed / (spawned + eps), aggregated across runs by day (sum).
#     Shows smoothed ratio with y=1 guide (overdraw if >1).
#     """
#     os.makedirs(outdir, exist_ok=True)
#     eps = 1e-9
#     fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharex=True, sharey=True)

#     for z, ax in zip(ZONES, axes):
#         cols = [f"z{z}_spawn", f"z{z}_cons"]
#         d = df[["day", *cols]].copy()
#         for c in d.columns:
#             d[c] = pd.to_numeric(d[c], errors="coerce")

#         daily = d.groupby("day")[cols].sum(min_count=1).sort_index()
#         ratio = daily[f"z{z}_cons"] / (daily[f"z{z}_spawn"] + eps)
#         ratio = S(ratio)

#         ax.plot(ratio.index.to_numpy(), ratio.to_numpy(), linewidth=2.0)
#         ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
#         ax.set_title(f"Zone {z} — cons/spawn{note}")
#         ax.set_xlabel("Day")
#         if z == ZONES[0]:
#             ax.set_ylabel("Ratio (consumed / spawned)")
#         ax.grid(True, alpha=.25)

#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, f"cons_spawn_ratio_by_zone{suf}.png"))
#     plt.close(fig)
# def plot_lag_ccf_spawn_cons(df: pd.DataFrame, outdir: str, max_lag: int = 10, note: str = ""):
#     """
#     Cross-correlation corr(spawn_t, cons_{t+lag}) for lag in [-max_lag, +max_lag],
#     using daily sums across runs.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df[["day", "total_spawned", "total_consumed"]].copy()
#     for c in d.columns:
#         d[c] = pd.to_numeric(d[c], errors="coerce")

#     daily = d.groupby("day").sum(min_count=1).sort_index()
#     x = daily["total_spawned"].to_numpy(dtype=float)
#     y = daily["total_consumed"].to_numpy(dtype=float)

#     def _corr_at_lag(a, b, lag):
#         if lag > 0:
#             a_, b_ = a[:-lag], b[lag:]
#         elif lag < 0:
#             a_, b_ = a[-lag:], b[:lag]
#         else:
#             a_, b_ = a, b
#         m = np.isfinite(a_) & np.isfinite(b_)
#         if m.sum() < 2:
#             return np.nan
#         return np.corrcoef(a_[m], b_[m])[0, 1]

#     lags = np.arange(-max_lag, max_lag + 1)
#     ccf  = np.array([_corr_at_lag(x, y, L) for L in lags], dtype=float)

#     fig, ax = plt.subplots(figsize=(8, 4))
#     # NOTE: no 'use_line_collection' on new Matplotlib
#     markerline, stemlines, baseline = ax.stem(lags, ccf)
#     # optional styling
#     plt.setp(stemlines, linewidth=1.5)
#     plt.setp(markerline, markersize=4)
#     baseline.set_visible(False)

#     ax.set_xlabel("Lag (days) — positive = spawn leads consumption")
#     ax.set_ylabel("Correlation")
#     ax.set_title(f"Lagged cross-correlation: spawn → consumption{note}")
#     ax.grid(True, alpha=.25)
#     ax.axhline(0, color="black", linewidth=1)
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "lag_ccf_spawn_cons.png"))
#     plt.close(fig)

# def plot_spawn_vs_cons_scatter(df: pd.DataFrame, outdir: str, note: str):
#     """
#     Scatter over all days (aggregated across runs by day via sum).
#     Adds linear fit and R^2.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df[["day", "total_spawned", "total_consumed"]].copy()
#     for c in d.columns:
#         d[c] = pd.to_numeric(d[c], errors="coerce")

#     daily = d.groupby("day").sum(min_count=1)
#     x = daily["total_spawned"].to_numpy(dtype=float)
#     y = daily["total_consumed"].to_numpy(dtype=float)

#     # drop NaNs
#     m = np.isfinite(x) & np.isfinite(y)
#     x, y = x[m], y[m]

#     if len(x) >= 2:
#         a, b = np.polyfit(x, y, 1)           # y ≈ a*x + b
#         yhat = a * x + b
#         r = np.corrcoef(x, y)[0, 1]
#         r2 = r * r
#     else:
#         a = b = r2 = np.nan
#         yhat = np.array([])

#     fig, ax = plt.subplots(figsize=(6.5, 5))
#     ax.scatter(x, y, s=25, alpha=0.6, edgecolors="none")
#     if yhat.size:
#         order = np.argsort(x)
#         ax.plot(x[order], yhat[order], color="black", linewidth=2, label=f"fit: y={a:.2f}x+{b:.1f}  (R²={r2:.2f})")
#         ax.legend()
#     ax.set_xlabel("Spawned (units/day)")
#     ax.set_ylabel("Consumed (units/day)")
#     ax.set_title(f"Spawn vs consumption (daily aggregates){note}")
#     ax.grid(True, alpha=.25)
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "spawn_vs_cons_scatter.png"))
#     plt.close(fig)
# def plot_extinction_histograms(df: pd.DataFrame, outdir: str):
#     """
#     First day where population <= 0 (per run). Runs that never go extinct are excluded
#     from the histogram but reported in the titles as 'alive at end'.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df[["run", "day", "blue_pop", "red_pop"]].copy()
#     for c in d.columns:
#         d[c] = pd.to_numeric(d[c], errors="coerce")
#     runs = np.sort(d["run"].dropna().unique())

#     def first_ext_day(house_col: str) -> pd.Series:
#         ext = {}
#         for r in runs:
#             s = d.loc[d["run"] == r, ["day", house_col]].sort_values("day")
#             hit = s.loc[s[house_col] <= 0, "day"]
#             ext[r] = hit.iloc[0] if not hit.empty else np.nan
#         return pd.Series(ext, name=f"{house_col}_ext_day")

#     b_ext = first_ext_day("blue_pop")
#     r_ext = first_ext_day("red_pop")

#     alive_b = int(b_ext.isna().sum())
#     alive_r = int(r_ext.isna().sum())

#     bins = np.arange(0, int(d["day"].max()) + 2, 2)  # 2-day bins

#     fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
#     axes[0].hist(b_ext.dropna(), bins=bins, color="blue", alpha=0.7, edgecolor="white")
#     axes[0].set_title(f"Blue extinction days (alive at end: {alive_b})")
#     axes[0].set_xlabel("Day"); axes[0].set_ylabel("# runs"); axes[0].grid(True, axis="y", alpha=.25)

#     axes[1].hist(r_ext.dropna(), bins=bins, color="red", alpha=0.7, edgecolor="white")
#     axes[1].set_title(f"Red extinction days (alive at end: {alive_r})")
#     axes[1].set_xlabel("Day"); axes[1].grid(True, axis="y", alpha=.25)

#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "extinction_hist.png"))
#     plt.close(fig)
# def plot_survival_curves(df: pd.DataFrame, outdir: str):
#     """
#     Fraction of runs still alive vs day (Blue/Red).
#     A run is 'alive' for a house if its population > 0 at that day.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     d = df[["run", "day", "blue_pop", "red_pop"]].copy()
#     for c in ["run", "day", "blue_pop", "red_pop"]:
#         d[c] = pd.to_numeric(d[c], errors="coerce")

#     days = np.sort(d["day"].dropna().unique())
#     runs = np.sort(d["run"].dropna().unique())

#     surv_blue, surv_red = [], []
#     for day in days:
#         sub = d[d["day"] == day]
#         alive_b = (sub["blue_pop"] > 0).sum()
#         alive_r = (sub["red_pop"]  > 0).sum()
#         # normalize by number of runs that reached that day
#         surv_blue.append(alive_b / len(runs))
#         surv_red.append(alive_r / len(runs))

#     fig, ax = plt.subplots(figsize=(9, 4))
#     ax.plot(days, surv_blue, color="blue", linewidth=2.2, label="Blue")
#     ax.plot(days, surv_red,  color="red",  linewidth=2.2, label="Red")
#     ax.set_ylim(0, 1.01)
#     ax.set_xlabel("Day"); ax.set_ylabel("Fraction of runs alive")
#     ax.set_title("Survival curves by house")
#     ax.grid(True, alpha=.25); ax.legend()
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "survival_curves.png"))
#     plt.close(fig)

# def plot_population_consumption_trust_per_run(df: pd.DataFrame, S, outdir: str, suf: str, note: str):
#     """
#     Une figure par run avec 5 sous-graphiques superposés :
#       1) Population (Blue vs Red)
#       2) Consommation par zone (Z0, Z1, Z2)
#       3) Confiance (Within vs Between)
#     """
#     os.makedirs(outdir, exist_ok=True)

#     for r in sorted(df["run"].dropna().unique()):
#         d = df[df["run"] == r].sort_values("day")

#         fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

#         # --- (1) Population ---
#         blue = pd.to_numeric(d["blue_pop"], errors="coerce")
#         red  = pd.to_numeric(d["red_pop"], errors="coerce")
#         axes[0].plot(d["day"], S(blue), color="blue", linewidth=2, label="Blue pop")
#         axes[0].plot(d["day"], S(red),  color="red",  linewidth=2, label="Red pop")
#         axes[0].set_ylabel("Population")
#         axes[0].set_title(f"Run {r} — Population")
#         axes[0].grid(True, alpha=0.25)
#         axes[0].legend()

#         # --- (2) Consommation par zone (stacked area) ---
#         z0 = pd.to_numeric(d["z0_cons"], errors="coerce")
#         z1 = pd.to_numeric(d["z1_cons"], errors="coerce")
#         z2 = pd.to_numeric(d["z2_cons"], errors="coerce")
#         z3 = pd.to_numeric(d["z3_cons"], errors="coerce")
#         z4 = pd.to_numeric(d["z4_cons"], errors="coerce")

#         axes[1].stackplot(
#             d["day"], z0, z1, z2,z3,z4,
#             labels=[f"Zone {z}" for z in ZONES],
#             colors=["#1f77b4", "#d62728", "#2ca02c"],
#             alpha=0.7
#         )
#         axes[1].set_ylabel("Consommation")
#         axes[1].set_title(f"Run {r} — Consommation par zone (stacked)")
#         axes[1].grid(True, alpha=0.25)
#         axes[1].legend(loc="upper left")
#         # --- (3) Confiance ---
#         within  = pd.to_numeric(d["within_trust"], errors="coerce")
#         between = pd.to_numeric(d["between_trust"], errors="coerce")
#         axes[2].plot(d["day"], S(within),  color="green",  linewidth=2, label="Within trust")
#         axes[2].plot(d["day"], S(between), color="purple", linewidth=2, label="Between trust")
#         axes[2].set_ylabel("Trust")
#         axes[2].set_title(f"Run {r} — Variation de confiance")
#         axes[2].grid(True, alpha=0.25)
#         axes[2].legend()

#         axes[2].set_xlabel("Day")
#         fig.tight_layout()
#         fig.savefig(os.path.join(outdir, f"pop_cons_trust_run{r}{suf}.png"))
#         plt.close(fig)

# def plot_spawn_vs_cons_by_family_per_run(df: pd.DataFrame, S, outdir: str, suf: str, note: str):
#     """
#     Spawn vs Consommation par famille (Blue/Red) : une figure par run,
#     avec 2 sous-graphiques verticaux.
#     """
#     os.makedirs(outdir, exist_ok=True)

#     for r in sorted(df["run"].dropna().unique()):
#         d = df[df["run"] == r].sort_values("day")

#         fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#         for ax, fam, color in zip(axes, ["blue", "red"], ["blue", "red"]):
#             sp = pd.to_numeric(d[f"{fam}_born"], errors="coerce")
#             co = (
#                 pd.to_numeric(d[f"z0_{fam}_cons"], errors="coerce")
#               + pd.to_numeric(d[f"z1_{fam}_cons"], errors="coerce")
#               + pd.to_numeric(d[f"z2_{fam}_cons"], errors="coerce")
#             )

#             ax.plot(d["day"], S(sp), color=color, linestyle="-", linewidth=2, label=f"{fam.capitalize()} spawn")
#             ax.plot(d["day"], S(co), color=color, linestyle="--", linewidth=2, label=f"{fam.capitalize()} cons")
#             ax.set_title(f"Run {r} — {fam.capitalize()} spawn vs consommation")
#             ax.set_ylabel("Units/day")
#             ax.grid(True, alpha=0.25)
#             ax.legend()

#         axes[-1].set_xlabel("Day")
#         fig.suptitle(f"Run {r} — Spawn vs consommation par famille{note}")
#         fig.tight_layout(rect=[0, 0, 1, 0.96])
#         fig.savefig(os.path.join(outdir, f"spawn_cons_by_family_run{r}{suf}.png"))
#         plt.close(fig)













# if __name__ == "__main__":
#     main()

# # batch_plots.py
# import os
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MaxNLocator

# # ── Helpers ────────────────────────────────────────────────
# def detect_zones(df: pd.DataFrame) -> list[int]:
#     """Detect unique zone indices dynamically from CSV column names."""
#     zones = sorted({
#         int(c[1]) for c in df.columns
#         if c.startswith("z") and "_cons" in c and "_blue" not in c
#     })
#     return zones


# def _agg_across_runs(d: pd.DataFrame, value_col: str, smooth):
#     v = pd.to_numeric(d[value_col], errors="coerce")
#     piv = d.assign(v=v).pivot_table(index="day", columns="run", values="v", aggfunc="mean")
#     x   = piv.index.to_numpy()
#     mean = piv.mean(axis=1)
#     p10  = pd.Series(np.nanpercentile(piv.to_numpy(), 10, axis=1), index=piv.index)
#     p90  = pd.Series(np.nanpercentile(piv.to_numpy(), 90, axis=1), index=piv.index)

#     if smooth:
#         mean = smooth(mean); p10 = smooth(p10); p90 = smooth(p90)

#     return x, mean.astype(float).to_numpy(), p10.astype(float).to_numpy(), p90.astype(float).to_numpy()

# def make_smoother(enabled: bool, window: int):
#     suffix = f"_sm{window}" if enabled and window > 1 else ""
#     title_note = f" (rolling {window})" if enabled and window > 1 else ""
#     def S(s: pd.Series) -> pd.Series:
#         if enabled and window > 1 and len(s) > 1:
#             return s.rolling(window, min_periods=1).mean()
#         return s
#     return S, suffix, title_note

# def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
#     d = df.copy()
#     for c in cols:
#         if c in d.columns:
#             d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
#     return d

# # ── Plots ─────────────────────────────────────────────────
# # (most functions unchanged, only ZONES replaced with zones=detect_zones(df))

# def plot_zone_spawn_vs_consumption(df: pd.DataFrame, S, outdir, suf, note):
#     zones = detect_zones(df)
#     cols = [f"z{z}_spawn" for z in zones] + [f"z{z}_cons" for z in zones]
#     d = _ensure_numeric(df, cols).sort_values(["run", "day"])

#     fig, axes = plt.subplots(1, len(zones), figsize=(5*len(zones), 4.5), sharex=True, sharey=True)
#     if len(zones) == 1: axes = [axes]

#     for z, ax in zip(zones, axes):
#         x, sp_mean, sp_p10, sp_p90 = _agg_across_runs(d, f"z{z}_spawn", S)
#         _, co_mean, co_p10, co_p90 = _agg_across_runs(d, f"z{z}_cons",  S)

#         ax.fill_between(x, sp_p10, sp_p90, color="#2ca02c", alpha=0.18)
#         ax.fill_between(x, co_p10, co_p90, color="#ff7f0e", alpha=0.18)
#         ax.plot(x, sp_mean, color="#2ca02c", linewidth=2.2, label="Spawned (mean)")
#         ax.plot(x, co_mean, color="#ff7f0e", linewidth=2.2, label="Consumed (mean)")
#         ax.set_title(f"Zone {z}"); ax.set_xlabel("Day"); ax.grid(True, alpha=.25)

#     axes[0].set_ylabel("Units / day")
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
#     fig.suptitle(f"Spawn vs consumption — mean ± 10–90% across runs{note}", y=1.12)
#     fig.tight_layout(rect=[0, 0, 1, 0.98])
#     fig.savefig(os.path.join(outdir, f"zone_spawn_cons_ALL_runs{suf}.png"))
#     plt.close(fig)

# def plot_zone_consumption_by_house(df: pd.DataFrame, S, outdir, suf, note):
#     zones = detect_zones(df)
#     cols = [f"z{z}_blue_cons" for z in zones] + [f"z{z}_red_cons" for z in zones]
#     d = _ensure_numeric(df, cols)

#     fig, axes = plt.subplots(1, len(zones), figsize=(5*len(zones), 4), sharex=True, sharey=True)
#     if len(zones) == 1: axes = [axes]

#     for z, ax in zip(zones, axes):
#         bcol, rcol = f"z{z}_blue_cons", f"z{z}_red_cons"
#         g = d.groupby("day")
#         b_q10, b_med, b_q90 = S(g[bcol].quantile(0.1)), S(g[bcol].median()), S(g[bcol].quantile(0.9))
#         r_q10, r_med, r_q90 = S(g[rcol].quantile(0.1)), S(g[rcol].median()), S(g[rcol].quantile(0.9))
#         x = b_med.index.to_numpy()
#         ax.fill_between(x, b_q10, b_q90, color="blue", alpha=0.15)
#         ax.plot(x, b_med, color="blue", linewidth=2, label="Blue median")
#         ax.fill_between(x, r_q10, r_q90, color="red", alpha=0.15)
#         ax.plot(x, r_med, color="red", linewidth=2, label="Red median")
#         ax.set_title(f"Zone {z} — per-house consumption{note}")
#         ax.set_xlabel("Day"); ax.grid(True, alpha=0.25)

#     axes[0].set_ylabel("Units / day")
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2)
#     fig.tight_layout(rect=[0, 0, 1, 0.94])
#     fig.savefig(os.path.join(outdir, f"zone_house_cons_all_runs{suf}.png"))
#     plt.close(fig)

# def plot_consumption_share(df: pd.DataFrame, outdir, suf, note):
#     zones = detect_zones(df)
#     cols = [f"z{z}_cons" for z in zones]
#     d = _ensure_numeric(df, cols).sort_values(["run","day"])

#     g = d.groupby("day")[cols].sum(min_count=1)
#     denom = g.sum(axis=1).replace(0, np.nan)
#     share = (g.div(denom, axis=0) * 100).fillna(0.0)

#     fig, ax = plt.subplots(figsize=(14, 4.5))
#     bottom = np.zeros(len(share))
#     for z in zones:
#         vals = share[f"z{z}_cons"].to_numpy()
#         ax.bar(share.index, vals, bottom=bottom, label=f"Zone {z}")
#         bottom += vals

#     ax.set_ylim(0, 100); ax.set_ylabel("Share of daily consumption (%)")
#     ax.set_xlabel("Day"); ax.set_title(f"Daily consumption mix by zone — 100% stacked (all runs){note}")
#     ax.grid(True, axis="y", alpha=0.25); ax.legend(ncol=len(zones), loc="upper center", bbox_to_anchor=(0.5, 1.12))
#     fig.tight_layout(); fig.savefig(os.path.join(outdir, f"zone_mix_daily_pct_ALL_runs{suf}.png"), dpi=120)
#     plt.close(fig)

# def plot_overall_zone_exploitation_totals(df: pd.DataFrame, outdir):
#     zones = detect_zones(df)
#     totals_blue = np.array([df[f"z{z}_blue_cons"].sum() for z in zones], dtype=float)
#     totals_red  = np.array([df[f"z{z}_red_cons"].sum()  for z in zones], dtype=float)
#     zone_totals = totals_blue + totals_red

#     x = np.arange(len(zones))
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.bar(x, totals_blue, width=0.6, label="Blue", color="blue")
#     ax.bar(x, totals_red,  width=0.6, bottom=totals_blue, label="Red", color="red")
#     ax.set_xticks(x); ax.set_xticklabels([f"Zone {z}" for z in zones])
#     ax.set_ylabel("Total units consumed"); ax.set_title("Overall exploitation by zone and family")
#     ax.grid(True, axis="y", alpha=.25); ax.legend()
#     fig.tight_layout(); fig.savefig(os.path.join(outdir, "overall_zone_exploitation_stacked.png"))
#     plt.close(fig)

# def plot_spawn_vs_cons_by_family_per_run(df: pd.DataFrame, S, outdir, suf, note):
#     zones = detect_zones(df)
#     os.makedirs(outdir, exist_ok=True)

#     for r in sorted(df["run"].dropna().unique()):
#         d = df[df["run"] == r].sort_values("day")
#         fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#         for ax, fam, color in zip(axes, ["blue", "red"], ["blue", "red"]):
#             sp = pd.to_numeric(d[f"{fam}_born"], errors="coerce")
#             co = sum(pd.to_numeric(d.get(f"z{z}_{fam}_cons"), errors="coerce") for z in zones)
#             ax.plot(d["day"], S(sp), color=color, linestyle="-", linewidth=2, label=f"{fam.capitalize()} spawn")
#             ax.plot(d["day"], S(co), color=color, linestyle="--", linewidth=2, label=f"{fam.capitalize()} cons")
#             ax.set_title(f"Run {r} — {fam.capitalize()} spawn vs consommation")
#             ax.set_ylabel("Units/day"); ax.grid(True, alpha=0.25); ax.legend()

#         axes[-1].set_xlabel("Day")
#         fig.suptitle(f"Run {r} — Spawn vs consommation par famille{note}")
#         fig.tight_layout(rect=[0, 0, 1, 0.96])
#         fig.savefig(os.path.join(outdir, f"spawn_cons_by_family_run{r}{suf}.png"))
#         plt.close(fig)

# # ── Main ────────────────────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", default="batch_results/all_runs_combined.csv")
#     ap.add_argument("--out", default="batch_results/plots")
#     ap.add_argument("--smooth", action="store_true")
#     ap.add_argument("--window", type=int, default=7)
#     args = ap.parse_args()

#     os.makedirs(args.out, exist_ok=True)
#     df = pd.read_csv(args.csv).sort_values(["run","day"])
#     zones = detect_zones(df)
#     print(f"Detected zones: {zones}")

#     S, suf, note = make_smoother(args.smooth, args.window)

#     # Call whichever plots you need
#     plot_zone_spawn_vs_consumption(df, S, args.out, suf, note)
#     plot_zone_consumption_by_house(df, S, args.out, suf, note)
#     plot_consumption_share(df, args.out, suf, note)
#     plot_overall_zone_exploitation_totals(df, args.out)
#     plot_spawn_vs_cons_by_family_per_run(df, S, args.out, suf, note)

#     print(f"Saved plots to: {args.out}")

# if __name__ == "__main__":
#     main()


# batch_plots.py
import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ───────────────────────── Helpers ─────────────────────────

# palette for up to many zones (cycled automatically)
ZONE_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def detect_zones(df: pd.DataFrame) -> list[int]:
    """
    Detect unique zone indices from column names like z3_cons, z10_blue_cons, ...
    Robust to 2+ digit indices.
    """
    zones = set()
    for c in df.columns:
        m = re.match(r"^z(\d+)_cons$", c)  # total consumption column per zone
        if m:
            zones.add(int(m.group(1)))
    return sorted(zones)

def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def make_smoother(enabled: bool, window: int):
    suffix = f"_sm{window}" if enabled and window > 1 else ""
    title_note = f" (rolling {window})" if enabled and window > 1 else ""
    def S(s: pd.Series) -> pd.Series:
        if isinstance(s, pd.Series) and enabled and window > 1 and len(s) > 1:
            return s.rolling(window, min_periods=1).mean()
        return s
    return S, suffix, title_note

def _agg_across_runs(d: pd.DataFrame, value_col: str, smooth):
    """
    Return x, mean, p10, p90 for a metric across runs per day.
    Always returns plain float numpy arrays.
    """
    v = pd.to_numeric(d[value_col], errors="coerce")
    piv = d.assign(v=v).pivot_table(index="day", columns="run", values="v", aggfunc="mean")
    if piv.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])
    x    = piv.index.to_numpy()
    mean = piv.mean(axis=1)
    p10  = pd.Series(np.nanpercentile(piv.to_numpy(), 10, axis=1), index=piv.index)
    p90  = pd.Series(np.nanpercentile(piv.to_numpy(), 90, axis=1), index=piv.index)
    if smooth:
        mean = smooth(mean); p10 = smooth(p10); p90 = smooth(p90)
    return x, mean.astype(float).to_numpy(), p10.astype(float).to_numpy(), p90.astype(float).to_numpy()

def _sum_series(series_list: list[pd.Series]) -> pd.Series:
    """Elementwise sum of a list of Series with fill_value=0 and aligned index."""
    if not series_list:
        return pd.Series(dtype=float)
    out = series_list[0].astype(float)
    for s in series_list[1:]:
        out = out.add(s.astype(float), fill_value=0.0)
    return out

# ───────────────────────── Plots ─────────────────────────

def plot_population(df: pd.DataFrame, S, outdir, suf, note,
                    show_band: bool = True, alpha_runs: float = 0.10):
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()

    plt.figure(figsize=(10, 5))
    # thin overlay per run
    for r in sorted(d["run"].dropna().unique()):
        sub = d[d.run == r]
        x  = sub.day.to_numpy()
        yb = pd.to_numeric(S(sub.blue_pop), errors="coerce").astype(float).to_numpy()
        yr = pd.to_numeric(S(sub.red_pop),  errors="coerce").astype(float).to_numpy()
        plt.plot(x, yb, color="blue", alpha=alpha_runs, linewidth=1, label="_nolegend_")
        plt.plot(x, yr, color="red",  alpha=alpha_runs, linewidth=1, label="_nolegend_")

    blue_piv = d.pivot_table(index="day", columns="run", values="blue_pop", aggfunc="mean")
    red_piv  = d.pivot_table(index="day", columns="run", values="red_pop",  aggfunc="mean")
    blue_mean, red_mean = S(blue_piv.mean(axis=1)), S(red_piv.mean(axis=1))

    x = blue_mean.index.to_numpy()
    plt.plot(x, blue_mean.astype(float).to_numpy(), color="blue", linewidth=2.5, label="Blue mean")
    plt.plot(x, red_mean.astype(float).to_numpy(),  color="red",  linewidth=2.5, label="Red mean")

    if show_band:
        for piv, color, label in [(blue_piv,"blue","Blue"), (red_piv,"red","Red")]:
            lo = S(piv.quantile(0.10, axis=1)).astype(float).to_numpy()
            hi = S(piv.quantile(0.90, axis=1)).astype(float).to_numpy()
            plt.fill_between(x, lo, hi, color=color, alpha=0.12, label=f"{label} 10–90%")

    plt.xlabel("Day"); plt.ylabel("Population")
    plt.title(f"Population — all runs{note}")
    plt.grid(True, alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pop_ALL_runs{suf}.png"))
    plt.close()

def plot_population_per_run(df: pd.DataFrame, S, outdir, suf, note):
    os.makedirs(outdir, exist_ok=True)
    for r in sorted(df["run"].dropna().unique()):
        d = df[df.run == r].sort_values("day")
        x  = d["day"].to_numpy()
        yb = pd.to_numeric(S(d["blue_pop"]), errors="coerce").astype(float).to_numpy()
        yr = pd.to_numeric(S(d["red_pop"]),  errors="coerce").astype(float).to_numpy()
        plt.figure(figsize=(9, 4))
        plt.plot(x, yb, color="blue", linewidth=2, label="Blue")
        plt.plot(x, yr, color="red",  linewidth=2, label="Red")
        plt.xlabel("Day"); plt.ylabel("Population")
        plt.title(f"Population — run {r}{note}")
        plt.grid(True, alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"pop_run{r}{suf}.png"))
        plt.close()

def plot_births_deaths_all_runs(df: pd.DataFrame, S, outdir, suf, note):
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()
    x, blue_b_mean, *_ = _agg_across_runs(d, "blue_born", S)
    _,  red_b_mean,  *_ = _agg_across_runs(d, "red_born",  S)
    _, blue_d_mean,  *_ = _agg_across_runs(d, "blue_dead", S)
    _,  red_d_mean,   *_ = _agg_across_runs(d, "red_dead",  S)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    l1, = ax1.plot(x, blue_b_mean, color="blue", linewidth=2.2, label="Blue births/day (mean)")
    l2, = ax1.plot(x, red_b_mean,  color="red",  linewidth=2.2, label="Red births/day (mean)")
    ax1.set_xlabel("Day"); ax1.set_ylabel("Births per day"); ax1.grid(True, alpha=.25)

    ax2 = ax1.twinx()
    l3, = ax2.plot(x, blue_d_mean, color="blue", linestyle="--", linewidth=2.2, label="Blue deaths (cum, mean)")
    l4, = ax2.plot(x, red_d_mean,  color="red",  linestyle="--", linewidth=2.2, label="Red deaths (cum, mean)")
    ax2.set_ylabel("Cumulative deaths")

    lines = [l1, l2, l3, l4]; labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")
    plt.title(f"Births & deaths — all runs (means only){note}")
    fig.tight_layout(); plt.savefig(os.path.join(outdir, f"births_deaths_ALL_runs{suf}.png")); plt.close(fig)

def plot_trust_agg(df: pd.DataFrame, S, outdir, suf, note):
    from matplotlib.patches import Patch
    os.makedirs(outdir, exist_ok=True)

    def _agg_col(col: str):
        s = pd.to_numeric(df[col], errors="coerce")
        g = s.groupby(df["day"])
        m, p10, p90 = g.mean().sort_index(), g.quantile(0.10).sort_index(), g.quantile(0.90).sort_index()
        return S(m), S(p10), S(p90)

    # A) Within vs Between
    w_m, w_p10, w_p90 = _agg_col("within_trust")
    b_m, b_p10, b_p90 = _agg_col("between_trust")
    x = sorted(set(w_m.index) | set(b_m.index))
    wM, wL, wU = w_m.reindex(x), w_p10.reindex(x), w_p90.reindex(x)
    bM, bL, bU = b_m.reindex(x), b_p10.reindex(x), b_p90.reindex(x)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, wL, wU, color="gray", alpha=0.18)
    ax.plot(x, wM, color="green", linewidth=2, label="Within mean")
    ax.fill_between(x, bL, bU, color="mediumorchid", alpha=0.18)
    ax.plot(x, bM, color="purple", linewidth=2, label="Between mean")
    band_legend = [Patch(facecolor="gray", alpha=0.18, label="Within 10–90%"),
                   Patch(facecolor="mediumorchid", alpha=0.18, label="Between 10–90%")]
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], *band_legend], loc="upper left")
    ax.set_xlabel("Day"); ax.set_ylabel("Trust"); ax.set_title(f"Trust — Within vs Between (all runs){note}")
    ax.grid(True, alpha=.25); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"trust_within_between_all_runs{suf}.png")); plt.close(fig)

    # B) Within Blue vs Within Red
    bl_m, bl_p10, bl_p90 = _agg_col("within_blue_trust")
    rd_m, rd_p10, rd_p90 = _agg_col("within_red_trust")
    x = sorted(set(bl_m.index) | set(rd_m.index))
    blM, blL, blU = bl_m.reindex(x), bl_p10.reindex(x), bl_p90.reindex(x)
    rdM, rdL, rdU = rd_m.reindex(x), rd_p10.reindex(x), rd_p90.reindex(x)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, blL, blU, color="blue", alpha=0.12); ax.plot(x, blM, color="blue", linewidth=2, label="Within Blue mean")
    ax.fill_between(x, rdL, rdU, color="red",  alpha=0.12); ax.plot(x, rdM, color="red",  linewidth=2, label="Within Red mean")
    band_legend = [Patch(facecolor="blue", alpha=0.12, label="Blue 10–90%"),
                   Patch(facecolor="red",  alpha=0.12, label="Red 10–90%")]
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], *band_legend], loc="upper left")
    ax.set_xlabel("Day"); ax.set_ylabel("Trust"); ax.set_title(f"Trust — Within Blue vs Within Red (all runs){note}")
    ax.grid(True, alpha=.25); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"trust_by_house_all_runs{suf}.png")); plt.close(fig)

def plot_total_spawn_vs_consumption(df: pd.DataFrame, S, outdir, suf, note):
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()
    d = _ensure_numeric(d, ["total_spawned", "total_consumed"])
    x, sp_mean, sp_p10, sp_p90 = _agg_across_runs(d, "total_spawned", S)
    _,  co_mean, co_p10, co_p90 = _agg_across_runs(d, "total_consumed", S)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(x, sp_p10, sp_p90, color="#2ca02c", alpha=0.18, label="Spawned 10–90%")
    ax.fill_between(x, co_p10, co_p90, color="#ff7f0e", alpha=0.18, label="Consumed 10–90%")
    ax.plot(x, sp_mean, color="#2ca02c", linewidth=2.2, label="Spawned (mean)")
    ax.plot(x, co_mean, color="#ff7f0e", linewidth=2.2, label="Consumed (mean)")
    ax.set_xlabel("Day"); ax.set_ylabel("Units / day")
    ax.set_title(f"Global spawn vs consumption — all runs (mean ± 10–90%){note}")
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"global_spawn_cons_ALL_runs{suf}.png")); plt.close(fig)

def plot_zone_spawn_vs_consumption(df: pd.DataFrame, S, outdir, suf, note):
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)
    cols = [f"z{z}_spawn" for z in zones] + [f"z{z}_cons" for z in zones]
    d = _ensure_numeric(df, cols).sort_values(["run", "day"])

    W = max(12, 4 * len(zones))
    fig, axes = plt.subplots(1, len(zones), figsize=(W, 4.5), sharex=True, sharey=True)
    if len(zones) == 1: axes = [axes]

    for idx, (z, ax) in enumerate(zip(zones, axes)):
        x, sp_mean, sp_p10, sp_p90 = _agg_across_runs(d, f"z{z}_spawn", S)
        _,  co_mean, co_p10, co_p90 = _agg_across_runs(d, f"z{z}_cons",  S)
        ax.fill_between(x, sp_p10, sp_p90, color="#2ca02c", alpha=0.18, label="Spawned 10–90%")
        ax.fill_between(x, co_p10, co_p90, color="#ff7f0e", alpha=0.18, label="Consumed 10–90%")
        ax.plot(x, sp_mean, color="#2ca02c", linewidth=2.2, label="Spawned (mean)")
        ax.plot(x, co_mean, color="#ff7f0e", linewidth=2.2, label="Consumed (mean)")
        ax.set_title(f"Zone {z}"); ax.set_xlabel("Day"); ax.grid(True, alpha=.25)
        if idx == 0: ax.set_ylabel("Units / day")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Spawn vs consumption — mean ± 10–90% across runs{note}", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(outdir, f"zone_spawn_cons_ALL_runs{suf}.png")); plt.close(fig)

def plot_zone_consumption_by_house(df: pd.DataFrame, S, outdir, suf, note):
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)
    cols = [f"z{z}_blue_cons" for z in zones] + [f"z{z}_red_cons" for z in zones]
    d = _ensure_numeric(df, cols)

    W = max(12, 4 * len(zones))
    fig, axes = plt.subplots(1, len(zones), figsize=(W, 4), sharex=True, sharey=True)
    if len(zones) == 1: axes = [axes]

    for idx, (z, ax) in enumerate(zip(zones, axes)):
        bcol, rcol = f"z{z}_blue_cons", f"z{z}_red_cons"
        tmp = d[["day", bcol, rcol]].dropna(subset=["day"]).copy()
        tmp["day"] = pd.to_numeric(tmp["day"], errors="coerce")
        tmp = tmp.dropna(subset=["day"]).sort_values("day")
        g = tmp.groupby("day", dropna=True)

        b_q10 = S(g[bcol].quantile(0.10)); b_med = S(g[bcol].median()); b_q90 = S(g[bcol].quantile(0.90))
        r_q10 = S(g[rcol].quantile(0.10)); r_med = S(g[rcol].median()); r_q90 = S(g[rcol].quantile(0.90))
        x = b_med.index.to_numpy()

        ax.fill_between(x, b_q10, b_q90, color="blue", alpha=0.15, label="Blue 10–90%")
        ax.plot(x, b_med, color="blue", linewidth=2, label="Blue median")
        ax.fill_between(x, r_q10, r_q90, color="red", alpha=0.15, label="Red 10–90%")
        ax.plot(x, r_med, color="red", linewidth=2, label="Red median")

        ax.set_title(f"Zone {z} — per-house consumption{note}")
        ax.set_xlabel("Day"); ax.grid(True, alpha=0.25)
        if idx == 0: ax.set_ylabel("Units / day")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, f"zone_house_cons_all_runs{suf}.png")); plt.close(fig)

def plot_consumption_share(df: pd.DataFrame, outdir: str, suf: str, note: str):
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)
    cols = [f"z{z}_cons" for z in zones]
    d = _ensure_numeric(df, cols).sort_values(["run","day"])

    g = d.groupby("day")[cols].sum(min_count=1)
    denom = g.sum(axis=1).replace(0, np.nan)
    share = (g.div(denom, axis=0) * 100).fillna(0.0)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    bottom = np.zeros(len(share))
    for i, z in enumerate(zones):
        vals = share[f"z{z}_cons"].to_numpy()
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        ax.bar(share.index, vals, bottom=bottom, color=color, label=f"Zone {z}")
        bottom += vals

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(12, integer=True))
    ax.set_ylabel("Share of daily consumption (%)")
    ax.set_xlabel("Day")
    ax.set_title(f"Daily consumption mix by zone — 100% stacked (all runs){note}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=min(6, len(zones)), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"zone_mix_daily_pct_ALL_runs{suf}.png"), dpi=120)
    plt.close(fig)

def plot_per_capita_all_runs(df: pd.DataFrame, S, outdir, suf, note):
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()
    zones = detect_zones(d)

    # Make columns numeric
    d["blue_pop"] = pd.to_numeric(d["blue_pop"], errors="coerce")
    d["red_pop"]  = pd.to_numeric(d["red_pop"],  errors="coerce")
    blue_series = [pd.to_numeric(d.get(f"z{z}_blue_cons"), errors="coerce") for z in zones if f"z{z}_blue_cons" in d]
    red_series  = [pd.to_numeric(d.get(f"z{z}_red_cons"),  errors="coerce") for z in zones if f"z{z}_red_cons"  in d]

    blue_cons = _sum_series(blue_series)
    red_cons  = _sum_series(red_series)

    blue_pop = d["blue_pop"].astype(float).where(d["blue_pop"] > 0, np.nan)
    red_pop  = d["red_pop"].astype(float).where(d["red_pop"]  > 0, np.nan)
    d["blue_pc"] = (blue_cons / blue_pop).astype(float)
    d["red_pc"]  = (red_cons  / red_pop ).astype(float)

    blue_stats = d.groupby("day")["blue_pc"].agg(mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
    red_stats  = d.groupby("day")["red_pc" ].agg(mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))

    all_days = pd.Index(sorted(set(blue_stats.index) | set(red_stats.index)))
    blue_stats, red_stats = blue_stats.reindex(all_days), red_stats.reindex(all_days)

    x         = all_days.to_numpy()
    blue_mean = pd.Series(S(blue_stats["mean"])).astype(float).to_numpy()
    blue_p10  = pd.Series(S(blue_stats["p10"])).astype(float).to_numpy()
    blue_p90  = pd.Series(S(blue_stats["p90"])).astype(float).to_numpy()
    red_mean  = pd.Series(S(red_stats["mean"])).astype(float).to_numpy()
    red_p10   = pd.Series(S(red_stats["p10"])).astype(float).to_numpy()
    red_p90   = pd.Series(S(red_stats["p90"])).astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, blue_p10, blue_p90, color="blue", alpha=0.15, label="Blue 10–90%")
    ax.fill_between(x, red_p10,  red_p90,  color="red",  alpha=0.12, label="Red 10–90%")
    ax.plot(x, blue_mean, color="blue", linewidth=2.2, label="Blue mean")
    ax.plot(x, red_mean,  color="red",  linewidth=2.2, label="Red mean")
    ax.set_xlabel("Day"); ax.set_ylabel("Per-capita consumption (units/person/day)")
    ax.set_title(f"Per-capita consumption — all runs (mean ± 10–90%){note}")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"per_capita_ALL_runs{suf}.png")); plt.close(fig)

def plot_overall_zone_exploitation_totals(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)
    totals_blue = np.array([pd.to_numeric(df[f"z{z}_blue_cons"], errors="coerce").sum() for z in zones], dtype=float)
    totals_red  = np.array([pd.to_numeric(df[f"z{z}_red_cons"],  errors="coerce").sum() for z in zones], dtype=float)

    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, totals_blue, width=0.6, label="Blue", color="blue")
    ax.bar(x, totals_red,  width=0.6, bottom=totals_blue, label="Red", color="red")
    ax.set_xticks(x); ax.set_xticklabels([f"Zone {z}" for z in zones])
    ax.set_ylabel("Total units consumed (all runs & days)")
    ax.set_title("Overall exploitation by zone and family")
    ax.grid(True, axis="y", alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "overall_zone_exploitation_stacked.png"))
    plt.close(fig)

def plot_zone_dominance_by_run(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)
    d = df.copy()
    d["run"] = pd.to_numeric(d["run"], errors="coerce")

    runs = np.sort(d["run"].dropna().unique())
    blue_wins, red_wins, ties, no_data = [], [], [], []

    for z in zones:
        shares = []
        for r in runs:
            sub = d[d["run"] == r]
            b_sum = pd.to_numeric(sub.get(f"z{z}_blue_cons"), errors="coerce").sum()
            r_sum = pd.to_numeric(sub.get(f"z{z}_red_cons"),  errors="coerce").sum()
            tot = b_sum + r_sum
            shares.append(np.nan if tot <= 0 else (b_sum / tot))
        s = pd.Series(shares, index=runs)
        blue_wins.append(int((s > 0.5).sum()))
        red_wins .append(int((s < 0.5).sum()))
        ties     .append(int((s == 0.5).sum()))
        no_data  .append(int(s.isna().sum()))

    x = np.arange(len(zones)); w = 0.22
    fig, ax = plt.subplots(figsize=(10, 4.5))
    b = ax.bar(x - w, blue_wins, width=w, color='blue',  label="Blue wins (runs)")
    r = ax.bar(x,       red_wins,  width=w, color='red',   label="Red wins (runs)")
    t = ax.bar(x + w,   ties,      width=w, color='grey',  label="Ties")
    nd = ax.bar(x, no_data, width=0.65, color='none', edgecolor='#666', linewidth=1, hatch='////', label="No data")

    ax.set_xticks(x); ax.set_xticklabels([f"Zone {z}" for z in zones])
    ax.set_ylabel("# of runs"); ax.set_title("Which family exploited each zone most (by run)")
    ax.grid(True, axis="y", alpha=.25); ax.legend(loc="upper left")

    totals = np.array(blue_wins) + np.array(red_wins) + np.array(ties) + np.array(no_data)
    def _annotate(bar_container, counts):
        for rect, c, tot in zip(bar_container, counts, totals):
            if tot > 0 and c > 0:
                pct = 100.0 * c / tot
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.05,
                        f"{c} ({pct:.0f}%)", ha='center', va='bottom', fontsize=9)
    _annotate(b, blue_wins); _annotate(r, red_wins); _annotate(t, ties)

    fig.tight_layout(); fig.savefig(os.path.join(outdir, "zone_dominance_by_run.png")); plt.close(fig)

def plot_cons_spawn_ratio_by_zone(df: pd.DataFrame, S, outdir: str, suf: str, note: str):
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)
    eps = 1e-9
    W = max(12, 4 * len(zones))
    fig, axes = plt.subplots(1, len(zones), figsize=(W, 4), sharex=True, sharey=True)
    if len(zones) == 1: axes = [axes]

    for idx, (z, ax) in enumerate(zip(zones, axes)):
        cols = [f"z{z}_spawn", f"z{z}_cons"]
        d = df[["day", *cols]].copy()
        for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
        daily = d.groupby("day")[cols].sum(min_count=1).sort_index()
        ratio = S(daily[f"z{z}_cons"] / (daily[f"z{z}_spawn"] + eps))
        ax.plot(ratio.index.to_numpy(), ratio.to_numpy(), linewidth=2.0)
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(f"Zone {z} — cons/spawn{note}")
        ax.set_xlabel("Day"); ax.grid(True, alpha=.25)
        if idx == 0: ax.set_ylabel("Ratio (consumed / spawned)")

    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"cons_spawn_ratio_by_zone{suf}.png")); plt.close(fig)

def plot_lag_ccf_spawn_cons(df: pd.DataFrame, outdir: str, max_lag: int = 10, note: str = ""):
    os.makedirs(outdir, exist_ok=True)
    d = df[["day", "total_spawned", "total_consumed"]].copy()
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    daily = d.groupby("day").sum(min_count=1).sort_index()
    x = daily["total_spawned"].to_numpy(dtype=float)
    y = daily["total_consumed"].to_numpy(dtype=float)

    def _corr_at_lag(a, b, lag):
        if lag > 0:   a_, b_ = a[:-lag], b[lag:]
        elif lag < 0: a_, b_ = a[-lag:], b[:lag]
        else:         a_, b_ = a, b
        m = np.isfinite(a_) & np.isfinite(b_)
        return np.corrcoef(a_[m], b_[m])[0, 1] if m.sum() >= 2 else np.nan

    lags = np.arange(-max_lag, max_lag + 1)
    ccf  = np.array([_corr_at_lag(x, y, L) for L in lags], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    markerline, stemlines, baseline = ax.stem(lags, ccf)   # no use_line_collection in new mpl
    plt.setp(stemlines, linewidth=1.5); plt.setp(markerline, markersize=4); baseline.set_visible(False)
    ax.set_xlabel("Lag (days) — positive = spawn leads consumption")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Lagged cross-correlation: spawn → consumption{note}")
    ax.grid(True, alpha=.25); ax.axhline(0, color="black", linewidth=1)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "lag_ccf_spawn_cons.png")); plt.close(fig)

def plot_spawn_vs_cons_scatter(df: pd.DataFrame, outdir: str, note: str):
    os.makedirs(outdir, exist_ok=True)
    d = df[["day", "total_spawned", "total_consumed"]].copy()
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    daily = d.groupby("day").sum(min_count=1)
    x = daily["total_spawned"].to_numpy(dtype=float)
    y = daily["total_consumed"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]

    if len(x) >= 2:
        a, b = np.polyfit(x, y, 1); yhat = a * x + b
        r2 = np.corrcoef(x, y)[0, 1] ** 2
    else:
        yhat = np.array([]); a = b = r2 = np.nan

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, s=25, alpha=0.6, edgecolors="none")
    if yhat.size:
        order = np.argsort(x)
        ax.plot(x[order], yhat[order], color="black", linewidth=2, label=f"fit: y={a:.2f}x+{b:.1f}  (R²={r2:.2f})")
        ax.legend()
    ax.set_xlabel("Spawned (units/day)"); ax.set_ylabel("Consumed (units/day)")
    ax.set_title(f"Spawn vs consumption (daily aggregates){note}")
    ax.grid(True, alpha=.25); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spawn_vs_cons_scatter.png")); plt.close(fig)

def plot_extinction_histograms(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    d = df[["run", "day", "blue_pop", "red_pop"]].copy()
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    runs = np.sort(d["run"].dropna().unique())

    def first_ext_day(house_col: str) -> pd.Series:
        ext = {}
        for r in runs:
            s = d.loc[d["run"] == r, ["day", house_col]].sort_values("day")
            hit = s.loc[s[house_col] <= 0, "day"]
            ext[r] = hit.iloc[0] if not hit.empty else np.nan
        return pd.Series(ext)

    b_ext, r_ext = first_ext_day("blue_pop"), first_ext_day("red_pop")
    alive_b, alive_r = int(b_ext.isna().sum()), int(r_ext.isna().sum())
    bins = np.arange(0, int(d["day"].max()) + 2, 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].hist(b_ext.dropna(), bins=bins, color="blue", alpha=0.7, edgecolor="white")
    axes[0].set_title(f"Blue extinction days (alive at end: {alive_b})")
    axes[0].set_xlabel("Day"); axes[0].set_ylabel("# runs"); axes[0].grid(True, axis="y", alpha=.25)
    axes[1].hist(r_ext.dropna(), bins=bins, color="red", alpha=0.7, edgecolor="white")
    axes[1].set_title(f"Red extinction days (alive at end: {alive_r})")
    axes[1].set_xlabel("Day"); axes[1].grid(True, axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "extinction_hist.png")); plt.close(fig)

def plot_survival_curves(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    d = df[["run", "day", "blue_pop", "red_pop"]].copy()
    for c in ["run", "day", "blue_pop", "red_pop"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    days = np.sort(d["day"].dropna().unique()); runs = np.sort(d["run"].dropna().unique())
    surv_blue, surv_red = [], []
    for day in days:
        sub = d[d["day"] == day]
        surv_blue.append((sub["blue_pop"] > 0).sum() / len(runs))
        surv_red.append( (sub["red_pop"]  > 0).sum() / len(runs))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(days, surv_blue, color="blue", linewidth=2.2, label="Blue")
    ax.plot(days, surv_red,  color="red",  linewidth=2.2, label="Red")
    ax.set_ylim(0, 1.01); ax.set_xlabel("Day"); ax.set_ylabel("Fraction of runs alive")
    ax.set_title("Survival curves by house"); ax.grid(True, alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "survival_curves.png")); plt.close(fig)

def plot_population_consumption_trust_per_run(df: pd.DataFrame, S, outdir: str, suf: str, note: str):
    """
    One figure per run with **3** stacked subplots:
      (1) Population (Blue vs Red)
      (2) Consumption by zone (stacked area)
      (3) Trust (Within vs Between)
    """
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)

    for r in sorted(df["run"].dropna().unique()):
        d = df[df["run"] == r].sort_values("day")
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

        # (1) Population
        blue = pd.to_numeric(d["blue_pop"], errors="coerce")
        red  = pd.to_numeric(d["red_pop"],  errors="coerce")
        axes[0].plot(d["day"], S(blue), color="blue", linewidth=2, label="Blue pop")
        axes[0].plot(d["day"], S(red),  color="red",  linewidth=2, label="Red pop")
        axes[0].set_ylabel("Population"); axes[0].set_title(f"Run {r} — Population")
        axes[0].grid(True, alpha=0.25); axes[0].legend()

        # (2) Zone consumption (stacked area)
        zone_series = [pd.to_numeric(d[f"z{z}_cons"], errors="coerce") for z in zones]
        colors = [ZONE_COLORS[i % len(ZONE_COLORS)] for i in range(len(zones))]
        axes[1].stackplot(d["day"], *zone_series, labels=[f"Zone {z}" for z in zones], colors=colors, alpha=0.75)
        axes[1].set_ylabel("Consumption"); axes[1].set_title(f"Run {r} — Consommation par zone (stacked)")
        axes[1].grid(True, alpha=0.25); axes[1].legend(loc="upper left")

        # (3) Trust
        within  = pd.to_numeric(d["within_trust"],  errors="coerce")
        between = pd.to_numeric(d["between_trust"], errors="coerce")
        axes[2].plot(d["day"], S(within),  color="green",  linewidth=2, label="Within trust")
        axes[2].plot(d["day"], S(between), color="purple", linewidth=2, label="Between trust")
        axes[2].set_ylabel("Trust"); axes[2].set_title(f"Run {r} — Variation de confiance")
        axes[2].grid(True, alpha=0.25); axes[2].legend()

        axes[-1].set_xlabel("Day")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"pop_cons_trust_run{r}{suf}.png")); plt.close(fig)

def plot_spawn_vs_cons_by_family_per_run(df: pd.DataFrame, S, outdir: str, suf: str, note: str):
    """
    Spawn vs Consumption by family (Blue/Red): one figure per run, 2 rows.
    """
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)

    for r in sorted(df["run"].dropna().unique()):
        d = df[df["run"] == r].sort_values("day")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for ax, fam, color in zip(axes, ["blue", "red"], ["blue", "red"]):
            sp = pd.to_numeric(d[f"{fam}_born"], errors="coerce")
            series_list = [pd.to_numeric(d[f"z{z}_{fam}_cons"], errors="coerce") for z in zones if f"z{z}_{fam}_cons" in d]
            co = _sum_series(series_list)
            ax.plot(d["day"], S(sp), color=color, linestyle="-",  linewidth=2, label=f"{fam.capitalize()} spawn")
            ax.plot(d["day"], S(co), color=color, linestyle="--", linewidth=2, label=f"{fam.capitalize()} cons")
            ax.set_title(f"Run {r} — {fam.capitalize()} spawn vs consommation")
            ax.set_ylabel("Units/day"); ax.grid(True, alpha=0.25); ax.legend()

        axes[-1].set_xlabel("Day")
        fig.suptitle(f"Run {r} — Spawn vs consommation par famille{note}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(outdir, f"spawn_cons_by_family_run{r}{suf}.png")); plt.close(fig)

# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default="batch_results/all_244_combined_1trust.csv")
    ap.add_argument("--out",    default="batch_results/plots_seed244_1trust")
    ap.add_argument("--smooth", action="store_true", help="Enable rolling-average smoothing")
    ap.add_argument("--window", type=int, default=7, help="Rolling window (days)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv).sort_values(["run","day"])
    S, suf, note = make_smoother(args.smooth, args.window)

    print(f"Detected zones: {detect_zones(df)}")

    # Core, non-duplicative plots
    plot_population(df, S, args.out, suf, note)
    plot_population_per_run(df, S, args.out, suf, note)
    plot_births_deaths_all_runs(df, S, args.out, suf, note)
    plot_trust_agg(df, S, args.out, suf, note)
    plot_total_spawn_vs_consumption(df, S, args.out, suf, note)
    plot_zone_spawn_vs_consumption(df, S, args.out, suf, note)
    plot_zone_consumption_by_house(df, S, args.out, suf, note)
    plot_consumption_share(df, args.out, suf, note)
    plot_per_capita_all_runs(df, S, args.out, suf, note)
    plot_overall_zone_exploitation_totals(df, args.out)
    plot_zone_dominance_by_run(df, args.out)

    # Diagnostics
    plot_survival_curves(df, args.out)
    plot_extinction_histograms(df, args.out)
    plot_spawn_vs_cons_scatter(df, args.out, note)
    plot_lag_ccf_spawn_cons(df, args.out, max_lag=10, note=note)
    plot_cons_spawn_ratio_by_zone(df, S, args.out, suf, note)

    # Per-run detail
    plot_population_consumption_trust_per_run(df, S, args.out, suf, note)
    plot_spawn_vs_cons_by_family_per_run(df, S, args.out, suf, note)

    print(f"Saved plots to: {args.out}")

if __name__ == "__main__":
    main()
