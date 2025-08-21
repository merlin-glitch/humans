


# batch_plots.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ZONES = [0, 1, 2]


def _agg_across_runs(d: pd.DataFrame, value_col: str, smooth):
    """Return x, mean, p10, p90 for a metric across runs per day."""
    v = pd.to_numeric(d[value_col], errors="coerce")
    piv = d.assign(v=v).pivot_table(index="day", columns="run", values="v", aggfunc="mean")
    x   = piv.index.to_numpy()
    mean = piv.mean(axis=1)
    p10  = pd.Series(np.nanpercentile(piv.to_numpy(), 10, axis=1), index=piv.index)
    p90  = pd.Series(np.nanpercentile(piv.to_numpy(), 90, axis=1), index=piv.index)

    if smooth:
        mean = smooth(mean)
        p10  = smooth(p10)
        p90  = smooth(p90)

    # ensure plain float arrays (Matplotlib can’t handle pd.NA)
    mean = pd.to_numeric(mean, errors="coerce").astype(float).to_numpy()
    p10  = pd.to_numeric(p10,  errors="coerce").astype(float).to_numpy()
    p90  = pd.to_numeric(p90,  errors="coerce").astype(float).to_numpy()
    return x, mean, p10, p90

def _runs(df: pd.DataFrame) -> list[int]:
    return sorted(df["run"].dropna().unique().tolist())

def make_smoother(enabled: bool, window: int):
    suffix = f"_sm{window}" if enabled and window > 1 else ""
    title_note = f" (rolling {window})" if enabled and window > 1 else ""
    def S(s: pd.Series) -> pd.Series:
        if enabled and window > 1 and len(s) > 1:
            return s.rolling(window, min_periods=1).mean()
        return s
    return S, suffix, title_note

def plot_population(df: pd.DataFrame, S, outdir, suf, note,
                    show_band: bool = True, alpha_runs: float = 0.10):
    """
    One figure for ALL runs:
      - thin translucent line per run (blue/red)
      - bold mean line per family (blue/red)
      - optional 10–90% band
    """
    os.makedirs(outdir, exist_ok=True)

    # Sort for safety
    df = df.sort_values(["run", "day"])

    plt.figure(figsize=(10, 5))

    # 1) Overlay each run lightly
    for r in sorted(df["run"].dropna().unique()):
        d = df[df.run == r]
        # Cast to float so matplotlib doesn't choke on pd.NA
        x  = d.day.to_numpy()
        yb = pd.to_numeric(S(d.blue_pop), errors="coerce").astype(float).to_numpy()
        yr = pd.to_numeric(S(d.red_pop),  errors="coerce").astype(float).to_numpy()

        plt.plot(x, yb, color="blue", alpha=alpha_runs, linewidth=1, label="_nolegend_")
        plt.plot(x, yr, color="red",  alpha=alpha_runs, linewidth=1, label="_nolegend_")

    # 2) Mean & bands across runs (pivot -> aggregate by day)
    blue_piv = df.pivot_table(index="day", columns="run", values="blue_pop", aggfunc="mean")
    red_piv  = df.pivot_table(index="day", columns="run", values="red_pop",  aggfunc="mean")

    # stats per day
    blue_mean = blue_piv.mean(axis=1)
    red_mean  = red_piv.mean(axis=1)

    if show_band:
        blue_lo = blue_piv.quantile(0.10, axis=1)
        blue_hi = blue_piv.quantile(0.90, axis=1)
        red_lo  = red_piv.quantile(0.10, axis=1)
        red_hi  = red_piv.quantile(0.90, axis=1)
        # smoothing
        blue_lo, blue_hi = S(blue_lo), S(blue_hi)
        red_lo,  red_hi  = S(red_lo),  S(red_hi)

    # smooth means
    blue_mean = S(blue_mean)
    red_mean  = S(red_mean)

    # Ensure numeric arrays (NaNs are fine)
    x = blue_mean.index.to_numpy()
    plt.plot(x, blue_mean.astype(float).to_numpy(), color="blue", linewidth=2.5, label="Blue mean")
    plt.plot(x, red_mean.astype(float).to_numpy(),  color="red",  linewidth=2.5, label="Red mean")

    if show_band:
        plt.fill_between(
            x,
            blue_lo.astype(float).to_numpy(),
            blue_hi.astype(float).to_numpy(),
            color="blue", alpha=0.12, label="Blue 10–90%"
        )
        plt.fill_between(
            x,
            red_lo.astype(float).to_numpy(),
            red_hi.astype(float).to_numpy(),
            color="red", alpha=0.12, label="Red 10–90%"
        )

    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.title(f"Population — all runs{note}")
    plt.grid(True, alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pop_ALL_runs{suf}.png"))
    plt.close()


def plot_births_deaths_all_runs(df: pd.DataFrame, S, outdir, suf, note):
    """
    ONE figure (no per-run files):
      • Solid lines: mean births/day (blue/red) on left axis
      • Dashed lines: mean cumulative deaths (blue/red) on right axis
      • Means only (no bands)
    """
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()

    # Means across runs (ignore percentiles)
    x, blue_b_mean, *_ = _agg_across_runs(d, "blue_born", S)
    _,  red_b_mean,  *_ = _agg_across_runs(d, "red_born",  S)
    _, blue_d_mean,  *_ = _agg_across_runs(d, "blue_dead", S)
    _,  red_d_mean,   *_ = _agg_across_runs(d, "red_dead",  S)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # births/day (left axis) — means only
    l1, = ax1.plot(x, blue_b_mean, color="blue", linewidth=2.2, label="Blue births/day (mean)")
    l2, = ax1.plot(x, red_b_mean,  color="red",  linewidth=2.2, label="Red births/day (mean)")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Births per day")
    ax1.grid(True, alpha=.25)

    # cumulative deaths (right axis) — means only
    ax2 = ax1.twinx()
    l3, = ax2.plot(x, blue_d_mean, color="blue", linestyle="--", linewidth=2.2, label="Blue deaths (cum, mean)")
    l4, = ax2.plot(x, red_d_mean,  color="red",  linestyle="--", linewidth=2.2, label="Red deaths (cum, mean)")
    ax2.set_ylabel("Cumulative deaths")

    # single legend
    lines = [l1, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title(f"Births & deaths — all runs (means only){note}")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"births_deaths_ALL_runs{suf}.png"))
    plt.close(fig)


def plot_trust_agg(df: pd.DataFrame, S, outdir, suf, note):
    """
    TWO figures total (aggregated across runs):
      1) Within vs Between (mean + 10–90% band for each)
      2) Within Blue vs Within Red (mean + 10–90% band for each)
    Colors: Blue=blue, Red=red; Within(overall)=black, Between=purple.
    """
    import numpy as np
    from matplotlib.patches import Patch

    def _agg_col(col: str):
        # mean / p10 / p90 by day across runs (safe numerics)
        s = pd.to_numeric(df[col], errors="coerce")
        g = s.groupby(df["day"])
        m   = g.mean().sort_index()
        p10 = g.quantile(0.10).sort_index()
        p90 = g.quantile(0.90).sort_index()
        # optional smoothing (keeps index)
        return S(m), S(p10), S(p90)

    # ---------- Figure A: Within vs Between ----------
    w_m, w_p10, w_p90 = _agg_col("within_trust")
    b_m, b_p10, b_p90 = _agg_col("between_trust")

    # unify x domain
    x = sorted(set(w_m.index) | set(b_m.index))
    wM, wL, wU = w_m.reindex(x), w_p10.reindex(x), w_p90.reindex(x)
    bM, bL, bU = b_m.reindex(x), b_p10.reindex(x), b_p90.reindex(x)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, wL, wU, color="gray", alpha=0.18)
    ax.plot(x, wM, color="green", linewidth=2, label="Within mean")
    ax.fill_between(x, bL, bU, color="mediumorchid", alpha=0.18)
    ax.plot(x, bM, color="purple", linewidth=2, label="Between mean")

    band_legend = [
        Patch(facecolor="gray", alpha=0.18, label="Within 10–90%"),
        Patch(facecolor="mediumorchid", alpha=0.18, label="Between 10–90%"),
    ]
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], *band_legend], loc="upper left")
    ax.set_xlabel("Day"); ax.set_ylabel("Trust")
    ax.set_title(f"Trust — Within vs Between (all runs){note}")
    ax.grid(True, alpha=.25); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"trust_within_between_all_runs{suf}.png"))
    plt.close(fig)

    # ---------- Figure B: Within Blue vs Within Red ----------
    bl_m, bl_p10, bl_p90 = _agg_col("within_blue_trust")
    rd_m, rd_p10, rd_p90 = _agg_col("within_red_trust")

    x = sorted(set(bl_m.index) | set(rd_m.index))
    blM, blL, blU = bl_m.reindex(x), bl_p10.reindex(x), bl_p90.reindex(x)
    rdM, rdL, rdU = rd_m.reindex(x), rd_p10.reindex(x), rd_p90.reindex(x)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, blL, blU, color="blue", alpha=0.12)
    ax.plot(x, blM, color="blue", linewidth=2, label="Within Blue mean")
    ax.fill_between(x, rdL, rdU, color="red", alpha=0.12)
    ax.plot(x, rdM, color="red", linewidth=2, label="Within Red mean")

    band_legend = [
        Patch(facecolor="blue", alpha=0.12, label="Blue 10–90%"),
        Patch(facecolor="red",  alpha=0.12, label="Red 10–90%"),
    ]
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], *band_legend], loc="upper left")
    ax.set_xlabel("Day"); ax.set_ylabel("Trust")
    ax.set_title(f"Trust — Within Blue vs Within Red (all runs){note}")
    ax.grid(True, alpha=.25); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"trust_by_house_all_runs{suf}.png"))
    plt.close(fig)


def plot_total_spawn_vs_consumption(df: pd.DataFrame, S, outdir, suf, note):
    """
    ONE figure (no per-run files):
      • Solid lines = mean across runs
      • Shaded = 10–90% across runs
    """
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()
    d = _ensure_numeric(d, ["total_spawned", "total_consumed"])

    x, sp_mean, sp_p10, sp_p90 = _agg_across_runs(d, "total_spawned", S)
    _,  co_mean, co_p10, co_p90 = _agg_across_runs(d, "total_consumed", S)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # bands
    ax.fill_between(x, sp_p10, sp_p90, color="#2ca02c", alpha=0.18, label="Spawned 10–90%")
    ax.fill_between(x, co_p10, co_p90, color="#ff7f0e", alpha=0.18, label="Consumed 10–90%")

    # means
    ax.plot(x, sp_mean, color="#2ca02c", linewidth=2.2, label="Spawned (mean)")
    ax.plot(x, co_mean, color="#ff7f0e", linewidth=2.2, label="Consumed (mean)")

    ax.set_xlabel("Day")
    ax.set_ylabel("Units / day")
    ax.set_title(f"Global spawn vs consumption — all runs (mean ± 10–90%){note}")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"global_spawn_cons_ALL_runs{suf}.png"))
    plt.close(fig)


def plot_zone_spawn_vs_consumption(df: pd.DataFrame, S, outdir, suf, note):
    """
    ONE PNG with 3 subplots (Zone 0/1/2):
      • Solid lines = mean across runs (spawned & consumed)
      • Shaded = 10–90% across runs (spawned & consumed)
    """
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()

    # make sure everything we need is numeric
    cols = []
    for z in ZONES:
        cols += [f"z{z}_spawn", f"z{z}_cons"]
    d = _ensure_numeric(d, cols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

    for z, ax in zip(ZONES, axes):
        # aggregate across runs for this zone
        x, sp_mean, sp_p10, sp_p90 = _agg_across_runs(d, f"z{z}_spawn", S)
        _,  co_mean, co_p10, co_p90 = _agg_across_runs(d, f"z{z}_cons",  S)

        # 10–90% bands
        ax.fill_between(x, sp_p10, sp_p90, color="#2ca02c", alpha=0.18, label="Spawned 10–90%")
        ax.fill_between(x, co_p10, co_p90, color="#ff7f0e", alpha=0.18, label="Consumed 10–90%")

        # mean lines
        ax.plot(x, sp_mean, color="#2ca02c", linewidth=2.2, label="Spawned (mean)")
        ax.plot(x, co_mean, color="#ff7f0e", linewidth=2.2, label="Consumed (mean)")

        ax.set_title(f"Zone {z}")
        ax.set_xlabel("Day")
        if z == ZONES[0]:
            ax.set_ylabel("Units / day")
        ax.grid(True, alpha=.25)

    # one legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(f"Spawn vs consumption — mean ± 10–90% across runs{note}", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(outdir, f"zone_spawn_cons_ALL_runs{suf}.png"))
    plt.close(fig)


def plot_zone_consumption_by_house(df: pd.DataFrame, S, outdir, suf, note):
    """
    One figure, 3 subplots (zones). For each zone we plot, across runs:
      - Blue median (solid blue) with 10–90% band (light blue)
      - Red  median (solid red)  with 10–90% band (light red)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ensure numeric columns (ignore missing if any)
    cols = []
    for z in ZONES:
        cols += [f"z{z}_blue_cons", f"z{z}_red_cons"]
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # we aggregate ACROSS runs at each day
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    for z, ax in zip(ZONES, axes):
        bcol = f"z{z}_blue_cons"
        rcol = f"z{z}_red_cons"

        tmp = d[["day", bcol, rcol]].dropna(subset=["day"]).copy()
        tmp["day"] = pd.to_numeric(tmp["day"], errors="coerce")
        tmp = tmp.dropna(subset=["day"]).sort_values("day")

        g = tmp.groupby("day", dropna=True)

        # quantiles/medians across runs per day
        b_q10 = g[bcol].quantile(0.10)
        b_med = g[bcol].median()
        b_q90 = g[bcol].quantile(0.90)

        r_q10 = g[rcol].quantile(0.10)
        r_med = g[rcol].median()
        r_q90 = g[rcol].quantile(0.90)

        # Apply optional smoothing (S returns a Series)
        b_q10, b_med, b_q90 = S(b_q10), S(b_med), S(b_q90)
        r_q10, r_med, r_q90 = S(r_q10), S(r_med), S(r_q90)

        x = b_med.index.to_numpy()

        # Blue band + median
        ax.fill_between(x, b_q10.to_numpy(), b_q90.to_numpy(), color="blue", alpha=0.15, label="Blue 10–90%")
        ax.plot(x, b_med.to_numpy(), color="blue", linewidth=2, label="Blue median")

        # Red band + median
        ax.fill_between(x, r_q10.to_numpy(), r_q90.to_numpy(), color="red", alpha=0.15, label="Red 10–90%")
        ax.plot(x, r_med.to_numpy(), color="red", linewidth=2, label="Red median")

        ax.set_title(f"Zone {z} — per-house consumption{note}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Units / day")
        ax.grid(True, alpha=0.25)

    # one shared legend (avoid duplicates)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2)
    fig.suptitle("Per-house consumption per zone — medians & 10–90% bands (across runs)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, f"zone_house_cons_all_runs{suf}.png"))
    plt.close(fig)


def plot_consumption_share(df: pd.DataFrame, outdir: str, suf: str, note: str):
    """
    ONE PNG:
      • x = day
      • each bar = 100% of that day's total consumption (all runs combined)
      • segments = share from Zone 0 / Zone 1 / Zone 2
    """
    os.makedirs(outdir, exist_ok=True)

    cols = ["run","day","z0_cons","z1_cons","z2_cons"]
    d = _ensure_numeric(df[cols], cols).sort_values(["run","day"])

    # Sum across runs for each day, then convert to % per day
    g = d.groupby("day")[["z0_cons","z1_cons","z2_cons"]].sum(min_count=1)
    denom = g.sum(axis=1).replace(0, np.nan)
    share = (g.div(denom, axis=0) * 100).fillna(0.0)

    # Stacked 100% bars
    x = share.index.to_numpy()
    z0 = share["z0_cons"].to_numpy()
    z1 = share["z1_cons"].to_numpy()
    z2 = share["z2_cons"].to_numpy()
    bottom1 = z0
    bottom2 = z0 + z1

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(x, z0, color="#1f77b4", label="Zone 0")  # blue-ish
    ax.bar(x, z1, bottom=bottom1, color="#d62728", label="Zone 1")  # red-ish
    ax.bar(x, z2, bottom=bottom2, color="#2ca02c", label="Zone 2")  # green

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(12, integer=True))
    ax.set_ylabel("Share of daily consumption (%)")
    ax.set_xlabel("Day")
    ax.set_title(f"Daily consumption mix by zone — 100% stacked (all runs){note}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"zone_mix_daily_pct_ALL_runs{suf}.png"), dpi=120)
    plt.close(fig)

def plot_per_capita_all_runs(df: pd.DataFrame, S, outdir, suf, note):
    """
    ONE PNG: per-capita consumption (blue/red) aggregated across runs.
    Solid lines = mean; translucent bands = 10–90% across runs.
    """
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()

    # Make sure needed columns are numeric
    cols = [
        "blue_pop", "red_pop",
        "z0_blue_cons", "z1_blue_cons", "z2_blue_cons",
        "z0_red_cons",  "z1_red_cons",  "z2_red_cons",
    ]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Total consumption per house/day
    blue_cons = d["z0_blue_cons"] + d["z1_blue_cons"] + d["z2_blue_cons"]
    red_cons  = d["z0_red_cons"]  + d["z1_red_cons"]  + d["z2_red_cons"]

    # Avoid division by zero -> NaN
    blue_pop = d["blue_pop"].astype(float).where(d["blue_pop"] > 0, np.nan)
    red_pop  = d["red_pop"].astype(float).where(d["red_pop"]  > 0, np.nan)

    # Per-capita time series per run/day
    d["blue_pc"] = (blue_cons / blue_pop).astype(float)
    d["red_pc"]  = (red_cons  / red_pop ).astype(float)

    # Aggregate across runs per day
    blue_stats = (
        d.groupby("day")["blue_pc"]
         .agg(mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
    )
    red_stats = (
        d.groupby("day")["red_pc"]
         .agg(mean="mean", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
    )

    # Align days in case of gaps
    all_days = pd.Index(sorted(set(blue_stats.index) | set(red_stats.index)))
    blue_stats = blue_stats.reindex(all_days)
    red_stats  = red_stats.reindex(all_days)

    # Optional smoothing (S) then convert to numpy
    x          = all_days.to_numpy()
    blue_mean  = pd.Series(S(blue_stats["mean"])).astype(float).to_numpy()
    blue_p10   = pd.Series(S(blue_stats["p10"])).astype(float).to_numpy()
    blue_p90   = pd.Series(S(blue_stats["p90"])).astype(float).to_numpy()
    red_mean   = pd.Series(S(red_stats["mean"])).astype(float).to_numpy()
    red_p10    = pd.Series(S(red_stats["p10"])).astype(float).to_numpy()
    red_p90    = pd.Series(S(red_stats["p90"])).astype(float).to_numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, blue_p10, blue_p90, color="blue", alpha=0.15, label="Blue 10–90%")
    ax.fill_between(x, red_p10,  red_p90,  color="red",  alpha=0.12, label="Red 10–90%")
    ax.plot(x, blue_mean, color="blue", linewidth=2.2, label="Blue mean")
    ax.plot(x, red_mean,  color="red",  linewidth=2.2, label="Red mean")

    ax.set_xlabel("Day")
    ax.set_ylabel("Per-capita consumption (units/person/day)")
    ax.set_title(f"Per-capita consumption — all runs (mean ± 10–90%){note}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"per_capita_ALL_runs{suf}.png"))
    plt.close(fig)


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return d

def plot_overall_zone_exploitation_totals(df: pd.DataFrame, outdir: str):
    """
    Stacked bars: per zone, total consumption by Blue and Red
    aggregated across ALL runs and ALL days, with % labels.
    """
    cols = [
        "z0_blue_cons","z1_blue_cons","z2_blue_cons",
        "z0_red_cons", "z1_red_cons", "z2_red_cons",
    ]
    d = _ensure_numeric(df, cols)

    totals_blue = np.array([d[f"z{z}_blue_cons"].sum() for z in ZONES], dtype=float)
    totals_red  = np.array([d[f"z{z}_red_cons"].sum()  for z in ZONES], dtype=float)
    zone_totals = totals_blue + totals_red

    x = np.arange(len(ZONES))
    fig, ax = plt.subplots(figsize=(8, 4))
    b = ax.bar(x, totals_blue, width=0.6, label="Blue", color="blue")
    r = ax.bar(x, totals_red,  width=0.6, bottom=totals_blue, label="Red", color="red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Zone {z}" for z in ZONES])
    ax.set_ylabel("Total units consumed (all runs & days)")
    ax.set_title("Overall exploitation by zone and family")
    ax.grid(True, axis="y", alpha=.25)
    ax.legend()

    # ── annotate % shares inside bars + total above each bar
    y_max = zone_totals.max() if len(zone_totals) else 0.0
    pad   = 0.03 * y_max  # space above bar for the total label

    for i, total in enumerate(zone_totals):
        if total <= 0:
            continue
        p_blue = 100.0 * totals_blue[i] / total
        p_red  = 100.0 * totals_red[i]  / total

        # Blue % centered in blue segment
        ax.text(
            x[i], totals_blue[i] * 0.5,
            f"{p_blue:.1f}%",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold"
        )
        # Red % centered in red segment
        ax.text(
            x[i], totals_blue[i] + totals_red[i] * 0.5,
            f"{p_red:.1f}%",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold"
        )
       
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "overall_zone_exploitation_stacked.png"))
    plt.close(fig)


def plot_zone_dominance_by_run(df: pd.DataFrame, outdir: str):
    """
    For each zone, count how many runs Blue consumed the majority vs Red.
    Robust to dtypes; we sum within run first, then compare.
    Adds % labels on top of bars.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    ZONES = [0, 1, 2]
    cols = [
        "z0_blue_cons", "z1_blue_cons", "z2_blue_cons",
        "z0_red_cons",  "z1_red_cons",  "z2_red_cons",
    ]

    d = df.copy()
    # ensure numeric, keep run
    d["run"] = pd.to_numeric(d["run"], errors="coerce")
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    runs = np.sort(d["run"].dropna().unique())

    blue_wins, red_wins, ties, no_data = [], [], [], []

    # optional: quick sanity print
    # print("Total per-zone (all runs):", d[cols].sum(numeric_only=True))

    for z in ZONES:
        shares = []
        for r in runs:
            sub = d[d["run"] == r]
            b_sum = sub[f"z{z}_blue_cons"].sum(skipna=True)
            r_sum = sub[f"z{z}_red_cons"].sum(skipna=True)
            tot = b_sum + r_sum
            if tot <= 0:
                shares.append(np.nan)        # no data for this run/zone
            else:
                shares.append(b_sum / tot)   # blue share for run r in zone z

        s = pd.Series(shares, index=runs)
        blue_wins.append(int((s > 0.5).sum()))
        red_wins .append(int((s < 0.5).sum()))
        ties     .append(int((s == 0.5).sum()))
        no_data  .append(int(s.isna().sum()))

    # ---- plot grouped bars with % labels ----
    x = np.arange(len(ZONES))
    w = 0.22

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b = ax.bar(x - w, blue_wins, width=w, color='blue',  label="Blue wins (runs)")
    r = ax.bar(x,       red_wins,  width=w, color='red',   label="Red wins (runs)")
    t = ax.bar(x + w,   ties,      width=w, color='grey',  label="Ties")

    # show “no data” as a thin hatched bar behind (if any)
    nd = ax.bar(x, no_data, width=0.65, color='none', edgecolor='#666',
                linewidth=1, hatch='////', label="No data")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Zone {z}" for z in ZONES])
    ax.set_ylabel("# of runs")
    ax.set_title("Which family exploited each zone most (by run)")
    ax.grid(True, axis="y", alpha=.25)
    ax.legend(loc="upper left")

    # annotate counts + percentages on each colored bar
    totals = np.array(blue_wins) + np.array(red_wins) + np.array(ties) + np.array(no_data)
    def _annotate(bar_container, counts):
        for rect, c, tot in zip(bar_container, counts, totals):
            if tot > 0 and c > 0:
                pct = 100.0 * c / tot
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.05,
                        f"{c} ({pct:.0f}%)", ha='center', va='bottom', fontsize=9)

    _annotate(b, blue_wins)
    _annotate(r, red_wins)
    _annotate(t, ties)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "zone_dominance_by_run.png"))
    plt.close(fig)


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default="batch_results/all_runs_combined.csv")
    ap.add_argument("--out",    default="batch_results/plots")
    ap.add_argument("--smooth", action="store_true", help="Enable rolling-average smoothing")
    ap.add_argument("--window", type=int, default=7, help="Rolling window (days)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv).sort_values(["run","day"])

    S, suf, note = make_smoother(args.smooth, args.window)

    plot_population(df, S, args.out, suf, note)
    plot_births_deaths_all_runs(df, S, args.out, suf, note)
    plot_trust_agg(df, S, args.out, suf, note)    
    plot_total_spawn_vs_consumption(df, S, args.out, suf, note)
    plot_zone_spawn_vs_consumption(df, S, args.out, suf, note)
    plot_zone_consumption_by_house(df, S, args.out, suf, note)
    plot_consumption_share(df, args.out, suf, note)
    plot_per_capita_all_runs(df, S, args.out, suf, note)
    plot_overall_zone_exploitation_totals(df, args.out)
    plot_zone_dominance_by_run(df, args.out)



    print(f"Saved plots to: {args.out}")

if __name__ == "__main__":
    main()
