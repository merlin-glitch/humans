



"""
batch_plot.py - Comprehensive plotting suite for batch simulation results

Part of the Human Society Simulation project.

Generates 20+ chart types including population dynamics, trust evolution,
zone exploitation patterns, survival curves, and resource flow analysis.
"""

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
    """
    Plot population dynamics over time.
    
    Note: This function calculates population changes as a proxy for births/deaths
    since birth/death columns are not available in the current CSV format.
    """
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()
    
    # Calculate population changes as a proxy for births/deaths
    d['blue_change'] = d.groupby('run')['blue_pop'].diff().fillna(0)
    d['red_change'] = d.groupby('run')['red_pop'].diff().fillna(0)
    
    x, blue_b_mean, *_ = _agg_across_runs(d, "blue_change", S)
    _,  red_b_mean,  *_ = _agg_across_runs(d, "red_change",  S)
    
    # For deaths, we'll use negative population changes
    d['blue_deaths'] = -d['blue_change'].where(d['blue_change'] < 0, 0)
    d['red_deaths'] = -d['red_change'].where(d['red_change'] < 0, 0)
    
    _, blue_d_mean,  *_ = _agg_across_runs(d, "blue_deaths", S)
    _,  red_d_mean,   *_ = _agg_across_runs(d, "red_deaths",  S)

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
    
    # Create a copy of the dataframe for modifications
    d = df.copy()

    def _agg_col(col: str):
        s = pd.to_numeric(d[col], errors="coerce")
        g = s.groupby(d["day"])
        m, p10, p90 = g.mean().sort_index(), g.quantile(0.10).sort_index(), g.quantile(0.90).sort_index()
        return S(m), S(p10), S(p90)

    # A) Within vs Between - use available trust columns
    # Calculate average within-trust from blue and red within-trust
    d['avg_within_trust'] = (d['within_blue_trust'] + d['within_red_trust']) / 2
    w_m, w_p10, w_p90 = _agg_col("avg_within_trust")
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

        # (3) Trust - use available trust columns
        within_avg = (pd.to_numeric(d["within_blue_trust"], errors="coerce") + 
                      pd.to_numeric(d["within_red_trust"], errors="coerce")) / 2
        between = pd.to_numeric(d["between_trust"], errors="coerce")
        axes[2].plot(d["day"], S(within_avg),  color="green",  linewidth=2, label="Within trust (avg)")
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
            # Since birth columns don't exist, we'll plot population instead
            pop = pd.to_numeric(d[f"{fam}_pop"], errors="coerce")
            series_list = [pd.to_numeric(d[f"z{z}_{fam}_cons"], errors="coerce") for z in zones if f"z{z}_{fam}_cons" in d]
            co = _sum_series(series_list)
            ax.plot(d["day"], S(pop), color=color, linestyle="-",  linewidth=2, label=f"{fam.capitalize()} population")
            ax.plot(d["day"], S(co), color=color, linestyle="--", linewidth=2, label=f"{fam.capitalize()} consumption")
            ax.set_title(f"Run {r} — {fam.capitalize()} population vs consumption")
            ax.set_ylabel("Units/day"); ax.grid(True, alpha=0.25); ax.legend()

        axes[-1].set_xlabel("Day")
        fig.suptitle(f"Run {r} — Population vs consumption by family{note}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(outdir, f"spawn_cons_by_family_run{r}{suf}.png")); plt.close(fig)

# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default="batch_results/all_244_combined_sim_function.csv")
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
