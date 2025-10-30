#!/usr/bin/env python3
"""
batch_plot.py - Streamlined plotting suite for batch simulation results

Generates essential plots for analyzing simulation outcomes:
- Population dynamics (aggregated across runs)
- Trust evolution (within/between houses)
- Resource flow (spawn vs consumption)
- Survival analysis

Part of the Human Society Simulation project.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ───────────────────────── Helpers ─────────────────────────

ZONE_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def detect_zones(df: pd.DataFrame) -> list[int]:
    """Detect unique zone indices from column names like z3_cons, z10_blue_cons."""
    zones = set()
    for c in df.columns:
        m = re.match(r"^z(\d+)_cons$", c)
        if m:
            zones.add(int(m.group(1)))
    return sorted(zones)

def make_smoother(enabled: bool, window: int):
    suffix = f"_sm{window}" if enabled and window > 1 else ""
    title_note = f" (rolling {window})" if enabled and window > 1 else ""
    def S(s: pd.Series) -> pd.Series:
        if isinstance(s, pd.Series) and enabled and window > 1 and len(s) > 1:
            return s.rolling(window, min_periods=1).mean()
        return s
    return S, suffix, title_note

def _agg_across_runs(d: pd.DataFrame, value_col: str, smooth):
    """Return x, mean, p10, p90 for a metric across runs per day."""
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

# ───────────────────────── Essential Plots ─────────────────────────

def plot_population_overview(df: pd.DataFrame, S, outdir, suf, note, alpha_runs: float = 0.10):
    """Population dynamics across all runs with confidence bands."""
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()

    plt.figure(figsize=(12, 6))
    
    # Thin overlay per run for context
    for r in sorted(d["run"].dropna().unique()):
        sub = d[d.run == r]
        x  = sub.day.to_numpy()
        yb = pd.to_numeric(S(sub.blue_pop), errors="coerce").astype(float).to_numpy()
        yr = pd.to_numeric(S(sub.red_pop),  errors="coerce").astype(float).to_numpy()
        plt.plot(x, yb, color="blue", alpha=alpha_runs, linewidth=1, label="_nolegend_")
        plt.plot(x, yr, color="red",  alpha=alpha_runs, linewidth=1, label="_nolegend_")

    # Aggregated statistics
    xb, mb, b10, b90 = _agg_across_runs(d, "blue_pop", S)
    xr, mr, r10, r90 = _agg_across_runs(d, "red_pop",  S)

    # Mean lines
    plt.plot(xb, mb, color="blue", linewidth=3, label="Blue (mean)")
    plt.plot(xr, mr, color="red",  linewidth=3, label="Red (mean)")

    # Confidence bands
    plt.fill_between(xb, b10, b90, color="blue", alpha=0.2, label="Blue (10-90%)")
    plt.fill_between(xr, r10, r90, color="red",  alpha=0.2, label="Red (10-90%)")

    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Population", fontsize=12)
    plt.title(f"Population Dynamics Across All Runs{note}", fontsize=14, fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"population_overview{suf}.png"), dpi=150)
    plt.close()


def plot_trust_evolution(df: pd.DataFrame, S, outdir, suf, note):
    """Trust evolution: within-house vs between-house."""
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Within-house trust
    ax = axes[0]
    xb, mb, b10, b90 = _agg_across_runs(d, "within_blue_trust", S)
    xr, mr, r10, r90 = _agg_across_runs(d, "within_red_trust",  S)
    
    ax.plot(xb, mb, "b-", linewidth=2, label="Blue")
    ax.fill_between(xb, b10, b90, color="blue", alpha=0.2)
    ax.plot(xr, mr, "r-", linewidth=2, label="Red")
    ax.fill_between(xr, r10, r90, color="red", alpha=0.2)
    
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Trust Score", fontsize=11)
    ax.set_title(f"Within-House Trust{note}", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Between-house trust
    ax = axes[1]
    xbw, mbw, bw10, bw90 = _agg_across_runs(d, "between_trust", S)
    
    ax.plot(xbw, mbw, color="purple", linewidth=2, label="Between Houses")
    ax.fill_between(xbw, bw10, bw90, color="purple", alpha=0.2)
    
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Trust Score", fontsize=11)
    ax.set_title(f"Between-House Trust{note}", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"trust_evolution{suf}.png"), dpi=150)
    plt.close()


def plot_resource_flow(df: pd.DataFrame, S, outdir, suf, note):
    """Global spawn vs consumption analysis."""
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()

    zones = detect_zones(d)
    
    # Calculate total spawn and consumption
    spawn_cols = [f"z{z}_spawned" for z in zones if f"z{z}_spawned" in d.columns]
    cons_cols  = [f"z{z}_cons" for z in zones if f"z{z}_cons" in d.columns]
    
    if not spawn_cols or not cons_cols:
        print("⚠️  No spawn/consumption data available")
        return
    
    d["total_spawn"] = d[spawn_cols].sum(axis=1)
    d["total_cons"]  = d[cons_cols].sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time series
    xs, ms, s10, s90 = _agg_across_runs(d, "total_spawn", S)
    xc, mc, c10, c90 = _agg_across_runs(d, "total_cons", S)
    
    ax1.plot(xs, ms, color="green", linewidth=2, label="Spawned")
    ax1.fill_between(xs, s10, s90, color="green", alpha=0.2)
    ax1.plot(xc, mc, color="orange", linewidth=2, label="Consumed")
    ax1.fill_between(xc, c10, c90, color="orange", alpha=0.2)
    
    ax1.set_xlabel("Day", fontsize=11)
    ax1.set_ylabel("Resource Units", fontsize=11)
    ax1.set_title(f"Global Resource Flow{note}", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative balance
    d_sorted = d.sort_values(["run", "day"])
    cumulative_balance = []
    
    for r in sorted(d["run"].dropna().unique()):
        sub = d_sorted[d_sorted.run == r].copy()
        sub["balance"] = (sub["total_spawn"] - sub["total_cons"]).cumsum()
        cumulative_balance.append(sub[["day", "balance"]].set_index("day")["balance"])
    
    if cumulative_balance:
        balance_df = pd.DataFrame(cumulative_balance).T
        mean_balance = balance_df.mean(axis=1)
        p10_balance = balance_df.quantile(0.1, axis=1)
        p90_balance = balance_df.quantile(0.9, axis=1)
        
        ax2.plot(mean_balance.index, mean_balance.values, color="teal", linewidth=2, label="Net Balance")
        ax2.fill_between(mean_balance.index, p10_balance.values, p90_balance.values, color="teal", alpha=0.2)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel("Day", fontsize=11)
        ax2.set_ylabel("Cumulative Balance", fontsize=11)
        ax2.set_title("Cumulative Resource Balance (Spawn - Consumption)", fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"resource_flow{suf}.png"), dpi=150)
    plt.close()


def plot_zone_exploitation(df: pd.DataFrame, outdir):
    """Zone exploitation patterns by house."""
    os.makedirs(outdir, exist_ok=True)
    zones = detect_zones(df)
    
    blue_cols = [f"z{z}_blue_cons" for z in zones if f"z{z}_blue_cons" in df.columns]
    red_cols  = [f"z{z}_red_cons" for z in zones if f"z{z}_red_cons" in df.columns]
    
    if not blue_cols or not red_cols:
        print("⚠️  No per-house consumption data available")
        return
    
    # Total consumption per zone per house
    blue_totals = df[blue_cols].sum().values
    red_totals  = df[red_cols].sum().values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart
    x = np.arange(len(zones))
    ax1.bar(x, blue_totals, color='blue', alpha=0.7, label='Blue')
    ax1.bar(x, red_totals, bottom=blue_totals, color='red', alpha=0.7, label='Red')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Z{z}" for z in zones])
    ax1.set_xlabel("Zone", fontsize=11)
    ax1.set_ylabel("Total Consumption", fontsize=11)
    ax1.set_title("Total Zone Exploitation by House", fontweight='bold')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Dominance percentage
    total_per_zone = blue_totals + red_totals
    blue_pct = np.where(total_per_zone > 0, 100 * blue_totals / total_per_zone, 50)
    
    colors = ['blue' if p > 60 else 'red' if p < 40 else 'purple' for p in blue_pct]
    ax2.bar(x, blue_pct, color=colors, alpha=0.7)
    ax2.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Equal')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Z{z}" for z in zones])
    ax2.set_xlabel("Zone", fontsize=11)
    ax2.set_ylabel("Blue Dominance (%)", fontsize=11)
    ax2.set_title("Zone Dominance (Blue %)", fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "zone_exploitation.png"), dpi=150)
    plt.close()


def plot_survival_analysis(df: pd.DataFrame, outdir):
    """Survival curves and extinction analysis."""
    os.makedirs(outdir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Survival curves
    for r in sorted(df["run"].dropna().unique()):
        sub = df[df.run == r].sort_values("day")
        total_pop = sub.blue_pop + sub.red_pop
        ax1.plot(sub.day, total_pop, alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel("Day", fontsize=11)
    ax1.set_ylabel("Total Population", fontsize=11)
    ax1.set_title("Survival Curves (All Runs)", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Extinction timing
    extinction_days = []
    for r in sorted(df["run"].dropna().unique()):
        sub = df[df.run == r].sort_values("day")
        extinct = sub[(sub.blue_pop == 0) | (sub.red_pop == 0)]
        if not extinct.empty:
            extinction_days.append(extinct.day.min())
    
    if extinction_days:
        ax2.hist(extinction_days, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Day of First Extinction", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title("Extinction Timing Distribution", fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No extinctions observed", ha='center', va='center', fontsize=14)
        ax2.set_title("Extinction Timing Distribution", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "survival_analysis.png"), dpi=150)
    plt.close()


def plot_per_capita_metrics(df: pd.DataFrame, S, outdir, suf, note):
    """Per-capita resource consumption."""
    os.makedirs(outdir, exist_ok=True)
    d = df.sort_values(["run", "day"]).copy()
    
    zones = detect_zones(d)
    cons_cols = [f"z{z}_cons" for z in zones if f"z{z}_cons" in d.columns]
    
    if not cons_cols:
        print("⚠️  No consumption data available")
        return
    
    d["total_cons"] = d[cons_cols].sum(axis=1)
    d["total_pop"] = d.blue_pop + d.red_pop
    d["per_capita"] = np.where(d.total_pop > 0, d.total_cons / d.total_pop, 0)
    
    plt.figure(figsize=(12, 6))
    
    x, m, p10, p90 = _agg_across_runs(d, "per_capita", S)
    
    plt.plot(x, m, color="teal", linewidth=2, label="Per-Capita Consumption")
    plt.fill_between(x, p10, p90, color="teal", alpha=0.2, label="10-90% Range")
    
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Resources per Human", fontsize=12)
    plt.title(f"Per-Capita Resource Consumption{note}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"per_capita_consumption{suf}.png"), dpi=150)
    plt.close()


# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate essential plots from batch simulation results")
    ap.add_argument("--csv",    default="batch_results/all_244_combined_sim_function.csv",
                    help="Path to combined CSV results")
    ap.add_argument("--out",    default="batch_results/plots",
                    help="Output directory for plots")
    ap.add_argument("--smooth", action="store_true", 
                    help="Enable rolling-average smoothing")
    ap.add_argument("--window", type=int, default=7, 
                    help="Rolling window size in days")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        return

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv).sort_values(["run","day"])
    S, suf, note = make_smoother(args.smooth, args.window)

    zones = detect_zones(df)
    print(f"📊 Analyzing {len(df['run'].unique())} runs over {df['day'].max()} days")
    print(f"🗺️  Detected {len(zones)} zones: {zones}")
    print(f"📁 Output directory: {args.out}\n")

    # Generate essential plots
    print("📈 Generating plots...")
    plot_population_overview(df, S, args.out, suf, note)
    print("  ✓ Population overview")
    
    plot_trust_evolution(df, S, args.out, suf, note)
    print("  ✓ Trust evolution")
    
    plot_resource_flow(df, S, args.out, suf, note)
    print("  ✓ Resource flow")
    
    plot_zone_exploitation(df, args.out)
    print("  ✓ Zone exploitation")
    
    plot_survival_analysis(df, args.out)
    print("  ✓ Survival analysis")
    
    plot_per_capita_metrics(df, S, args.out, suf, note)
    print("  ✓ Per-capita metrics")

    print(f"\n✅ All plots saved to: {args.out}")

if __name__ == "__main__":
    main()
