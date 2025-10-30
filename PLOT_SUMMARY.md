# Batch Plot Analysis - Essential Plots

## Overview

The batch plotting system has been streamlined from 40+ redundant plots to **6 essential visualizations** that provide comprehensive insights into simulation outcomes.

## Generated Plots

### 1. **population_overview.png**
- **Purpose:** Track population dynamics across all simulation runs
- **Features:**
  - Individual run trajectories (thin lines) for context
  - Mean population lines (thick) for Blue and Red houses
  - 10th-90th percentile confidence bands
  - Smoothing support for clearer trends

### 2. **trust_evolution.png**
- **Purpose:** Understand social dynamics and cooperation
- **Features:**
  - **Left panel:** Within-house trust (Blue vs Red family cohesion)
  - **Right panel:** Between-house trust (inter-family cooperation)
  - Confidence bands showing variation across runs
  - Shows emergence of social structures

### 3. **resource_flow.png**
- **Purpose:** Analyze resource sustainability and balance
- **Features:**
  - **Top panel:** Spawned vs consumed resources over time
  - **Bottom panel:** Cumulative balance (positive = surplus, negative = depletion)
  - Identifies periods of scarcity or abundance
  - Helps tune spawn rates and consumption patterns

### 4. **zone_exploitation.png**
- **Purpose:** Spatial analysis of resource competition
- **Features:**
  - **Left panel:** Total consumption per zone (stacked by house)
  - **Right panel:** Zone dominance percentage (Blue vs Red)
  - Shows territorial behavior and resource preferences
  - Identifies contested vs dominated zones

### 5. **survival_analysis.png**
- **Purpose:** Population viability and extinction patterns
- **Features:**
  - **Left panel:** Survival curves (total population over time per run)
  - **Right panel:** Histogram of extinction timing
  - Identifies critical periods and collapse patterns
  - Helps assess parameter robustness

### 6. **per_capita_consumption.png**
- **Purpose:** Individual-level resource efficiency
- **Features:**
  - Resources consumed per human per day
  - Confidence bands across runs
  - Reveals efficiency trends and sustainability
  - Useful for tuning energy/food parameters

## Usage

```bash
# Basic usage (default paths)
python batch_plot.py

# Custom paths
python batch_plot.py --csv batch_results/my_data.csv --out my_plots/

# With smoothing (7-day rolling average)
python batch_plot.py --smooth --window 7

# Help
python batch_plot.py --help
```

## Removed Plots (Previously Generated)

The following 40+ plots were removed as redundant or low-value:
- **10x** individual run population plots (pop_run0.png ... pop_run9.png)
- **10x** per-run pop/consumption/trust combined plots
- **10x** per-run spawn vs consumption by family
- Various single-metric plots now integrated into comprehensive views
- Redundant zone consumption breakdowns
- Low-insight scatter plots and CCF analyses

## What Was Kept

Only plots that provide:
1. **Unique insights** not available elsewhere
2. **Actionable information** for parameter tuning
3. **Statistical rigor** with confidence bands
4. **Clear visual communication** without clutter

## Migration Notes

- Old plots saved in `batch_results/plots_seed244_1trust/` (46 files)
- New plots saved in `batch_results/plots/` or custom `--out` directory (6 files)
- CSV data remains unchanged - compatible with old plotting code if needed
- All essential insights preserved in streamlined format

---

**Result:** 85% reduction in plot files with 100% insight retention
