# Visualization.py Usage Guide

## Overview
The `visualization.py` file creates **heatmap visualizations** showing how Blue and Red families exploit different resource zones on the simulation map.

## Quick Start

### 1. **Generate Data First**
```bash
# Run batch simulations to create data
python batch_simulation.py
```

### 2. **Run the Fixed Visualization**
```bash
# Use the corrected version
python visualization_fixed.py
```

### 3. **View Results**
The script will create a heatmap image showing:
- **Blue areas**: Where Blue families consume more resources
- **Red areas**: Where Red families consume more resources  
- **Purple areas**: Where both families consume equally
- **Zone percentages**: Shows the consumption split per zone

## What the Visualization Shows

### 🎯 **Purpose**
- **Resource Competition**: See which zones each family dominates
- **Spatial Patterns**: Understand geographic resource exploitation
- **Family Behavior**: Compare Blue vs Red family strategies

### 📊 **Visual Elements**
- **Base Map**: Your simulation map (e.g., `3_spots.png`)
- **Heat Overlay**: Color-coded consumption patterns
- **Zone Labels**: Percentage breakdown per zone
- **Legend**: Blue=Blue family, Red=Red family, Purple=Both

## Customization Options

### **Basic Usage**
```python
from visualization_fixed import plot_heatmap_for_run

plot_heatmap_for_run(
    data_path="batch_results/all_244_combined_sim_function.csv",
    coords_path="resources_coords.csv", 
    map_path="images/3_spots.png",
    run=0,  # Which simulation run to visualize
    save_path="my_heatmap.png"
)
```

### **Advanced Parameters**
```python
plot_heatmap_for_run(
    data_path="your_data.csv",
    coords_path="your_coords.csv",
    map_path="your_map.png",
    run=0,                    # Simulation run number
    scale_x=0.55,            # Horizontal scaling
    scale_y=1.0,             # Vertical scaling  
    offset_x=1,              # Horizontal offset
    offset_y=0,              # Vertical offset
    radius=12,               # Heat radius around zones
    sigma=5.0,               # Heat spread (higher = more spread)
    alpha=0.6,               # Heat transparency (0-1)
    save_path="output.png"   # Output file
)
```

## Troubleshooting

### ❌ **Common Issues**

1. **"Data file not found"**
   - Run `python batch_simulation.py` first
   - Check the file path in the script

2. **"Map file not found"** 
   - Use an existing map: `images/3_spots.png`
   - Check available maps in the `images/` folder

3. **"No rows found for this run"**
   - Check that the run number exists in your data
   - Try `run=0` (first run)

4. **"Missing columns for zone"**
   - The data format might be different
   - Check your CSV column names

### ✅ **Solutions**

1. **Generate coordinates file**:
   ```python
   from visualization_fixed import create_coords_file_from_data
   create_coords_file_from_data("your_data.csv", "coords.csv")
   ```

2. **Use correct file paths**:
   ```python
   # Check what files you have
   import os
   print("Available data files:")
   for f in os.listdir("batch_results/"):
       if f.endswith('.csv'):
           print(f"  - batch_results/{f}")
   ```

3. **Check your data structure**:
   ```python
   import pandas as pd
   df = pd.read_csv("batch_results/all_244_combined_sim_function.csv")
   print("Columns:", df.columns.tolist())
   print("Available runs:", df['run'].unique())
   ```

## Example Output

The visualization will show:
- A map with colored heat overlays
- Zone labels showing consumption percentages
- A title indicating which run is displayed
- Saved as a PNG file for further analysis

## Next Steps

1. **Experiment with different runs**: Try `run=1, run=2, etc.`
2. **Adjust parameters**: Change `alpha`, `radius`, `sigma` for different effects
3. **Compare families**: Look for patterns in Blue vs Red consumption
4. **Analyze zones**: See which zones are most contested

This visualization helps you understand the **spatial dynamics** of resource competition in your simulation! 🎯
