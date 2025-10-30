# Zone Coordinates Guide

## How Zone Coordinates Work in Your Simulation

### 🎯 **Where Coordinates Come From**

Zone coordinates are **automatically generated** during simulation setup from your map image. Here's how:

1. **Map Analysis**: The simulation reads your map image (e.g., `3_spots.png`)
2. **Zone Detection**: Uses `identify_zones()` to find food zones by color
3. **Coordinate Extraction**: Uses `extract_resource_coords_from_zones()` to get actual (x,y) positions
4. **Storage**: Coordinates are stored in the simulation's zone data structures

### 📍 **How to Generate Coordinates File**

#### **Method 1: From Simulation Map (Recommended)**
```bash
# Generate coordinates from your actual map
python generate_coords.py
```

This will:
- Read your map image (`images/3_spots.png`)
- Identify actual food zones
- Extract real coordinates
- Save to `resources_coords.csv`

#### **Method 2: From Batch Data (Fallback)**
```bash
# Generate coordinates from batch simulation data
python -c "from generate_coords import generate_coords_from_batch_data; generate_coords_from_batch_data('batch_results/all_244_combined_sim_function.csv')"
```

### 🔍 **Understanding the Coordinate System**

Your simulation uses a **grid-based coordinate system**:

- **Map Dimensions**: 100x60 cells (from `config.py`)
- **Zone IDs**: 0, 1, 2, etc. (food zones only)
- **Coordinates**: (x, y) where x=0-99, y=0-59
- **Food Zones**: Identified by color `(54, 109, 70)` in your map

### 📊 **Coordinate File Format**

The `resources_coords.csv` file contains:
```csv
zone_id,x,y
0,10,15
0,11,15
0,12,15
1,25,30
1,26,30
...
```

Where:
- `zone_id`: Which food zone (0, 1, 2, etc.)
- `x`: Horizontal position (0-99)
- `y`: Vertical position (0-59)

### 🛠️ **How to Use Coordinates**

#### **For Visualization**
```python
# The visualization script will automatically generate coordinates
python visualization_fixed.py
```

#### **Manual Coordinate Generation**
```python
from generate_coords import generate_coords_from_simulation

# Generate from your map
coords_file = generate_coords_from_simulation(
    map_path="images/3_spots.png",
    output_path="my_coords.csv"
)
```

#### **Check Your Coordinates**
```python
import pandas as pd

# Load and inspect coordinates
coords = pd.read_csv("resources_coords.csv")
print(f"Total coordinates: {len(coords)}")
print(f"Zones: {coords['zone_id'].unique()}")
print(f"X range: {coords['x'].min()}-{coords['x'].max()}")
print(f"Y range: {coords['y'].min()}-{coords['y'].max()}")
```

### 🎨 **Visualization Process**

1. **Load Map**: Your simulation map as background
2. **Load Coordinates**: Zone positions from `resources_coords.csv`
3. **Load Data**: Consumption data from batch results
4. **Create Heatmap**: Overlay consumption patterns on map
5. **Save/Display**: Generate final visualization

### 🔧 **Troubleshooting Coordinates**

#### **❌ "No coordinates found"**
- Run `python generate_coords.py` first
- Check that your map file exists
- Verify zone detection is working

#### **❌ "Missing zone data"**
- Ensure batch simulation has run
- Check that zones are being detected
- Verify map has food zones (dark green areas)

#### **❌ "Coordinates don't match map"**
- Check map dimensions in `config.py`
- Verify zone detection parameters
- Ensure map has proper food zone colors

### 📁 **File Structure**

```
your_project/
├── images/
│   └── 3_spots.png          # Your map image
├── batch_results/
│   └── all_244_combined_sim_function.csv  # Simulation data
├── resources_coords.csv      # Generated coordinates
├── generate_coords.py        # Coordinate generator
└── visualization_fixed.py    # Visualization script
```

### 🚀 **Quick Start**

1. **Run simulation**:
   ```bash
   python batch_simulation.py
   ```

2. **Generate coordinates**:
   ```bash
   python generate_coords.py
   ```

3. **Create visualization**:
   ```bash
   python visualization_fixed.py
   ```

### 💡 **Pro Tips**

- **Real Coordinates**: Use `generate_coords.py` for actual zone positions
- **Map Matching**: Coordinates should match your map's food zones
- **Zone Count**: Check that you have the expected number of zones
- **Coordinate Range**: Should be within your map dimensions (0-99, 0-59)

The coordinates are **automatically generated** from your simulation map - you don't need to create them manually! 🎯
