"""
visualization_fixed.py - Fixed heatmap visualization for zone consumption patterns

Part of the Human Society Simulation project.

Generates spatial heatmaps showing Blue vs Red family resource exploitation
patterns overlaid on the simulation map.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_coords_file_from_data(data_path: str, output_path: str = "resources_coords.csv"):
    """
    Create a coordinates file from simulation data.
    This uses the proper coordinate generation from the simulation.
    """
    print(f"[INFO] Creating coordinates file from {data_path}")
    
    # Import the coordinate generation function
    try:
        from generate_coords import generate_coords_from_batch_data
        return generate_coords_from_batch_data(data_path, output_path)
    except ImportError:
        print("[WARNING] generate_coords.py not found, using fallback method")
        
        # Fallback method
        df = pd.read_csv(data_path)
        
        # Get unique zones from column names
        zone_cols = [col for col in df.columns if col.startswith('z') and '_spawn' in col]
        zones = []
        for col in zone_cols:
            zone_num = int(col.split('_')[0][1:])  # Extract zone number from 'z0_spawn'
            zones.append(zone_num)
        
        zones = sorted(set(zones))
        print(f"[INFO] Found zones: {zones}")
        
        # Create synthetic coordinates (you can replace this with real coordinates)
        coords_data = []
        for i, zone in enumerate(zones):
            # Create a grid of coordinates for each zone
            # This is a simplified approach - replace with actual zone coordinates
            for x in range(10):  # 10x10 grid per zone
                for y in range(10):
                    coords_data.append({
                        'zone_id': zone,
                        'x': x + i * 15,  # Offset zones horizontally
                        'y': y
                    })
        
        coords_df = pd.DataFrame(coords_data)
        coords_df.to_csv(output_path, index=False)
        print(f"[INFO] Created coordinates file: {output_path}")
        return output_path

def plot_heatmap_for_run(
    data_path: str,
    coords_path: str,
    map_path: str,
    run: int = 0,
    scale_x: float = 0.55,
    scale_y: float = 1.0,
    offset_x: int = 1,
    offset_y: int = 0,
    radius: int = 12,
    sigma: float = 5.0,
    alpha: float = 0.6,
    save_path: str = None,
):
    print(f"[INFO] Loading data from {data_path}, coords from {coords_path}, map from {map_path}")

    # 1. Load data
    df = pd.read_csv(data_path)
    coords_df = pd.read_csv(coords_path)
    base_map = np.array(Image.open(map_path).convert("RGB"))
    print(f"[INFO] Data loaded: {len(df)} rows, {len(coords_df)} coords, map shape={base_map.shape}")

    # Select run
    df_run = df[df["run"] == run]
    print(f"[INFO] Selected run={run}, {len(df_run)} rows")

    if df_run.empty:
        print("[WARNING] No rows found for this run! Exiting.")
        return

    # Detect zones
    zones = coords_df["zone_id"].unique()
    print(f"[INFO] Found {len(zones)} unique zones in coords")

    # Blue/Red consumption totals per zone
    blue_cons = {}
    red_cons = {}
    for z in zones:
        blue_col = f"z{z}_blue_cons"
        red_col  = f"z{z}_red_cons"
        if blue_col not in df_run or red_col not in df_run:
            print(f"[WARNING] Missing columns for zone {z}: {blue_col}, {red_col}")
            continue
        blue_cons[z] = df_run[blue_col].sum()
        red_cons[z]  = df_run[red_col].sum()
    print(f"[INFO] Consumption dicts built: {len(blue_cons)} zones with data")

    # 2. Compute centroids
    MAP_WIDTH = coords_df["x"].max() + 1
    MAP_HEIGHT = coords_df["y"].max() + 1
    img_h, img_w = base_map.shape[0], base_map.shape[1]
    print(f"[INFO] Coord bounds: width={MAP_WIDTH}, height={MAP_HEIGHT}, image={img_w}x{img_h}")

    centroids = {}
    for z in zones:
        subset = coords_df[coords_df["zone_id"] == z]
        if len(subset) == 0:
            print(f"[WARNING] No coords for zone {z}")
            continue
        cy, cx = subset["x"].mean(), subset["y"].mean()
        cx = int(cx / MAP_WIDTH  * img_w * scale_x) + offset_x
        cy = int(cy / MAP_HEIGHT * img_h * scale_y) + offset_y
        centroids[z] = (cx, cy)
    print(f"[INFO] Centroids computed for {len(centroids)} zones")

    # 3. Heatmap layers
    heat_blue = np.zeros((img_h, img_w), dtype=np.float32)
    heat_red  = np.zeros((img_h, img_w), dtype=np.float32)

    for z in zones:
        if z not in centroids or z not in blue_cons:
            continue
        cx, cy = centroids[z]
        b_val = blue_cons[z]
        r_val = red_cons[z]
        max_val = max(b_val, r_val)
        if max_val == 0:
            continue
        b_norm = b_val / max_val
        r_norm = r_val / max_val
        # Debug values
        print(f"[DEBUG] Zone {z}: centroid=({cx},{cy}), blue={b_val}, red={r_val}, norm=({b_norm:.2f},{r_norm:.2f})")

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < img_w and 0 <= y < img_h:
                    w = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    heat_blue[y, x] += b_norm * w
                    heat_red[y, x]  += r_norm * w

    # 4. RGB heatmap overlay
    heat_rgb = np.zeros_like(base_map, dtype=np.float32)
    heat_rgb[..., 0] = heat_red
    heat_rgb[..., 2] = heat_blue
    heat_rgb = np.clip(heat_rgb, 0, 1)

    overlay = (1 - alpha) * base_map / 255.0 + alpha * heat_rgb

    # 5. Plot
    print(f"[INFO] Plotting overlay, alpha={alpha}, save_path={save_path}")
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Consumption Heatmap - Run {run}\n(Blue=Blue Family, Red=Red Family, Purple=Both)")
    plt.axis("off")

    # Percentages per zone
    for z in zones:
        if z in centroids and z in blue_cons:
            cx, cy = centroids[z]
            b_val, r_val = blue_cons[z], red_cons[z]
            total = b_val + r_val
            if total > 0:
                b_pct = 100 * b_val / total
                r_pct = 100 * r_val / total
                text = f"Zone {z}\nB:{b_pct:.1f}%\nR:{r_pct:.1f}%"
                plt.text(cx, cy - 5, text,
                         color="white", fontsize=9, ha="center", va="center",
                         bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[INFO] Figure saved to {save_path}")
    else:
        plt.show()
        print("[INFO] Figure displayed")

def main():
    """Main function to run the visualization"""
    
    # File paths
    data_path = "batch_results/all_244_combined_sim_function.csv"
    coords_path = "resources_coords.csv"
    map_path = "images/5_spots_fixed.png"  # Use the actual map file
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        print("Please run batch_simulation.py first to generate data.")
        return
    
    if not os.path.exists(coords_path):
        print(f"[INFO] Creating coordinates file...")
        create_coords_file_from_data(data_path, coords_path)
    
    if not os.path.exists(map_path):
        print(f"[ERROR] Map file not found: {map_path}")
        print("Available maps:")
        for img in os.listdir("images/"):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - images/{img}")
        return
    
    # Create visualization for run 0
    print("[INFO] Creating heatmap visualization...")
    plot_heatmap_for_run(
        data_path=data_path,
        coords_path=coords_path,
        map_path=map_path,
        run=0,
        save_path="heatmap_run0.png"
    )
    
    print("[INFO] Visualization complete!")

if __name__ == "__main__":
    main()
