"""
generate_coords.py - Generate zone coordinates from simulation data

This script extracts actual zone coordinates from your simulation data
and creates the coordinates file needed for visualization.py
"""

import os
import pandas as pd
import numpy as np
from resource_manager import identify_zones, extract_resource_coords_from_zones

def generate_coords_from_simulation(map_path: str, output_path: str = "resources_coords.csv"):
    """
    Generate coordinates file from actual simulation zone data.
    
    Args:
        map_path: Path to the map image used in simulation
        output_path: Where to save the coordinates CSV file
    """
    print(f"[INFO] Generating coordinates from map: {map_path}")
    
    # Identify zones from the map (same as simulation does)
    zone_map, food_zone_ids = identify_zones(map_path, min_size=1, tol=20)
    
    print(f"[INFO] Found {len(food_zone_ids)} food zones: {food_zone_ids}")
    
    # Extract coordinates for each zone
    all_coords = []
    for zone_id in food_zone_ids:
        coords, _ = extract_resource_coords_from_zones(zone_map, food_zone_id=zone_id)
        
        # Convert to (x, y) format and add zone_id
        for coord in coords:
            y, x = coord  # Note: coords are (y, x) from np.where
            all_coords.append({
                'zone_id': zone_id,
                'x': int(x),
                'y': int(y)
            })
        
        print(f"[INFO] Zone {zone_id}: {len(coords)} coordinates")
    
    # Save to CSV
    coords_df = pd.DataFrame(all_coords)
    coords_df.to_csv(output_path, index=False)
    
    print(f"[INFO] Saved {len(all_coords)} coordinates to {output_path}")
    print(f"[INFO] Coordinate ranges: x=[{coords_df['x'].min()}-{coords_df['x'].max()}], y=[{coords_df['y'].min()}-{coords_df['y'].max()}]")
    
    return output_path

def generate_coords_from_batch_data(data_path: str, output_path: str = "resources_coords.csv"):
    """
    Generate coordinates file from batch simulation data.
    This creates synthetic coordinates based on the zones found in the data.
    """
    print(f"[INFO] Generating coordinates from batch data: {data_path}")
    
    # Load the batch data
    df = pd.read_csv(data_path)
    
    # Find zone columns
    zone_cols = [col for col in df.columns if col.startswith('z') and '_spawn' in col]
    zones = []
    for col in zone_cols:
        zone_num = int(col.split('_')[0][1:])  # Extract zone number from 'z0_spawn'
        zones.append(zone_num)
    
    zones = sorted(set(zones))
    print(f"[INFO] Found zones in data: {zones}")
    
    # Create coordinates for each zone
    all_coords = []
    for zone_id in zones:
        # Create a grid of coordinates for each zone
        # This is a simplified approach - you can make it more realistic
        zone_size = 20  # 20x20 grid per zone
        start_x = zone_id * 25  # Offset zones horizontally
        
        for x in range(start_x, start_x + zone_size):
            for y in range(zone_size):
                all_coords.append({
                    'zone_id': zone_id,
                    'x': x,
                    'y': y
                })
        
        print(f"[INFO] Zone {zone_id}: {zone_size}x{zone_size} grid at ({start_x}, 0)")
    
    # Save to CSV
    coords_df = pd.DataFrame(all_coords)
    coords_df.to_csv(output_path, index=False)
    
    print(f"[INFO] Saved {len(all_coords)} coordinates to {output_path}")
    return output_path

def main():
    """Main function to generate coordinates"""
    
    # Try to generate from actual simulation map first
    map_path = "images/3_spots.png"
    data_path = "batch_results/all_244_combined_sim_function.csv"
    
    if os.path.exists(map_path):
        print("[INFO] Using actual simulation map to generate coordinates...")
        try:
            generate_coords_from_simulation(map_path, "resources_coords.csv")
            return
        except Exception as e:
            print(f"[WARNING] Failed to generate from map: {e}")
            print("[INFO] Falling back to batch data method...")
    
    # Fallback to batch data method
    if os.path.exists(data_path):
        print("[INFO] Using batch simulation data to generate coordinates...")
        generate_coords_from_batch_data(data_path, "resources_coords.csv")
    else:
        print(f"[ERROR] Neither map file ({map_path}) nor data file ({data_path}) found!")
        print("Please run batch_simulation.py first to generate data.")

if __name__ == "__main__":
    main()
