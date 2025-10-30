"""
ui_simulation.py - Interactive UI simulation with real-time visualization

Part of the Human Society Simulation project.

Provides a pygame-based interactive interface for running simulations
with real-time visualization, parameter controls, and export functionality.
Includes both menu-driven and programmatic interfaces.
"""

import os
import random
import csv
from typing import List, Tuple, Dict, Optional
import numpy as np
import pygame, pygame_menu

from ui_components import (
    Slider,
    handle_pause_event,
    create_action_buttons,
    handle_action_buttons,
    draw_action_buttons
)
from config import *
from human import *               # Human, House, draw_human, etc.
from trust_system import TrustSystem
from social_mechanics import *
from resource_manager import (
    resources,                 # 3D grid [H,W,2] (life, amount)
    life_span_ressource,
    map_manage,
    resource_spawn_interval_inverse,
    display_house_storage,
    identify_zones,
    add_resource,
    set_active_palette
)
from simulation_utils import build_world, run_single_tick, should_move_house, find_best_house_location
from map_color_tools import extract_colors_and_prompt, _unique_colors_rgb, _quantize_image, _rgb_to_css4_name

# Single source of truth for the UI map image path
MAP_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images", "desert_oasis.png")

# ────────────────────────────────────────────────
# Map Color Analysis Functions
# ────────────────────────────────────────────────

def analyze_map_colors(map_path: str) -> Dict[str, str]:
    """
    Analyze colors in the map and return a color-to-category mapping.
    
    Args:
        map_path: Path to the map image file
        
    Returns:
        Dictionary mapping color names to categories (food, terrain, obstacle, etc.)
    """
    try:
        # If a saved mapping exists for this map, load and reuse it
        save_path = f"{map_path}_color_analysis.json"
        if os.path.isfile(save_path):
            import json
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept both formats: {"mapping": {...}} or raw mapping dict
            if isinstance(data, dict) and "mapping" in data and isinstance(data["mapping"], dict):
                return data["mapping"]
            if isinstance(data, dict):
                return data

        # Otherwise, extract and categorize colors interactively once, then save
        color_mapping = extract_colors_and_prompt(
            map_path,
            max_colors=32,
            categories=[
                "border", "grass",
                "house_blue", "house_red",
                "food_1", "food_2",
                "water", "obstacle",
                "ignore", "other",
            ],
            default_category="grass",
            save_path=save_path
        )
        return color_mapping
    except Exception as e:
        print(f"Error analyzing map colors: {e}")
        return {}

def build_palette_from_color_mapping(map_path: str, color_mapping: Dict[str, str]) -> Dict[Tuple[int,int,int], int]:
    """
    Build a palette mapping RGB -> zone_id from the interactive color mapping.
    Categories map to zone IDs per NEW_PALETTE semantics:
      border->0, grass->1, house_blue->2, house_red->3, food_1->4, food_2->5.
      water/obstacle/other/ignore default to grass (1) unless border explicitly set.
    """
    from PIL import Image
    img = Image.open(map_path)
    colors = _unique_colors_rgb(img)

    used_names = set()
    rgb_key_to_category: Dict[Tuple[int,int,int], str] = {}
    for color in colors:
        hex_color = "#%02x%02x%02x" % color
        name_guess, _ = _rgb_to_css4_name(color)
        key_name = name_guess
        if key_name in used_names:
            key_name = f"{name_guess} ({hex_color})"
        used_names.add(key_name)
        category = color_mapping.get(key_name)
        if category:
            rgb_key_to_category[color] = category

    def cat_to_zone(cat: str) -> int:
        if cat == "border":
            return 0
        if cat == "grass":
            return 1
        if cat == "house_blue":
            return 2
        if cat == "house_red":
            return 3
        if cat == "food_1":
            return 4
        if cat == "food_2":
            return 5
        # default fallbacks
        if cat == "obstacle":
            return 0
        return 1

    palette: Dict[Tuple[int,int,int], int] = {}
    for rgb, cat in rgb_key_to_category.items():
        palette[rgb] = cat_to_zone(cat)
    return palette

def get_map_color_info(map_path: str) -> Dict:
    """
    Get detailed information about colors in the map.
    
    Args:
        map_path: Path to the map image file
        
    Returns:
        Dictionary with color analysis information
    """
    from PIL import Image
    
    try:
        img = Image.open(map_path)
        colors = _unique_colors_rgb(img)
        
        # If too many colors, quantize the image
        if len(colors) > 32:
            quantized_img = _quantize_image(img, max_colors=32)
            colors = _unique_colors_rgb(quantized_img)
        
        return {
            "total_colors": len(colors),
            "colors": colors,
            "image_size": img.size,
            "image_mode": img.mode
        }
    except Exception as e:
        print(f"Error getting map color info: {e}")
        return {}

def display_color_legend(screen, color_mapping: Dict[str, str], font: pygame.font.Font):
    """
    Display a legend showing the color categories found in the map.
    
    Args:
        screen: Pygame screen surface
        color_mapping: Dictionary mapping color names to categories
        font: Pygame font for text rendering
    """
    if not color_mapping:
        return
    
    # Group colors by category
    categories = {}
    for color_name, category in color_mapping.items():
        if category not in categories:
            categories[category] = []
        categories[category].append(color_name)
    
    # Display legend
    y_offset = 50
    for category, color_names in categories.items():
        text = f"{category}: {', '.join(color_names[:3])}"  # Show first 3 colors
        if len(color_names) > 3:
            text += f" (+{len(color_names)-3} more)"
        
        surf = font.render(text, True, (255, 255, 255))
        screen.blit(surf, (10, y_offset))
        y_offset += 20

# ────────────────────────────────────────────────
# House Movement Analysis
# ────────────────────────────────────────────────

def export_house_movement_log(movement_log: List[Dict], filename: str = "house_movements.csv"):
    """
    Export house movement statistics to CSV for analysis.
    
    Args:
        movement_log: List of movement event dictionaries
        filename: Output CSV filename
    """
    if not movement_log:
        print("No house movements to export")
        return
    
    import csv
    with open(filename, 'w', newline='') as f:
        fieldnames = ['day', 'house', 'old_x', 'old_y', 'new_x', 'new_y', 
                     'distance', 'storage_S', 'travel_D', 'food_F', 
                     'P_move', 'inertia', 'population']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(movement_log)
    
    print(f"Exported {len(movement_log)} house movement events to {filename}")
    
    # Print summary statistics
    by_house = {}
    for event in movement_log:
        house_name = event['house']
        if house_name not in by_house:
            by_house[house_name] = {'count': 0, 'total_distance': 0.0}
        by_house[house_name]['count'] += 1
        by_house[house_name]['total_distance'] += event['distance']
    
    print("\nHouse Movement Summary:")
    for house_name, stats in by_house.items():
        avg_distance = stats['total_distance'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {house_name}: {stats['count']} moves, avg distance: {avg_distance:.2f} cells")

def plot_house_movement_analysis(movement_log: List[Dict], save_dir: str = "house_movement_plots"):
    """
    Create comprehensive visualizations of house movement patterns.
    
    Args:
        movement_log: List of movement event dictionaries
        save_dir: Directory to save plot images
    """
    if not movement_log:
        print("No house movements to visualize")
        return
    
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Separate data by house
    blue_events = [e for e in movement_log if e['house'] == 'Blue']
    red_events = [e for e in movement_log if e['house'] == 'Red']
    
    # Create a multi-panel figure with better layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('House Movement Analysis', fontsize=16, fontweight='bold')
    
    # 1. Movement trajectory (position over time)
    ax = axes[0, 0]
    if blue_events:
        days_b = [e['day'] for e in blue_events]
        x_b = [e['new_x'] for e in blue_events]
        y_b = [e['new_y'] for e in blue_events]
        ax.plot(x_b, y_b, 'b-o', label='Blue house', markersize=4, alpha=0.7)
        ax.scatter([blue_events[0]['old_x']], [blue_events[0]['old_y']], 
                  color='blue', s=200, marker='s', label='Blue start', zorder=5)
    if red_events:
        days_r = [e['day'] for e in red_events]
        x_r = [e['new_x'] for e in red_events]
        y_r = [e['new_y'] for e in red_events]
        ax.plot(x_r, y_r, 'r-o', label='Red house', markersize=4, alpha=0.7)
        ax.scatter([red_events[0]['old_x']], [red_events[0]['old_y']], 
                  color='red', s=200, marker='s', label='Red start', zorder=5)
    ax.set_xlabel('X Position (cells)')
    ax.set_ylabel('Y Position (cells)')
    ax.set_title('House Movement Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # No invert - standard math coordinates (0,0) at bottom-left
    
    # 2. Storage factor (S) over time - higher is better (well-fed house)
    ax = axes[0, 1]
    if blue_events:
        days_b = [e['day'] for e in blue_events]
        ax.plot(days_b, [e['storage_S'] for e in blue_events], 'b-o', label='Blue', linewidth=2, markersize=5)
    if red_events:
        days_r = [e['day'] for e in red_events]
        ax.plot(days_r, [e['storage_S'] for e in red_events], 'r-o', label='Red', linewidth=2, markersize=5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-level')
    ax.set_xlabel('Day')
    ax.set_ylabel('Storage Level (0-1)')
    ax.set_title('Storage (S): House Food Reserves\n(Low → Pressure to Move)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 3. Travel distance factor (D) over time - higher means more energy wasted
    ax = axes[1, 0]
    if blue_events:
        days_b = [e['day'] for e in blue_events]
        ax.plot(days_b, [e['travel_D'] for e in blue_events], 'b-o', label='Blue', linewidth=2, markersize=5)
    if red_events:
        days_r = [e['day'] for e in red_events]
        ax.plot(days_r, [e['travel_D'] for e in red_events], 'r-o', label='Red', linewidth=2, markersize=5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-level')
    ax.set_xlabel('Day')
    ax.set_ylabel('Travel Distance (0-1)')
    ax.set_title('Travel (D): Avg Daily Distance\n(High → Pressure to Move Closer)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 4. Local food availability (F) over time - higher means more nearby food
    ax = axes[1, 1]
    if blue_events:
        days_b = [e['day'] for e in blue_events]
        ax.plot(days_b, [e['food_F'] for e in blue_events], 'b-o', label='Blue', linewidth=2, markersize=5)
    if red_events:
        days_r = [e['day'] for e in red_events]
        ax.plot(days_r, [e['food_F'] for e in red_events], 'r-o', label='Red', linewidth=2, markersize=5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-level')
    ax.set_xlabel('Day')
    ax.set_ylabel('Food Availability (0-1)')
    ax.set_title('Food (F): Nearby Resources\n(Low → Pressure to Move to Better Area)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 5. Combined pressure (why they moved)
    ax = axes[0, 1]
    if blue_events:
        days_b = [e['day'] for e in blue_events]
        # Calculate pressure: higher values = more pressure to move
        pressure_b = [
            HOUSE_RELOC_ALPHA1*(1-e['storage_S']) + 
            HOUSE_RELOC_ALPHA2*e['travel_D'] + 
            HOUSE_RELOC_ALPHA3*(1-e['food_F'])
            for e in blue_events
        ]
        ax.plot(days_b, pressure_b, 'b-o', label='Blue', linewidth=2, markersize=5)
    if red_events:
        days_r = [e['day'] for e in red_events]
        pressure_r = [
            HOUSE_RELOC_ALPHA1*(1-e['storage_S']) + 
            HOUSE_RELOC_ALPHA2*e['travel_D'] + 
            HOUSE_RELOC_ALPHA3*(1-e['food_F'])
            for e in red_events
        ]
        ax.plot(days_r, pressure_r, 'r-o', label='Red', linewidth=2, markersize=5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Move Pressure Score')
    ax.set_title('Why Houses Moved:\nCombined Pressure\n(Low Storage + High Travel + Low Food)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Movement events timeline
    ax = axes[2, 0]
    if blue_events:
        days_b = [e['day'] for e in blue_events]
        ax.scatter(days_b, [1]*len(days_b), color='blue', alpha=0.7, s=100, marker='|', linewidths=3, label='Blue moves')
    if red_events:
        days_r = [e['day'] for e in red_events]
        ax.scatter(days_r, [0]*len(days_r), color='red', alpha=0.7, s=100, marker='|', linewidths=3, label='Red moves')
    ax.set_xlabel('Day')
    ax.set_ylabel('House')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Red', 'Blue'])
    ax.set_title('When Houses Moved (Timeline)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 7. Distance moved per event
    ax = axes[2, 1]
    if blue_events:
        days_b = [e['day'] for e in blue_events]
        dist_b = [e['distance'] for e in blue_events]
        ax.bar(days_b, dist_b, color='blue', alpha=0.6, width=2, label='Blue')
    if red_events:
        days_r = [e['day'] for e in red_events]
        dist_r = [e['distance'] for e in red_events]
        ax.bar(days_r, dist_r, color='red', alpha=0.6, width=2, label='Red')
    ax.set_xlabel('Day')
    ax.set_ylabel('Distance Moved (cells)')
    ax.set_title('How Far Each Move Was\n(Max 2 cells per night)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'house_movement_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved house movement plots to {plot_path}")
    plt.close()
    
    # Create a second figure for 2D spatial heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Plot all positions visited by each house
    if blue_events:
        all_x_b = [blue_events[0]['old_x']] + [e['new_x'] for e in blue_events]
        all_y_b = [blue_events[0]['old_y']] + [e['new_y'] for e in blue_events]
        ax2.scatter(all_x_b, all_y_b, c=range(len(all_x_b)), cmap='Blues', 
                   s=100, alpha=0.6, edgecolors='blue', linewidths=2, label='Blue house')
        ax2.plot(all_x_b, all_y_b, 'b-', alpha=0.3, linewidth=1)
        
    if red_events:
        all_x_r = [red_events[0]['old_x']] + [e['new_x'] for e in red_events]
        all_y_r = [red_events[0]['old_y']] + [e['new_y'] for e in red_events]
        ax2.scatter(all_x_r, all_y_r, c=range(len(all_x_r)), cmap='Reds',
                   s=100, alpha=0.6, edgecolors='red', linewidths=2, label='Red house')
        ax2.plot(all_x_r, all_y_r, 'r-', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('X Position (cells)')
    ax2.set_ylabel('Y Position (cells)')
    ax2.set_title('House Movement Spatial Trajectory (darker = later)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # No invert - standard math coordinates (0,0) at bottom-left
    
    trajectory_path = os.path.join(save_dir, 'house_trajectories.png')
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    print(f"Saved house trajectories to {trajectory_path}")
    plt.close()
    
    # Create a third figure: house movements overlaid on food zone map
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    
    # Get zone_map from a human (all humans share same codes)
    if movement_log:
        # Try to get zone_map from simulation context
        # We'll need to pass it or reconstruct it from the map
        try:
            # Import here to avoid circular dependencies
            from resource_manager import identify_zones
            from simulation_utils import cached_identify
            
            # Get the map path from the first move event or use default
            # For now, use the global MAP_IMAGE_PATH
            zone_map, food_ids = cached_identify(MAP_IMAGE_PATH, min_size=1, tol=20)
            
            # Create a color map for visualization
            H, W = zone_map.shape
            map_colors = np.zeros((H, W, 3), dtype=np.uint8)
            
            # Color the map: grass=tan, walls=black, food1=light green, food2=dark green
            map_colors[zone_map == 0] = [0, 0, 0]        # walls/border
            map_colors[zone_map == 1] = [222, 197, 158]  # grass (tan)
            
            # Food type 1 zones (41-79) - light green
            food1_mask = (zone_map >= 41) & (zone_map < 80)
            map_colors[food1_mask] = [144, 238, 144]  # light green
            
            # Food type 2 zones (81+) - darker green
            food2_mask = zone_map >= 81
            map_colors[food2_mask] = [34, 139, 34]  # forest green
            
            # Display the map with (0,0) at bottom-left
            ax3.imshow(map_colors, origin='lower', aspect='equal', interpolation='nearest')
            
            # Overlay house trajectories
            if blue_events:
                all_x_b = [blue_events[0]['old_x']] + [e['new_x'] for e in blue_events]
                all_y_b = [blue_events[0]['old_y']] + [e['new_y'] for e in blue_events]
                ax3.plot(all_x_b, all_y_b, 'b-', linewidth=3, alpha=0.8, label='Blue path')
                ax3.scatter(all_x_b, all_y_b, c=range(len(all_x_b)), cmap='Blues',
                           s=150, alpha=0.9, edgecolors='darkblue', linewidths=2, zorder=10)
                # Mark start and end
                ax3.scatter([all_x_b[0]], [all_y_b[0]], color='blue', s=300, marker='s', 
                           edgecolors='white', linewidths=2, label='Blue start', zorder=11)
                ax3.scatter([all_x_b[-1]], [all_y_b[-1]], color='navy', s=300, marker='*', 
                           edgecolors='white', linewidths=2, label='Blue end', zorder=11)
            
            if red_events:
                all_x_r = [red_events[0]['old_x']] + [e['new_x'] for e in red_events]
                all_y_r = [red_events[0]['old_y']] + [e['new_y'] for e in red_events]
                ax3.plot(all_x_r, all_y_r, 'r-', linewidth=3, alpha=0.8, label='Red path')
                ax3.scatter(all_x_r, all_y_r, c=range(len(all_x_r)), cmap='Reds',
                           s=150, alpha=0.9, edgecolors='darkred', linewidths=2, zorder=10)
                # Mark start and end
                ax3.scatter([all_x_r[0]], [all_y_r[0]], color='red', s=300, marker='s',
                           edgecolors='white', linewidths=2, label='Red start', zorder=11)
                ax3.scatter([all_x_r[-1]], [all_y_r[-1]], color='darkred', s=300, marker='*',
                           edgecolors='white', linewidths=2, label='Red end', zorder=11)
            
            ax3.set_xlabel('X Position (cells)')
            ax3.set_ylabel('Y Position (cells)')
            ax3.set_title('House Movements Overlaid on Food Zone Map\n(Light Green=Food1/Regenerative, Dark Green=Food2/Finite)')
            ax3.legend(loc='upper right', fontsize=10)
            
            # Add custom legend for map colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=[222/255, 197/255, 158/255], edgecolor='black', label='Grass/Terrain'),
                Patch(facecolor=[144/255, 238/255, 144/255], edgecolor='black', label='Food Zone 1 (Regenerative)'),
                Patch(facecolor=[34/255, 139/255, 34/255], edgecolor='black', label='Food Zone 2 (Finite)'),
                Patch(facecolor='black', edgecolor='black', label='Walls/Border'),
            ]
            ax3.legend(handles=legend_elements, loc='lower left', fontsize=9, title='Map Legend')
            
            food_map_path = os.path.join(save_dir, 'house_movements_on_food_map.png')
            plt.savefig(food_map_path, dpi=150, bbox_inches='tight')
            print(f"Saved house movements on food map to {food_map_path}")
            plt.close()
            
        except Exception as e:
            print(f"Could not generate food zone map overlay: {e}")

# ────────────────────────────────────────────────
# HUD helpers
# ────────────────────────────────────────────────
def display_human_counts(screen, humans: List[Human], font: pygame.font.Font):
    """Display alive/dead human counts on screen, broken down by house."""
    # Total counts
    alive = sum(h.alive for h in humans)
    dead  = len(humans) - alive
    
    # Per-house counts
    blue_alive = sum(1 for h in humans if h.alive and h.home.color == (0, 0, 128))
    blue_dead = sum(1 for h in humans if not h.alive and h.home.color == (0, 0, 128))
    red_alive = sum(1 for h in humans if h.alive and h.home.color == (255, 0, 0))
    red_dead = sum(1 for h in humans if not h.alive and h.home.color == (255, 0, 0))
    
    # Display all counts
    x = MAP_WIDTH*CELL_SIZE - 10
    y_base = MAP_HEIGHT*CELL_SIZE - 10
    
    lines = [
        (f"Alive: {alive}   Dead: {dead}", (255, 255, 255)),
        (f"Blue alive: {blue_alive}   Blue dead: {blue_dead}", (100, 100, 255)),
        (f"Red alive: {red_alive}   Red dead: {red_dead}", (255, 100, 100)),
    ]
    
    for i, (text, color) in enumerate(lines):
        surf = font.render(text, True, color)
        y = y_base - (len(lines) - 1 - i) * (surf.get_height() + 4)
        rect = surf.get_rect(bottomright=(x, y))
        screen.blit(surf, rect)

def draw_houses_overlay(screen, houses: List[House]):
    """Draw house squares at their current positions as an overlay."""
    import pygame
    for house in houses:
        # Make house square 4x bigger for visibility
        house_display_size = HOUSE_SIZE * CELL_SIZE * 4
        rx = house.x * CELL_SIZE - (house_display_size - CELL_SIZE) // 2
        ry = house.y * CELL_SIZE - (house_display_size - CELL_SIZE) // 2
        rect = pygame.Rect(rx, ry, house_display_size, house_display_size)
        pygame.draw.rect(screen, house.color, rect, 3)  # Thicker border too

def draw_legend(screen, cell_size: int, font: pygame.font.Font):
    """Draw legend showing energy and bag capacity indicators."""
    entries = [
        ("energy",      (0, 255,   0)),
        ("bag_capacity",(255,   0,   0)),
    ]
    margin  = cell_size
    swatch  = 2*cell_size
    spacing = int(swatch * 0.6)
    line_h  = swatch + spacing
    screen_h = MAP_HEIGHT * cell_size
    start_x  = margin
    start_y  = screen_h - margin - line_h * len(entries)

    for i, (label, color) in enumerate(entries):
        y = start_y + i * line_h
        rect = pygame.Rect(start_x, y, swatch, swatch)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (0,0,0), rect, 2)
        text_surf = font.render(label, True, (255,255,255))
        text_pos  = (start_x + swatch + spacing,
                     y + (swatch - text_surf.get_height())//2)
        screen.blit(text_surf, text_pos)

# ────────────────────────────────────────────────
# Main Simulation Functions
# ────────────────────────────────────────────────

def start_interactive_simulation():
    """Start the interactive pygame simulation with menu and controls."""
    # Build world using shared utility
    map_path = MAP_IMAGE_PATH
    
    # Analyze map colors and build a runtime palette interactively
    print("Analyzing map colors...")
    color_mapping = analyze_map_colors(map_path)
    runtime_palette = build_palette_from_color_mapping(map_path, color_mapping)
    if runtime_palette:
        set_active_palette(runtime_palette)
        print(f"Applied runtime palette with {len(runtime_palette)} color entries")
    color_info = get_map_color_info(map_path)
    print(f"Found {color_info.get('total_colors', 0)} unique colors in map")
    
    world = build_world(
        seed=None, 
        map_path=map_path, 
        min_size=1, 
        tol=20, 
        precomputed=None, 
        n_days=200  # Default for UI mode
    )
    
    # Extract world components
    zone_map = world["zone_map"]
    houses = world["houses"]
    humans = world["humans"]
    trust_system = world["trust_system"]
    food_cells = world["food_cells"]
    last_storage = world["last_storage"]

    # Initialize pygame (reuse existing initialization)
    pygame.init()
    screen = pygame.display.set_mode((MAP_WIDTH*CELL_SIZE +200, MAP_HEIGHT*CELL_SIZE+200))
    pygame.display.set_caption("Human Society Simulation")
    font = pygame.font.Font(None, max(12, CELL_SIZE*4))

    # Build static terrain using original map image (prefer fidelity over palette tiles)
    try:
        orig_img = pygame.image.load(map_path)
        orig_img = pygame.transform.scale(orig_img, (MAP_WIDTH*CELL_SIZE, MAP_HEIGHT*CELL_SIZE))
        static_layer = orig_img.convert()
    except Exception:
        # Fallback to palette rendering if direct load fails
        static_layer = map_manage(zone_map)

    # UI widgets
    current_population = len([h for h in humans if h.alive])
    slider = Slider((MAP_WIDTH*CELL_SIZE+60, 50, 30, MAP_HEIGHT*CELL_SIZE-160), 1, 500, current_population, orientation='vertical', label='H', quantize_to_int=True)
    speed_slider = Slider(((MAP_WIDTH*CELL_SIZE)//4, MAP_HEIGHT*CELL_SIZE+120, (MAP_WIDTH*CELL_SIZE)//2, 20), 0.1, 6.0, 1.0, orientation='horizontal')
    # Trust threshold slider (0..1) next to population slider (to its right)
    # Default to None (no override) - only applies if user moves the slider
    trust_threshold = None  # Will use default 0.55 if not touched
    trust_slider = Slider((MAP_WIDTH*CELL_SIZE+140, 50, 30, MAP_HEIGHT*CELL_SIZE-160), 0.0, 1.0, 0.55, orientation='vertical', label='T', quantize_to_int=False)
    trust_slider_touched = False  # Track if user has moved it
    action_rects = create_action_buttons(speed_slider)

    # bookkeeping
    last_storage = {h: h.storage for h in houses}
    total_shares = 0
    clock, paused = pygame.time.Clock(), False
    # Simulation time bookkeeping
    tick_accum = 0.0  # accumulates speed into whole simulation ticks
    tick_count = 0    # absolute tick counter
    day_tick = DAY_LENGTH * 0.3
    next_id = len(humans)  # humans already created by build_world
    target_population = current_population  # Track desired population from slider
    prev_is_day = False
    pick_history: List[int] = []
    last_mated: Dict[Tuple[int,int], int] = {}
    births_by_pair: Dict[Tuple[int,int], int] = {}
    deaths_today: List[int] = []
    day = 0
    next_day_tick = DAY_LENGTH
    
    # Initialize house attributes for adaptive relocation
    for house in houses:
        if not hasattr(house, "inertia"):
            house.inertia = 0.0
    
    # Food respawn toggle (runtime control)
    food_respawn_enabled = True
    # Trust mechanism toggle (runtime control)
    trust_enabled = True
    
    # House movement tracking
    house_movement_log = []  # List of dicts tracking each move event
    
    # Population control: slider is for initial setup only
    simulation_started = False  # Once sim starts, slider becomes read-only display
    
    # Initialize day/night cycle variables (for rendering even when paused)
    is_day = True
    cycle_pos = 0.0

    # CSV init
    blue_file = "blue_population.csv"
    red_file  = "red_population.csv"
    for fn in (blue_file, red_file):
        with open(fn, 'w', newline='') as f:
            csv.writer(f).writerow(["day", "population"])

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            # Toggle food respawn with 'R' key
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                food_respawn_enabled = not food_respawn_enabled
                print(f"Food respawn: {'ENABLED' if food_respawn_enabled else 'DISABLED'}")
            # Toggle trust mechanism with 'T' key
            if e.type == pygame.KEYDOWN and e.key == pygame.K_t:
                trust_enabled = not trust_enabled
                print(f"Trust system: {'ENABLED' if trust_enabled else 'DISABLED'}")
            # Allow user to place houses by clicking (up to 2 houses)
            if len(houses) < 2 and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                if 0 <= mx < MAP_WIDTH*CELL_SIZE and 0 <= my < MAP_HEIGHT*CELL_SIZE:
                    gx = int(mx // CELL_SIZE)
                    gy = int(my // CELL_SIZE)
                    if len(houses) == 0:
                        new_house = House(gx, gy, (0, 0, 128))  # blue
                        houses.append(new_house)
                        last_storage[new_house] = 0  # Initialize storage tracking
                        new_house.inertia = 0.0  # Initialize inertia
                        print(f"Blue house placed at ({gx}, {gy})")
                    elif len(houses) == 1:
                        new_house = House(gx, gy, (255, 0, 0))  # red
                        houses.append(new_house)
                        last_storage[new_house] = 0  # Initialize storage tracking
                        new_house.inertia = 0.0  # Initialize inertia
                        print(f"Red house placed at ({gx}, {gy})")
                        # After placing both houses, spawn initial humans around them
                        total_to_spawn = max(Nbre_HUMANS, 2)
                        per_house = total_to_spawn // 2
                        remainder = total_to_spawn - per_house * 2
                        next_local_id = len(humans)
                        print(f"Spawning {total_to_spawn} humans ({per_house} per house)...")
                        for idx, house in enumerate(houses):
                            count = per_house + (1 if idx < remainder else 0)
                            for _ in range(count):
                                x = max(0, min(MAP_WIDTH - 1, house.x + random.randint(-3, 3)))
                                y = max(0, min(MAP_HEIGHT - 1, house.y + random.randint(-3, 3)))
                                sex = random.choice(['M', 'F'])
                                new_h = Human(next_local_id, sex, x, y, house, zone_map)
                                humans.append(new_h)
                                trust_system.init_human(next_local_id)
                                next_local_id += 1
                        print(f"Spawned {len(humans)} total humans")
                        # Update next_id counter
                        next_id = next_local_id
            paused = handle_pause_event(e, paused)
            
            # Track if user moves trust slider
            old_trust_val = trust_slider.value
            slider.handle_event(e)
            speed_slider.handle_event(e)
            trust_slider.handle_event(e)
            
            # Detect if trust slider was actually moved by user
            if abs(trust_slider.value - old_trust_val) > 0.01:
                trust_slider_touched = True
                trust_threshold = float(trust_slider.value)
                print(f"Trust threshold set to {trust_threshold:.2f} by user")
            def export_all_data():
                export_trust_matrix(trust_system, humans)
                export_house_movement_log(house_movement_log)
                plot_house_movement_analysis(house_movement_log)
            
            handle_action_buttons(
                e, action_rects,
                on_reset=lambda: start_interactive_simulation(),
                on_export=export_all_data
            )

        # Handle population changes from slider
        current_alive = sum(1 for h in humans if h.alive)
        target_pop = int(slider.value)
        
        # Once simulation starts (unpaused and has moved), mark it
        if not simulation_started and not paused and tick_count > 0:
            simulation_started = True
            print(f"Simulation started with {current_alive} humans.")
        
        # Slider behavior:
        # - Before start: fully interactive
        # - While running: locked (display only)
        # - While paused: "cheat mode" - interactive again!
        can_adjust_population = (not simulation_started) or paused
        
        # Lock slider to current population when running
        if not can_adjust_population:
            slider.value = float(current_alive)
        
        if can_adjust_population and target_pop != target_population:
            # Initial setup: slider can add/remove humans
            target_population = target_pop
            
            if target_pop > current_alive:
                # Add humans
                to_add = target_pop - current_alive
                for _ in range(to_add):
                    if houses:  # Ensure we have houses to assign to
                        # Choose a random house
                        house = random.choice(houses)
                        # Find empty position near the house
                        for _ in range(10):  # Try up to 10 times
                            x = house.x + random.randint(-5, 5)
                            y = house.y + random.randint(-5, 5)
                            if (0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT and 
                                zone_map[y, x] != 0):  # Not in walls
                                # Choose random sex for new human
                                sex = random.choice(['M', 'F'])
                                new_human = Human(next_id, sex, x, y, house, zone_map)
                                humans.append(new_human)
                                trust_system.init_human(next_id)
                                next_id += 1
                                break
            elif target_pop < current_alive:
                # Remove humans (kill the most recent ones)
                to_remove = current_alive - target_pop
                alive_humans = [h for h in humans if h.alive]
                for h in alive_humans[-to_remove:]:  # Remove from the end
                    h.alive = False
        else:
            # After simulation starts: update slider to show current population (read-only)
            slider.value = current_alive
            target_population = current_alive

        if not paused:
            # Accumulate slider speed into whole simulation ticks
            tick_accum += float(speed_slider.value)
            sim_steps = int(tick_accum)
            if sim_steps > 0:
                tick_accum -= sim_steps

            # Run sim_steps simulation ticks this frame
            for _ in range(sim_steps):
                tick_count += 1
                # Day/Night cycle (70% day, 30% night): advance exactly 1 tick
                cycle_pos = ((tick_count - 1) % DAY_LENGTH) / DAY_LENGTH
                is_day = cycle_pos < 0.7

                # Dawn: adaptive house relocation + leadership competition + mating (at start of each day)
                if ((tick_count - 1) % DAY_LENGTH) == 0 and houses:
                    current_day = (tick_count - 1) // DAY_LENGTH
                    
                    # Flush trust updates from previous day
                    trust_system.flush()
                    
                    # Trust decay (forgetting) - apply every TRUST_DECAY_INTERVAL days
                    if trust_enabled and current_day % TRUST_DECAY_INTERVAL == 0 and current_day > 0:
                        # Decay all trust relationships
                        for h in humans:
                            if h.alive:
                                trust_system.init_human(h.id)
                                # Get all known relationships for this human
                                if h.id in trust_system.hints:
                                    data = trust_system.hints[h.id]
                                    size = data.get("size", 0)
                                    if size == 0:
                                        # Initialize size if not set
                                        size = len(data.get("index", {}))
                                        data["size"] = size
                                    
                                    # Decay trust for all known relationships
                                    for other_id in list(data.get("index", {}).keys()):
                                        trust_system.increase_trust(
                                            trustor_id=h.id,
                                            trustee_id=other_id,
                                            increment=-TRUST_DECAY_AMOUNT,
                                            refresh=False
                                        )
                        trust_system.flush()
                        # Removed trust decay console output per user request
                        # print(f"💔 Day {current_day}: Trust decay applied (-{TRUST_DECAY_AMOUNT} to all relationships)")
                    
                    # Adaptive house relocation at dawn (before competition)
                    for house in houses:
                        # Calculate decision factors before move
                        members = [h for h in humans if h.home is house and h.alive]
                        S = min(1.0, house.storage / MAX_HOUSE_STORAGE if MAX_HOUSE_STORAGE else 0.0)
                        dists = [getattr(h, "daily_travel", 0.0) for h in members]
                        D = min(1.0, sum(dists) / len(dists) / HOUSE_MAX_TRAVEL_PER_DAY) if dists else 0.0
                        
                        # Local food availability
                        y0 = max(0, house.y - HOUSE_LOCAL_RADIUS)
                        y1 = min(resources.shape[0]-1, house.y + HOUSE_LOCAL_RADIUS)
                        x0 = max(0, house.x - HOUSE_LOCAL_RADIUS)
                        x1 = min(resources.shape[1]-1, house.x + HOUSE_LOCAL_RADIUS)
                        local_food = resources[y0:y1+1, x0:x1+1, 1].sum()
                        max_cells = (y1-y0+1)*(x1-x0+1) * MAX_FOOD_PER_CELL
                        F = min(1.0, local_food / max_cells if max_cells > 0 else 0.0)
                        
                        P_move = should_move_house(house, humans, resources)
                        
                        if random.random() < P_move:
                            old_x, old_y = house.x, house.y
                            new_x, new_y = find_best_house_location(house, humans, resources)
                            distance_moved = ((new_x - old_x)**2 + (new_y - old_y)**2)**0.5
                            
                            # Log the movement with decision factors
                            house_name = "Blue" if house.color == (0, 0, 128) else "Red" if house.color == (255, 0, 0) else str(house.color)
                            house_movement_log.append({
                                'day': current_day,
                                'house': house_name,
                                'old_x': old_x,
                                'old_y': old_y,
                                'new_x': new_x,
                                'new_y': new_y,
                                'distance': distance_moved,
                                'storage_S': S,
                                'travel_D': D,
                                'food_F': F,
                                'P_move': P_move,
                                'inertia': getattr(house, 'inertia', 0.0),
                                'population': len(members)
                            })
                            
                            house.x, house.y = new_x, new_y
                            house.inertia = 0.0
                            # Move all humans in house
                            for h in humans:
                                if h.home is house and h.alive:
                                    h.home_x = new_x
                                    h.home_y = new_y
                                    h.x = new_x
                                    h.y = new_y
                        else:
                            # Increase inertia if house didn't move
                            house.inertia = min(1.0, getattr(house, "inertia", 0.0) + HOUSE_INERTIA_STEP)
                    
                    # Reset daily travel metrics
                    for h in humans:
                        if h.alive:
                            h.daily_travel = 0.0
                    
                    # Competition for leadership (only if trust enabled)
                    if trust_enabled:
                        # Use slider value only if user has touched it, otherwise use default 0.55
                        leadership_threshold = trust_threshold if trust_slider_touched else 0.55
                        for house in houses:
                            family = [h for h in humans if h.home is house and h.alive]
                            if family:
                                run_competition(family, trust_system, threshold=leadership_threshold)
                    
                    # Mating logic (end of day) - requires trust system to be enabled
                    if ENABLE_MATING and trust_enabled and humans:
                        residents_by_house = {house: [] for house in houses}
                        for h in humans:
                            if h.alive:
                                residents_by_house[h.home].append(h)
                        
                        total_births_today = 0
                        mating_attempts = 0
                        energy_failures = 0
                        trust_failures = 0
                        
                        for house, residents in residents_by_house.items():
                            for i in range(len(residents)):
                                for j in range(i + 1, len(residents)):
                                    h1, h2 = residents[i], residents[j]
                                    mating_attempts += 1
                                    
                                    # Energy check
                                    if h1.energy < ENERGY_COST or h2.energy < ENERGY_COST:
                                        energy_failures += 1
                                        continue
                                    # Cooldown check
                                    pair = tuple(sorted((h1.id, h2.id)))
                                    if tick_count - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                                        continue
                                    # Attempt mating (will check trust internally)
                                    # Use lower threshold (0.6) to allow mating with reasonable trust
                                    created, next_id = to_mate(
                                        h1, h2, trust_system, humans, zone_map, next_id,
                                        threshold=0.6, energy_cost=ENERGY_COST
                                    )
                                    if created:
                                        last_mated[pair] = tick_count
                                        births_by_pair[pair] = births_by_pair.get(pair, 0) + created
                                        total_births_today += created
                                    else:
                                        # Failed due to trust
                                        trust_failures += 1
                        
                        # Removed baby birth console output per user request
                        # if total_births_today > 0:
                        #     new_pop = len([h for h in humans if h.alive])
                        #     print(f"🍼 Day {current_day}: {total_births_today} babies born! Population: {new_pop}")
                        # elif current_day % 10 == 0 and mating_attempts > 0:
                        #     print(f"📊 Day {current_day}: No births. Attempts: {mating_attempts}, Energy fails: {energy_failures}, Trust fails: {trust_failures}")

                # Resource decay
                life_span_ressource()

                # Humans act using standardized logic (one tick)
                # Pass None for trust_system if trust is disabled (each human for themselves)
                picked_this_tick, shared_this_tick, per_family_consumption = run_single_tick(
                    humans, houses, trust_system if trust_enabled else None, resources, is_day, last_storage, 
                    use_occupancy_map=True
                )
                total_shares += shared_this_tick

                # Track deaths
                for h in humans:
                    if not h.alive and h.id not in deaths_today:
                        deaths_today.append(h.id)

                # Adaptive spawn based on recent picks (only if respawn enabled)
                if food_respawn_enabled:
                    pick_history.append(picked_this_tick)
                    if len(pick_history) > 30:
                        pick_history.pop(0)
                    avg_pick_rate = (sum(pick_history) / len(pick_history)) if pick_history else 0.0
                    interval = resource_spawn_interval_inverse(avg_pick_rate)
                    if interval and food_cells and int(tick_count) % interval == 0:
                        spawned_now = 0
                        for _ in range(FOOD_SPAWN_COUNT):
                            x, y = random.choice(food_cells)
                            if resources[y, x, 1] < FOOD_STACK:
                                add_resource(x, y)
                                spawned_now += 1

                # Track per-human travel (continuous, accumulated per tick)
                for h in humans:
                    if h.alive:
                        h.daily_travel = getattr(h, "daily_travel", 0.0)
                        h._last_x = getattr(h, "_last_x", h.x)
                        h._last_y = getattr(h, "_last_y", h.y)
                        movement = ((h.x - h._last_x) ** 2 + (h.y - h._last_y) ** 2) ** 0.5
                        h.daily_travel += movement
                        h._last_x = h.x
                        h._last_y = h.y
        
        # ---- Drawing ---- (outside pause block so it updates even when paused)
        screen.blit(static_layer, (0, 0))

        # dynamic food as green squares
        nz = np.argwhere(resources[:, :, 1] > 0)
        for (y, x) in nz:
            qty  = int(resources[y, x, 1])
            frac = max(0.0, min(1.0, qty / FOOD_STACK))
            s = max(2, int(CELL_SIZE * 0.7 * frac))
            rx = x * CELL_SIZE + (CELL_SIZE - s) // 2
            ry = y * CELL_SIZE + (CELL_SIZE - s) // 2
            pygame.draw.rect(screen, (0, 255, 80), pygame.Rect(rx, ry, s, s))

        # humans
        for h in (hh for hh in humans if hh.alive):
            draw_human(screen, h, CELL_SIZE, font)

        # houses overlay (moves when houses relocate)
        draw_houses_overlay(screen, houses)

        # UI
        display_human_counts(screen, humans, font)
        draw_legend(screen, CELL_SIZE, font)
        display_house_storage(screen, houses, CELL_SIZE, font)
        slider.draw(screen, font)
        speed_slider.draw(screen, font)
        trust_slider.draw(screen, font)
        draw_action_buttons(screen, action_rects, font)
        # Day counter
        current_day = tick_count // DAY_LENGTH
        screen.blit(font.render(f"Day: {current_day}", True, (255, 255, 255)), (10, 10))
        # Food respawn status indicator (below speed slider)
        respawn_status = "ON" if food_respawn_enabled else "OFF"
        respawn_color = (0, 255, 0) if food_respawn_enabled else (255, 0, 0)
        respawn_y = MAP_HEIGHT*CELL_SIZE + 180  # Below speed slider
        screen.blit(font.render(f"Respawn [R]: {respawn_status}", True, respawn_color), ((MAP_WIDTH*CELL_SIZE)//4, respawn_y))
        # Trust mechanism status indicator (next to respawn)
        trust_status = "ON" if trust_enabled else "OFF"
        trust_color = (0, 255, 0) if trust_enabled else (255, 0, 0)
        screen.blit(font.render(f"Trust [T]: {trust_status}", True, trust_color), ((MAP_WIDTH*CELL_SIZE)//4 + 200, respawn_y))

        # If no houses exist yet, instruct the user how to place them
        if not houses:
            msg = "Click to place BLUE house, then RED house"
            screen.blit(font.render(msg, True, (255, 255, 0)), (10, 50))

        # overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0))
        if is_day:
            brightness = 0.3 + (cycle_pos / 0.7) * 0.7
        else:
            brightness = 1.0 - ((cycle_pos - 0.7) / 0.3) * 0.7
        overlay.set_alpha(int((1 - brightness) * 200))
        screen.blit(overlay, (0, 0))

        pygame.display.flip()

        # Fixed frame rate; speed affects number of sim ticks per frame, not movement per tick
        clock.tick(60)

    # Export data and generate plots when simulation ends
    print("\n" + "="*50)
    print("Simulation ended. Exporting final data...")
    print("="*50)
    export_trust_matrix(trust_system, humans)
    export_house_movement_log(house_movement_log)
    plot_house_movement_analysis(house_movement_log)
    print("\n✅ All data exported successfully!")
    
    pygame.quit()

def simulate_ui(*, num_days: int, seed: Optional[int], map_path: str,
                min_size: int, tol: int, precomputed=None):
    """
    Programmatic UI simulation - runs simulation with pygame visualization.
    
    Args:
        num_days: Number of simulation days to run
        seed: Random seed for reproducible results (optional)
        map_path: Path to the map image file
        min_size: Minimum zone size for zone detection
        tol: Color tolerance for zone identification
        precomputed: Pre-computed world data (optional)
    """
    world = build_world(seed=seed, map_path=map_path, min_size=min_size,
                        tol=tol, precomputed=precomputed, n_days=num_days)

    zone_map = world["zone_map"]
    houses   = world["houses"]
    humans   = world["humans"]
    trust    = world["trust_system"]
    day_tick = world["day_tick"]
    food_zone_ids = world["food_zone_ids"]
    per_zone = world["per_zone"]
    food_cells = world["food_cells"]
    last_storage = world["last_storage"]
    pick_history_global = world["pick_history_global"]

    pygame.quit(); pygame.init()
    screen = pygame.display.set_mode((MAP_WIDTH*CELL_SIZE, MAP_HEIGHT*CELL_SIZE))
    pygame.display.set_caption("Programmatic Simulation")
    font = pygame.font.Font(None, max(12, CELL_SIZE*4))

    static_layer = map_manage(zone_map)
    clock = pygame.time.Clock()
    prev_is_day = False
    tick_f = 0.0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # Day/night cycle
        tick_f += 1.0
        day_tick = (day_tick + 1.0) % DAY_LENGTH
        cycle_pos = day_tick / DAY_LENGTH
        is_day = (cycle_pos < 0.7)

        # Dawn competition for red house
        if not prev_is_day and is_day and houses:
            for house in houses:
                # Apply leadership competition to both blue and red houses
                fam = [h for h in humans if h.alive and h.home is house]
                if fam: 
                    run_competition(fam, trust, threshold=0.55)  # programmatic mode keeps default
        prev_is_day = is_day

        life_span_ressource()

        # Use standardized agent logic with occupancy map for performance
        picked_this_tick, shared_this_tick, per_family_consumption = run_single_tick(
            humans, houses, trust, resources, is_day, last_storage,
            use_occupancy_map=True  # Programmatic UI uses occupancy map for efficiency
        )

        # Compact away dead humans (don't render/iterate them again)
        if any(not hh.alive for hh in humans):
            humans[:] = [hh for hh in humans if hh.alive]

        # Resource spawning
        pick_history_global.append(picked_this_tick)
        if len(pick_history_global) > 30:
            pick_history_global.pop(0)
        avg_pick = (sum(pick_history_global) / len(pick_history_global)) if pick_history_global else 0.0
        interval = resource_spawn_interval_inverse(avg_pick)
        if interval and food_cells and int(tick_f) % interval == 0:
            for _ in range(FOOD_SPAWN_COUNT):
                x, y = random.choice(food_cells)
                if resources[y, x, 1] < FOOD_STACK:
                    add_resource(x, y)

        # Rendering
        screen.blit(static_layer, (0, 0))
        for h in humans:
            if h.alive:
                draw_human(screen, h, CELL_SIZE, font)
        
        # Food rendering
        nz = np.argwhere(resources[:, :, 1] > 0)
        for (y, x) in nz:
            qty = int(resources[y, x, 1])
            frac = max(0.0, min(1.0, qty / FOOD_STACK))
            s = max(2, int(CELL_SIZE * 0.7 * frac))
            rx = x * CELL_SIZE + (CELL_SIZE - s) // 2
            ry = y * CELL_SIZE + (CELL_SIZE - s) // 2
            pygame.draw.rect(screen, (0, 255, 80), pygame.Rect(rx, ry, s, s))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# ────────────────────────────────────────────────
# Menu Functions
# ────────────────────────────────────────────────

def show_menu():
    """Show the pygame menu for starting simulations."""
    pygame.init()
    surface = pygame.display.set_mode((800, 550))
    start_requested = False

    menu = pygame_menu.Menu('Human Society Simulation', 800, 550, theme=pygame_menu.themes.THEME_DARK)
    
    def on_start():
        nonlocal start_requested
        start_requested = True
        menu.disable()
    
    menu.add.button('Start Interactive Simulation', on_start)
    menu.add.button('Quit', pygame_menu.events.EXIT)
    menu.mainloop(surface)

    if start_requested:
        start_interactive_simulation()

def analyze_map_interactive():
    """Interactive map color analysis."""
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Map Color Analysis")
    font = pygame.font.Font(None, 24)
    
    map_path = MAP_IMAGE_PATH
    
    # Get color information
    color_info = get_map_color_info(map_path)
    color_mapping = analyze_map_colors(map_path)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((0, 0, 0))
        
        # Display color information
        y = 20
        screen.blit(font.render(f"Map: {os.path.basename(map_path)}", True, (255, 255, 255)), (20, y))
        y += 30
        
        screen.blit(font.render(f"Total Colors: {color_info.get('total_colors', 0)}", True, (255, 255, 255)), (20, y))
        y += 30
        
        screen.blit(font.render(f"Image Size: {color_info.get('image_size', (0, 0))}", True, (255, 255, 255)), (20, y))
        y += 30
        
        screen.blit(font.render(f"Image Mode: {color_info.get('image_mode', 'Unknown')}", True, (255, 255, 255)), (20, y))
        y += 40
        
        # Display color categories
        if color_mapping:
            screen.blit(font.render("Color Categories:", True, (255, 255, 0)), (20, y))
            y += 30
            
            categories = {}
            for color_name, category in color_mapping.items():
                if category not in categories:
                    categories[category] = []
                categories[category].append(color_name)
            
            for category, color_names in categories.items():
                text = f"{category}: {', '.join(color_names[:5])}"
                if len(color_names) > 5:
                    text += f" (+{len(color_names)-5} more)"
                screen.blit(font.render(text, True, (200, 200, 200)), (40, y))
                y += 25
        
        screen.blit(font.render("Press ESC or close window to return to menu", True, (100, 100, 100)), (20, 550))
        
        pygame.display.flip()
    
    pygame.quit()

# ────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────

if __name__ == "__main__":
    show_menu()
