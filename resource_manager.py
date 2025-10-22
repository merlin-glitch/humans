
"""
new_ressource.py - Resource management and zone detection

Part of the Human Society Simulation project.

Handles food spawning/decay, zone identification from map images,
and adaptive respawn dynamics based on consumption patterns.
"""

import os
import numpy as np
import cv2
import pygame
from typing import Tuple, Dict, Optional
from config import *
from social_mechanics import draw_human

# resources[y, x, 0] = lifetime
# resources[y, x, 1] = food_left
resources = np.zeros((MAP_HEIGHT, MAP_WIDTH, 2), dtype=np.int32)

FOOD_COLOR_BGR = (54, 109, 70)  # from config

# ─────────────── Resource utils ───────────────
def extract_resource_coords_from_zones(zone_map, food_zone_id=4):
    """Return coordinates of cells belonging to given food zone."""
    # Food zone ID 4 corresponds to food zones in the NEW_PALETTE (config.py)
    food_mask = (zone_map == food_zone_id)
    coords = np.column_stack(np.where(food_mask))
    return coords, food_mask.astype(np.uint8) * 255

def add_resource(x: int, y: int, life: int = FOOD_LIFETIME, food: int = FOOD_STACK) -> None:
    """Spawn resource at (x,y) with given life and food amount."""
    resources[y, x, 0] = life
    resources[y, x, 1] = min(FOOD_STACK, resources[y, x, 1] + food)
    #print(f"[RESOURCE ADDED] at ({x},{y}) life={life}, food={resources[y, x, 1]}")

def remove_resource(x: int, y: int) -> None:
    """Remove resource at (x,y) coordinates."""
    if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH:
        resources[y, x] = [0, 0]


def life_span_ressource() -> None:
    """
    Update resource lifetimes and remove expired resources.

    Decreases the lifetime of all resources by 1 tick and removes any
    resources that have expired (lifetime <= 0). This simulates natural
    food decay and spoilage over time.

    Note:
        This function modifies the global resources array in place.
        Should be called once per simulation tick.

    Example:
        >>> initial_food = total_food()
        >>> life_span_ressource()
        >>> remaining_food = total_food()  # Should be <= initial_food
    """
    resources[:, :, 0] -= 1
    dead = resources[:, :, 0] <= 0
    resources[dead] = 0

def total_food() -> int:
    """
    Calculate the total amount of food remaining on the map.

    Returns the sum of all food units currently available across all
    cells in the simulation world. This is useful for monitoring
    resource availability and depletion rates.

    Returns:
        Total number of food units remaining across all zones

    Example:
        >>> current_food = total_food()
        >>> print(f"Total food remaining: {current_food} units")
    """
    return resources[:, :, 1].sum()

# ─────────────── Zone detection ───────────────
def identify_zones(map_image_path: str, min_size: int = 30, tol: int = 30) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Identify and label zones from a map image.
    
    Processes a PNG image to identify different terrain zones based on color
    mapping defined in NEW_PALETTE. Uses connected component analysis to
    merge small zones and relabel food zones with unique IDs.
    
    Args:
        map_image_path: Path to the PNG map image
        min_size: Minimum zone size to keep as separate (default 30)
        tol: Color tolerance for zone identification (default 30)
        
    Returns:
        Tuple of (zone_map, food_zone_ids) where:
        - zone_map: 2D numpy array with zone IDs for each cell
        - food_zone_ids: Dictionary mapping zone names to IDs
        
    Raises:
        FileNotFoundError: If map image file doesn't exist
        ValueError: If parameters are invalid
        OSError: If image file cannot be read
        
    Example:
        >>> zone_map, food_ids = identify_zones("images/map.png")
        >>> print(f"Found {len(food_ids)} food zones")
        >>> print(f"Zone map shape: {zone_map.shape}")
    """
    # Input validation
    if not isinstance(map_image_path, str):
        raise TypeError(f"Map image path must be a string, got {type(map_image_path)}")
    
    if not map_image_path.strip():
        raise ValueError("Map image path cannot be empty")
    
    if not os.path.exists(map_image_path):
        raise FileNotFoundError(f"Map image file not found: {map_image_path}")
    
    if not isinstance(min_size, int) or min_size <= 0:
        raise ValueError(f"min_size must be a positive integer, got {min_size}")
    
    if not isinstance(tol, int) or tol < 0 or tol > 255:
        raise ValueError(f"tol must be an integer between 0-255, got {tol}")
    
    try:
        img = cv2.imread(map_image_path)
        if img is None:
            raise OSError(f"Could not read image file: {map_image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (MAP_WIDTH, MAP_HEIGHT), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        raise OSError(f"Error processing image {map_image_path}: {e}") from e

    zone_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int32)

    # tolerant mapping for all colors except pure black (id 0)
    for rgb, zone_id in NEW_PALETTE.items():
        rgb = np.array(rgb, dtype=np.int16)
        if zone_id == 0:
            # keep border strict
            mask = np.all(img == rgb, axis=-1)
        else:
            low  = np.clip(rgb - tol, 0, 255).astype(np.uint8)
            high = np.clip(rgb + tol, 0, 255).astype(np.uint8)
            mask = cv2.inRange(img, low, high) > 0
        zone_map[mask] = zone_id

    # Fill any stray 0s inside the map with grass (id 1) – preserve the 1-cell outer frame
    inner = zone_map[1:-1, 1:-1]
    inner[inner == 0] = 1
    zone_map[1:-1, 1:-1] = inner

    # Connected components just for food zones (id 4), then relabel to >= 41 as you had
    food_mask = (zone_map == 4).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(food_mask)
    sizes, centroids = {}, {}
    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        sizes[i] = xs.size
        centroids[i] = (np.mean(xs), np.mean(ys))

    large = {i for i, s in sizes.items() if s >= min_size}
    small = {i for i, s in sizes.items() if s < min_size}

    base_id = 40
    food_ids, mapping = {}, {}
    for j, i in enumerate(sorted(large), start=1):
        new_id = base_id + j
        mapping[i] = new_id
        food_ids[f"food_zone_{j}"] = new_id

    for i in small:
        cx, cy = centroids[i]
        nearest = min(large, key=lambda j: (centroids[j][0]-cx)**2 + (centroids[j][1]-cy)**2)
        mapping[i] = mapping[nearest]

    for i in range(1, num_labels):
        zone_map[labels == i] = mapping[i]

    print("Zones nourriture identifiées :", food_ids)
    return zone_map, food_ids

# ─────────────── Resource respawn dynamics ───────────────
cooldown_state = {}

def resource_spawn_interval_inverse(
    f_avg: float,
    zone_id: int = 0,
    zone_map=None,
    I_max: int = 50,   # Further reduced from 100 - much faster spawning
    k: float = 2.0,    # Increased from 1.0 - much more responsive to consumption
    I_min: int = 2,    # Reduced from 5 - very fast minimum spawning
    stock_today=None,
    spawn_baseline: int = FOOD_SPAWN_COUNT,
    reset: bool = False,
    cooldown_days: int = 2  # Reduced from 5 - very short cooldown periods
) -> Optional[int]:
    """
    Compute adaptive respawn interval based on consumption patterns.
    
    Implements an adaptive spawning system that adjusts food respawn rates
    based on recent consumption. Higher consumption leads to faster respawn,
    while low consumption leads to slower respawn. Includes cooldown periods
    when zones become severely depleted.
    
    Args:
        f_avg: Average consumption rate for this zone
        zone_id: Unique identifier for the zone
        zone_map: Map array for stock calculation (optional)
        I_max: Maximum respawn interval (default 200)
        k: Responsiveness factor (default 0.6)
        I_min: Minimum respawn interval (default 10)
        stock_today: Current food stock in zone (optional)
        spawn_baseline: Baseline spawn amount for threshold calc
        reset: Reset cooldown state (default False)
        cooldown_days: Cooldown period when depleted (default 10)
        
    Returns:
        Respawn interval in ticks, or None if zone is in cooldown
        
    Example:
        >>> interval = resource_spawn_interval_inverse(5.2, zone_id=1)
        >>> if interval and tick % interval == 0:
        ...     spawn_food_in_zone(1)
    """
    global cooldown_state

    if reset:
        cooldown_state = {}
        return I_min

    if stock_today is None and zone_map is not None:
        stock_today = resources[:, :, 1][zone_map == zone_id].sum()

    remaining = cooldown_state.get(zone_id, 0)
    if stock_today is not None:
        if remaining > 0:
            cooldown_state[zone_id] = max(0, remaining - 1)
        else:
            threshold = max(1, int(0.10 * spawn_baseline))
            if stock_today < threshold:
                cooldown_state[zone_id] = cooldown_days

    if cooldown_state.get(zone_id, 0) > 0:
        return None

    interval = I_max / (1.0 + k * max(0.0, f_avg))
    return max(I_min, int(interval))

# ─────────────── Drawing / Pygame helpers ───────────────
def map_draw(image_path: str):
    """Return zone_map array (OpenCV-based)."""
    zone_map, _ = identify_zones(image_path)
    return zone_map

def map_manage(zone_map):
    surf = pygame.Surface((MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
    rev_color = {v: k for k, v in NEW_PALETTE.items()}

    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            zone_id = zone_map[y, x]
            if zone_id >= 40:
                # show food zones in the palette dark green (id 4)
                color = rev_color.get(4, (54, 109, 70))
            else:
                # fall back to grass (id 1) instead of dark gray/black
                color = rev_color.get(zone_id, rev_color.get(1, (94, 185, 30)))
            pygame.draw.rect(
                surf, color,
                pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )
    return surf




def pixel_update(screen, static_layer, humans, font):
    """Draw static terrain layer + live food + humans."""
    screen.blit(static_layer, (0, 0))

    # draw food stacks dynamically
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if resources[y, x, 1] > 0:
                frac = resources[y, x, 1] / FOOD_STACK
                color = (int(0 + 200*frac), int(255*frac), int(0 + 50*frac))
                pygame.draw.circle(
                    screen, color,
                    (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2),
                    CELL_SIZE//2
                )

    # draw humans
    for h in humans:
        if h.alive:
            draw_human(screen, h, CELL_SIZE, font)

def display_house_storage(screen, houses, cell_size, font):
    """Draw storage info for each house."""
    for i, house in enumerate(houses):
        txt = f"House {i} storage: {house.storage}"
        surf = font.render(txt, True, (255,255,0))
        screen.blit(surf, (10, 30 + i*20))
