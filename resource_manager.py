
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
from human import draw_human

# resources[y, x, 0] = lifetime
# resources[y, x, 1] = food_left
resources = np.zeros((MAP_HEIGHT, MAP_WIDTH, 2), dtype=np.int32)

# Optional runtime-override palette provided by UI/tools
# Maps RGB tuples to zone IDs, same semantics as NEW_PALETTE
ACTIVE_PALETTE: Optional[dict] = None

def set_active_palette(palette: dict) -> None:
    """Override the color→zone mapping used by zone detection and drawing."""
    global ACTIVE_PALETTE
    ACTIVE_PALETTE = dict(palette)

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
def add_resource_infinite(x: int, y: int, food: int = 1) -> None:
    """Spawn non-decaying resource at (x,y); lifetime -1 indicates no decay."""
    resources[y, x, 0] = -1
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
    # Decrement lifetime for decaying resources only (lifetime >= 0)
    mask_decay = resources[:, :, 0] >= 0
    resources[:, :, 0][mask_decay] -= 1
    dead = resources[:, :, 0] == 0
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

    # Use runtime palette override when provided
    palette = ACTIVE_PALETTE if ACTIVE_PALETTE is not None else NEW_PALETTE

    # tolerant mapping for all colors except pure black (id 0)
    for rgb, zone_id in palette.items():
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

    # Connected components for food_1 (id 4) and food_2 (id 5)
    # food_1 relabeled to 41+; food_2 relabeled to 81+
    food_mask_1 = (zone_map == 4).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(food_mask_1)
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
        food_ids[f"food1_zone_{j}"] = new_id

    for i in small:
        cx, cy = centroids[i]
        nearest = min(large, key=lambda j: (centroids[j][0]-cx)**2 + (centroids[j][1]-cy)**2)
        mapping[i] = mapping[nearest]

    for i in range(1, num_labels):
        zone_map[labels == i] = mapping[i]

    # Repeat for food_2 (id 5)
    food_mask_2 = (zone_map == 5).astype(np.uint8)
    num_labels2, labels2 = cv2.connectedComponents(food_mask_2)
    sizes2, centroids2 = {}, {}
    for i in range(1, num_labels2):
        ys, xs = np.where(labels2 == i)
        if xs.size == 0:
            continue
        sizes2[i] = xs.size
        centroids2[i] = (np.mean(xs), np.mean(ys))
    large2 = {i for i, s in sizes2.items() if s >= min_size}
    small2 = {i for i, s in sizes2.items() if s < min_size}
    base_id2 = 80
    mapping2 = {}
    for j, i in enumerate(sorted(large2), start=1):
        new_id = base_id2 + j
        mapping2[i] = new_id
        food_ids[f"food2_zone_{j}"] = new_id
    for i in small2:
        cx, cy = centroids2[i]
        nearest = min(large2, key=lambda j: (centroids2[j][0]-cx)**2 + (centroids2[j][1]-cy)**2)
        mapping2[i] = mapping2[nearest]
    for i in range(1, num_labels2):
        zone_map[labels2 == i] = mapping2[i]

    print("Zones nourriture identifiées :", food_ids)
    return zone_map, food_ids

# ─────────────── Resource respawn dynamics ───────────────
cooldown_state = {}

def resource_spawn_interval_inverse(
    f_avg: float,
    zone_id: int = 0,
    zone_map=None,
    I_max: int = 100,  # maximum interval (slowest respawn)
    k: float = 4.0,    # responsiveness to consumption
    I_min: int = 20,   # minimum interval (fastest respawn when consumption is low)
    stock_today=None,
    spawn_baseline: int = FOOD_SPAWN_COUNT,
    reset: bool = False,
    cooldown_days: int = 8  # longer cooldown when overexploited
) -> Optional[int]:
    """
    Compute adaptive respawn interval based on consumption patterns.
    
    Implements an adaptive spawning system where HIGHER consumption leads to
    SLOWER respawn (resource depletion pressure). Lower consumption allows
    faster recovery. Includes cooldown periods when zones become severely depleted.
    
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
            # Trigger cooldown aggressively when stock drops below 40% of baseline
            threshold = max(1, int(0.40 * spawn_baseline))
            if stock_today < threshold:
                cooldown_state[zone_id] = cooldown_days

    if cooldown_state.get(zone_id, 0) > 0:
        return None

    # INVERTED LOGIC: Higher consumption (f_avg) → LONGER interval (slower respawn)
    # This simulates resource depletion pressure
    interval = I_min + (I_max - I_min) * (f_avg / (f_avg + 1.0))
    return max(I_min, min(I_max, int(interval)))

# ─────────────── Drawing / Pygame helpers ───────────────
def map_draw(image_path: str):
    """Return zone_map array (OpenCV-based)."""
    zone_map, _ = identify_zones(image_path)
    return zone_map

def map_manage(zone_map):
    surf = pygame.Surface((MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
    palette = ACTIVE_PALETTE if ACTIVE_PALETTE is not None else NEW_PALETTE
    rev_color = {v: k for k, v in palette.items()}

    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            zone_id = zone_map[y, x]
            if 41 <= zone_id < 80:
                # food_1 zones in palette dark green
                color = rev_color.get(4, (54, 109, 70))
            elif zone_id >= 80:
                # food_2 zones in a distinct cyan/teal if not defined
                color = rev_color.get(5, (0, 200, 200))
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
        # Use friendly names based on color
        if house.color == (0, 0, 128):
            name = "Blue"
        elif house.color == (255, 0, 0):
            name = "Red"
        else:
            name = f"{house.color}"
        txt = f"House {name} storage: {house.storage}"
        surf = font.render(txt, True, (255,255,0))
        screen.blit(surf, (10, 30 + i*20))
