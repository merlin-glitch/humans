"""
simulate_utils.py - Shared utilities for simulation setup and world building

Part of the Human Society Simulation project.

Contains world building functions, zone detection caching, house construction,
and shared helper functions used by both UI and headless simulation modes.
"""

import os
import random
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np

from config import *  # MAP_WIDTH, MAP_HEIGHT, CELL_SIZE, FOOD_*, DAY_LENGTH, NEW_PALETTE, etc.
from human import Human, House
from trust_system import TrustSystem
from resource_manager import (
    resources, life_span_ressource, map_manage,
    resource_spawn_interval_inverse, identify_zones, extract_resource_coords_from_zones,
    add_resource_infinite,
)
from simulate_utils import seed_food

import math
from config import HOUSE_RELOC_ALPHA1, HOUSE_RELOC_ALPHA2, HOUSE_RELOC_ALPHA3, HOUSE_RELOC_BETA, HOUSE_RELOC_E_MOVE, MAX_HOUSE_STORAGE, HOUSE_MAX_TRAVEL_PER_DAY, HOUSE_LOCAL_RADIUS, HOUSE_INERTIA_STEP, HOUSE_INERTIA_MAX, MAX_FOOD_PER_CELL

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def should_move_house(house, humans, resources):
    # 1. Storage S: normalized
    S = min(1.0, house.storage / MAX_HOUSE_STORAGE if MAX_HOUSE_STORAGE else 0.0)

    # 2. D = avg daily travel for this house's humans, normalized
    dists = []
    members = [h for h in humans if h.home is house]
    for h in members:
        dist_today = getattr(h, "daily_travel", 0.0)
        dists.append(dist_today)
    if dists:
        D = min(1.0, sum(dists) / len(dists) / HOUSE_MAX_TRAVEL_PER_DAY)
    else:
        D = 0.0

    # 3. F = local food ratio
    y0 = max(0, house.y - HOUSE_LOCAL_RADIUS)
    y1 = min(resources.shape[0]-1, house.y + HOUSE_LOCAL_RADIUS)
    x0 = max(0, house.x - HOUSE_LOCAL_RADIUS)
    x1 = min(resources.shape[1]-1, house.x + HOUSE_LOCAL_RADIUS)
    local_food = resources[y0:y1+1, x0:x1+1, 1].sum()
    max_cells = (y1-y0+1)*(x1-x0+1) * MAX_FOOD_PER_CELL
    F = local_food / max_cells if max_cells > 0 else 0.0
    F = min(1.0, F)

    # Inertia term (house.inertia, increases if no move, reset to 0 when move)
    if not hasattr(house, "inertia"):
        house.inertia = 0.0

    x = HOUSE_RELOC_ALPHA1*(1-S) + HOUSE_RELOC_ALPHA2*D + HOUSE_RELOC_ALPHA3*(1-F) - HOUSE_RELOC_BETA*HOUSE_RELOC_E_MOVE
    P_move = sigmoid(x) * (1.0 - min(house.inertia, HOUSE_INERTIA_MAX))
    return P_move  # return probability, caller can sample determination

def find_best_house_location(house, humans, resources, max_move_distance=2):
    """
    Scan points in radius, return (x, y) maximizing F (local food).
    Only allows relocation to grass/terrain (zone id 1), not food zones or walls.
    Movement is gradual: maximum max_move_distance cells per relocation.
    """
    from resource_manager import identify_zones
    # Get zone_map to check terrain type (avoid placing house on food/walls)
    # For performance, assume zone_map is accessible via house.codes or a global
    # We'll use the codes from a human in that house as a proxy for zone_map
    members = [h for h in humans if h.home is house and h.alive]
    if not members:
        return (house.x, house.y)
    zone_map = members[0].codes  # All humans share same codes (zone_map)
    
    radius = HOUSE_LOCAL_RADIUS
    best_score = -1
    best_pos = (house.x, house.y)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = house.x + dx, house.y + dy
            if not (0 <= nx < resources.shape[1] and 0 <= ny < resources.shape[0]):
                continue
            # Check terrain: only allow grass (zone id 1) or non-restricted terrain
            # Avoid food zones (4, 5, and 41+, 81+), walls (0), and existing houses (2, 3)
            zone_id = zone_map[ny, nx]
            if zone_id == 0:  # walls/border
                continue
            if zone_id in (2, 3):  # existing house zones
                continue
            if zone_id == 4 or zone_id == 5:  # base food IDs
                continue
            if zone_id >= 40:  # food zones (41+ for food1, 81+ for food2)
                continue
            
            # Calculate local food density (nearby food, not on the house spot)
            y0 = max(0, ny - 2)
            y1 = min(resources.shape[0] - 1, ny + 2)
            x0 = max(0, nx - 2)
            x1 = min(resources.shape[1] - 1, nx + 2)
            local_food = resources[y0:y1+1, x0:x1+1, 1].sum()
            if local_food > best_score:
                best_score = local_food
                best_pos = (nx, ny)
    
    # Gradual movement: move towards best_pos but max max_move_distance cells per step
    target_x, target_y = best_pos
    dx = target_x - house.x
    dy = target_y - house.y
    distance = (dx**2 + dy**2) ** 0.5
    
    if distance <= max_move_distance:
        # Close enough, move to exact position
        return best_pos
    else:
        # Move max_move_distance cells towards target
        ratio = max_move_distance / distance
        new_x = int(house.x + dx * ratio)
        new_y = int(house.y + dy * ratio)
        # Ensure new position is valid terrain
        if 0 <= new_x < resources.shape[1] and 0 <= new_y < resources.shape[0]:
            zone_id = zone_map[new_y, new_x]
            if zone_id != 0 and zone_id not in (2, 3, 4, 5) and zone_id < 40:
                return (new_x, new_y)
        # If invalid, stay put
        return (house.x, house.y)

# === Integrate call at end of day in the simulation ===
# Put this after the main tick/day loop, before trust/mating (see how births/deaths are logged)
# (Integration spot is left as a comment for clarity)
# For each house:
#   P_move = should_move_house(house, humans, resources)
#   if random.random() < P_move:
#      new_x, new_y = find_best_house_location(house, humans, resources)
#      relocate house + all members
#      house.inertia = 0.0
#   else:
#      house.inertia = min(1.0, house.inertia + HOUSE_INERTIA_STEP)

# When relocating, update house.x/.y, all h.home_x/home_y/h.x/h.y for matching humans.

# Feature flags (fallbacks if not in config.py)
PER_ZONE_RESPAWN = bool(globals().get('PER_ZONE_RESPAWN', True))
COLLECT_METRICS  = bool(globals().get('COLLECT_METRICS', True))
ENABLE_MATING    = bool(globals().get('ENABLE_MATING', True))

# ───────────────── helpers shared by both modes ─────────────────


_IDENTIFY_CACHE = {}

def cached_identify(map_path: str, min_size: int, tol: int,
                    *, use_cache: bool = True, reset: bool = False):
    """
    Cached wrapper for zone identification to improve performance.

    Provides a simple caching mechanism for the expensive zone identification
    process. Repeated calls with the same parameters return cached results
    instead of re-processing the map image.

    Args:
        map_path: Path to the map image file
        min_size: Minimum zone size for identification
        tol: Color tolerance for zone detection
        use_cache: If False, forces fresh computation (default: True)
        reset: If True, clears the cache before processing (default: False)

    Returns:
        Tuple of (zone_map, food_zone_ids) from identify_zones()

    Note:
        Cache has a maximum size of 8 entries to prevent memory issues.
        Uses absolute file paths as cache keys to handle relative paths correctly.

    Example:
        >>> # First call - computes and caches result
        >>> zones, food_ids = cached_identify("map.png", 30, 20)
        >>> # Second call - returns cached result
        >>> zones, food_ids = cached_identify("map.png", 30, 20)
    """
    if reset:
        _IDENTIFY_CACHE.clear()

    if not use_cache:
        return identify_zones(map_path, min_size=min_size, tol=tol)

    key = (os.path.abspath(map_path), int(min_size), int(tol))
    if key in _IDENTIFY_CACHE:
        return _IDENTIFY_CACHE[key]

    result = identify_zones(map_path, min_size=min_size, tol=tol)
    # optional tiny cap like the old maxsize=8
    if len(_IDENTIFY_CACHE) >= 8:
        # drop an arbitrary (oldest-unknown) entry; keep it simple
        _IDENTIFY_CACHE.pop(next(iter(_IDENTIFY_CACHE)))
    _IDENTIFY_CACHE[key] = result
    return result


def build_occupancy(humans: List[Human]) -> Dict[Tuple[int, int], set]:
    """Build occupancy map for efficient neighbor finding."""
    occ = defaultdict(set)
    for h in humans:
        if h.alive:
            occ[(h.x, h.y)].add(h)
    return occ

_NEIGH_OFFS = (-1, 0, 1)

def iter_peers(h: Human, occ: Dict[Tuple[int, int], set]):
    x, y = h.x, h.y
    same = occ.get((x, y))
    if same:
        for p in same:
            if p is not h:
                yield p
    get = occ.get
    for dx in _NEIGH_OFFS:
        nx = x + dx
        for dy in _NEIGH_OFFS:
            if dx == 0 and dy == 0:
                continue
            bucket = get((nx, y + dy))
            if bucket:
                for p in bucket:
                    yield p

def _rev_palette() -> Dict[int, Tuple[int, int, int]]:
    """Create reverse mapping from zone codes to RGB colors."""
    return {code: rgb for rgb, code in NEW_PALETTE.items()}

def build_houses(zone_map: np.ndarray) -> List[House]:
    rev = _rev_palette()
    houses: List[House] = []
    # Zone IDs: 2 = blue house, 3 = red house (from NEW_PALETTE in config.py)
    for zid in (2, 3):  # blue, red
        ys, xs = np.where(zone_map == zid)
        if xs.size == 0:
            continue
        cx = (xs.min() + xs.max()) // 2
        cy = (ys.min() + ys.max()) // 2
        houses.append(House(int(cx), int(cy), rev.get(zid)))
    return houses

# seed_food function is imported from simulate_utils.py


def draw_offscreen(surface, static_layer, font, resources_arr, humans):
    # background
    surface.blit(static_layer, (0, 0))
    # food
    nz = np.argwhere(resources_arr[:, :, 1] > 0)
    import pygame
    for (y, x) in nz:
        qty  = int(resources_arr[y, x, 1])
        frac = max(0.0, min(1.0, qty / FOOD_STACK))
        s = max(2, int(CELL_SIZE * 0.7 * frac))
        rx = x * CELL_SIZE + (CELL_SIZE - s)//2
        ry = y * CELL_SIZE + (CELL_SIZE - s)//2
        pygame.draw.rect(surface, (0, 255, 80), pygame.Rect(rx, ry, s, s))
    # humans
    from social_mechanics import draw_human
    for h in (hh for hh in humans if hh.alive):
        draw_human(surface, h, CELL_SIZE, font)

# ───────────────── world builder ─────────────────

def _load_color_mapping_if_exists(map_path: str) -> bool:
    """
    Load color mapping from JSON file if it exists for the given map.
    Uses the SAME logic as UI to ensure consistency.
    Sets ACTIVE_PALETTE in resource_manager to use the saved color categories.
    
    Args:
        map_path: Path to the map image file
        
    Returns:
        True if JSON was loaded successfully, False otherwise
    """
    import json
    from resource_manager import set_active_palette
    
    # Construct JSON path from map path
    json_path = map_path + "_color_analysis.json"
    
    if not os.path.exists(json_path):
        return False
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Accept both formats: {"mapping": {...}} or {"color_mapping": {...}}
        color_mapping = data.get('mapping', data.get('color_mapping', {}))
        if not color_mapping:
            return False
        
        # Use the EXACT same function that UI uses to build palette
        # This is in ui_simulation.py but we'll inline the logic here for headless use
        import cv2
        from map_color_tools import _rgb_to_css4_name
        
        # Load image and extract colors
        img = cv2.imread(map_path)
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (MAP_WIDTH, MAP_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # Get unique colors
        pixels = img.reshape(-1, 3)
        unique_colors_np = np.unique(pixels, axis=0)
        # Convert numpy types to Python ints for proper color matching
        unique_colors = [tuple(int(x) for x in c) for c in unique_colors_np]
        
        # Build RGB -> category mapping
        rgb_key_to_category = {}
        used_names = set()
        for color in unique_colors:
            name_guess, _ = _rgb_to_css4_name(color)
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            key_name = name_guess
            if key_name in used_names:
                key_name = f"{name_guess} ({hex_color})"
            used_names.add(key_name)
            category = color_mapping.get(key_name)
            if category:
                rgb_key_to_category[color] = category
        
        # Build palette: RGB -> zone_id
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
            # Unknown categories default to grass
            return 1
        
        palette = {rgb: cat_to_zone(cat) for rgb, cat in rgb_key_to_category.items()}
        
        # Ensure at least one grass color exists
        if not any(v == 1 for v in palette.values()):
            palette[(94, 185, 30)] = 1
        
        if palette:
            set_active_palette(palette)
            print(f"✅ Loaded color mapping from {os.path.basename(json_path)} ({len(palette)} colors)")
            print(f"   Houses: Blue={any(v==2 for v in palette.values())}, Red={any(v==3 for v in palette.values())}")
            return True
        
    except Exception as e:
        print(f"⚠️  Could not load color mapping from {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False

def build_world(*, seed: Optional[int], map_path: str, min_size: int, tol: int,
                precomputed: Optional[tuple], n_days: int):
    """
    Build a complete simulation world with all necessary components.

    Creates and initializes all simulation components including the zone map,
    houses, human agents, trust system, and resource management structures.
    This function serves as the central world builder for both UI and headless
    simulation modes.

    Args:
        seed: Random seed for reproducible results (None for random)
        map_path: Path to the PNG map image file
        min_size: Minimum zone size for zone identification
        tol: Color tolerance for zone detection
        precomputed: Pre-computed (zone_map, food_ids) tuple to skip identification
        n_days: Number of simulation days (affects some initialization)

    Returns:
        Dictionary containing all world components:
        - 'zone_map': 2D numpy array with zone IDs
        - 'houses': List of House objects
        - 'humans': List of Human objects
        - 'trust_system': TrustSystem instance
        - 'next_id': Next available human ID
        - 'food_zone_ids': List of food zone identifiers
        - 'per_zone': Per-zone tracking data (if enabled)
        - 'food_cells': List of (x,y) food spawn coordinates
        - 'last_storage': Dictionary tracking house storage levels
        - 'pick_history_global': Global resource consumption history
        - 'day_tick': Current day tick counter

    Example:
        >>> world = build_world(seed=42, map_path="map.png", min_size=30, 
        ...                    tol=20, precomputed=None, n_days=100)
        >>> humans = world["humans"]
        >>> print(f"Created {len(humans)} human agents")
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    # fresh grid
    resources[:, :, :] = 0

    # Try to load JSON color mapping for consistency with UI
    _load_color_mapping_if_exists(map_path)

    if precomputed is not None:
        zone_map, food_ids = precomputed
    else:
        zone_map, food_ids = cached_identify(map_path, min_size, tol)

    H, W = zone_map.shape
    houses = build_houses(zone_map)

    # Split food types by naming convention from identify_zones
    food1_ids = [v for k, v in food_ids.items() if str(k).startswith('food1_')]
    food2_ids = [v for k, v in food_ids.items() if str(k).startswith('food2_')]

    food_cells_type1 = [(x, y) for y in range(H) for x in range(W) if zone_map[y, x] in food1_ids]
    food_cells_type2 = [(x, y) for y in range(H) for x in range(W) if zone_map[y, x] in food2_ids]
    # Backwards compat: combined cells if needed elsewhere
    food_zone_ids = food1_ids + food2_ids
    food_cells = food_cells_type1 + food_cells_type2

    # Spawn initial food:
    # - Type1 (regenerative): seed modestly (uses decaying resources + respawn)
    # - Type2 (finite, non-decaying): set lifetime = -1 and do not respawn
    from random import shuffle
    if food_cells_type1:
        seed_food(food_cells_type1, INITIAL_FOOD_COUNT, resources)
    # For type2, place INITIAL_FOOD_COUNT//2 units distributed one per cell first
    if food_cells_type2:
        temp = list(food_cells_type2)
        shuffle(temp)
        k = min(len(temp), max(1, INITIAL_FOOD_COUNT // 2))
        for (x, y) in temp[:k]:
            add_resource_infinite(x, y, food=1)

    humans: List[Human] = []
    next_id = 0
    if houses:
        base, extra = divmod(Nbre_HUMANS, len(houses))
        extra_idx = random.randrange(len(houses)) if extra else -1
        for i, house in enumerate(houses):
            for _ in range(base + (1 if i == extra_idx else 0)):
                humans.append(Human(
                    human_id=next_id,
                    sex=random.choice(['homme', 'femme']),
                    x=house.x, y=house.y,
                    home=house, codes=zone_map
                ))
                next_id += 1

    trust_system = TrustSystem()
    last_storage = {h: h.storage for h in houses}
    pick_history_global: deque[int] = deque(maxlen=30)
    day_tick = DAY_LENGTH * 0.3

    per_zone = None
    if PER_ZONE_RESPAWN and food1_ids:
        zones = []
        # Only type1 zones are managed by respawn dynamics
        for zid in food1_ids:
            coords, _ = extract_resource_coords_from_zones(zone_map, food_zone_id=zid)
            zones.append([(int(x), int(y)) for (y, x) in coords])

        N_ZONES = len(zones)
        zid_to_idx = {zid: i for i, zid in enumerate(food_zone_ids)}
        pick_histories = [deque(maxlen=30) for _ in range(N_ZONES)]
        areas = np.array([len(c) for c in zones], dtype=float)
        areas[areas == 0] = 1.0
        scale = areas / areas.mean()
        I_MAX = (120 * scale).astype(int).clip(80, None).tolist()
        K_GAIN = [0.9] * N_ZONES
        I_MIN = [2] * N_ZONES
        SPAWN_COUNT = (60 * scale).astype(int).clip(6, None).tolist()

        for zid in food_zone_ids:
            resource_spawn_interval_inverse(0.0, zid, zone_map, reset=True)

        per_zone = dict(
            zones=zones, N_ZONES=N_ZONES, zid_to_idx=zid_to_idx,
            pick_histories=pick_histories,
            I_MAX=I_MAX, K_GAIN=K_GAIN, I_MIN=I_MIN, SPAWN_COUNT=SPAWN_COUNT,
            spawned_today=[0]*N_ZONES, consumed_today=[0]*N_ZONES,
            blue_consumed_today=[0]*N_ZONES, red_consumed_today=[0]*N_ZONES
        )

    world = dict(
        zone_map=zone_map, food_zone_ids=food_zone_ids, food_cells=food_cells,
        houses=houses, humans=humans, trust_system=trust_system, next_id=next_id,
        last_storage=last_storage, pick_history_global=pick_history_global,
        day_tick=day_tick, n_days=n_days, per_zone=per_zone
    )
    return world


def run_single_tick(humans: List[Human], houses: List[House], trust_system: Optional[TrustSystem], 
                   resources, is_day: bool, last_storage: Dict[House, int], 
                   use_occupancy_map: bool = False) -> Tuple[int, int, Dict[str, int]]:
    """
    Run a single simulation tick with standardized agent behavior.
    
    Args:
        humans: List of human agents
        houses: List of house objects
        trust_system: Trust system for tracking relationships (or None to disable trust)
        resources: Resource grid
        is_day: Whether it's currently day time
        last_storage: Previous storage levels for trust calculations
        use_occupancy_map: Whether to use occupancy map for neighbor finding
        
    Returns:
        Tuple of (picked_count, shared_count, per_family_consumption) where
        per_family_consumption is a dict mapping family colors to consumption counts
    """
    from social_mechanics import boost_house_trust
    
    picked_count = 0
    shared_count = 0
    per_family_consumption = {"blue": 0, "red": 0}  # Track consumption by family
    
    # Build occupancy map if requested (for performance optimization)
    occ = None
    if use_occupancy_map:
        occ = build_occupancy(humans)
    
    for h in humans:
        if not h.alive:
            continue
            
        old_pos = (h.x, h.y)
        
        # Determine neighbors to pass to step()
        if use_occupancy_map and occ is not None:
            peers = iter_peers(h, occ)
        else:
            peers = humans  # Pass all humans (simpler but potentially slower)
            
        # Standardized agent step with consistent parameters
        picked, shared = h.step(
            resources, houses, peers, trust_system,
            is_day=is_day, 
            action_cost=ACTION_COST, 
            food_gain=FOOD_GAIN, 
            decay_rate=ENERGY_DECAY_RATE
        )
        
        # Track per-family consumption
        if picked is not None:
            # Human consumed food, determine family color
            if h.home.color == (0, 0, 128):  # Blue house
                per_family_consumption["blue"] += 1
            elif h.home.color == (255, 0, 0):  # Red house
                per_family_consumption["red"] += 1
        
        # Update occupancy map if using it
        if use_occupancy_map and occ is not None:
            new_pos = (h.x, h.y)
            if not h.alive:
                # Remove from old position
                bucket = occ.get(old_pos)
                if bucket:
                    bucket.discard(h)
                    if not bucket: 
                        occ.pop(old_pos, None)
            elif new_pos != old_pos:
                # Move between positions
                bucket = occ.get(old_pos)
                if bucket:
                    bucket.discard(h)
                    if not bucket: 
                        occ.pop(old_pos, None)
                occ[new_pos].add(h)
        
        # Count picks and shares
        if picked is not None:
            picked_count += 1
        if shared:
            shared_count += 1
            
        # Update trust when humans share resources (only if trust_system is provided)
        if trust_system is not None and h.home.storage > last_storage[h.home]:
            boost_house_trust(trust_system, h, humans, increment=TRUST_INCREMENT)
            last_storage[h.home] = h.home.storage
    
    return picked_count, shared_count, per_family_consumption
