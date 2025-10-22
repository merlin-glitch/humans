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
)
from simulate_utils import seed_food

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

    if precomputed is not None:
        zone_map, food_ids = precomputed
    else:
        zone_map, food_ids = cached_identify(map_path, min_size, tol)

    H, W = zone_map.shape
    houses = build_houses(zone_map)

    food_zone_ids = list(food_ids.values())
    food_cells = [(x, y) for y in range(H) for x in range(W)
                  if zone_map[y, x] in food_zone_ids]

    # Spawn initial food, but don't fill cells to capacity
    # This leaves room for continuous spawning during simulation
    seed_food(food_cells, INITIAL_FOOD_COUNT, resources)

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
    if PER_ZONE_RESPAWN and food_zone_ids:
        zones = []
        for zid in food_zone_ids:
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


def run_single_tick(humans: List[Human], houses: List[House], trust_system: TrustSystem, 
                   resources, is_day: bool, last_storage: Dict[House, int], 
                   use_occupancy_map: bool = False) -> Tuple[int, int, Dict[str, int]]:
    """
    Run a single simulation tick with standardized agent behavior.
    
    Args:
        humans: List of human agents
        houses: List of house objects
        trust_system: Trust system for tracking relationships
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
            
        # Update trust when humans share resources
        if h.home.storage > last_storage[h.home]:
            boost_house_trust(trust_system, h, humans, increment=TRUST_INCREMENT)
            last_storage[h.home] = h.home.storage
    
    return picked_count, shared_count, per_family_consumption
