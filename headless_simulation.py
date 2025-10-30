"""
headless_simulation.py - Optimized headless simulation runner

Part of the Human Society Simulation project.

Provides high-performance batch simulation capabilities without UI overhead.
Optimized for running multiple simulations with different parameters and seeds.
"""

# headless_simulation.py
import os
import random
import numpy as np
from typing import Optional

from config import *  # constants like MAP_WIDTH, CELL_SIZE, DAY_LENGTH, etc.
from social_mechanics import boost_house_trust, run_competition, to_mate, _avg_pairwise_trust
from simulation_utils import (
    PER_ZONE_RESPAWN, COLLECT_METRICS, ENABLE_MATING,
    build_world, draw_offscreen, seed_food, run_single_tick,
    should_move_house, find_best_house_location
)
from resource_manager import (
    resources, life_span_ressource, map_manage, resource_spawn_interval_inverse, add_resource
)

def simulate_headless(*, num_days: int, seed: Optional[int],
                      map_path: str, min_size: int, tol: int,
                      precomputed=None, preview_every: Optional[int] = None,
                      preview_dir: str = "previews"):
    """
    Headless simulation, optimized:
      - no occupancy map / neighbor buckets per tick
      - O(1) zone lookup via precomputed cell_to_zone
      - per-zone pick counting; per-zone respawn
      - preview frames optional (off by default)
    """
    # ---- Build world (map, houses, humans, trust, per-zone params) ----
    world = build_world(seed=seed, map_path=map_path, min_size=min_size,
                        tol=tol, precomputed=precomputed, n_days=num_days)

    zone_map        = world["zone_map"]
    houses          = world["houses"]
    humans          = world["humans"]
    trust           = world["trust_system"]
    next_id         = world["next_id"]
    food_zone_ids   = world["food_zone_ids"]
    per_zone        = world["per_zone"]
    food_cells      = world["food_cells"]
    last_storage    = world["last_storage"]
    pick_history_global = world["pick_history_global"]

    # Precompute (x,y)->zone_idx mapping once for O(1) lookups
    if PER_ZONE_RESPAWN and per_zone and food_zone_ids:
        zones = per_zone["zones"]  # list[list[(x,y)]]
        N_ZONES = per_zone["N_ZONES"]
        cell_to_zone = {}
        for zi, comp in enumerate(zones):
            for (x, y) in comp:
                cell_to_zone[(x, y)] = zi
    else:
        zones = []
        N_ZONES = 0
        cell_to_zone = {}

    # ---- Metrics (daily) ----
    days = []; blue_pop = []; red_pop = []
    within_blue_trust = []; within_red_trust = []; between_trust = []
    blue_born = []; red_born = []; blue_dead = []; red_dead = []
    if PER_ZONE_RESPAWN and COLLECT_METRICS and per_zone:
        zone_spawned_daily  = [[] for _ in range(N_ZONES)]
        zone_consumed_daily = [[] for _ in range(N_ZONES)]
        # Track per-family consumption per zone
        zone_blue_consumed_daily = [[] for _ in range(N_ZONES)]
        zone_red_consumed_daily = [[] for _ in range(N_ZONES)]

    # ---- Optional offscreen previews (no window) ----
    offscreen = None
    if preview_every:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        pygame.init()
        os.makedirs(preview_dir, exist_ok=True)
        offscreen = {
            "pg": pygame,
            "screen": pygame.Surface((MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE)),
            "font": pygame.font.Font(None, max(12, CELL_SIZE * 4)),
            "static": map_manage(zone_map),
            "frame_dir": preview_dir,
        }

    total_ticks = int(num_days * DAY_LENGTH)
    last_mated = {}
    births_by_pair = {}
    
    # Initialize house attributes for adaptive relocation
    for house in houses:
        if not hasattr(house, "inertia"):
            house.inertia = 0.0

    for t in range(1, total_ticks + 1):
        # Day/night cycle: 70% day, 30% night
        is_day = ((t - 1) % DAY_LENGTH) / DAY_LENGTH < 0.7

        # Track per-human travel for adaptive house relocation
        for h in humans:
            if h.alive:
                h.daily_travel = getattr(h, "daily_travel", 0.0)
                h._last_x = getattr(h, "_last_x", h.x)
                h._last_y = getattr(h, "_last_y", h.y)
                movement = ((h.x - h._last_x) ** 2 + (h.y - h._last_y) ** 2) ** 0.5
                h.daily_travel += movement
                h._last_x = h.x
                h._last_y = h.y

        # Dawn: adaptive house relocation + leadership competition (at start of each day)
        if ((t - 1) % DAY_LENGTH) == 0 and houses:
            # Adaptive house relocation at dawn (before competition)
            for house in houses:
                P_move = should_move_house(house, humans, resources)
                if random.random() < P_move:
                    old_x, old_y = house.x, house.y
                    new_x, new_y = find_best_house_location(house, humans, resources)
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
            
            # Competition for leadership
            for house in houses:
                # Apply leadership competition to both blue and red houses
                fam = [h for h in humans if h.alive and h.home is house]
                if fam:
                    run_competition(fam, trust, threshold=0.55)

        # Resource decay
        life_span_ressource()

        # Per-zone pick counters for this tick
        if N_ZONES:
            picks_this_tick = [0] * N_ZONES
        else:
            picks_this_tick = None

        # ---- Human actions using standardized logic ----
        picked_this_tick_total, shared_this_tick, per_family_consumption = run_single_tick(
            humans, houses, trust, resources, is_day, last_storage,
            use_occupancy_map=True  # Enable occupancy map for 2-5x performance boost
        )
        
        # Count picks per zone (fast dict lookup)
        if N_ZONES:
            # Need to track individual picks for zone counting
            for h in humans:
                if not h.alive:
                    continue
                # Get the most recent pick position from the human's memory
                if hasattr(h, 'last_pick_pos') and h.last_pick_pos is not None:
                    px, py = h.last_pick_pos
                    zi = cell_to_zone.get((px, py))
                    if zi is not None:
                        picks_this_tick[zi] += 1
                        if COLLECT_METRICS:
                            per_zone["consumed_today"][zi] += 1
                            # Track per-family consumption per zone
                            if h.home.color == (0, 0, 128):  # Blue house
                                per_zone["blue_consumed_today"][zi] += 1
                            elif h.home.color == (255, 0, 0):  # Red house
                                per_zone["red_consumed_today"][zi] += 1

        # Compact away dead humans once per tick (no draw/compute on corpses)
        if any(not hh.alive for hh in humans):
            humans[:] = [hh for hh in humans if hh.alive]

        # ---- Adaptive per-zone respawn (type1 food only, overexploitation-aware) ----
        # Only respawn if ENABLE_FOOD_RESPAWN flag is True
        if ENABLE_FOOD_RESPAWN and PER_ZONE_RESPAWN and per_zone and food_zone_ids and N_ZONES:
            pick_histories = per_zone["pick_histories"]
            I_MAX = per_zone["I_MAX"]; K_GAIN = per_zone["K_GAIN"]; I_MIN = per_zone["I_MIN"]
            SPAWN_COUNT = per_zone["SPAWN_COUNT"]
            
            for zi in range(N_ZONES):
                pick_histories[zi].append(picks_this_tick[zi] if picks_this_tick else 0)
                avg_pick = (sum(pick_histories[zi])/len(pick_histories[zi])) if pick_histories[zi] else 0.0
                stock = resources[:, :, 1][zone_map == food_zone_ids[zi]].sum() if zi < len(food_zone_ids) else 0
                interval = resource_spawn_interval_inverse(
                    avg_pick, zone_id=food_zone_ids[zi], zone_map=zone_map,
                    I_max=I_MAX[zi], k=K_GAIN[zi], I_min=I_MIN[zi], stock_today=stock
                )
                if interval and (t % interval == 0):
                    spawned_now = 0
                    for _ in range(SPAWN_COUNT[zi]):
                        if zones[zi]:
                            x, y = random.choice(zones[zi])
                            if resources[y, x, 1] < FOOD_STACK:
                                add_resource(x, y)
                                spawned_now += 1
                    if COLLECT_METRICS:
                        per_zone["spawned_today"][zi] += spawned_now
        elif ENABLE_FOOD_RESPAWN:
            # Legacy global respawn path (only if respawn enabled)
            pick_history_global.append(picked_this_tick_total)
            avg_pick = (sum(pick_history_global)/len(pick_history_global)) if pick_history_global else 0.0
            interval = resource_spawn_interval_inverse(avg_pick)
            if interval and food_cells and (t % interval == 0):
                seed_food(food_cells, FOOD_SPAWN_COUNT, resources)

        # ---- Optional preview frame ----
        if offscreen and preview_every and (t % preview_every == 0):
            pygame = offscreen["pg"]
            draw_offscreen(offscreen["screen"], offscreen["static"], offscreen["font"], resources, humans)
            pygame.image.save(offscreen["screen"], f'{offscreen["frame_dir"]}/frame_{t:06d}.png')

        # ---- End-of-day bookkeeping ----
        if (t % DAY_LENGTH) == 0:
            day = t // DAY_LENGTH
            trust.flush()
            
            # Trust decay (forgetting) - apply every TRUST_DECAY_INTERVAL days
            if day % TRUST_DECAY_INTERVAL == 0 and day > 0:
                for h in humans:
                    if h.alive:
                        trust.init_human(h.id)
                        if h.id in trust.hints:
                            data = trust.hints[h.id]
                            size = data.get("size", 0)
                            if size == 0:
                                size = len(data.get("index", {}))
                                data["size"] = size
                            for other_id in list(data.get("index", {}).keys()):
                                trust.increase_trust(
                                    trustor_id=h.id,
                                    trustee_id=other_id,
                                    increment=-TRUST_DECAY_AMOUNT,
                                    refresh=False
                                )
                trust.flush()

            # Mating (same logic; trust threshold/energy cost unchanged)
            if ENABLE_MATING and humans:
                residents_by_house = {house: [] for house in houses}
                for h in humans:
                    if h.alive:
                        residents_by_house[h.home].append(h)
                for house, residents in residents_by_house.items():
                    for i in range(len(residents)):
                        for j in range(i + 1, len(residents)):
                            h1, h2 = residents[i], residents[j]
                            if h1.energy < ENERGY_COST or h2.energy < ENERGY_COST:
                                continue
                            pair = tuple(sorted((h1.id, h2.id)))
                            if t - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                                continue
                            created, next_id = to_mate(
                                h1, h2, trust, humans, zone_map, next_id,
                                threshold=0.7, energy_cost=ENERGY_COST
                            )
                            if created:
                                last_mated[pair] = t
                                births_by_pair[pair] = births_by_pair.get(pair, 0) + created

            # Daily metrics
            if COLLECT_METRICS:
                blue_alive = sum(1 for h in humans if h.alive and len(houses) >= 1 and h.home is houses[0])
                red_alive  = sum(1 for h in humans if h.alive and len(houses) >= 2 and h.home is houses[1])
                days.append(day)
                blue_pop.append(blue_alive); red_pop.append(red_alive)

                blue_ids = [h.id for h in humans if h.alive and len(houses) >= 1 and h.home is houses[0]]
                red_ids  = [h.id for h in humans if h.alive and len(houses) >= 2 and h.home is houses[1]]

                within_blue_trust.append(_avg_pairwise_trust(blue_ids, trust) if blue_ids else 0.0)
                within_red_trust.append(_avg_pairwise_trust(red_ids, trust) if red_ids else 0.0)
                if blue_ids and red_ids:
                    s = 0.0; n = 0
                    for i in blue_ids:
                        for j in red_ids:
                            s += trust.trust_score(i, j); n += 1
                    between_trust.append(s / n if n else 0.0)
                else:
                    between_trust.append(0.0)

                if PER_ZONE_RESPAWN and per_zone and food_zone_ids and N_ZONES:
                    for z in range(N_ZONES):
                        zone_spawned_daily[z].append(per_zone["spawned_today"][z])
                        zone_consumed_daily[z].append(per_zone["consumed_today"][z])
                        # Save per-family consumption data
                        zone_blue_consumed_daily[z].append(per_zone.get("blue_consumed_today", [0] * N_ZONES)[z])
                        zone_red_consumed_daily[z].append(per_zone.get("red_consumed_today", [0] * N_ZONES)[z])
                    per_zone["spawned_today"]  = [0] * N_ZONES
                    per_zone["consumed_today"] = [0] * N_ZONES
                    per_zone["blue_consumed_today"] = [0] * N_ZONES
                    per_zone["red_consumed_today"] = [0] * N_ZONES

    # Cleanup preview
    if offscreen:
        offscreen["pg"].quit()

    # ---- Return shape unchanged ----
    if COLLECT_METRICS:
        if PER_ZONE_RESPAWN and per_zone and food_zone_ids and N_ZONES:
            return (days, blue_pop, red_pop,
                    within_blue_trust, within_red_trust, between_trust,
                    zone_spawned_daily, zone_consumed_daily, 
                    zone_blue_consumed_daily, zone_red_consumed_daily)
        else:
            return (days, blue_pop, red_pop,
                    within_blue_trust, within_red_trust, between_trust)
    return None
