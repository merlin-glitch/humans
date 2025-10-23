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
    add_resource
)
from simulation_utils import build_world, run_single_tick

# ────────────────────────────────────────────────
# HUD helpers
# ────────────────────────────────────────────────
def display_human_counts(screen, humans: List[Human], font: pygame.font.Font):
    """Display alive/dead human counts on screen."""
    alive = sum(h.alive for h in humans)
    dead  = len(humans) - alive
    text  = f"Alive: {alive}   Dead: {dead}"
    surf  = font.render(text, True, (255,255,255))
    x = MAP_WIDTH*CELL_SIZE - 10
    y = MAP_HEIGHT*CELL_SIZE - 10
    rect = surf.get_rect(bottomright=(x, y))
    screen.blit(surf, rect)

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
    map_path = os.path.join(os.path.dirname(__file__), "images", "3_spots.png")
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

    # Build static terrain once for speed
    static_layer = map_manage(zone_map)

    # UI widgets
    current_population = len([h for h in humans if h.alive])
    slider = Slider((MAP_WIDTH*CELL_SIZE+100, 50, 30, MAP_HEIGHT*CELL_SIZE-100), 1, 500, current_population, orientation='vertical')
    speed_slider = Slider(((MAP_WIDTH*CELL_SIZE)//4, MAP_HEIGHT*CELL_SIZE+120, (MAP_WIDTH*CELL_SIZE)//2, 20), 0.1, 6.0, 1.0, orientation='horizontal')
    action_rects = create_action_buttons(speed_slider)

    # bookkeeping
    last_storage = {h: h.storage for h in houses}
    total_shares = 0
    clock, paused = pygame.time.Clock(), False
    tick = 0.0
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
            paused = handle_pause_event(e, paused)
            slider.handle_event(e)
            speed_slider.handle_event(e)
            handle_action_buttons(
                e, action_rects,
                on_reset=lambda: start_interactive_simulation(),
                on_export=lambda: export_trust_matrix(trust_system, humans)
            )

        # Handle population changes from slider
        current_alive = sum(1 for h in humans if h.alive)
        target_pop = int(slider.value)
        
        if target_pop != target_population:
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

        if not paused:
            # Day/Night cycle (70% day, 30% night)
            day_tick = (day_tick + speed_slider.value) % DAY_LENGTH
            cycle_pos = day_tick / DAY_LENGTH
            is_day = (cycle_pos < 0.7)  # Magic number 0.7 = 70% day time

            # Dawn competition for red house (only at day transition)
            if not prev_is_day and is_day and houses:
                for house in houses:
                    # Apply leadership competition to both blue and red houses
                    family = [h for h in humans if h.home is house and h.alive]
                    if family:
                        run_competition(family, trust_system)
            prev_is_day = is_day

            # Resource decay
            life_span_ressource()

            # Humans act using standardized logic
            picked_this_tick, shared_this_tick, per_family_consumption = run_single_tick(
                humans, houses, trust_system, resources, is_day, last_storage, 
                use_occupancy_map=True  # Enable occupancy map for better performance
            )
            total_shares += shared_this_tick
            
            # Track deaths
            for h in humans:
                if not h.alive and h.id not in deaths_today:
                    deaths_today.append(h.id)

            # Adaptive spawn
            pick_history.append(picked_this_tick)
            if len(pick_history) > 30:
                pick_history.pop(0)
            avg_pick_rate = (sum(pick_history) / len(pick_history)) if pick_history else 0.0
            interval = resource_spawn_interval_inverse(avg_pick_rate)
            if interval and food_cells and int(tick) % interval == 0:
                spawned_now = 0
                for _ in range(FOOD_SPAWN_COUNT):
                    x, y = random.choice(food_cells)
                    if resources[y, x, 1] < FOOD_STACK:
                        add_resource(x, y)
                        spawned_now += 1

            # ---- Drawing ----
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

            # UI
            display_human_counts(screen, humans, font)
            draw_legend(screen, CELL_SIZE, font)
            display_house_storage(screen, houses, CELL_SIZE, font)
            slider.draw(screen, font)
            speed_slider.draw(screen, font)
            draw_action_buttons(screen, action_rects, font)
            screen.blit(font.render(f"Shares: {total_shares}", True, (255, 255, 0)), (10, 10))
            screen.blit(font.render(f"Target: {target_population}", True, (255, 255, 255)), (10, 30))

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

        tick += speed_slider.value
        clock.tick(int(30 * speed_slider.value))

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
                    run_competition(fam, trust)
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

# ────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────

if __name__ == "__main__":
    show_menu()
