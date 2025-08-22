
import random
from typing import List, Tuple, Dict
import numpy as np
import pygame, pygame_menu
import csv

from menu import (
    Slider,
    handle_pause_event,
    create_action_buttons,
    handle_action_buttons,
    draw_action_buttons
)
from config import *
from human import *
from caracteristics import TrustSystem
from ressource import *
from common import *

def display_human_counts(screen, humans: List[Human], font: pygame.font.Font):
    """
    Displays the count of alive and dead humans on the game screen.

    This function calculates the number of alive and dead Human objects in the provided list,
    renders this information as text using the given font, and displays it at the bottom-right
    corner of the game screen.

    Args:
        screen (pygame.Surface): The surface on which to render the text.
        humans (List[Human]): A list of Human objects to count alive and dead from.
        font (pygame.font.Font): The font used to render the text.

    # This function is used to visually display the current counts of alive and dead humans
    # in the simulation/game, updating the information in real-time on the screen.
    """
    alive = sum(h.alive for h in humans)
    dead  = len(humans) - alive
    text  = f"Alive: {alive}   Dead: {dead}"
    surf  = font.render(text, True, (255,255,255))
    x = MAP_WIDTH*CELL_SIZE - 10
    y = MAP_HEIGHT*CELL_SIZE - 10
    rect = surf.get_rect(bottomright=(x, y))
    screen.blit(surf, rect)

def draw_legend(screen, cell_size: int, font: pygame.font.Font):
    entries = [
        ("energy",      (0, 255,   0)),
        ("sleep",       (0, 128, 255)),
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

def resource_spawn_interval_inverse(f_avg: float, I_max: int = 200, k: float = 0.5, I_min: int = 10) -> int:
    interval = I_max / (1 + k * f_avg)
    return max(I_min, int(interval))

def start_simulation(params):
    """
    Starts and runs the main simulation loop for the human resource-sharing environment.

    This function initializes the simulation environment, including terrain, houses, resources, and human agents.
    It manages the main event loop using pygame, handling user input, updating simulation state, and rendering the environment.
    The simulation models resource collection, sharing, trust dynamics, mating, and population tracking among humans living in houses.
    It also logs population statistics and exports trust matrices for analysis.

    Args:
        params (dict): Dictionary of simulation parameters (e.g., initial human count, resource settings, etc.).

    Side Effects:
        - Opens a pygame window for visualization.
        - Writes population statistics to CSV files.
        - Exports trust matrices and generates plots at the end of the simulation.

    Note:
        The function runs until the user closes the pygame window.
    """
    terrain_txt = map_draw('3_spots.png')
    codes       = np.loadtxt(terrain_txt, dtype=int)
    H, W        = codes.shape
    rev_color   = {code: rgb for rgb, code in map_draw.__globals__['NEW_PALETTE'].items()}

    houses: List[House] = []
    for safe_code in (2, 3):
        pts = [(x,y) for y in range(H) for x in range(W) if codes[y,x]==safe_code]
        if not pts: continue
        avg_x = sum(x for x,y in pts)//len(pts)
        avg_y = sum(y for x,y in pts)//len(pts)
        houses.append(House(avg_x, avg_y, rev_color.get(safe_code)))

    pygame.quit(); pygame.init()
    screen = pygame.display.set_mode((MAP_WIDTH*CELL_SIZE, MAP_HEIGHT*CELL_SIZE))
    pygame.display.set_caption("Simulation ressources, maisons et humains")
    font = pygame.font.Font(None, max(12, CELL_SIZE*4))

    food_zones = [(x,y) for y in range(H) for x in range(W) if codes[y,x] in {4,5}]
    resources, food_birth = [], {}
    while len(resources) < INITIAL_FOOD_COUNT:
        x,y = random.choice(food_zones)
        if sum(1 for r in resources if (r.x,r.y)==(x,y)) < FOOD_STACK:
            resources.append(Resource(x,y, life=FOOD_LIFETIME))
            food_birth.setdefault((x,y), 0)

    humans: List[Human] = []
    humanid = 0
    base, extra = divmod(Nbre_HUMANS, len(houses))
    extra_idx = random.randrange(len(houses)) if extra else -1
    for i, house in enumerate(houses):
        count = base + (1 if i==extra_idx else 0)
        for _ in range(count):
            humans.append(Human(human_id=humanid,
                                sex=random.choice(['homme','femme']),
                                x=house.x, y=house.y,
                                home=house, codes=codes))
            humanid += 1

    trust_system = TrustSystem()

    slider = Slider((MAP_WIDTH*CELL_SIZE-80, 50, 30, MAP_HEIGHT*CELL_SIZE-100), 1, 500, Nbre_HUMANS, orientation='vertical')
    speed_slider = Slider(((MAP_WIDTH*CELL_SIZE)//4, MAP_HEIGHT*CELL_SIZE-120, (MAP_WIDTH*CELL_SIZE)//2, 20), 0.1, 6.0, 1.0, orientation='horizontal')
    action_rects = create_action_buttons(speed_slider)

    last_storage = {h: h.storage for h in houses}
    total_shares = 0
    clock, paused = pygame.time.Clock(), False
    tick = 0
    day_tick = DAY_LENGTH * 0.3
    next_id = humanid
    prev_slider_val = slider.value
    prev_is_day = False
    pick_history = []

    last_mated: Dict[Tuple[int,int], int] = {}
    births_by_pair = {}
    deaths_today = []
    day = 0
    next_day_tick = DAY_LENGTH

    blue_file = "blue_population.csv"
    red_file  = "red_population.csv"
    for fn in (blue_file, red_file):
        with open(fn, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["day", "population"])

    running = True
    # Main simulation loop: handles events, updates state, and draws everything
    while running:
        # --- Event Handling ---
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False  # Exit simulation if window is closed
            paused = handle_pause_event(e, paused)  # Toggle pause on key event
            slider.handle_event(e)                  # Handle population slider
            speed_slider.handle_event(e)            # Handle speed slider
            handle_action_buttons(
                e, action_rects,
                on_reset=lambda: start_simulation(params),  # Restart simulation
                on_export=lambda: export_trust_matrix(trust_system, humans)  # Export trust matrix
            )

        if not paused:
            # --- Day/Night Cycle Update ---
            day_tick = (day_tick + speed_slider.value) % DAY_LENGTH
            cycle_pos = day_tick / DAY_LENGTH
            is_day = (cycle_pos < 0.7)

            # At dawn, run competition in each house to select leader/memory spot
            if not prev_is_day and is_day:
                for house in houses:
                    family = [h for h in humans if h.home is house]
                    run_competition(family, trust_system)
            prev_is_day = is_day

            # --- Resource Updates ---
            for r in resources:
                r.update()
            resources = [r for r in resources if r.is_alive()]

            # --- Human Actions ---
            picked_this_tick = 0
            for h in humans:
                if not h.alive:
                    deaths_today.append(h.id)
                    continue
                picked, shared = h.step(
                    resources, houses, humans, trust_system,
                    is_day=is_day, action_cost=0.01, food_gain=1.0, decay_rate=0.005
                )
                if picked is not None:
                    food_birth.pop(picked, None)
                    picked_this_tick += 1
                if shared:
                    total_shares += 1
                # If house storage increased, boost trust for all house members
                new_s = h.home.storage
                if new_s > last_storage[h.home]:
                    boost_house_trust(trust_system, h, humans, increment=0.1)
                    last_storage[h.home] = new_s
                # Reward last night's leader if applicable
                lid = getattr(h, "_last_night_leader", None)
                if lid is not None and h.bag > 0:
                    share = h.bag * 0.1
                    h.bag -= share
                    leader = next(x for x in humans if x.id == lid)
                    leader.bag += share
                    trust_system.increase_trust(h.id, lid, increment=+0.01)
                    if not hasattr(h, "paid_leader"):
                        h.paid_leader = []
                    h.paid_leader.append((lid, share))
                    del h._last_night_leader

            # --- Track Resource Pick Rate ---
            pick_history.append(picked_this_tick)
            if len(pick_history) > 30:
                pick_history.pop(0)
            avg_pick_rate = sum(pick_history) / len(pick_history)

            # --- Resource Spawning (dynamic interval) ---
            current_interval = resource_spawn_interval_inverse(avg_pick_rate)
            if tick % current_interval == 0:
                for _ in range(FOOD_SPAWN_COUNT):
                    x, y = random.choice(food_zones)
                    if sum(1 for r in resources if (r.x, r.y) == (x, y)) < FOOD_STACK:
                        resources.append(Resource(x, y, life=FOOD_LIFETIME))
                        food_birth[(x, y)] = tick

            # --- Mating (within each house, with cooldown and energy check) ---
            residents_by_house = {house: [] for house in houses}
            for h in humans:
                if h.alive:
                    residents_by_house[h.home].append(h)

            for house, residents in residents_by_house.items():
                for i in range(len(residents)):
                    for j in range(i+1, len(residents)):
                        h1, h2 = residents[i], residents[j]
                        if h1.energy < ENERGY_COST or h2.energy < ENERGY_COST:
                            continue
                        pair = tuple(sorted((h1.id, h2.id)))
                        if tick - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                            continue
                        created, next_id = to_mate(
                            h1, h2, trust_system, humans, codes, next_id,
                            threshold=0.6, energy_cost=ENERGY_COST
                        )
                        if created:
                            last_mated[pair] = tick
                            births_by_pair[pair] = births_by_pair.get(pair, 0) + created

            # --- Daily Logging ---
            if tick >= next_day_tick:
                log_daily_population(humans, houses, day, births_by_pair, deaths_today)
                births_by_pair.clear()
                deaths_today.clear()
                log_population_by_house(day, humans, houses[0], blue_file)
                log_population_by_house(day, humans, houses[1], red_file)
                day += 1
                next_day_tick += DAY_LENGTH

            # --- Drawing Section ---
            screen.fill((0, 0, 0))
            # Draw terrain
            for y in range(H):
                for x in range(W):
                    pygame.draw.rect(
                        screen,
                        rev_color.get(codes[y, x], (50, 50, 50)),
                        pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )
            # Draw houses
            for hse in houses:
                pygame.draw.rect(
                    screen, hse.color,
                    pygame.Rect(hse.x * CELL_SIZE, hse.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
            # Draw humans and UI overlays
            alive = [h for h in humans if h.alive]
            static_layer = map_manage(codes)   # new signature: only codes
            pixel_update(screen, static_layer, alive, font)

            display_human_counts(screen, humans, font)
            draw_legend(screen, CELL_SIZE, font)
            display_house_storage(screen, houses, CELL_SIZE, font)
            slider.draw(screen, font)
            speed_slider.draw(screen, font)
            draw_action_buttons(screen, action_rects, font)
            screen.blit(font.render(f"Shares: {total_shares}", True, (255, 255, 0)), (10, 10))
            # Day/night overlay
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0))
            if is_day:
                brightness = 0.3 + (cycle_pos / 0.7) * 0.7
            else:
                brightness = 1.0 - ((cycle_pos - 0.7) / 0.3) * 0.7
            overlay.set_alpha(int((1 - brightness) * 200))
            screen.blit(overlay, (0, 0))
            pygame.display.flip()

        # --- Advance Simulation Time ---
        tick += speed_slider.value
        clock.tick(int(30 * speed_slider.value))

    pygame.quit()
    export_trust_matrix(trust_system, humans, "trust_matrix.csv")
    plot_avg_trust_per_house(humans, trust_system)
    plot_within_vs_between_trust(humans, trust_system)
    plot_sharing_counts(humans, trust_system)
    plot_population_variation(red_file,blue_file)

def show_menu():
    pygame.init()
    surface = pygame.display.set_mode((800, 550))
    start_requested = False

    menu = pygame_menu.Menu('Paramètres', 800, 550, theme=pygame_menu.themes.THEME_DARK)
    def on_start():
        nonlocal start_requested
        start_requested = True
        menu.disable()
    menu.add.button('Start', on_start)
    menu.add.button('Quit', pygame_menu.events.EXIT)
    menu.mainloop(surface)
    pygame.quit()

    if start_requested:
        start_simulation(params)

params = {
    'map_width': MAP_WIDTH,
    'map_height': MAP_HEIGHT,
    'cell_size': CELL_SIZE,
    'food_count': FOOD_STACK,
    'max_life':   FOOD_LIFETIME,
    'spawn_interval': SPAWN_INTERVAL,
    'house_count':   2,
    'house_size':    1,
    'min_house_distance': 0
}

if __name__ == "__main__":
    show_menu()
