
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
from new_ressource import (
    resources, life_span_ressource,map_draw,map_manage, pixel_update , resource_spawn_interval_inverse, display_house_storage,identify_zones
)
from ressource import *
from common import *

# def display_human_counts(screen, humans: List[Human], font: pygame.font.Font):
#     alive = sum(h.alive for h in humans)
#     dead  = len(humans) - alive
#     text  = f"Alive: {alive}   Dead: {dead}"
#     surf  = font.render(text, True, (255,255,255))
#     x = MAP_WIDTH*CELL_SIZE - 10
#     y = MAP_HEIGHT*CELL_SIZE - 10
#     rect = surf.get_rect(bottomright=(x, y))
#     screen.blit(surf, rect)

# def draw_legend(screen, cell_size: int, font: pygame.font.Font):
#     entries = [
#         ("energy",      (0, 255,   0)),
#         ("sleep",       (0, 128, 255)),
#         ("bag_capacity",(255,   0,   0)),
#     ]
#     margin  = cell_size
#     swatch  = 2*cell_size
#     spacing = int(swatch * 0.6)
#     line_h  = swatch + spacing
#     screen_h = MAP_HEIGHT * cell_size
#     start_x  = margin
#     start_y  = screen_h - margin - line_h * len(entries)+100

#     for i, (label, color) in enumerate(entries):
#         y = start_y + i * line_h
#         rect = pygame.Rect(start_x, y, swatch, swatch)
#         pygame.draw.rect(screen, color, rect)
#         pygame.draw.rect(screen, (0,0,0), rect, 2)
#         text_surf = font.render(label, True, (255,255,255))
#         text_pos  = (start_x + swatch + spacing,
#                      y + (swatch - text_surf.get_height())//2)
#         screen.blit(text_surf, text_pos)


# def start_simulation(params):
#     import os
#     from new_ressource import identify_zones

#     # --- Load map and zones ---
#     zone_map, food_ids = identify_zones(
#         os.path.join(os.path.dirname(__file__), "images", "5_spots.png")
#     )
#     codes = zone_map
#     H, W = codes.shape
#     rev_color = {code: rgb for rgb, code in map_draw.__globals__['NEW_PALETTE'].items()}

#     # --- Houses ---
#     houses: List[House] = []
#     for safe_code in (2, 3):
#         pts = [(x,y) for y in range(H) for x in range(W) if codes[y,x]==safe_code]
#         if not pts:
#             continue
#         avg_x = sum(x for x,y in pts)//len(pts)
#         avg_y = sum(y for x,y in pts)//len(pts)
#         houses.append(House(avg_x, avg_y, rev_color.get(safe_code)))

#     pygame.quit(); pygame.init()

#     # extra space for UI
#     UI_MARGIN_BOTTOM = 120
#     UI_MARGIN_RIGHT  = 150

#     screen = pygame.display.set_mode(
#         (MAP_WIDTH*CELL_SIZE + UI_MARGIN_RIGHT,
#          MAP_HEIGHT*CELL_SIZE + UI_MARGIN_BOTTOM)
#     )
#     pygame.display.set_caption("Simulation ressources, maisons et humains")
#     font = pygame.font.Font(None, max(12, CELL_SIZE*4))

#     # --- Build food zones ---
#     food_zones = []
#     for zid in food_ids.values():
#         food_zones.extend([(x, y) for y in range(H) for x in range(W) if codes[y, x] == zid])

#     # --- Seed initial food resources ---
#     spawned = 0
#     while spawned < INITIAL_FOOD_COUNT and food_zones:
#         x, y = random.choice(food_zones)
#         if resources[y, x, 1] < FOOD_STACK:
#             add_resource(x, y, life=FOOD_LIFETIME)
#             spawned += 1

#     # --- Humans ---
#     humans: List[Human] = []
#     humanid = 0
#     base, extra = divmod(Nbre_HUMANS, len(houses))
#     extra_idx = random.randrange(len(houses)) if extra else -1
#     for i, house in enumerate(houses):
#         count = base + (1 if i==extra_idx else 0)
#         for _ in range(count):
#             humans.append(Human(human_id=humanid,
#                                 sex=random.choice(['homme','femme']),
#                                 x=house.x, y=house.y,
#                                 home=house, codes=codes))
#             humanid += 1

#     trust_system = TrustSystem()

#     # UI widgets
#     slider = Slider((MAP_WIDTH*CELL_SIZE+80, 50, 30, MAP_HEIGHT*CELL_SIZE-100),
#                     1, 500, Nbre_HUMANS, orientation='vertical')
#     speed_slider = Slider(((MAP_WIDTH*CELL_SIZE)//4, MAP_HEIGHT*CELL_SIZE+30,
#                            (MAP_WIDTH*CELL_SIZE)//2, 20),
#                           0.1, 6.0, 1.0, orientation='horizontal')
#     action_rects = create_action_buttons(speed_slider)

#     # bookkeeping
#     last_storage = {h: h.storage for h in houses}
#     total_shares = 0
#     clock, paused = pygame.time.Clock(), False
#     tick = 0
#     day_tick = DAY_LENGTH * 0.3
#     next_id = humanid
#     prev_is_day = False
#     pick_history = []
#     last_mated: Dict[Tuple[int,int], int] = {}
#     births_by_pair = {}
#     deaths_today = []
#     day = 0
#     next_day_tick = DAY_LENGTH

#     blue_file = "blue_population.csv"
#     red_file  = "red_population.csv"
#     for fn in (blue_file, red_file):
#         with open(fn, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["day", "population"])

#     # --- Simulation Loop ---
#     running = True
#     while running:
#         # --- Event Handling ---
#         for e in pygame.event.get():
#             if e.type == pygame.QUIT:
#                 running = False
#             paused = handle_pause_event(e, paused)
#             slider.handle_event(e)
#             speed_slider.handle_event(e)
#             handle_action_buttons(
#                 e, action_rects,
#                 on_reset=lambda: start_simulation(params),  # restart simulation
#                 on_export=lambda: export_trust_matrix(trust_system, humans)
#             )

#         if not paused:
#             # --- Day/Night Cycle ---
#             day_tick = (day_tick + speed_slider.value) % DAY_LENGTH
#             cycle_pos = day_tick / DAY_LENGTH
#             is_day = (cycle_pos < 0.7)

#             if not prev_is_day and is_day:
#                 for house in houses:
#                     # Only run for red houses
#                     if house.color == (255, 0, 0):  
#                         family = [h for h in humans if h.home is house]
#                         run_competition(family, trust_system)
#             prev_is_day = is_day

#             # --- Resource Updates ---
#             life_span_ressource()

#             # --- Human Actions (THIS is the movement!) ---
#             picked_this_tick = 0
#             for h in humans:
#                 if not h.alive:
#                     deaths_today.append(h.id)
#                     continue
#                 picked, shared = h.step(
#                     resources, houses, humans, trust_system,
#                     is_day=is_day, action_cost=0.01, food_gain=1.0, decay_rate=0.005
#                 )
#                 if picked is not None:
#                     picked_this_tick += 1
#                 if shared:
#                     total_shares += 1
#                 new_s = h.home.storage
#                 if new_s > last_storage[h.home]:
#                     boost_house_trust(trust_system, h, humans, increment=0.1)
#                     last_storage[h.home] = new_s
#                 lid = getattr(h, "_last_night_leader", None)
#                 if lid is not None and h.bag > 0:
#                     share = h.bag * 0.1
#                     h.bag -= share
#                     leader = next(x for x in humans if x.id == lid)
#                     leader.bag += share
#                     trust_system.increase_trust(h.id, lid, increment=+0.01)
#                     if not hasattr(h, "paid_leader"):
#                         h.paid_leader = []
#                     h.paid_leader.append((lid, share))
#                     del h._last_night_leader

#             # --- Track Pick Rate ---
#             pick_history.append(picked_this_tick)
#             if len(pick_history) > 30:
#                 pick_history.pop(0)
#             avg_pick_rate = sum(pick_history) / len(pick_history)

#             # --- Resource Spawning ---
#             current_interval = resource_spawn_interval_inverse(avg_pick_rate)
#             if tick % current_interval == 0 and food_zones:
#                 for _ in range(FOOD_SPAWN_COUNT):
#                     x, y = random.choice(food_zones)
#                     if resources[y, x, 1] < FOOD_STACK:
#                         add_resource(x, y, life=FOOD_LIFETIME)

#             # --- Mating ---
#             residents_by_house = {house: [] for house in houses}
#             for h in humans:
#                 if h.alive:
#                     residents_by_house[h.home].append(h)
#             for house, residents in residents_by_house.items():
#                 for i in range(len(residents)):
#                     for j in range(i+1, len(residents)):
#                         h1, h2 = residents[i], residents[j]
#                         if h1.energy < ENERGY_COST or h2.energy < ENERGY_COST:
#                             continue
#                         pair = tuple(sorted((h1.id, h2.id)))
#                         if tick - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
#                             continue
#                         created, next_id = to_mate(
#                             h1, h2, trust_system, humans, codes, next_id,
#                             threshold=0.6, energy_cost=ENERGY_COST
#                         )
#                         if created:
#                             last_mated[pair] = tick
#                             births_by_pair[pair] = births_by_pair.get(pair, 0) + created

#             # --- Daily Logging ---
#             if tick >= next_day_tick:
#                 log_daily_population(humans, houses, day, births_by_pair, deaths_today)
#                 births_by_pair.clear()
#                 deaths_today.clear()
#                 log_population_by_house(day, humans, houses[0], blue_file)
#                 log_population_by_house(day, humans, houses[1], red_file)
#                 day += 1
#                 next_day_tick += DAY_LENGTH


#             # --- Drawing -------------------------------------------------
#             # Build the static terrain (houses + bright-green zones, no food)
#             static_layer = map_manage(codes)

#             # 1) draw static terrain
#             screen.blit(static_layer, (0, 0))

#             # 2) draw food stacks dynamically from the shared numpy array
#             #    (high-contrast yellow; size shrinks with remaining stack)
#             food_cells = np.argwhere(resources[:, :, 1] > 0)  # rows of [y, x]
#             for y, x in food_cells:
#                 qty  = int(resources[y, x, 1])
#                 frac = max(0.0, min(1.0, qty / FOOD_STACK))
#                 # square size proportional to remaining stack
#                 s = max(2, int(CELL_SIZE * 0.7 * frac))
#                 rx = x * CELL_SIZE + (CELL_SIZE - s) // 2
#                 ry = y * CELL_SIZE + (CELL_SIZE - s) // 2
#                 pygame.draw.rect(
#                     screen,
#                     (255, 230, 40),  # bright yellow so it pops against the green zone
#                     pygame.Rect(rx, ry, s, s)
#                 )

#             # 3) draw humans (big and on top of food)
#             alive = [h for h in humans if h.alive]
#             for h in alive:
#                 draw_human(screen, h, CELL_SIZE, font)

#             # 4) UI overlays (legend, sliders, counters, buttons)
#             display_human_counts(screen, humans, font)
#             draw_legend(screen, CELL_SIZE, font)
#             display_house_storage(screen, houses, CELL_SIZE, font)
#             slider.draw(screen, font)
#             speed_slider.draw(screen, font)
#             draw_action_buttons(screen, action_rects, font)
#             screen.blit(font.render(f"Shares: {total_shares}", True, (255, 255, 0)), (10, 10))

#             # Optional: day/night overlay last
#             overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
#             overlay.fill((0, 0, 0))
#             if is_day:
#                 brightness = 0.3 + (cycle_pos / 0.7) * 0.7
#             else:
#                 brightness = 1.0 - ((cycle_pos - 0.7) / 0.3) * 0.7
#             overlay.set_alpha(int((1 - brightness) * 200))
#             screen.blit(overlay, (0, 0))

#             pygame.display.flip()
#         tick += speed_slider.value
#         clock.tick(int(30 * speed_slider.value))

#     pygame.quit()

# def show_menu():
#     pygame.init()
#     surface = pygame.display.set_mode((800, 550))
#     start_requested = False

#     menu = pygame_menu.Menu('Paramètres', 800, 550, theme=pygame_menu.themes.THEME_DARK)

#     def on_start():
#         nonlocal start_requested
#         start_requested = True
#         menu.disable()

#     menu.add.button('Start', on_start)
#     menu.add.button('Quit', pygame_menu.events.EXIT)
#     menu.mainloop(surface)
#     pygame.quit()

#     if start_requested:
#         result = start_simulation(params)
#         if result == "reset":
#             show_menu()   # restart clean


# params = {
#     'map_width': MAP_WIDTH,
#     'map_height': MAP_HEIGHT,
#     'cell_size': CELL_SIZE,
#     'food_count': FOOD_STACK,
#     'max_life':   FOOD_LIFETIME,
#     'spawn_interval': SPAWN_INTERVAL,
#     'house_count':   2,
#     'house_size':    1,
#     'min_house_distance': 0
# }

# if __name__ == "__main__":
#     show_menu()



# ───────────────────────────────────────────────────────────
# HUD helpers
# ───────────────────────────────────────────────────────────
def display_human_counts(screen, humans: List[Human], font: pygame.font.Font):
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
        ("energy", (0, 255, 0)),
        ("bag",    (255, 0, 0)),
    ]
    margin  = cell_size
    swatch  = 2*cell_size
    spacing = int(swatch * 0.6)
    line_h  = swatch + spacing
    screen_h = MAP_HEIGHT * cell_size
    start_x  = margin
    # dropped lower per your preference
    start_y  = screen_h - margin - line_h * len(entries) + 100

    for i, (label, color) in enumerate(entries):
        y = start_y + i * line_h
        rect = pygame.Rect(start_x, y, swatch, swatch)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (0,0,0), rect, 2)
        text_surf = font.render(label, True, (255,255,255))
        text_pos  = (start_x + swatch + spacing,
                     y + (swatch - text_surf.get_height())//2)
        screen.blit(text_surf, text_pos)

# ───────────────────────────────────────────────────────────
# Main Simulation
# ───────────────────────────────────────────────────────────
def start_simulation(params):
    # clear shared resources grid so reset is clean
    resources[:, :, :] = 0

    # Load map and detect zones
    zone_map, food_ids = identify_zones(
        os.path.join(os.path.dirname(__file__), "images", "5_spots.png")
    )
    codes = zone_map
    H, W = codes.shape

    # Build houses from zone ids 2 (blue) and 3 (red), using centroids
    houses: List[House] = []
    for zid in (2, 3):
        ys, xs = np.where(codes == zid)
        if xs.size == 0:
            continue

        # # METHOD A: centroid with proper rounding (not floor)
        # cx = int(np.round(xs.mean()))
        # cy = int(np.round(ys.mean()))

        ## METHOD B (alternative): exact midpoint of the house bounding box
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        cx = (minx + maxx) // 2
        cy = (miny + maxy) // 2

        color = (0, 0, 255) if zid == 2 else (255, 0, 0)
        houses.append(House(cx, cy, color))


    pygame.quit(); pygame.init()

    # give space for UI so it’s not on the map
    UI_MARGIN_BOTTOM = 120
    UI_MARGIN_RIGHT  = 150

    screen = pygame.display.set_mode(
        (MAP_WIDTH*CELL_SIZE + UI_MARGIN_RIGHT,
         MAP_HEIGHT*CELL_SIZE + UI_MARGIN_BOTTOM)
    )
    pygame.display.set_caption("Simulation ressources, maisons et humains")
    font = pygame.font.Font(None, max(12, CELL_SIZE*4))

    # Precompute cells belonging to food zones (ids relabeled >= 41)
    food_zone_cells: List[Tuple[int, int]] = []
    for zid in food_ids.values():
        food_zone_cells.extend(
            (x, y) for y in range(H) for x in range(W) if codes[y, x] == zid
        )

    # Seed initial food into the shared numpy grid
    spawned = 0
    while spawned < INITIAL_FOOD_COUNT and food_zone_cells:
        x, y = random.choice(food_zone_cells)
        if resources[y, x, 1] < FOOD_STACK:
            add_resource(x, y, life=FOOD_LIFETIME)
            spawned += 1

    # Humans
    humans: List[Human] = []
    humanid = 0
    if houses:
        base, extra = divmod(Nbre_HUMANS, len(houses))
        extra_idx = random.randrange(len(houses)) if extra else -1
        for i, house in enumerate(houses):
            count = base + (1 if i == extra_idx else 0)
            for _ in range(count):
                humans.append(Human(
                    human_id=humanid,
                    sex=random.choice(['homme','femme']),
                    x=house.x, y=house.y,
                    home=house, codes=codes
                ))
                humanid += 1

    trust_system = TrustSystem()

    # UI widgets (placed in margins)
    slider = Slider((MAP_WIDTH*CELL_SIZE + 80, 50, 30, MAP_HEIGHT*CELL_SIZE - 100),
                    1, 500, Nbre_HUMANS, orientation='vertical')
    speed_slider = Slider(((MAP_WIDTH*CELL_SIZE)//4, MAP_HEIGHT*CELL_SIZE + 30,
                           (MAP_WIDTH*CELL_SIZE)//2, 20),
                          0.1, 6.0, 1.0, orientation='horizontal')
    action_rects = create_action_buttons(speed_slider)

    # bookkeeping
    last_storage = {h: h.storage for h in houses}
    total_shares = 0
    clock, paused = pygame.time.Clock(), False
    tick = 0.0
    day_tick = DAY_LENGTH * 0.3
    next_id = humanid
    prev_is_day = False
    pick_history: List[int] = []
    last_mated: Dict[Tuple[int, int], int] = {}
    births_by_pair: Dict[Tuple[int, int], int] = {}
    deaths_today: List[int] = []
    day = 0
    next_day_tick = DAY_LENGTH

    # build static terrain once (perf)
    static_layer = map_manage(codes)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    reset_requested = False
    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    # CSV headers (optional; kept for consistency)
    blue_file = "blue_population.csv"
    red_file  = "red_population.csv"
    for fn in (blue_file, red_file):
        with open(fn, 'w', newline='') as f:
            csv.writer(f).writerow(["day", "population"])

    # Main loop
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
                on_reset=request_reset,  # proper reset
                on_export=lambda: export_trust_matrix(trust_system, humans)
            )

        if reset_requested:
            pygame.event.clear()
            pygame.display.flip()
            pygame.quit()
            return "reset"

        if not paused:
            # Day/night
            day_tick = (day_tick + speed_slider.value) % DAY_LENGTH
            cycle_pos = day_tick / DAY_LENGTH
            is_day = (cycle_pos < 0.7)

            # Dawn competition (RED house only)
            if not prev_is_day and is_day and houses:
                for house in houses:
                    if house.color == (255, 0, 0):
                        family = [h for h in humans if h.home is house and h.alive]
                        if family:
                            run_competition(family, trust_system)
            prev_is_day = is_day

            # Resource aging
            life_span_ressource()

            # Humans act
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
                    picked_this_tick += 1
                if shared:
                    total_shares += 1

                # Trust bump if storage grew
                new_s = h.home.storage
                if new_s > last_storage[h.home]:
                    boost_house_trust(trust_system, h, humans, increment=0.1)
                    last_storage[h.home] = new_s

                # optional leader tip-out
                lid = getattr(h, "_last_night_leader", None)
                if lid is not None and h.bag > 0:
                    share = h.bag * 0.1
                    h.bag -= share
                    leader = next((x for x in humans if x.id == lid and x.alive), None)
                    if leader:
                        leader.bag += share
                        trust_system.increase_trust(h.id, lid, increment=+0.01)
                    if not hasattr(h, "paid_leader"):
                        h.paid_leader = []
                    h.paid_leader.append((lid, share))
                    del h._last_night_leader

            # Pick-rate history for adaptive spawn
            pick_history.append(picked_this_tick)
            if len(pick_history) > 30:
                pick_history.pop(0)
            avg_pick_rate = (sum(pick_history) / len(pick_history)) if pick_history else 0.0

            # Spawn food adaptively inside food zones
            current_interval = resource_spawn_interval_inverse(avg_pick_rate)
            if current_interval and food_zone_cells and int(tick) % current_interval == 0:
                for _ in range(FOOD_SPAWN_COUNT):
                    x, y = random.choice(food_zone_cells)
                    if resources[y, x, 1] < FOOD_STACK:
                        add_resource(x, y, life=FOOD_LIFETIME)

            # Mating
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
                        if int(tick) - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                            continue
                        created, next_id = to_mate(
                            h1, h2, trust_system, humans, codes, next_id,
                            threshold=0.6, energy_cost=ENERGY_COST
                        )
                        if created:
                            last_mated[pair] = int(tick)
                            births_by_pair[pair] = births_by_pair.get(pair, 0) + created

            # Daily logging (optional, guarded)
            if int(tick) >= next_day_tick:
                if houses:
                    log_daily_population(humans, houses, day, births_by_pair, deaths_today)
                    births_by_pair.clear()
                    deaths_today.clear()
                    if len(houses) >= 2:
                        log_population_by_house(day, humans, houses[0], blue_file)
                        log_population_by_house(day, humans, houses[1], red_file)
                day += 1
                next_day_tick += DAY_LENGTH

            # ── drawing ──────────────────────────────────────────────
            # 1) static terrain
            screen.blit(static_layer, (0, 0))

            # 2) dynamic food (draw only cells with qty > 0; size scales with qty)
            for (y, x) in np.argwhere(resources[:, :, 1] > 0):
                qty  = int(resources[y, x, 1])
                frac = max(0.0, min(1.0, qty / FOOD_STACK))
                s = max(2, int(CELL_SIZE * 0.7 * frac))
                rx = x * CELL_SIZE + (CELL_SIZE - s) // 2
                ry = y * CELL_SIZE + (CELL_SIZE - s) // 2
                pygame.draw.rect(screen, (255, 230, 40), pygame.Rect(rx, ry, s, s))

            # 3) humans
            for h in (x for x in humans if x.alive):
                draw_human(screen, h, CELL_SIZE, font)

            # 4) UI and overlay
            display_human_counts(screen, humans, font)
            draw_legend(screen, CELL_SIZE, font)
            display_house_storage(screen, houses, CELL_SIZE, font)
            slider.draw(screen, font)
            speed_slider.draw(screen, font)
            draw_action_buttons(screen, action_rects, font)
            screen.blit(font.render(f"Shares: {total_shares}", True, (255, 255, 0)), (10, 10))

            # day/night overlay
            if is_day:
                brightness = 0.3 + (cycle_pos / 0.7) * 0.7
            else:
                brightness = 1.0 - ((cycle_pos - 0.7) / 0.3) * 0.7
            overlay.fill((0, 0, 0))
            overlay.set_alpha(int((1 - brightness) * 200))
            screen.blit(overlay, (0, 0))

            pygame.display.flip()

        tick += speed_slider.value
        clock.tick(int(30 * speed_slider.value))

    pygame.quit()


# ───────────────────────────────────────────────────────────
# Menu boot
# ───────────────────────────────────────────────────────────
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
        result = start_simulation(params)
        if result == "reset":
            show_menu()   # restart clean


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