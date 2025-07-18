

# main.py
import random
from typing import List, Tuple, Dict

import numpy as np
import pygame
import pygame_menu

from menu import (
    Slider,
    handle_pause_event,
    create_action_buttons,
    handle_action_buttons,
    draw_action_buttons
)
from config import (
    MAP_WIDTH, MAP_HEIGHT, CELL_SIZE,
    INITIAL_FOOD_COUNT, FOOD_STACK, FOOD_LIFETIME,
    SPAWN_INTERVAL, FOOD_SPAWN_COUNT, Nbre_HUMANS
)
from human import Human, House, draw_human
from caracteristics import TrustSystem
from ressource import (
    Resource, map_draw, map_manage,
    pixel_update, display_house_storage
)
from common import (
    export_trust_matrix,
    export_house_contributions,
    plot_avg_trust_per_house,
    plot_within_vs_between_trust,
    plot_sharing_counts,
    boost_house_trust
)

def display_human_counts(screen, humans: List[Human], font: pygame.font.Font):
    """
    Renders alive/dead counts in the bottom‐right corner.
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
    """
    Draws a larger legend in the bottom-left corner:
      ▉ energy   (green)
      ▉ sleep    (blue)
      ▉ bag_cap  (red)
    """
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

# ─── simulation parameters (will be mutated by sliders) ─────────────────────
params = {
    'map_width': MAP_WIDTH,
    'map_height': MAP_HEIGHT,
    'cell_size': CELL_SIZE,
    'food_count': FOOD_STACK,
    'max_life':   FOOD_LIFETIME,
    'spawn_interval': SPAWN_INTERVAL,
    'house_count':   2,          # you only had two safe‐zones 2&3
    'house_size':    1,          # we treat each avg‐pixel as one
    'min_house_distance': 0
}

def start_simulation(params):
    # --- build & draw the static map from PNG → codes
    terrain_txt = map_draw('easy_food.png')
    codes       = np.loadtxt(terrain_txt, dtype=int)
    H, W        = codes.shape
    rev_color   = {code: rgb for rgb, code in map_draw.__globals__['TERRAIN_PALETTE'].items()}

    # --- find the two house‐centers & their colors
    houses: List[House] = []
    for safe_code in (2, 3):
        pts = [(x,y) for y in range(H) for x in range(W) if codes[y,x]==safe_code]
        if not pts: continue
        avg_x = sum(x for x,y in pts)//len(pts)
        avg_y = sum(y for x,y in pts)//len(pts)
        house_color = rev_color.get(safe_code, (150,75,0))
        houses.append(House(avg_x, avg_y, house_color))

    # --- Pygame init
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((MAP_WIDTH*CELL_SIZE, MAP_HEIGHT*CELL_SIZE))
    pygame.display.set_caption("Simulation ressources, maisons et humains")
    font = pygame.font.Font(None, max(12, CELL_SIZE*4))

    # --- food‐zones
    food_zone_codes = {4,5}
    food_zones = [(x,y) for y in range(H) for x in range(W) if codes[y,x] in food_zone_codes]

    # --- seed initial food
    resources, food_birth = [], {}
    while len(resources)<INITIAL_FOOD_COUNT:
        x,y = random.choice(food_zones)
        if sum(1 for r in resources if (r.x,r.y)==(x,y)) < FOOD_STACK:
            resources.append(Resource(x,y, life=FOOD_LIFETIME))
            food_birth.setdefault((x,y), 0)

    # --- spawn humans evenly
    total_to_spawn = Nbre_HUMANS
    base, extra = divmod(total_to_spawn, len(houses))
    extra_idx   = random.randrange(len(houses)) if extra else -1

    humans: List[Human] = []
    humanid = 0
    for idx, house in enumerate(houses):
        count = base + (1 if idx==extra_idx else 0)
        for _ in range(count):
            humans.append(Human(human_id=humanid,
                                sex=random.choice(['homme','femme']),
                                x=house.x, y=house.y,
                                home=house, codes=codes))
            humanid += 1

    trust_system = TrustSystem()

    # --- sliders & reset/export buttons
    slider = Slider(
      rect=(MAP_WIDTH*CELL_SIZE-80, 50, 30, MAP_HEIGHT*CELL_SIZE-100),
      min_val=1, max_val=500, initial=Nbre_HUMANS,
      orientation='vertical'
    )
    speed_slider = Slider(
      rect=((MAP_WIDTH*CELL_SIZE)//4,
            MAP_HEIGHT*CELL_SIZE-120,
            (MAP_WIDTH*CELL_SIZE)//2,
            20),
      min_val=0.1, max_val=10.0, initial=1.0,
      orientation='horizontal'
    )
    action_rects = create_action_buttons(speed_slider)

    last_storage = {house:house.storage for house in houses}
    total_shares = 0
    clock, tick, paused = pygame.time.Clock(), 0, False


# ─── simulation main loop ─────────────────────────────────────────────────
    DAY_LENGTH   = 2000  # ticks per full day→night cycle
    tick         = DAY_LENGTH // 4    # start at 0.25 of the cycle → bright morning
    total_shares = 0
    running      = True

    while running:
        # 1) Event handling
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            paused = handle_pause_event(e, paused)
            slider.handle_event(e)
            speed_slider.handle_event(e)

            # Reset & Export buttons
            handle_action_buttons(
                e,
                action_rects,
                on_reset = lambda: start_simulation(params),
                on_export = lambda: export_trust_matrix(trust_system, humans, "trust_matrix.csv")
            )

        if not paused:
            # 2) Day/night calculation
            cycle_pos = (tick % DAY_LENGTH) / DAY_LENGTH    # [0,1)
            is_day    = (cycle_pos < 0.5)                  # first half = day
            # linear brightness: 0.2→1.0 during day, then 1.0→0.2 at night
            if is_day:
                brightness = 0.4 + (cycle_pos / 0.5) * 0.8
            else:
                brightness = 1.0 - ((cycle_pos - 0.5) / 0.5) * 0.8

            # 3) Adjust total human count per house (slider)
            target     = int(slider.value)
            num_houses = len(houses)
            base, extra = divmod(target, num_houses)
            desired = {
                house: base + (1 if i < extra else 0)
                for i, house in enumerate(houses)
            }
            counts = {house: 0 for house in houses}
            for h in humans:
                counts[h.home] += 1

            # spawn or cull
            for house, want in desired.items():
                have = counts[house]
                if have < want:
                    for _ in range(want - have):
                        humans.append(Human(
                            human_id=humanid,
                            sex=random.choice(['homme','femme']),
                            x=house.x, y=house.y,
                            home=house, codes=codes
                        ))
                        humanid += 1
                elif have > want:
                    to_kill = have - want
                    for idx in range(len(humans)-1, -1, -1):
                        if to_kill == 0:
                            break
                        if humans[idx].home is house:
                            del humans[idx]
                            to_kill -= 1

            # 4) Update & spawn resources
            for r in resources:
                r.update()
            resources = [r for r in resources if r.is_alive()]
            if tick % SPAWN_INTERVAL == 0:
                for _ in range(FOOD_SPAWN_COUNT):
                    x, y = random.choice(food_zones)
                    if sum(1 for r in resources if (r.x, r.y) == (x, y)) < FOOD_STACK:
                        resources.append(Resource(x, y, life=FOOD_LIFETIME))
                        food_birth[(x, y)] = tick

            # 5) Humans act
            for h in humans:
                picked, shared = h.step(
                    resources, houses, humans, trust_system,
                    is_day=is_day,
                    action_cost=0.01,
                    food_gain=1.0,
                    decay_rate=0.005
                )

                # clean up eaten food
                if picked is not None:
                    food_birth.pop(picked, None)

                # count sharing events
                if shared:
                    total_shares += 1

                # detect new deposit → boost house trust
                new_store = h.home.storage
                if new_store > last_storage[h.home]:
                    boost_house_trust(trust_system, h, humans, increment=0.1)
                    last_storage[h.home] = new_store

            # 6) Draw static map & houses
            screen.fill((0, 0, 0))
            for y in range(H):
                for x in range(W):
                    color_static = rev_color.get(codes[y, x], (50, 50, 50))
                    pygame.draw.rect(
                        screen, color_static,
                        pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )
            for house in houses:
                pygame.draw.rect(
                    screen, house.color,
                    pygame.Rect(house.x * CELL_SIZE, house.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

            # 7) Draw dynamic layers (food + humans + UI)
            managed      = map_manage(codes, food_birth, tick)
            alive_humans = [h for h in humans if h.alive]
            pixel_update(screen, managed, alive_humans, font)

            display_human_counts(screen, humans, font)
            draw_legend(screen, CELL_SIZE, font)
            display_house_storage(screen, houses, CELL_SIZE, font)

            slider.draw(screen, font)
            speed_slider.draw(screen, font)
            draw_action_buttons(screen, action_rects, font)

            # share counter
            share_text = font.render(f"Shares: {total_shares}", True, (255,255,0))
            screen.blit(share_text, (10, 10))

            # 8) Night‐time overlay
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            alpha   = int((1.0 - brightness) * 200)  # 0..200
            # fill solid black…
            overlay.fill((0, 0, 0))
            # …then set the whole surface’s transparency
            overlay.set_alpha(alpha)
            screen.blit(overlay, (0, 0))

            pygame.display.flip()
            tick += 1
            clock.tick(int(30 * speed_slider.value))

        # ─── after simulation ends ────────────────────────────────────────────────
    export_trust_matrix(trust_system, humans, "trust_matrix.csv")
    for h in humans:
        trust_system.display_trust_summary(h.id)

    export_house_contributions(humans, houses[0],
                                "contributions_house1.csv",
                                "contribution_to_house1")
    export_house_contributions(humans, houses[1],
                                "contributions_house2.csv",
                                "contribution_to_house2")
    pygame.quit()
    plot_avg_trust_per_house(humans, trust_system)
    plot_within_vs_between_trust(humans, trust_system)
    plot_sharing_counts(humans, trust_system)

    

def show_menu():
    import pygame, pygame_menu
 
    # (and whatever else you need imported here)

    pygame.init()
    surface = pygame.display.set_mode((800, 550))

    # This flag tells us whether Start was clicked
    start_requested = False

    # Build the menu
    menu = pygame_menu.Menu(
        title='Paramètres',
        width=800,
        height=550,
        theme=pygame_menu.themes.THEME_DARK
    )

    # The Start callback
    def on_start():
        nonlocal start_requested   # <— you **must** do this
        start_requested = True
        menu.disable()             # <— this breaks out of menu.mainloop()

    # Add the two buttons
    menu.add.button('Start', on_start)
    menu.add.button('Quit',  pygame_menu.events.EXIT)

    # Run the menu loop (blocks here until disabled or EXIT)
    menu.mainloop(surface)

    # Tear down the menu’s pygame
    pygame.quit()

    # If Start was clicked, launch your sim:
    if start_requested:
        start_simulation(params)

        

if __name__ == "__main__":
    show_menu()
