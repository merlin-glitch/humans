
# main.py
import random

import numpy as np
import pygame

from menu import show_menu
from config import *
from human import Human
from caracteristics import TrustSystem
from ressource import *

#-----------------display human count-----------------
def display_human_counts(screen, humans, font):
    """
    Renders alive/dead counts in the bottom‐right corner.
    """
    alive = sum(h.alive for h in humans)
    dead  = len(humans) - alive
    text  = f"Alive: {alive}   Dead: {dead}"
    surf  = font.render(text, True, (255,255,255))
    # position 10px from bottom‐right
    x = MAP_WIDTH*CELL_SIZE - 10
    y = MAP_HEIGHT*CELL_SIZE - 10
    rect = surf.get_rect(bottomright=(x, y))
    screen.blit(surf, rect)    

def draw_legend(screen, cell_size, font):
    """
    Draws a larger legend in the bottom-left corner:
      ▉ energy   (green)
      ▉ slp      (blue)
      ▉ bag_cap  (red)
    """
    # Legend entries (label, color)
    entries = [
        ("energy",  (0, 255, 0)),
        ("sleep",     (0, 128, 255)),
        ("bag_capacity", (255,   0,   0)),
    ]

    # Sizing & spacing
    margin    = cell_size  # distance from edges
    swatch    = 2*cell_size  # square size
    spacing   = swatch *0.6
    line_h    = swatch + spacing


    # Starting coords: bottom-left, going up
    screen_h = MAP_HEIGHT * cell_size
    start_x  = margin
    start_y  = screen_h - margin - line_h * len(entries)

    for i, (label, color) in enumerate(entries):
        y = start_y + i * line_h
        # Draw color swatch
        rect = pygame.Rect(start_x, y, swatch, swatch)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (0,0,0), rect, 2)  # border

        # Draw label to the right of the swatch
        text_surf = font.render(label, True, (255,255,255))
        text_pos  = (start_x + swatch + spacing, y + (swatch - text_surf.get_height()) // 2)
        screen.blit(text_surf, text_pos)


# ─── 1) Main Simulation ──────────────────────────────────────────────────────
def start_simulation(params):# check menu.py ==> params= {'map_width': MAP_WIDTH,'map_height': MAP_HEIGHT,'cell_size': CELL_SIZE,'food_count': FOOD_COUNT,'max_life': MAX_LIFE,'spawn_interval': SPAWN_INTERVAL,'house_count': HOUSE_COUNT,'house_size': HOUSE_SIZE,'min_house_distance': MIN_HOUSE_DISTANCE}
    # build codes from PNG/text
    #terrain_txt = map_draw('beach1.png')
    terrain_txt = map_draw('easy_food.png')
    codes       = np.loadtxt(terrain_txt, dtype=int)
    H, W        = codes.shape

    # invert TERRAIN_PALETTE for drawing static ( from code to rgb color to show the static parts of the map)
    rev_color = {code: rgb for rgb, code in TERRAIN_PALETTE.items()}

    # extract houses from safe‐zone colors (2 & 3) 
    houses = []
    for safe_code in (2, 3):
        pts = [(x, y) for y in range(H) for x in range(W) if codes[y, x] == safe_code]
        if not pts:
            continue
        avg_x = sum(x for x, y in pts) // len(pts)
        avg_y = sum(y for x, y in pts) // len(pts)
        houses.append(House(avg_x, avg_y))

    # init pygame
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
    pygame.display.set_caption("Simulation ressources, maisons et humains")
    font = pygame.font.Font(None, max(12, CELL_SIZE * 4))

    # determine food‐zone tiles (codes 4 & 5)
    food_zone_codes = {4, 5}
    food_zones = [(x, y)
                  for y in range(H) for x in range(W)
                  if codes[y, x] in food_zone_codes]

    
    # initial resources (only from food_zones)
    resources, food_birth = [], {}
    seeded = 0
    while seeded < INITIAL_FOOD_COUNT:
        x, y = random.choice(food_zones)
        # only add if under stack limit
        if sum(1 for r in resources if (r.x, r.y) == (x, y)) < FOOD_STACK:
            resources.append(Resource(x, y, life=FOOD_LIFETIME))
            food_birth.setdefault((x, y), 0)
            seeded += 1

    # spawn humans: for now using house coordinates to avoid spawning in -1 tiles
    humans = []
    humanid = 0
    for house in houses:
        spawn_x = min(MAP_WIDTH - 1, house.x + 2)
        spawn_y = min(MAP_HEIGHT - 1, house.y + 2)
        for _ in range(Nbre_HUMANS):
            humans.append(Human(
                human_id=humanid,
                sex=random.choice(['homme', 'femme']),
                x=spawn_x, y=spawn_y,
                home=house,
                codes=codes
            ))
            humanid += 1

    trust_system = TrustSystem()
    

    clock, tick = pygame.time.Clock(), 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # update + cull resources
        for r in resources: # update de chaque ressource r dans resources[]
            r.update()
        resources = [r for r in resources if r.is_alive()] # rebuilds resources[] by only keeping the r alive
        alive_coords = {(r.x, r.y) for r in resources}
        for coord in list(food_birth.keys()): # on enleve de la carte les food morts 
            if coord not in alive_coords:
                food_birth.pop(coord, None)

        # # periodic spawn (only in food_zones)

        # every SPAWN_INTERVAL ticks, drop FOOD_SPAWN_COUNT new bits of food
        if tick % SPAWN_INTERVAL == 0:
            for _ in range(FOOD_SPAWN_COUNT):
                x, y = random.choice(food_zones)
                if sum(1 for r in resources if (r.x, r.y) == (x, y)) < FOOD_STACK:
                    resources.append(Resource(x, y, life=FOOD_LIFETIME))
                    food_birth[(x, y)] = tick


        # humans act
        for h in humans:
            picked = h.step(resources, houses, humans)
            if picked is not None:
                food_birth.pop(picked, None)
    



        # # ─── update trust on meeting/sharing memory ──────────────────────────

         # ─── update trust on every meeting ───────────────────────────────────
        for h1 in humans:
            for h2 in humans:
                if h1 is not h2 and (h1.x, h1.y) == (h2.x, h2.y):
                    # register the meeting (success or failure decided inside)
                    trust_system.update_on_meeting(h1, h2, resources)
                    # clear last_collected so it only counts once
                    h1.last_collected = None
                    h2.last_collected = None


#------------------to debug-------------------
        # # After all pair updates, count total meetings recorded
        # total_meetings = sum(ts for stats in trust_system.hints.values()
        #                     for _, (_, ts) in stats["pair_stats"].items())
        # print(f"Total pairwise interactions recorded so far: {total_meetings}")
        
        # # show trust_matrix each 5 ticks
        # if tick % 5 == 0:
        #     ids = sorted(h.id for h in humans)
        #     print(f"\nTick {tick} — Trust Matrix")
        #     header = "   " + "  ".join(f"{i:2d}" for i in ids)
        #     print(header)
        #     for row_id in ids:
        #         row_scores = [
        #             f"{trust_system.trust_score(row_id, col_id):.1f}"
        #             for col_id in ids
        #         ]
        #         print(f"{row_id:2d} " + "  ".join(row_scores))
#---------------------------------------------------------------------

        # draw static terrain & houses
        screen.fill((0, 0, 0))
        for y in range(H):
            for x in range(W):
                color_static_map = rev_color.get(codes[y, x], (50, 50, 50))
                pygame.draw.rect(screen, color_static_map,
                                 pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for h in houses:
            pygame.draw.rect(screen,
                             (150, 75, 0),
                             pygame.Rect(h.x*CELL_SIZE, h.y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw dynamic food & humans
        managed = map_manage(codes, food_birth, tick)
        alive_humans = [h for h in humans if h.alive]
        pixel_update(screen, managed, alive_humans, font)

        # display alive/dead counts
        display_human_counts(screen, humans, font)
        #display legend
        draw_legend(screen, CELL_SIZE, font)
        #display house storage
        display_house_storage(screen, houses, CELL_SIZE, font)

        pygame.display.flip()
        tick += 1
        clock.tick(25)

    # ─── after loop ends, export CSV ─────────────────────────────────────────
    trust_system.export_trust_matrix(humans, "/home/oussama/ants/humans/trust_matrix.csv")
    print("\n=== Final Trust Summaries ===")
    for h in humans:
        trust_system.display_trust_summary(h.id)

    pygame.quit()


if __name__ == "__main__":
    
    show_menu()
