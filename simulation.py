
# simulations_fast.py
import random
import numpy as np
from tqdm import trange
from itertools import combinations

from human import Human, House
from caracteristics import TrustSystem
from ressource import Resource, map_draw, resources, resource_array, number_of_resources
from common import (
    boost_house_trust,
    to_mate,
    average_trust_within_vs_between,
)

# bring in your constants
from config import (
    MAP_WIDTH, MAP_HEIGHT, CELL_SIZE,
    INITIAL_FOOD_COUNT, FOOD_STACK, FOOD_LIFETIME,
    SPAWN_INTERVAL, FOOD_SPAWN_COUNT, Nbre_HUMANS
)



def run_simulation(num_days: int):
    """
    Run headlessly for `num_days`, returning:
      - days:            List[int]
      - blue_pop:        List[int]
      - red_pop:         List[int]
      - within_trust:    List[float]
      - between_trust:   List[float]
    """
    # ── SETUP ────────────────────────────────────────────────────────
    terrain_txt = map_draw('easy_food.png')
    codes       = np.loadtxt(terrain_txt, dtype=int)
    H, W        = codes.shape

    # find the two houses (code 2=blue,3=red)
    terrain_palette = map_draw.__globals__['TERRAIN_PALETTE']
    # TERRAIN_PALETTE maps RGB→code, so invert to get code→RGB
    rev_color = { code: rgb for rgb, code in terrain_palette.items() }
    houses = []
    for safe_code in (2,3):
        pts = [(x,y) for y in range(H) for x in range(W) if codes[y,x]==safe_code]
        avg_x = sum(x for x,y in pts)//len(pts)
        avg_y = sum(y for x,y in pts)//len(pts)
        houses.append(House(avg_x, avg_y, rev_color[safe_code]))

    # seed initial resources
    food_zones = [(x,y) for y in range(H) for x in range(W) if codes[y,x] in {4,5}]
    food_birth =  {}

    total_resources  = 0
    while total_resources < INITIAL_FOOD_COUNT:
        x,y = random.choice(food_zones)
        if (number_of_resources(x, y)==0):
            resources[total_resources, :] = [x, y, FOOD_LIFETIME]
            food_birth[(x,y)] = 0
            total_resources += 1

    # spawn humans evenly:
    humans = []
    humanid = 0
    base, extra = divmod(Nbre_HUMANS, len(houses))
    extra_idx = random.randrange(len(houses)) if extra else -1
    for i, home in enumerate(houses):
        for _ in range(base + (1 if i==extra_idx else 0)):
            humans.append(Human(
                human_id=humanid,
                sex=random.choice(['homme','femme']),
                x=home.x, y=home.y,
                home=home, codes=codes
            ))
            humanid += 1

    trust_system    = TrustSystem()
    DAY_LENGTH      = 500
    ENERGY_COST     = 5.0
    MATING_COOLDOWN = DAY_LENGTH  # 1 day cooldown
    last_mated      = {}
    next_id         = humanid
    last_storage    = {h: h.storage for h in houses}

    # ── STORAGE FOR DAILY OUTPUT ────────────────────────────────────
    days            = []
    blue_pop        = []
    red_pop         = []
    within_trust    = []
    between_trust   = []

    total_ticks = num_days * DAY_LENGTH

    # ── MAIN LOOP (no graphics) ────────────────────────────────────
    for tick in trange(total_ticks, desc="Simulating", unit="tick"):
        # — spawn new food periodically
        if tick % SPAWN_INTERVAL == 0:
            for _ in range(FOOD_SPAWN_COUNT):
                x,y = random.choice(food_zones)

                if number_of_resources(x, y) < FOOD_STACK:
                    resource_array[total_resources, :] = [x, y, FOOD_LIFETIME]
                    food_birth[(x,y)] = tick

        resources = resource_array[total_resources, :]

        # — update existing food lifetimes

        resources[:, 2] = -1

        #We remove all elements not used
        indices_to_remove = np.where(resources[:, 2] == 0)[0] # We get the first coordinates ( index ) of the ressource qithout lifetime of 0
        vectors_to_remove = resources [indices_to_remove]
        #We pu everything in end position in place of the elements to removes
        end_positions = np.arange(total_resources - indices_to_remove.size, total_resources)
        resources[indices_to_remove] = resources[end_positions]

        total_resources -= indices_to_remove.size
        
        #new dimension without consumed resources
        resources = resource_array[total_resources, :]
   
        # — each human acts
        for h in humans:
            if not h.alive:
                continue
            picked, shared = h.step(
                resources, houses, humans, trust_system,
                is_day=True,
                action_cost=0.01,
                food_gain=1.0,
                decay_rate=0.0005
            )
            # detect deposit → boost
            if h.home.storage > last_storage[h.home]:
                boost_house_trust(trust_system, h, humans, increment=0.1)
                last_storage[h.home] = h.home.storage

        # — end‑of‑day processing
        if tick and (tick % DAY_LENGTH) == 0:
            day = tick // DAY_LENGTH
            days.append(day)

            # populations
            blue_pop.append(sum(1 for h in humans if h.home is houses[0] and h.alive))
            red_pop .append(sum(1 for h in humans if h.home is houses[1] and h.alive))

            # record trust
            aw, ab = average_trust_within_vs_between(humans, trust_system)
            within_trust .append(aw)
            between_trust.append(ab)

            # perform mating once per day
            for h1,h2 in combinations(humans,2):
                if h1.home is not h2.home: 
                    continue
                if h1.energy < ENERGY_COST or h2.energy < ENERGY_COST:
                    continue
                pair = tuple(sorted((h1.id,h2.id)))
                if tick - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                    continue
                created, next_id = to_mate(
                    h1, h2, trust_system, humans, codes,
                    next_id,
                    threshold=0.6,
                    energy_cost=ENERGY_COST
                )
                if created:
                    last_mated[pair] = tick

    # ── return the five aligned series ───────────────────────────────
    return days, blue_pop, red_pop, within_trust, between_trust


if __name__ == "__main__":
    run_simulation(100)
