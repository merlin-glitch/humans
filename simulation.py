
import random
from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict

from human import Human, House
from caracteristics import TrustSystem
from ressource import Resource, map_draw
from common import (
    boost_house_trust,
    to_mate,
    average_trust_within_vs_between,
    run_competition,
)

from config import (
    INITIAL_FOOD_COUNT, FOOD_STACK, FOOD_LIFETIME,
    SPAWN_INTERVAL, FOOD_SPAWN_COUNT, Nbre_HUMANS,
    MAP_WIDTH, MAP_HEIGHT, CELL_SIZE,
    DAY_LENGTH, ENERGY_COST, MATING_COOLDOWN
)

def resource_spawn_interval_inverse(f_avg: float, I_max: int = 200, k: float = 0.5, I_min: int = 10) -> int:
    interval = I_max / (1 + k * f_avg)
    return max(I_min, int(interval))

def run_simulation(num_days: int, seed: int = None) -> Tuple[List[int], List[int], List[int], List[float], List[float]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    terrain_txt = map_draw('3_spots.png')
    codes = np.loadtxt(terrain_txt, dtype=int)
    H, W = codes.shape
    rev_color = {code: rgb for rgb, code in map_draw.__globals__['NEW_PALETTE'].items()}

    houses: List[House] = []
    for safe_code in (2, 3):
        pts = [(x, y) for y in range(H) for x in range(W) if codes[y, x] == safe_code]
        if pts:
            avg_x = sum(x for x, y in pts) // len(pts)
            avg_y = sum(y for x, y in pts) // len(pts)
            houses.append(House(avg_x, avg_y, rev_color[safe_code]))

    food_zones = [(x, y) for y in range(H) for x in range(W) if codes[y, x] in {4, 5}]
    resources: List[Resource] = []
    food_count = defaultdict(int)
    food_birth = {}
    while len(resources) < INITIAL_FOOD_COUNT:
        x, y = random.choice(food_zones)
        if food_count[(x, y)] < FOOD_STACK:
            resources.append(Resource(x, y, life=FOOD_LIFETIME))
            food_birth[(x, y)] = 0
            food_count[(x, y)] += 1

    humans: List[Human] = []
    humanid = 0
    base, extra = divmod(Nbre_HUMANS, len(houses))
    extra_idx = random.randrange(len(houses)) if extra else -1
    for i, house in enumerate(houses):
        count = base + (1 if i == extra_idx else 0)
        for _ in range(count):
            humans.append(Human(
                human_id=humanid,
                sex=random.choice(['homme', 'femme']),
                x=house.x, y=house.y,
                home=house, codes=codes
            ))
            humanid += 1

    trust_system = TrustSystem()
    next_id = humanid
    last_storage = {h: h.storage for h in houses}
    last_mated: Dict[Tuple[int, int], int] = {}
    human_by_id = {h.id: h for h in humans}

    tick = 0
    day_tick = DAY_LENGTH * 0.3
    prev_is_day = False
    pick_history = []
    pick_total = 0

    deaths_today = []
    births_by_pair = {}

    days, blue_pop, red_pop, within_trust, between_trust = [], [], [], [], []
    total_ticks = num_days * DAY_LENGTH

    food_spawned = 0
    food_picked = 0
    food_spawn_history = []
    food_pick_history = []

    while tick < total_ticks:


        day_tick = (day_tick + 1) % DAY_LENGTH
        cycle_pos = day_tick / DAY_LENGTH
        is_day = cycle_pos < 0.7

        if not prev_is_day and is_day:
            for house in houses:
                family = [h for h in humans if h.home is house and h.alive]
                run_competition(family, trust_system)
        prev_is_day = is_day

        for r in resources:
            r.update()
        resources = [r for r in resources if r.is_alive()]
        food_count.clear()
        for r in resources:
            food_count[(r.x, r.y)] += 1

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
                food_picked += 1

            new_storage = h.home.storage
            if new_storage > last_storage[h.home]:
                boost_house_trust(trust_system, h, humans, increment=0.1)
                last_storage[h.home] = new_storage

            lid = getattr(h, "_last_night_leader", None)
            if lid is not None and h.bag > 0:
                share = h.bag * 0.1
                h.bag -= share
                leader = human_by_id.get(lid)
                if leader:
                    leader.bag += share
                    trust_system.increase_trust(h.id, lid, increment=0.01)
                del h._last_night_leader

        pick_total += picked_this_tick
        pick_history.append(picked_this_tick)
        if len(pick_history) > 30:
            pick_total -= pick_history.pop(0)
        avg_pick_rate = pick_total / len(pick_history)
        current_interval = resource_spawn_interval_inverse(avg_pick_rate)

        if tick % current_interval == 0:
            for _ in range(FOOD_SPAWN_COUNT):
                x, y = random.choice(food_zones)
                if food_count[(x, y)] < FOOD_STACK:
                    resources.append(Resource(x, y, life=FOOD_LIFETIME))
                    food_birth[(x, y)] = tick
                    food_count[(x, y)] += 1
                    food_spawned += 1  #

        for house in houses:
            eligible = [h for h in humans if h.alive and h.home is house and h.energy >= ENERGY_COST]
            for i in range(len(eligible)):
                for j in range(i + 1, len(eligible)):
                    h1, h2 = eligible[i], eligible[j]
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

        if tick % DAY_LENGTH == 0 and tick != 0:
            day = tick // DAY_LENGTH
            alive = [h for h in humans if h.alive]
            days.append(day)
            blue_pop.append(sum(1 for h in alive if h.home is houses[0]))
            red_pop.append(sum(1 for h in alive if h.home is houses[1]))
            aw, ab = average_trust_within_vs_between(humans, trust_system)
            food_spawn_history.append(food_spawned)
            food_pick_history.append(food_picked)
            food_spawned = 0
            food_picked = 0

            within_trust.append(aw)
            between_trust.append(ab)
            deaths_today.clear()
            births_by_pair.clear()

        tick += 1

    return days, blue_pop, red_pop, within_trust, between_trust, food_spawn_history, food_pick_history
