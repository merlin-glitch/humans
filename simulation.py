
# import random
# import numpy as np
# from tqdm import trange
# from typing import Dict, List, Tuple, Optional
# from collections import deque
# from scipy.sparse import dok_matrix

# import os
# import matplotlib.pyplot as plt

# from itertools import combinations
# from human import Human, House
# from caracteristics import TrustSystem
# from ressource import (
#     map_draw, add_resource, number_of_resources, remove_resource,
#     tick_resources_and_compact, active_resources,
#     resource_spawn_interval_inverse,
# )
# from common import boost_house_trust, to_mate, average_trust_within_vs_between, run_competition
# from config import (
#     INITIAL_FOOD_COUNT, FOOD_STACK, FOOD_LIFETIME,
#     Nbre_HUMANS, DAY_LENGTH, ENERGY_COST, MATING_COOLDOWN,
# )

# # ───────────────────────── Adapters ─────────────────────────
# class _ResourceProxy:
#     __slots__ = ("_buf", "_idx")
#     def __init__(self, buf: np.ndarray, idx: int):
#         self._buf = buf
#         self._idx = idx
#     @property
#     def x(self):   return int(self._buf[self._idx, 0])
#     @property
#     def y(self):   return int(self._buf[self._idx, 1])
#     @property
#     def life(self):return int(self._buf[self._idx, 2])
#     def update(self): pass
#     def is_alive(self): return self.life > 0

# class NumpyResourcesView:
#     def __len__(self):
#         return active_resources().shape[0]

#     def __iter__(self):
#         buf = active_resources()
#         for i in range(buf.shape[0]):
#             yield _ResourceProxy(buf, i)

#     def __getitem__(self, idx: int):
#         buf = active_resources()
#         if idx < 0: idx += buf.shape[0]
#         if idx < 0 or idx >= buf.shape[0]:
#             raise IndexError("resource index out of range")
#         return _ResourceProxy(buf, idx)

#     def remove(self, item):
#         remove_resource(item.x, item.y)

# # ───────────────────── helpers ──────────────────────
# def _avg_pairwise_trust(ids: List[int], ts: TrustSystem) -> float:
#     n = 0
#     s = 0.0
#     L = len(ids)
#     for i in range(L):
#         for j in range(i+1, L):
#             s += ts.trust_score(ids[i], ids[j])
#             n += 1
#     return s / n if n else 0.0



# def _ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)

# # ───────────────────── Zones & helpers ──────────────────────
# def _find_food_zones(codes: np.ndarray, min_distance: int = 3) -> List[List[Tuple[int,int]]]:
#     H, W = codes.shape
#     food = (codes == 4) | (codes == 5)
#     seen = np.zeros_like(food, dtype=bool)
#     zones: List[List[Tuple[int,int]]] = []
#     for y in range(H):
#         for x in range(W):
#             if not food[y, x] or seen[y, x]:
#                 continue
#             stack = [(x, y)]
#             seen[y, x] = True
#             comp: List[Tuple[int,int]] = []
#             while stack:
#                 cx, cy = stack.pop()
#                 comp.append((cx, cy))
#                 for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
#                     nx, ny = cx + dx, cy + dy
#                     if 0 <= nx < W and 0 <= ny < H and food[ny, nx] and not seen[ny, nx]:
#                         seen[ny, nx] = True
#                         stack.append((nx, ny))
#             zones.append(comp)
#     zones = merge_close_zones(zones, min_distance=min_distance)
#     zones.sort(key=lambda pts: (sum(x for x,_ in pts)/len(pts)))
#     return zones

# def merge_close_zones(zones, min_distance=3):
#     merged = []
#     while zones:
#         z = zones.pop()
#         zx = [x for x,_ in z]; zy = [y for _,y in z]
#         zx1, zx2, zy1, zy2 = min(zx), max(zx), min(zy), max(zy)
#         cluster = z
#         to_remove = []
#         for other in zones:
#             ox = [x for x,_ in other]; oy = [y for _,y in other]
#             ox1, ox2, oy1, oy2 = min(ox), max(ox), min(oy), max(oy)
#             if zx1-min_distance <= ox2 and zx2+min_distance >= ox1 and zy1-min_distance <= oy2 and zy2+min_distance >= oy1:
#                 cluster += other
#                 to_remove.append(other)
#         for o in to_remove: zones.remove(o)
#         merged.append(cluster)
#     return merged

# def _spawn_in_zone(zone_cells: List[Tuple[int,int]], how_many: int) -> int:
#     spawned = 0
#     max_attempts = max(10, how_many * 10)
#     for _ in range(max_attempts):
#         if spawned >= how_many: break
#         x, y = random.choice(zone_cells)
#         if number_of_resources(x, y) < FOOD_STACK:
#             add_resource(x, y, life=FOOD_LIFETIME)
#             spawned += 1
#     return spawned

# # ───────────────────── Main headless sim ────────────────────
# def run_simulation(
#     num_days: int,
#     seed: Optional[int] = None,
#     return_zone_series: bool = False,
#     return_final_state: bool = False,
#     progress: bool = True,
# ):
#     if seed is not None:
#         random.seed(seed); np.random.seed(seed)

#     # Map & codes
#     terrain_txt = map_draw('5_spots.png')
#     codes = np.loadtxt(terrain_txt, dtype=int)
#     H, W = codes.shape

#     # Houses
#     palette = map_draw.__globals__.get('NEW_PALETTE') or map_draw.__globals__['TERRAIN_PALETTE']
#     rev_color = { code: rgb for rgb, code in palette.items() }
#     houses: List[House] = []
#     for safe_code in (2, 3):
#         pts = [(x, y) for y in range(H) for x in range(W) if codes[y, x] == safe_code]
#         avg_x = sum(x for x, y in pts) // len(pts)
#         avg_y = sum(y for x, y in pts) // len(pts)
#         houses.append(House(avg_x, avg_y, rev_color.get(safe_code, (255,255,255))))

#     # Zones
#     zones = _find_food_zones(codes)
#     N_ZONES = len(zones)
#     cell_to_zone: Dict[Tuple[int,int], int] = {pos: i for i, comp in enumerate(zones) for pos in comp}

#     # Seed food
#     food_cells = list(cell_to_zone.keys())
#     spawned0 = 0
#     while spawned0 < INITIAL_FOOD_COUNT and food_cells:
#         x, y = random.choice(food_cells)
#         if number_of_resources(x, y) < FOOD_STACK:
#             add_resource(x, y, life=FOOD_LIFETIME)
#             spawned0 += 1

#     # Humans
#     humans: List[Human] = []
#     next_id = 0
#     base, extra = divmod(Nbre_HUMANS, len(houses))
#     extra_idx = random.randrange(len(houses)) if extra else -1
#     for i, home in enumerate(houses):
#         for _ in range(base + (1 if i == extra_idx else 0)):
#             humans.append(Human(next_id, random.choice(['homme','femme']), home.x, home.y, home, codes))
#             next_id += 1

#     trust_system = TrustSystem()

#     # Birth/death
#     initial_blue = np.sum(home==0)
#     initial_red  = np.sum(home==1)
#     born_blue_today = born_red_today = 0
#     cum_born_blue = cum_born_red = 0
#     born_blue_daily, born_red_daily = [], []
#     dead_blue_cum, dead_red_cum = [], []

#     last_mated: Dict[Tuple[int,int], int] = {}
#     last_storage = {h: h.storage for h in houses}

#     # Per-zone tracking
#     TICK_WINDOW = 30
#     pick_histories = [deque(maxlen=TICK_WINDOW) for _ in range(N_ZONES)]
#     day_spawn_acc  = [0] * N_ZONES
#     day_cons_acc   = [0] * N_ZONES
#     zone_spawned_daily  = [[] for _ in range(N_ZONES)]
#     zone_consumed_daily = [[] for _ in range(N_ZONES)]
#     day_cons_acc_by_house = [[0, 0] for _ in range(N_ZONES)]
#     zone_consumed_daily_by_house = [[[], []] for _ in range(N_ZONES)]

#     # Respawn params
#     areas = np.array([len(c) for c in zones], dtype=float)
#     scale = areas / areas.mean()
#     I_MAX = (220 * scale).astype(int).clip(80, None).tolist()
#     K_GAIN = [0.6] * N_ZONES
#     I_MIN = [10] * N_ZONES
#     SPAWN_COUNT = (60 * scale).astype(int).clip(6, None).tolist()

#     resources_view = NumpyResourcesView()
#     resource_spawn_interval_inverse(0.0, reset=True)

#     # Daily outputs
#     days, blue_pop, red_pop = [], [], []
#     within_trust, between_trust = [], []
#     within_blue_trust, within_red_trust = [], []

#     total_ticks = int(num_days * DAY_LENGTH)
#     iterator = trange(1, total_ticks + 1, desc="Simulating", unit="tick", disable=not progress)

#     for tick in iterator:
#         is_day = ((tick - 1) % DAY_LENGTH) / DAY_LENGTH < 0.7

#         # # Dawn: select leaders
#         # if ((tick - 1) % DAY_LENGTH) == 0:
#         #     for fam in (0,1):
#         #         fam_members = [h for h in humans if h.alive and (0 if h.home is houses[0] else 1)==fam]
#         #         if fam_members:
#         #             run_competition(fam_members, trust_system, threshold=0.55)
        
#         # Dawn: select leaders (only red)
#         if ((tick - 1) % DAY_LENGTH) == 0:
#             red_family = [h for h in humans if h.alive and h.home is houses[1]]
#             if red_family:
#                 run_competition(red_family, trust_system, threshold=0.55)

#         tick_resources_and_compact()
#         picks_this_tick = [0] * N_ZONES

#         # Human actions
#         for h in humans:
#             if not h.alive:
#                 continue
#             picked, shared = h.step(resources_view, houses, humans, trust_system,
#                                     is_day=is_day, action_cost=0.005, food_gain=1.0, decay_rate=0.001)
#             if picked is not None:
#                 z = cell_to_zone.get(picked)
#                 if z is not None:
#                     picks_this_tick[z] += 1
#                     day_cons_acc[z] += 1
#                     idx = 0 if h.home is houses[0] else 1
#                     day_cons_acc_by_house[z][idx] += 1
#             if h.home.storage > last_storage[h.home]:
#                 boost_house_trust(trust_system, h, humans, increment=0.1)
#                 last_storage[h.home] = h.home.storage

#         for z in range(N_ZONES):
#             pick_histories[z].append(picks_this_tick[z])
#             f_avg = np.mean(pick_histories[z]) if pick_histories[z] else 0.0
#             interval = resource_spawn_interval_inverse(f_avg, I_max=I_MAX[z], k=K_GAIN[z], I_min=I_MIN[z], zone_id=z)
#             if interval is not None and (tick % interval == 0):
#                 day_spawn_acc[z] += _spawn_in_zone(zones[z], SPAWN_COUNT[z])

#         # End-of-day
#         if (tick % DAY_LENGTH) == 0:
#             day = tick // DAY_LENGTH
#             # efficient mating
#             for fam in (0,1):
#                 candidates = [h for h in humans if h.alive and (0 if h.home is houses[0] else 1)==fam and h.energy>=ENERGY_COST]
#                 random.shuffle(candidates)
#                 for i in range(0, len(candidates)-1, 2):
#                     h1, h2 = candidates[i], candidates[i+1]
#                     pair = tuple(sorted((h1.id, h2.id)))
#                     if tick - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
#                         continue
#                     created, next_id = to_mate(h1, h2, trust_system, humans, codes, next_id, threshold=0.6, energy_cost=ENERGY_COST)
#                     if created:
#                         last_mated[pair] = tick
#                         if fam==0: born_blue_today+=1
#                         else: born_red_today+=1

#             days.append(day)
#             blue_alive = sum(1 for h in humans if h.alive and h.home is houses[0])
#             red_alive  = sum(1 for h in humans if h.alive and h.home is houses[1])
#             blue_pop.append(blue_alive); red_pop.append(red_alive)

#             # trust averages
#             blue_ids = [h.id for h in humans if h.alive and h.home is houses[0]]
#             red_ids  = [h.id for h in humans if h.alive and h.home is houses[1]]
#             within_blue_trust.append(_avg_pairwise_trust(blue_ids, trust_system))
#             within_red_trust.append(_avg_pairwise_trust(red_ids, trust_system))
#             within_trust.append((within_blue_trust[-1]+within_red_trust[-1])/2)
#             if blue_ids and red_ids:
                
#                 s = 0.0
#                 n = 0
#                 for i in blue_ids:
#                     for j in red_ids:
#                         s += trust_system.trust_score(i, j)
#                         n += 1
#                 between_trust.append(s / n if n else 0.0)
#             else:
#                 between_trust.append(0.0)

#             cum_born_blue += born_blue_today; cum_born_red += born_red_today
#             dead_blue_cum.append(initial_blue + cum_born_blue - blue_alive)
#             dead_red_cum.append(initial_red + cum_born_red - red_alive)
#             born_blue_daily.append(born_blue_today); born_red_daily.append(born_red_today)
#             born_blue_today = born_red_today = 0

#             for z in range(N_ZONES):
#                 zone_spawned_daily[z].append(day_spawn_acc[z])
#                 zone_consumed_daily[z].append(day_cons_acc[z])
#                 zone_consumed_daily_by_house[z][0].append(day_cons_acc_by_house[z][0])
#                 zone_consumed_daily_by_house[z][1].append(day_cons_acc_by_house[z][1])
#                 day_spawn_acc[z]=0; day_cons_acc[z]=0; day_cons_acc_by_house[z]=[0,0]

#     if return_zone_series:
#         total_spawned = [sum(vals) for vals in zip(*zone_spawned_daily)] if zone_spawned_daily[0] else []
#         total_picked  = [sum(vals) for vals in zip(*zone_consumed_daily)] if zone_consumed_daily[0] else []
#         ret = (days, blue_pop, red_pop, within_trust, between_trust,
#                total_spawned, total_picked, zone_spawned_daily, zone_consumed_daily,
#                zone_consumed_daily_by_house, within_blue_trust, within_red_trust,
#                dead_blue_cum, dead_red_cum, born_blue_daily, born_red_daily)
#     else:
#         ret = (days, blue_pop, red_pop, within_trust, between_trust,
#                within_blue_trust, within_red_trust,
#                dead_blue_cum, dead_red_cum, born_blue_daily, born_red_daily)

#     if return_final_state:
#         ret = (*ret, (humans, trust_system))
#     return ret


# # ──────────────────────── Plotting ─────────────────────────
# RESULTS_DIR  = "batch_results"
# os.makedirs(RESULTS_DIR, exist_ok=True)
# COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")


# def plot_zone_spawn_vs_consumption_single_run(
#     num_days: int = 100,
#     seed: Optional[int] = None,
#     out_dir: str = RESULTS_DIR,
#     zone_names: Optional[list[str]] = None,
# ) -> None:
#     _ensure_dir(out_dir)
#     result = run_simulation(num_days=num_days, seed=seed, return_zone_series=True)

#     days, blue, red, _within, _between = result[:5]
#     _total_spawned, _total_picked, zone_spawned_daily, zone_consumed_daily = result[5:9]
#     zone_consumed_by_house = result[9] if len(result) >= 10 else None

#     N_ZONES = len(zone_spawned_daily)
#     if not zone_names or len(zone_names) != N_ZONES:
#         zone_names = [f"Zone {z}" for z in range(N_ZONES)]

#     # Population variation
#     plt.figure(figsize=(8, 4))
#     plt.plot(days, blue, label="Blue House", linewidth=2, color='blue')
#     plt.plot(days, red,  label="Red House",  linewidth=2, color='red')
#     plt.xlabel("Day")
#     plt.ylabel("Alive humans")
#     plt.title("Population Variation — Blue vs Red")
#     plt.grid(True, alpha=0.25)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, "population_variation.png"))
#     plt.close()

#     # Spawn vs Consumption per zone
#     for z in range(N_ZONES):
#         plt.figure(figsize=(8, 4))
#         plt.plot(days, zone_spawned_daily[z],  label="Spawn",       linewidth=1.8, color='green')
#         plt.plot(days, zone_consumed_daily[z], label="Consumption", linewidth=1.8, color='steelblue')
#         plt.xlabel("Day")
#         plt.ylabel("Count per day")
#         plt.title(f"{zone_names[z]} — Spawn vs Consumption")
#         plt.grid(True, alpha=0.25)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, f"zone{z}_spawn_vs_consumption.png"))
#         plt.close()

#     # Per-house consumption
#     if zone_consumed_by_house is not None:
#         fig, axes = plt.subplots(1, N_ZONES, figsize=(5*N_ZONES, 4), sharex=True)
#         if N_ZONES == 1:
#             axes = [axes]
#         for z, ax in enumerate(axes):
#             blue_cons = zone_consumed_by_house[z][0]
#             red_cons  = zone_consumed_by_house[z][1]
#             ax.plot(days, blue_cons, label="Blue consumed", linewidth=1.8, color='blue')
#             ax.plot(days, red_cons,  label="Red consumed",  linewidth=1.8, color='red')
#             ax.set_title(f"{zone_names[z]} — per-house consumption")
#             ax.set_xlabel("Day")
#             ax.set_ylabel("Units / day")
#             ax.grid(True, alpha=0.25)
#             ax.legend()
#         fig.tight_layout()
#         plt.savefig(os.path.join(out_dir, "consumption_by_house_per_zone.png"))
#         plt.close()
#     else:
#         print("Note: per-house consumption not returned by run_simulation.")

# if __name__ == "__main__":
#     plot_zone_spawn_vs_consumption_single_run(num_days=5, seed=42)
#     print(f"Zone spawn/consumption figures saved to: {RESULTS_DIR}")




import random
import numpy as np
from tqdm import trange
from typing import Dict, List, Tuple, Optional
from collections import deque
import os
import matplotlib.pyplot as plt

from human import Human, House
from caracteristics import TrustSystem
from common import boost_house_trust, to_mate, run_competition
from config import (
    INITIAL_FOOD_COUNT, FOOD_STACK, FOOD_LIFETIME,
    Nbre_HUMANS, DAY_LENGTH, ENERGY_COST, MATING_COOLDOWN,
)
from new_ressource import (
    identify_zones, add_resource,
    life_span_ressource, total_food, resources,
    resource_spawn_interval_inverse,extract_resource_coords_from_zones
)

# ───────────────────── helpers ──────────────────────
def _avg_pairwise_trust(ids: List[int], ts: TrustSystem) -> float:
    n, s = 0, 0.0
    L = len(ids)
    for i in range(L):
        for j in range(i+1, L):
            s += ts.trust_score(ids[i], ids[j])
            n += 1
    return s / n if n else 0.0

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ───────────────────── Main headless sim ────────────────────
def run_simulation(
    num_days: int,
    seed: Optional[int] = None,
    return_zone_series: bool = False,
    return_final_state: bool = False,
    progress: bool = True,
):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    # Zones de nourriture
    img_path = os.path.join(os.path.dirname(__file__), "images", "5_spots.png")
    zone_map, food_ids = identify_zones(img_path)
    zones = [extract_resource_coords_from_zones(zone_map, food_zone_id=zid)[0] 
            for zid in food_ids.values()]

    N_ZONES = len(zones)
    cell_to_zone: Dict[Tuple[int, int], int] = {tuple(pos): i for i, comp in enumerate(zones) for pos in comp}

    # Maisons (safe zones : code 2 et 3 dans ton NEW_PALETTE)
    houses: List[House] = []
    for safe_code in (2, 3):
        pts = np.argwhere(zone_map == safe_code)
        if len(pts) == 0:
            continue
        avg_x, avg_y = pts[:, 1].mean(), pts[:, 0].mean()
        houses.append(House(int(avg_x), int(avg_y), (0, 0, 255) if safe_code == 3 else (255, 0, 0)))

    # Seed nourriture
    food_cells = list(cell_to_zone.keys())
    spawned0 = 0
    while spawned0 < INITIAL_FOOD_COUNT and food_cells:
        y,x = random.choice(food_cells)
        if resources[y,x, 1] < FOOD_STACK:
            add_resource(x, y, life=FOOD_LIFETIME)
            spawned0 += 1

    # Humains
    humans: List[Human] = []
    next_id = 0
    base, extra = divmod(Nbre_HUMANS, len(houses))
    extra_idx = random.randrange(len(houses)) if extra else -1
    for i, home in enumerate(houses):
        for _ in range(base + (1 if i == extra_idx else 0)):
            humans.append(Human(next_id, random.choice(['homme', 'femme']), home.x, home.y, home, zone_map))
            next_id += 1

    trust_system = TrustSystem()

    # Birth/death
    born_blue_today = born_red_today = 0
    cum_born_blue = cum_born_red = 0
    born_blue_daily, born_red_daily = [], []
    dead_blue_cum, dead_red_cum = [], []
    last_mated: Dict[Tuple[int, int], int] = {}
    last_storage = {h: h.storage for h in houses}

    # Tracking zones
    TICK_WINDOW = 30
    pick_histories = [deque(maxlen=TICK_WINDOW) for _ in range(N_ZONES)]
    day_spawn_acc  = [0] * N_ZONES
    day_cons_acc   = [0] * N_ZONES
    zone_spawned_daily  = [[] for _ in range(N_ZONES)]
    zone_consumed_daily = [[] for _ in range(N_ZONES)]
    day_cons_acc_by_house = [[0, 0] for _ in range(N_ZONES)]
    zone_consumed_daily_by_house = [[[], []] for _ in range(N_ZONES)]

    # Respawn params
    areas = np.array([len(c) for c in zones], dtype=float)
    scale = areas / areas.mean()
    I_MAX = (220 * scale).astype(int).clip(80, None).tolist()
    K_GAIN = [0.6] * N_ZONES
    I_MIN = [10] * N_ZONES
    SPAWN_COUNT = (60 * scale).astype(int).clip(6, None).tolist()

    # Reset cooldowns for all food zones
    for zid in food_ids.values():
        resource_spawn_interval_inverse(0.0, zid, zone_map, reset=True)


    # Daily outputs
    days, blue_pop, red_pop = [], [], []
    within_trust, between_trust = [], []
    within_blue_trust, within_red_trust = [], []

    total_ticks = int(num_days * DAY_LENGTH)
    iterator = trange(1, total_ticks + 1, desc="Simulating", unit="tick", disable=not progress)

    for tick in iterator:
        is_day = ((tick - 1) % DAY_LENGTH) / DAY_LENGTH < 0.7

        # Dawn: leaders rouges
        if ((tick - 1) % DAY_LENGTH) == 0:
            red_family = [h for h in humans if h.alive and h.home is houses[1]]
            if red_family:
                run_competition(red_family, trust_system, threshold=1)

        # Tick ressources
        life_span_ressource()
        picks_this_tick = [0] * N_ZONES

        # Actions des humains
        for h in humans:
            if not h.alive:
                continue
            picked, shared = h.step(resources, houses, humans, trust_system,
                                    is_day=is_day, action_cost=0.005, food_gain=1.0, decay_rate=0.001)
            if picked is not None:
                z = cell_to_zone.get(picked)
                if z is not None:
                    picks_this_tick[z] += 1
                    day_cons_acc[z] += 1
                    idx = 0 if h.home is houses[0] else 1
                    day_cons_acc_by_house[z][idx] += 1
            if h.home.storage > last_storage[h.home]:
                boost_house_trust(trust_system, h, humans, increment=0.1)
                last_storage[h.home] = h.home.storage

        # Respawn nourriture
        for z in range(N_ZONES):
            pick_histories[z].append(picks_this_tick[z])
            f_avg = np.mean(pick_histories[z]) if pick_histories[z] else 0.0
            interval = resource_spawn_interval_inverse(
                f_avg,
                zone_id=z,
                zone_map=zone_map,
                I_max=I_MAX[z],
                k=K_GAIN[z],
                I_min=I_MIN[z]
            )
            if interval is not None and (tick % interval == 0):
                spawned = 0
                for _ in range(SPAWN_COUNT[z]):
                    y, x = random.choice(zones[z])
                    add_resource(x, y, life=FOOD_LIFETIME)
                    spawned += 1
                day_spawn_acc[z] += spawned

        # End-of-day
        if (tick % DAY_LENGTH) == 0:
            day = tick // DAY_LENGTH

            # Mating
            for fam in (0, 1):
                candidates = [h for h in humans if h.alive and (0 if h.home is houses[0] else 1) == fam and h.energy >= ENERGY_COST]
                random.shuffle(candidates)
                for i in range(0, len(candidates)-1, 2):
                    h1, h2 = candidates[i], candidates[i+1]
                    pair = tuple(sorted((h1.id, h2.id)))
                    if tick - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                        continue
                    created, next_id = to_mate(h1, h2, trust_system, humans, zone_map, next_id, threshold=0.6, energy_cost=ENERGY_COST)
                    if created:
                        last_mated[pair] = tick
                        if fam == 0: born_blue_today += 1
                        else: born_red_today += 1

            days.append(day)
            blue_alive = sum(1 for h in humans if h.alive and h.home is houses[0])
            red_alive  = sum(1 for h in humans if h.alive and h.home is houses[1])
            blue_pop.append(blue_alive); red_pop.append(red_alive)

            # Trust
            blue_ids = [h.id for h in humans if h.alive and h.home is houses[0]]
            red_ids  = [h.id for h in humans if h.alive and h.home is houses[1]]
            within_blue_trust.append(_avg_pairwise_trust(blue_ids, trust_system))
            within_red_trust.append(_avg_pairwise_trust(red_ids, trust_system))
            within_trust.append((within_blue_trust[-1] + within_red_trust[-1]) / 2)
            if blue_ids and red_ids:
                s = 0.0; n = 0
                for i in blue_ids:
                    for j in red_ids:
                        s += trust_system.trust_score(i, j)
                        n += 1
                between_trust.append(s / n if n else 0.0)
            else:
                between_trust.append(0.0)

            dead_blue_cum.append(cum_born_blue + born_blue_today - blue_alive)
            dead_red_cum.append(cum_born_red + born_red_today - red_alive)
            born_blue_daily.append(born_blue_today); born_red_daily.append(born_red_today)
            cum_born_blue += born_blue_today; cum_born_red += born_red_today
            born_blue_today = born_red_today = 0

            for z in range(N_ZONES):
                zone_spawned_daily[z].append(day_spawn_acc[z])
                zone_consumed_daily[z].append(day_cons_acc[z])
                zone_consumed_daily_by_house[z][0].append(day_cons_acc_by_house[z][0])
                zone_consumed_daily_by_house[z][1].append(day_cons_acc_by_house[z][1])
                day_spawn_acc[z] = 0; day_cons_acc[z] = 0; day_cons_acc_by_house[z] = [0, 0]

    if return_zone_series:
        total_spawned = [sum(vals) for vals in zip(*zone_spawned_daily)] if zone_spawned_daily[0] else []
        total_picked  = [sum(vals) for vals in zip(*zone_consumed_daily)] if zone_consumed_daily[0] else []
        ret = (days, blue_pop, red_pop, within_trust, between_trust,
               total_spawned, total_picked, zone_spawned_daily, zone_consumed_daily,
               zone_consumed_daily_by_house, within_blue_trust, within_red_trust,
               dead_blue_cum, dead_red_cum, born_blue_daily, born_red_daily)
    else:
        ret = (days, blue_pop, red_pop, within_trust, between_trust,
               within_blue_trust, within_red_trust,
               dead_blue_cum, dead_red_cum, born_blue_daily, born_red_daily)

    if return_final_state:
        ret = (*ret, (humans, trust_system))
    return ret

# ──────────────────────── Plotting ─────────────────────────
RESULTS_DIR  = "bath_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_zone_spawn_vs_consumption_single_run(
    num_days: int = 100,
    seed: Optional[int] = None,
    out_dir: str = RESULTS_DIR,
    zone_names: Optional[list[str]] = None,
) -> None:
    _ensure_dir(out_dir)
    result = run_simulation(num_days=num_days, seed=seed, return_zone_series=True)

    days, blue, red, _within, _between = result[:5]
    _total_spawned, _total_picked, zone_spawned_daily, zone_consumed_daily = result[5:9]
    zone_consumed_by_house = result[9] if len(result) >= 10 else None

    N_ZONES = len(zone_spawned_daily)
    if not zone_names or len(zone_names) != N_ZONES:
        zone_names = [f"Zone {z}" for z in range(N_ZONES)]

    # Population
    plt.figure(figsize=(8, 4))
    plt.plot(days, blue, label="Blue House", linewidth=2, color='blue')
    plt.plot(days, red,  label="Red House",  linewidth=2, color='red')
    plt.xlabel("Day"); plt.ylabel("Alive humans")
    plt.title("Population Variation — Blue vs Red")
    plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "population_variation.png"))
    plt.close()

    # Spawn vs Consumption
    for z in range(N_ZONES):
        plt.figure(figsize=(8, 4))
        plt.plot(days, zone_spawned_daily[z],  label="Spawn",       linewidth=1.8, color='green')
        plt.plot(days, zone_consumed_daily[z], label="Consumption", linewidth=1.8, color='steelblue')
        plt.xlabel("Day"); plt.ylabel("Count per day")
        plt.title(f"{zone_names[z]} — Spawn vs Consumption")
        plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"zone{z}_spawn_vs_consumption.png"))
        plt.close()

    # Per-house consumption
    if zone_consumed_by_house is not None:
        fig, axes = plt.subplots(1, N_ZONES, figsize=(5*N_ZONES, 4), sharex=True)
        if N_ZONES == 1:
            axes = [axes]
        for z, ax in enumerate(axes):
            blue_cons = zone_consumed_by_house[z][0]
            red_cons  = zone_consumed_by_house[z][1]
            ax.plot(days, blue_cons, label="Blue consumed", linewidth=1.8, color='blue')
            ax.plot(days, red_cons,  label="Red consumed",  linewidth=1.8, color='red')
            ax.set_title(f"{zone_names[z]} — per-house consumption")
            ax.set_xlabel("Day"); ax.set_ylabel("Units / day")
            ax.grid(True, alpha=0.25); ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "consumption_by_house_per_zone.png"))
        plt.close()

if __name__ == "__main__":
    plot_zone_spawn_vs_consumption_single_run(num_days=100, seed=42)
    print(f"Zone spawn/consumption figures saved to: {RESULTS_DIR}")
