
# simulation.py
import random
import numpy as np
from tqdm import trange
from itertools import combinations
from collections import deque
from typing import Dict, List, Tuple, Optional
from human import Human, House
from caracteristics import TrustSystem
import os
import matplotlib.pyplot as plt

from ressource import (
    map_draw, add_resource, number_of_resources, remove_resource,
    tick_resources_and_compact, active_resources,
    resource_spawn_interval_inverse,   # <- stateful: supports reset/zone_id/stock_today
)

from common import boost_house_trust, to_mate, average_trust_within_vs_between
from config import (
    INITIAL_FOOD_COUNT, FOOD_STACK, FOOD_LIFETIME,
    Nbre_HUMANS, DAY_LENGTH, ENERGY_COST, MATING_COOLDOWN,
)

# ───────────────────────── Adapters ─────────────────────────

class _ResourceProxy:
    __slots__ = ("_buf", "_idx")
    def __init__(self, buf: np.ndarray, idx: int):
        self._buf = buf; self._idx = idx
    @property
    def x(self) -> int:   return int(self._buf[self._idx, 0])
    @property
    def y(self) -> int:   return int(self._buf[self._idx, 1])
    @property
    def life(self) -> int:return int(self._buf[self._idx, 2])
    def update(self) -> None: pass
    def is_alive(self) -> bool: return self.life > 0

class NumpyResourcesView:
    def __len__(self) -> int:
        return active_resources().shape[0]
    def __iter__(self):
        buf = active_resources()
        for i in range(buf.shape[0]):
            yield _ResourceProxy(buf, i)
    def __getitem__(self, idx: int) -> _ResourceProxy:
        buf = active_resources()
        if idx < 0: idx += buf.shape[0]
        if idx < 0 or idx >= buf.shape[0]:
            raise IndexError("resource index out of range")
        return _ResourceProxy(buf, idx)
    def remove(self, item: _ResourceProxy):
        remove_resource(item.x, item.y)
# ───────────────────── helpers ──────────────────────
# Average pairwise trust over a list of human ids
def _avg_pairwise_trust(ids: List[int], ts: TrustSystem) -> float:
    n = 0
    s = 0.0
    L = len(ids)
    for i in range(L):
        for j in range(i+1, L):
            s += ts.trust_score(ids[i], ids[j])
            n += 1
    return s / n if n else 0.0
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ───────────────────── Zones & helpers ──────────────────────

def _find_food_zones(codes: np.ndarray) -> List[List[Tuple[int,int]]]:
    H, W = codes.shape
    food = (codes == 4) | (codes == 5)
    seen = np.zeros_like(food, dtype=bool)
    zones: List[List[Tuple[int,int]]] = []
    for y in range(H):
        for x in range(W):
            if not food[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            comp: List[Tuple[int,int]] = []
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and food[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            zones.append(comp)
    if len(zones) > 3:
        zones = sorted(zones, key=lambda pts: (-len(pts), sum(x for x,_ in pts)/len(pts)))[:3]
    zones.sort(key=lambda pts: (sum(x for x,_ in pts)/len(pts)))
    return zones[:3]

def _spawn_in_zone(zone_cells: List[Tuple[int,int]], how_many: int) -> int:
    spawned = 0
    max_attempts = max(10, how_many * 10)
    for _ in range(max_attempts):
        if spawned >= how_many: break
        x, y = random.choice(zone_cells)
        if number_of_resources(x, y) < FOOD_STACK:
            add_resource(x, y, life=FOOD_LIFETIME)
            spawned += 1
    return spawned

# ───────────────────── Main headless sim ────────────────────

def run_simulation(
    num_days: int,
    seed: Optional[int] = None,
    return_zone_series: bool = False,
    return_final_state: bool = False,
    progress: bool = True,          # show inner tqdm progress bar
):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    # Map & codes
    terrain_txt = map_draw('3_spots.png')
    codes       = np.loadtxt(terrain_txt, dtype=int)
    H, W        = codes.shape

    # Houses (codes 2 & 3)
    palette = map_draw.__globals__.get('NEW_PALETTE') or map_draw.__globals__['TERRAIN_PALETTE']
    rev_color = { code: rgb for rgb, code in palette.items() }
    houses: List[House] = []
    for safe_code in (2, 3):
        pts = [(x, y) for y in range(H) for x in range(W) if codes[y, x] == safe_code]
        if not pts:
            raise ValueError(f"No cells with SAFE code {safe_code} found on the map.")
        avg_x = sum(x for x, y in pts) // len(pts)
        avg_y = sum(y for x, y in pts) // len(pts)
        houses.append(House(avg_x, avg_y, rev_color.get(safe_code, (255,255,255))))

    # Zones
    zones = _find_food_zones(codes)
    if len(zones) != 3:
        raise AssertionError(f"Expected 3 zones, found {len(zones)}")
    cell_to_zone: Dict[Tuple[int,int], int] = {pos: i for i, comp in enumerate(zones) for pos in comp}

    # Seed initial food
    food_cells = list(cell_to_zone.keys())
    spawned0 = 0
    while spawned0 < INITIAL_FOOD_COUNT and food_cells:
        x, y = random.choice(food_cells)
        if number_of_resources(x, y) < FOOD_STACK:
            add_resource(x, y, life=FOOD_LIFETIME)
            spawned0 += 1

    # Humans 

    humans: List[Human] = []
    next_id = 0
    base, extra = divmod(Nbre_HUMANS, len(houses))
    extra_idx = random.randrange(len(houses)) if extra else -1
    for i, home in enumerate(houses):
        for _ in range(base + (1 if i == extra_idx else 0)):
            humans.append(Human(next_id, random.choice(['homme', 'femme']), home.x, home.y, home, codes))
            next_id += 1

    # --- Birth/death accounting (per house) ---
    initial_blue = sum(1 for hu in humans if hu.home is houses[0])
    initial_red  = sum(1 for hu in humans if hu.home is houses[1])

    born_blue_today = 0
    born_red_today  = 0
    cum_born_blue   = 0
    cum_born_red    = 0

    born_blue_daily = []   # births on that day (blue)
    born_red_daily  = []   # births on that day (red)
    dead_blue_cum   = []   # cumulative deaths up to that day (blue)
    dead_red_cum    = []   # cumulative deaths up to that day (red)

    trust_system = TrustSystem()
    last_mated: Dict[Tuple[int,int], int] = {}
    last_storage = {h: h.storage for h in houses}

    # Per-zone tracking
    TICK_WINDOW = 30
    pick_histories = [deque(maxlen=TICK_WINDOW) for _ in range(3)]
    day_spawn_acc  = [0, 0, 0]
    day_cons_acc   = [0, 0, 0]
    zone_spawned_daily  = [[], [], []]
    zone_consumed_daily = [[], [], []]

    day_cons_acc_by_house      = [[0, 0] for _ in range(3)]   # 3 zones × 2 houses
    zone_consumed_daily_by_house = [[[], []] for _ in range(3)]

    # Respawn params
    I_MAX   = [220, 200, 180]
    K_GAIN  = [0.40, 0.60, 0.80]
    I_MIN   = [10,  10,  10]
    SPAWN_COUNT = [25, 20, 15]

    resources_view = NumpyResourcesView()

    # Reset per-zone disable state (in ressource.py)
    resource_spawn_interval_inverse(0.0, reset=True)

    # Daily outputs
    days, blue_pop, red_pop = [], [], []
    within_trust, between_trust = [], []
    within_blue_trust, within_red_trust = [], [] 

    # Robust day boundaries
    assert DAY_LENGTH > 0, f"DAY_LENGTH must be > 0 (got {DAY_LENGTH})"
    total_ticks = int(num_days * DAY_LENGTH)

    iterator = trange(1, total_ticks + 1, desc="Simulating", unit="tick", disable=not progress)
    for tick in iterator:
        # Day/night (~70% day)
        cycle_pos = ((tick - 1) % DAY_LENGTH) / DAY_LENGTH
        is_day = (cycle_pos < 0.7)

        tick_resources_and_compact()
        picks_this_tick = [0, 0, 0]

        # --- humans act ---
        for h in humans:
            if not h.alive:
                continue
            picked, shared = h.step(
                resources_view,
                houses, humans, trust_system,
                is_day=is_day,
                action_cost=0.005,
                food_gain=1.0,
                decay_rate=0.01
            )
            if picked is not None:
                z = cell_to_zone.get(picked)
                if z is not None:
                    picks_this_tick[z] += 1
                    day_cons_acc[z]    += 1
                    # Track consumption by house
                    house_idx = 0 if h.home is houses[0] else 1
                    day_cons_acc_by_house[z][house_idx] += 1

            if h.home.storage > last_storage[h.home]:
                boost_house_trust(trust_system, h, humans, increment=0.1)
                last_storage[h.home] = h.home.storage

        for z in range(3):
            pick_histories[z].append(picks_this_tick[z])

        # Respawn per zone (skip disabled zones; interval=None when disabled)
        for z in range(3):
            f_avg = (sum(pick_histories[z]) / len(pick_histories[z])) if pick_histories[z] else 0.0
            interval = resource_spawn_interval_inverse(
                f_avg, I_max=I_MAX[z], k=K_GAIN[z], I_min=I_MIN[z], zone_id=z
            )
            if interval is not None and (tick % interval == 0):
                spawned = _spawn_in_zone(zones[z], SPAWN_COUNT[z])
                day_spawn_acc[z] += spawned
        # End-of-day
        if (tick % DAY_LENGTH) == 0:
            day = tick // DAY_LENGTH

            # 1) Do mating first so today's births exist before we count pop/deaths
            for h1, h2 in combinations(humans, 2):
                if h1.home is not h2.home:
                    continue
                if h1.energy < ENERGY_COST or h2.energy < ENERGY_COST:
                    continue
                pair = tuple(sorted((h1.id, h2.id)))
                if tick - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                    continue
                created, next_id = to_mate(
                    h1, h2, trust_system, humans, codes,
                    next_id, threshold=0.6, energy_cost=ENERGY_COST
                )
                if created:
                    last_mated[pair] = tick
                    # parents are same-house by the check above
                    if h1.home is houses[0]:
                        born_blue_today += 1
                    else:
                        born_red_today += 1

            # 2) Record day index
            days.append(day)

            # 3) Populations (post-birth)
            blue_alive = sum(1 for hu in humans if hu.home is houses[0] and hu.alive)
            red_alive  = sum(1 for hu in humans if hu.home is houses[1] and hu.alive)
            blue_pop.append(blue_alive)
            red_pop .append(red_alive)

            # 4) Trust (overall + within each house)
            aw, ab = average_trust_within_vs_between(humans, trust_system)
            within_trust.append(aw)
            between_trust.append(ab)

            blue_ids = [h.id for h in humans if h.alive and h.home is houses[0]]
            red_ids  = [h.id for h in humans if h.alive and h.home is houses[1]]
            within_blue_trust.append(_avg_pairwise_trust(blue_ids, trust_system))
            within_red_trust .append(_avg_pairwise_trust(red_ids,  trust_system))

            # 5) Births/deaths bookkeeping (now consistent with today's births)
            cum_born_blue += born_blue_today
            cum_born_red  += born_red_today

            dead_blue_cum.append(initial_blue + cum_born_blue - blue_alive)
            dead_red_cum .append(initial_red  + cum_born_red  - red_alive)

            born_blue_daily.append(born_blue_today)
            born_red_daily .append(born_red_today)

            # reset daily birth counters
            born_blue_today = 0
            born_red_today  = 0

            # 6) Per-zone totals (+ per-house consumption)
            for z in range(3):
                zone_spawned_daily[z].append(day_spawn_acc[z])
                zone_consumed_daily[z].append(day_cons_acc[z])
                zone_consumed_daily_by_house[z][0].append(day_cons_acc_by_house[z][0])  # blue
                zone_consumed_daily_by_house[z][1].append(day_cons_acc_by_house[z][1])  # red
                day_spawn_acc[z] = 0
                day_cons_acc[z]  = 0
                day_cons_acc_by_house[z] = [0, 0]

            # 7) Update zone enable/disable from current stock
            present = [0, 0, 0]
            A = active_resources()
            for i in range(A.shape[0]):
                x = int(A[i, 0]); y = int(A[i, 1])
                z = cell_to_zone.get((x, y))
                if z is not None and 0 <= z < 3:
                    present[z] += 1
            for z in range(3):
                resource_spawn_interval_inverse(0.0, zone_id=z, stock_today=present[z])

            # (optional) debug print
            # status = []
            # for z in range(3):
            #     disabled = (resource_spawn_interval_inverse(0.0, zone_id=z) is None)
            #     status.append("DISABLED" if disabled else "active")
            # print(f"[day {day:03}] stock per zone: Z0={present[0]} ({status[0]}), "
            #       f"Z1={present[1]} ({status[1]}), Z2={present[2]} ({status[2]}); "
            #       f"TOTAL={sum(present)} | blue={blue_alive} red={red_alive}")


            # mating once/day
            for h1, h2 in combinations(humans, 2):
                if h1.home is not h2.home: continue
                if h1.energy < ENERGY_COST or h2.energy < ENERGY_COST: continue
                pair = tuple(sorted((h1.id, h2.id)))
                if tick - last_mated.get(pair, -MATING_COOLDOWN) < MATING_COOLDOWN:
                    continue
                created, next_id = to_mate(
                    h1, h2, trust_system, humans, codes,
                    next_id, threshold=0.6, energy_cost=ENERGY_COST
                )
                if created:
                    last_mated[pair] = tick

                    # parents are same-house by check above
                    if h1.home is houses[0]:
                        born_blue_today += 1
                    else:
                        born_red_today += 1


    # assemble return values
    if return_zone_series:
        total_spawned = [sum(vals) for vals in zip(*zone_spawned_daily)] if zone_spawned_daily[0] else []
        total_picked  = [sum(vals) for vals in zip(*zone_consumed_daily)] if zone_consumed_daily[0] else []
        ret = (
            days, blue_pop, red_pop, within_trust, between_trust,
            total_spawned, total_picked,
            zone_spawned_daily, zone_consumed_daily,
            zone_consumed_daily_by_house,
            within_blue_trust, within_red_trust,
            dead_blue_cum, dead_red_cum,
            born_blue_daily, born_red_daily,
        )

    else:
        ret = (
        days, blue_pop, red_pop, within_trust, between_trust,
        within_blue_trust, within_red_trust,
        dead_blue_cum, dead_red_cum,
        born_blue_daily, born_red_daily,
        )

    if return_final_state:
        ret = (*ret, (humans, trust_system))

    return ret

# ──────────────────────── Plotting ─────────────────────────

RESULTS_DIR  = "batch_results"
COMBINED_CSV = os.path.join(RESULTS_DIR, "all_runs_combined.csv")



def plot_zone_spawn_vs_consumption_single_run(
    num_days: int = 100,
    seed: int | None = 0,
    out_dir: str = RESULTS_DIR,
    zone_names: list[str] | None = None,
) -> None:
    """Run one headless simulation and plot:
       - daily Spawn vs Consumption for each zone (3 PNGs)
       - population variation (blue & red in the same figure; 1 PNG)
       - per-house consumption per zone (one figure with 3 subplots) if available
    """
    _ensure_dir(out_dir)

    # Run once and unpack robustly (supports 9- or 10-item returns)
    result = run_simulation(num_days=num_days, seed=seed, return_zone_series=True)

    # core 5
    days, blue, red, _within, _between = result[:5]
    # next 4
    _total_spawned, _total_picked, zone_spawned_daily, zone_consumed_daily = result[5:9]
    # optional 10th: per-house consumption [zone][house_idx][day]
    zone_consumed_by_house = result[9] if len(result) >= 10 else None

    # Provide default labels if not given
    if not zone_names or len(zone_names) != 3:
        zone_names = ["Zone 0", "Zone 1", "Zone 2"]

    # --- Population variation (both houses on the same figure) ---
    plt.figure(figsize=(8, 4))
    plt.plot(days, blue, label="Blue House", linewidth=2, color ='blue')
    plt.plot(days, red,  label="Red House",  linewidth=2 , color ='red')
    plt.xlabel("Day")
    plt.ylabel("Alive humans")
    plt.title("Population Variation — Blue vs Red")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "population_variation.png"))
    plt.close()

    # --- Spawn vs Consumption per zone ---
    for z in range(3):
        plt.figure(figsize=(8, 4))
        plt.plot(days, zone_spawned_daily[z],   label="Spawn",       linewidth=1.8 , color ='green') 
        plt.plot(days, zone_consumed_daily[z],  label="Consumption", linewidth=1.8)
        plt.xlabel("Day")
        plt.ylabel("Count per day")
        plt.title(f"{zone_names[z]} — Spawn vs Consumption")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"zone{z}_spawn_vs_consumption.png"))
        plt.close()

    # --- NEW: One figure with 3 subplots showing per-house consumption in each zone ---
    if zone_consumed_by_house is not None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
        for z, ax in enumerate(axes):
            blue_cons = zone_consumed_by_house[z][0]  # house 0 (blue)
            red_cons  = zone_consumed_by_house[z][1]  # house 1 (red)
            ax.plot(days, blue_cons, label="Blue consumed", linewidth=1.8, color='blue')
            ax.plot(days, red_cons,  label="Red consumed",  linewidth=1.8 ,color='red')
            ax.set_title(f"{zone_names[z]} — per-house consumption")
            ax.set_xlabel("Day")
            ax.set_ylabel("Units / day")
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "consumption_by_house_per_zone.png"))
        plt.close()
    else:
        print("Note: per-house consumption not returned by run_simulation .")






if __name__ == "__main__":
    plot_zone_spawn_vs_consumption_single_run(num_days=400, seed=42)
    print(f"Zone spawn/consumption figures saved to: {RESULTS_DIR}")