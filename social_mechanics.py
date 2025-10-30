"""
common.py - Shared utilities for simulation logic

Part of the Human Society Simulation project.

Contains trust boosting, mating mechanics, competition resolution,
CSV export functions, and plotting helpers.
"""

# Standard library imports
import csv
import os
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

# Third-party imports
import matplotlib.pyplot as plt
import pygame

# Local imports
from human import Human, House
from trust_system import TrustSystem
# Avoid module-level import to prevent circular dependency; import in-function when needed


def boost_house_trust(
    trust_system: TrustSystem,
    contributor: Human,
    humans: List[Human],
    increment: float = 0.001,
) -> None:
    """
    Increase trust in a contributor among their housemates.
    
    When an agent deposits food into their house storage, this function
    increases the trust that all other housemates have for the contributor.
    This simulates gratitude and recognition for contributions to the group.
    
    Args:
        trust_system: The trust system managing relationships
        contributor: The human who made a contribution (deposited food)
        humans: List of all humans in the simulation
        increment: Amount to increase trust by (default 0.001)
        
    Example:
        >>> # When human deposits food to house storage
        >>> boost_house_trust(trust_system, contributor, all_humans, 0.1)
        >>> # All housemates now trust the contributor more
    """
    trust_system.init_human(contributor.id)

    home = contributor.home
    ids = [r.id for r in humans if r is not contributor and r.home is home]
    if not ids:
        return

    for rid in ids:
        trust_system.increase_trust(
            trustor_id=rid,
            trustee_id=contributor.id,
            increment=increment,
            refresh=False,  # defer recompute
        )






def export_trust_matrix(
    trust_system: TrustSystem,
    human_list: List[Human],
    filename: str = "trust_matrix.csv"
) -> None:
    """
    Export the complete trust matrix to a CSV file.

    Creates a CSV file where each row represents a trustor (agent whose trust
    we're measuring) and each column represents a trustee (agent being trusted).
    The values are continuous trust scores ranging from 0.0 to 1.0.

    Args:
        trust_system: The trust system containing all relationship data
        human_list: List of human agents to include in the matrix
        filename: Output CSV filename (default: "trust_matrix.csv")

    Example:
        >>> export_trust_matrix(trust_system, humans, "my_trust_matrix.csv")
        >>> # Creates a CSV with human IDs as headers and trust scores as values
    """
    ids = sorted(h.id for h in human_list)
    id_to_human = {h.id: h for h in human_list}
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + ids)
        for row_id in ids:
            row = [row_id]
            for col_id in ids:
                if row_id == col_id:
                    # Use per-agent self-confidence as self-trust; default neutral 0.5
                    h = id_to_human.get(row_id)
                    self_trust = getattr(h, "self_confidence", 0.5) if h is not None else 0.5
                    row.append(f"{float(self_trust):.3f}")
                else:
                    row.append(f"{trust_system.trust_score(row_id, col_id):.3f}")
            writer.writerow(row)


def _avg_pairwise_trust(ids: List[int], trust_system: TrustSystem) -> float:
    """
    Calculate the average pairwise trust score for a group of agents.

    Computes the mean trust score across all ordered pairs (i, j) where i != j
    within the given list of agent IDs. This metric represents the overall
    level of trust within a group.

    Args:
        ids: List of human agent IDs to analyze
        trust_system: Trust system containing relationship data

    Returns:
        Average trust score (0.0 to 1.0), or 0.0 if fewer than 2 agents

    Example:
        >>> blue_ids = [1, 2, 3, 4]
        >>> avg_trust = _avg_pairwise_trust(blue_ids, trust_system)
        >>> print(f"Average trust within blue family: {avg_trust:.3f}")
    """
    if len(ids) < 2:
        return 0.0
    s = 0.0
    n = 0
    for i in ids:
        for j in ids:
            if i == j:
                continue
            s += trust_system.trust_score(i, j)
            n += 1
    return s / n if n else 0.0


def average_trust_per_house(
    humans: List[Human],
    trust_system: TrustSystem
) -> Dict[Tuple[int,int], float]:
    """
    For each unique house coordinate, compute the mean trust score
    between every *ordered* pair of distinct humans sharing that house.
    Returns a dict: (house.x, house.y) -> avg_trust.
    """
    # group by house
    by_house: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for h in humans:
        by_house[(h.home.x, h.home.y)].append(h.id)

    avg: Dict[Tuple[int,int], float] = {}
    for house_coord, ids in by_house.items():
        scores = []
        for i in ids:
            for j in ids:
                if i == j: continue
                scores.append(trust_system.trust_score(i, j))
        avg[house_coord] = sum(scores) / len(scores) if scores else 0.0
    return avg

def average_trust_within_vs_between(
    humans: List[Human],
    trust_system: TrustSystem
) -> Tuple[float, float]:
    """
    Returns (avg_within, avg_between):
      avg_within  = mean trust for all ordered pairs (i,j) in same house
      avg_between = mean trust for all ordered pairs (i,j) in different houses
    """
    within, between = [], []
    for h1 in humans:
        for h2 in humans:
            if h1.id == h2.id: continue
            score = trust_system.trust_score(h1.id, h2.id)
            if h1.home is h2.home:
                within.append(score)
            else:
                between.append(score)
    aw = sum(within)/len(within) if within else 0.0
    ab = sum(between)/len(between) if between else 0.0
    return aw, ab


def sharing_counts_within_vs_between(
    humans: List[Human],
    trust_system: TrustSystem
) -> Tuple[int, int]:
    """
    Returns (shares_within, shares_between):
      shares_X = sum of succ counts in pair_stats for those pairs.
      Only counts pairs where both trustor and trustee are still in `humans`.
    """
    # map each known human ID to its home
    id_to_home = {h.id: h.home for h in humans}
    within = between = 0

    for trustor_id, data in trust_system.hints.items():
        # skip trustors no longer in humans
        home_tor = id_to_home.get(trustor_id)
        if home_tor is None:
            continue

        for trustee_id, (succ, _) in data["pair_stats"].items():
            # skip trustees no longer in humans
            home_tee = id_to_home.get(trustee_id)
            if home_tee is None or trustor_id == trustee_id:
                continue

            if home_tor is home_tee:
                within += succ
            else:
                between += succ

    return within, between


# ─── now the plotting helpers ────────────────────────────────────────

def plot_avg_trust_per_house(humans, trust_system):
    """
    Plot the average trust per house as a bar chart, coloring each bar
    by its house’s RGB and labeling each bar’s x‐tick with its coordinate.
    """
    from collections import OrderedDict
    import matplotlib.pyplot as plt

    # 1) compute & sort
    avg = average_trust_per_house(humans, trust_system)
    avg = OrderedDict(sorted(avg.items()))
    coords = list(avg.keys())          # e.g. [(74,14), (73,32)]
    values = list(avg.values())

    # 2) map coord → house color
    coord_to_color = {}
    for h in humans:
        coord = (h.home.x, h.home.y)
        if coord in avg:
            coord_to_color.setdefault(coord, h.home.color)

    # 3) normalize colors
    bar_colors = [
        tuple(c/255.0 for c in coord_to_color[coord])
        for coord in coords
    ]

    # 4) plot
    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values, color=bar_colors)

    # 5) label x‐ticks with coordinates
    ax.set_xticks(range(len(coords)))
    # Show color name ("Blue" or "Red") instead of RGB
    def color_name(rgb):
        if rgb == (0, 0, 128):
            return "Blue"
        elif rgb == (255, 0, 0):
            return "Red"
        else:
            return str(rgb)
    ax.set_xticklabels([color_name(coord_to_color[coord]) for coord in coords])

    ax.set_xlabel("Houses")
    ax.set_ylabel("Average trust")
    ax.set_title("Average trust among members of each house")
    plt.tight_layout()
    plt.show()

def plot_within_vs_between_trust(humans, trust_system):
    aw, ab = average_trust_within_vs_between(humans, trust_system)
    plt.figure()
    plt.bar(["Within", "Between"], [aw, ab])
    plt.ylabel("Average trust")
    plt.title("Within‑ vs Between‑house trust")
    plt.show()

def plot_sharing_counts(humans, trust_system):
    w, b = sharing_counts_within_vs_between(humans, trust_system)
    plt.figure()
    plt.bar(["Within", "Between"], [w, b])
    plt.ylabel("Total successful shares")
    plt.title("Food‑sharing count: Within vs Between houses")
    plt.show()




def export_house_contributions(humans, house, filename, col_name):
    """
    Exporte dans `filename` un CSV à deux colonnes :
      human_id_from_house_X, col_name
    Ne contient que les humains dont h.home est exactement `house`.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"human_id_from_house_{1 if col_name.endswith('1') else 2}", col_name])
        for h in humans:
            if h.home is house:
                writer.writerow([h.id, getattr(h, 'contributed', 0)])

def print_contribution_board(humans: list[Human]):
    """
    Nicely print to stdout each human’s contribution.
    """
    print(f"{'ID':>3}  {'House':>7}  {'Contrib':>8}")
    print("-"*24)
    for h in humans:
        house_coord = f"({h.home.x},{h.home.y})"
        print(f"{h.id:3d}  {house_coord:>7}  {h.contributed:8d}")





def to_mate(
    h1: Human,
    h2: Human,
    trust_system: TrustSystem,
    humans: List[Human],
    codes,                # your numpy map array
    next_id: int,
    threshold: float = 0.65,
    energy_cost: float = 5.0
) -> Tuple[int, int]:
    """
    Attempt mating between two humans if conditions are met.
    
    Checks if two humans can mate based on mutual trust, energy requirements,
    and cohabitation. If successful, creates 1-2 offspring and adds them to
    the population. Both parents pay energy cost regardless of number of children.
    
    Args:
        h1: First potential parent
        h2: Second potential parent  
        trust_system: Trust system for checking mutual trust
        humans: List of all humans (children will be added here)
        codes: Terrain map array for child placement
        next_id: Next available human ID for children
        threshold: Minimum mutual trust required (default 0.7)
        energy_cost: Energy cost for each parent (default 5.0)
        
    Returns:
        Tuple of (num_children_created, updated_next_id)
        
    Example:
        >>> children, next_id = to_mate(human1, human2, trust, all_humans, 
        ...                           zone_map, next_id, threshold=0.8)
        >>> print(f"Created {children} children, next ID is {next_id}")
    """

    # 1) must be co‑residents
    if h1.home is not h2.home:
        return 0, next_id

    # 1a) optional adjacency check (uncomment if you care about physical proximity)
    # if max(abs(h1.x-h2.x), abs(h1.y-h2.y)) > 1:
    #     return 0, next_id

    # 2) mutual trust check - both must trust each other above threshold
    t12 = trust_system.trust_score(h1.id, h2.id)
    t21 = trust_system.trust_score(h2.id, h1.id)
    if t12 < threshold or t21 < threshold:  # default threshold = 0.7 (high trust)
        return 0, next_id

    # 3) energy check
    if h1.energy < energy_cost or h2.energy < energy_cost:
        return 0, next_id

    # 4) pay the energy cost
    h1.energy -= energy_cost
    h2.energy -= energy_cost

    # 5) decide how many kids (1 or 2) - random choice simulates natural variation
    num_kids = random.choice([1, 2])
    for _ in range(num_kids):
        sex = random.choice(['homme', 'femme'])
        # place newborn at parents' home center
        child = Human(
            human_id=next_id,
            sex=sex,
            x=h1.home.x,   # or h1.home_x for exact center
            y=h1.home.y,
            home=h1.home,
            codes=codes
        )
        humans.append(child)

        # register the new child in the trust system
        trust_system.init_human(next_id)

        next_id += 1

    return num_kids, next_id



def log_daily_population(
    humans: List[Human],
    houses: List[House],
    day: int,
    births_by_pair: Dict[Tuple[int,int], int],
    deaths_today: List[int]
) -> None:
    """
    Append one row per house to its CSV file.
    Each row has: day, initial_population, births, deaths.
    """
    # map human_id → house for quick lookups
    id_to_home = {h.id: h.home for h in humans}

    # for each house, compute stats
    for i, house in enumerate(houses, start=1):
        # 1) initial pop at start of this day
        init_pop = sum(1 for h in humans if h.home is house)

        # 2) births: sum up from any pair that share this same house
        births = 0
        for (a, b), cnt in births_by_pair.items():
            # if *both* parents lived here that day, we count their kids here
            if id_to_home.get(a) is house and id_to_home.get(b) is house:
                births += cnt

        # 3) deaths: how many of today's deaths belonged to this house?
        deaths = sum(
            1
            for dead_id in deaths_today
            if id_to_home.get(dead_id) is house
        )

        # 4) write to CSV (filename per‐house, e.g. 'daily_stats_<color>.csv')
        filename = f"daily_stats_{house.color[0]}_{house.color[1]}_{house.color[2]}.csv"
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([day, init_pop, births, deaths])
            
def log_population_by_house(
    day: int,
    humans: List[Human],
    house: House,
    filename: str
) -> None:
    """
    Appends a line (day, population) to `filename`.
    If the file did not exist or was empty, first writes a header row.
    """
    # count alive humans in that house
    pop = sum(1 for h in humans if h.home is house and h.alive)

    is_new = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["day", "population"])
        writer.writerow([day, pop])



def load_population_series(filename: str) -> Tuple[List[int],List[int]]:
    days = []
    pops = []
    import csv
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(int(row["day"]))
            pops.append(int(row["population"]))
    return days, pops

def plot_population_variation(blue_file: str, red_file: str) -> None:
    days_b, pop_b = load_population_series(blue_file)
    days_r, pop_r = load_population_series(red_file)
    # assume days_b == days_r
    x = range(len(days_b))
    width = 0.4
    plt.figure()
    plt.bar([i - width/2 for i in x], pop_b, width, label="Blue", color="blue")
    plt.bar([i + width/2 for i in x], pop_r, width, label="Red",  color="red")
    plt.xticks(x, days_b)
    plt.xlabel("Day")
    plt.ylabel("Alive population")
    plt.title("Population variation over time")
    plt.legend()
    plt.show()



def run_competition(
    family: List[Human],
    trust_system: TrustSystem,
    threshold: float = 0.55
) -> None:
    """
    Run leadership competition within a family group.
    
    Each family member evaluates others based on trust scores. Members
    with trust scores above the threshold become potential leaders.
    If a trusted member has a valid memory_spot (resource location),
    other members will follow them the next day.
    
    Args:
        family: List of humans in the same house
        trust_system: Trust system for evaluating relationships
        threshold: Minimum trust score to become a leader (default 0.55)
        
    Note:
        This simulates social hierarchy formation through trust-based
        leadership selection. Only affects the red house family.
        
    Example:
        >>> red_family = [h for h in humans if h.home.color == (255, 0, 0)]
        >>> run_competition(red_family, trust_system, threshold=0.6)
    """
 
    for member in family:
        # 1) On récupère la paire (autre, son score) pour tous les co‐résidents
        scores = [
            (other, trust_system.trust_score(member.id, other.id))
            for other in family
            if other is not member
        ]
        if not scores:
            # seul dans la maison
            member.next_day_target = None
            member._last_night_leader = None
            continue

        # 2) On trouve celui en qui member a le plus confiance
        leader, best_score = max(scores, key=lambda t: t[1])


        # 3) Si ce score ≥ threshold *ET* que leader a bien une memory_spot
        if best_score >= threshold and leader.memory_spot:
            member.next_day_target    = leader.memory_spot
            member._last_night_leader = leader.id
        else:
            # pas de leader fiable → retour à l’exploration aléatoire
            member.next_day_target    = None
            member._last_night_leader = None


    # After the loop, print all members who follow each leader
    followers_by_leader = {}
    for member in family:
        if member._last_night_leader is not None:
            followers_by_leader.setdefault(member._last_night_leader, []).append(member.id)

    # Evaluate leader coordinate correctness and adjust trust/benefits
    for leader_id, follower_ids in followers_by_leader.items():
        leader = next((h for h in family if h.id == leader_id), None)
        if not leader or not leader.memory_spot:
            continue
        lx, ly = leader.memory_spot
        # Local import to avoid circular dependency
        from resource_manager import resources
        h, w, _ = resources.shape
        # clamp window
        x0, x1 = max(0, lx-2), min(w-1, lx+2)
        y0, y1 = max(0, ly-2), min(h-1, ly+2)
        # correctness: any food present in spot neighborhood
        correct = resources[ly, lx, 1] > 0 or (resources[y0:y1+1, x0:x1+1, 1].sum() > 0)

        # optional self-confidence attribute
        if not hasattr(leader, "self_confidence"):
            leader.self_confidence = 0.5  # type: ignore[attr-defined]

        if correct:
            # followers trust leader more
            for fid in follower_ids:
                trust_system.increase_trust(trustor_id=fid, trustee_id=leader_id, increment=0.03, refresh=False)
            # leader self-confidence up
            leader.self_confidence = min(1.0, leader.self_confidence + 0.05)  # type: ignore[attr-defined]
            # leaders take 10%: approximate by adding 1 unit per 10 followers to leader's bag
            bonus = max(1, int(len(follower_ids) * 0.1))
            leader.bag = min(leader.bag_capacity, leader.bag + bonus)
        else:
            # followers trust leader less
            for fid in follower_ids:
                trust_system.increase_trust(trustor_id=fid, trustee_id=leader_id, increment=-0.03, refresh=False)
            # lower self-confidence
            leader.self_confidence = max(0.0, leader.self_confidence - 0.05)  # type: ignore[attr-defined]

    # flush cached trust lists once after batch updates
    trust_system.flush()



def draw_human(screen, human: Human, cell_size: int, font: pygame.font.Font):
    """
    Draws a human as a circle, colored by its home.color,
    with energy/sleep/bag bars underneath.
    """
    color = human.home.color
    cx = human.x * cell_size + cell_size // 2
    cy = human.y * cell_size + cell_size // 2
    r  = cell_size * 1.4
    # body
    pygame.draw.circle(screen, color, (cx, cy), r)
    # bar metrics
    bar_w = cell_size * 2
    bar_h = max(4, cell_size // 4)
    bx = cx - bar_w // 2
    y1 = cy + r + 2
    y2 = y1 + bar_h + 2
    y3 = y2 + bar_h + 2
    # backgrounds
    bg1 = pygame.Rect(bx, y1, bar_w, bar_h)
    # bg2 = pygame.Rect(bx, y2, bar_w, bar_h)
    bg3 = pygame.Rect(bx, y2, bar_w, bar_h)
    pygame.draw.rect(screen, (50, 50, 50), bg1)
    # pygame.draw.rect(screen, (50, 50, 50), bg2)
    pygame.draw.rect(screen, (50, 50, 50), bg3)
    # fills
    e_frac = human.energy / human.max_energy if human.max_energy>0 else 0
    # s_frac = human.sleep_count / human.max_sleep_count if human.max_sleep_count>0 else 0
    b_frac = human.bag / human.bag_capacity if human.bag_capacity>0 else 0
    pygame.draw.rect(screen, (0,255,0), (bx, y1, int(bar_w*e_frac), bar_h))
    # pygame.draw.rect(screen, (0,128,255), (bx, y2, int(bar_w*s_frac), bar_h))
    pygame.draw.rect(screen, (255,0,0), (bx, y2, int(bar_w*b_frac), bar_h))
    # borders
    pygame.draw.rect(screen, (0,0,0), bg1, 1)
    # pygame.draw.rect(screen, (0,0,0), bg2, 1)
    pygame.draw.rect(screen, (0,0,0), bg3, 1)





