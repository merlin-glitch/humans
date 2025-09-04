# common.py 

import csv
from typing import List , Tuple,  Dict
from human import *
from caracteristics import *
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os


def boost_house_trust(
    trust_system: TrustSystem,
    contributor: Human,
    humans: List[Human],
    increment: float = 0.001
) -> None:
    """
    Whenever `contributor` deposits food, bump by `increment`
    the trust score of every other human sharing that same house.
    
    Now uses float‐based increase_trust under the hood (no rounding).
    """
    # ensure the contributor is known
    trust_system.init_human(contributor.id)

    for receiver in humans:
        if receiver is contributor:
            continue
        # only boost those in the same house
        if receiver.home is contributor.home:
            # this will internally do:
            #   succ_float += increment * total
            #   clamp, recompute ratio, etc.
            trust_system.increase_trust(
                trustor_id=receiver.id,
                trustee_id=contributor.id,
                increment=increment
            )


def export_trust_matrix(
    trust_system: TrustSystem,
    human_list: List[Human],
    filename: str = "trust_matrix.csv"
) -> None:
    """
    Write a CSV with human IDs as headers and
    cells containing continuous trust scores in [-1,1].
    """
    ids = sorted(h.id for h in human_list)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + ids)
        for row_id in ids:
            row = [row_id]
            for col_id in ids:
                row.append(f"{trust_system.trust_score(row_id, col_id):.3f}")
            writer.writerow(row)




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
    threshold: float = 0.7,
    energy_cost: float = 5.0
) -> Tuple[int, int]:
    """
    If h1 and h2 share the same house, are adjacent, have mutual trust ≥ threshold,
    and each has ≥ energy_cost energy, they spend that energy to produce 1–2 children.
    Appends the new Humans to `humans` and registers them in `trust_system`.

    Returns:
      (num_children_created, new_next_id)
    """

    # 1) must be co‑residents
    if h1.home is not h2.home:
        return 0, next_id

    # 1a) optional adjacency check (uncomment if you care about physical proximity)
    # if max(abs(h1.x-h2.x), abs(h1.y-h2.y)) > 1:
    #     return 0, next_id

    # 2) mutual trust check
    t12 = trust_system.trust_score(h1.id, h2.id)
    t21 = trust_system.trust_score(h2.id, h1.id)
    if t12 < threshold or t21 < threshold:
        return 0, next_id

    # 3) energy check
    if h1.energy < energy_cost or h2.energy < energy_cost:
        return 0, next_id

    # 4) pay the energy cost
    h1.energy -= energy_cost
    h2.energy -= energy_cost

    # 5) decide how many kids (1 or 2)
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
    Chaque membre du groupe regarde qui, parmi les autres, il
    estime le plus (trust_score ≥ threshold). Si cette personne
    a une memory_spot valide, il la suit demain.
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

    for leader_id, followers in followers_by_leader.items():
        leader = next((h for h in family if h.id == leader_id), None)
        family_color = "Blue" if leader and leader.home.color == (0, 0, 128) else "Red" if leader and leader.home.color == (255, 0, 0) else str(leader.home.color) if leader else "Unknown"
        #print(f"Leader {leader_id} ({family_color}) is followed by: {followers}")