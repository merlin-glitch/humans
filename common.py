# common.py 

import csv
from typing import List
from human import Human
from caracteristics import TrustSystem


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


import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple, Dict, List
from human import Human
from caracteristics import TrustSystem

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
    avg = average_trust_per_house(humans, trust_system)
    houses, values = zip(*sorted(avg.items()))
    labels = [f"{x},{y}" for x,y in houses]
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("House (x,y)")
    plt.ylabel("Average trust")
    plt.title("Average trust among members of each house")
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
