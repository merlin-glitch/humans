
# """
# confiance.py: Implements a TrustSystem for simulated Human agents.
# Manages continuous trust values in [-1.0, 1.0], tracks interaction stats,
# and exports a trust matrix as CSV.
# """






  # type: ignore







import csv
from typing import List, Optional, Dict, Set, Tuple


class TrustSystem:
    """
    Maintains trust relationships between Human agents based on shared memory spots.

    Data structures:
        hints: Dict[int, Dict[str, object]]
            Maps human_id -> {
                "favorite_people": List[int],  # top‑5 trusted agents by score
                "trusted": Set[int],           # Agents with positive trust score
                "not_trusted": Set[int],       # Agents with negative trust score
                "pair_stats": Dict[int, Tuple[int,int]]
                    # other_id -> (success_count, total_interactions)
            }
    Trust score calculation:
        successes / total_interactions -> ratio in [0,1]
        mapped to [-1,1] by: 2*ratio - 1


    i switched it back from [-1,1] to [0,1] 
    """

    def __init__(self) -> None:
        """Initialize the hints dictionary empty."""
        self.hints: Dict[int, Dict[str, object]] = {}

    def init_human(self, h_id: int) -> None:
        """
        Ensure the trust record for human h_id exists.
        If not, create default entries with empty sets and stats.
        """
        if h_id not in self.hints:
            self.hints[h_id] = {
                "favorite_people": [],      # will hold up to 5 IDs
                "trusted": set(),
                "not_trusted": set(),
                "pair_stats": {}            # other_id -> (succ, total)
            }

    
    def trust_score(self, h_id: int, other_id: int) -> float:
        """
        Compute continuous trust score for h_id towards other_id.
        Formula: 2*(succ/total) - 1, in [-1,1].
        """
        self.init_human(h_id)
        stats: Dict[int, Tuple[int,int]] = self.hints[h_id]["pair_stats"]  # type: ignore
        if other_id not in stats:
            return 0.5
        succ, total = stats[other_id]
        if total == 0:
            return 0.0
         #changed to [0,1] intervalle
        return succ / total

    def _refresh_trust_lists(self, h_id: int) -> None:
        """
        Recompute:
          - trusted: other_ids with score > 0.5
          - not_trusted: other_ids with score < 0.5
          - favorite_people: top‑5 other_ids by descending score
        """
        data = self.hints[h_id]
        stats: Dict[int, Tuple[int,int]] = data["pair_stats"]  # type: ignore

        trusted = set()
        not_trusted = set()
        scored: List[Tuple[int,float]] = []

        for other_id, (succ, total) in stats.items():
            score = 0.0 if total == 0 else succ / total
            if score > 0.5:
                trusted.add(other_id)
            elif score < 0.5:
                not_trusted.add(other_id)
            scored.append((other_id, score))

        # sort by score descending and take up to 5 IDs
        top5 = [hid for hid, _ in sorted(scored, key=lambda t: t[1], reverse=True)][:5]

        data["trusted"] = trusted
        data["not_trusted"] = not_trusted
        data["favorite_people"] = top5

    def nbre_contacted(self, h_id: int) -> int:
        """Return total number of interactions for h_id."""
        self.init_human(h_id)
        stats: Dict[int, Tuple[int,int]] = self.hints[h_id]["pair_stats"]  # type: ignore
        return sum(total for _, total in stats.values())

    def display_trust_summary(self, h_id: int) -> None:
        """
        Print a summary for h_id:
          - Total interactions
          - Trusted (>0)
          - Distrusted (<0)
          - Top‑5 trusted IDs
          - All individual scores
        """
        self.init_human(h_id)
        data = self.hints[h_id]
        # print(f"Human {h_id} summary:")
        # print(f"  Interactions: {self.nbre_contacted(h_id)}")
        # print(f"  Trusted (>0): {sorted(data['trusted'])}")
        # print(f"  Distrusted (<0): {sorted(data['not_trusted'])}")
        # print(f"  Top‑5 trusted: {data['favorite_people']}")
        # raw scores
        for other_id, (succ, total) in data["pair_stats"].items():  # type: ignore
            score = 0.0 if total == 0 else succ/total
            #print(f"    -> {other_id}: {score:.3f}")


    def increase_trust(
        self,
        trustor_id: int,
        trustee_id: int,
        increment: float = 0.01
    ) -> None:
        """
        Bump by `increment` the trust that trustor_id has in trustee_id,
        keeping the underlying ratio precise (no integer rounding).
        Now trust_score ∈ [0,1], so we clamp into [0,1].
        """
        # 1) Ensure both sides exist
        self.init_human(trustor_id)
        self.init_human(trustee_id)

        stats = self.hints[trustor_id]["pair_stats"]

        # 2) Fetch previous (succ, total); default to (0.0, 1)
        succ, total = stats.get(trustee_id, (0.0, 1))

        # 3) Compute old score (succ/total ∈ [0,1])
        old_score = self.trust_score(trustor_id, trustee_id)

        # 4) Add your increment, clamp to [0,1]
        new_score = min(1.0, max(0.0, old_score + increment))

        # 5) Now ratio==score, so…
        new_ratio = new_score

        # 6) Translate back into a float success count over the same total
        new_succ = new_ratio * total

        # 7) Store back
        stats[trustee_id] = (new_succ, total)

        # 8) Refresh derived lists
        self._refresh_trust_lists(trustor_id)
            


    def update_on_meeting(
        self,
        h1: 'Human',
        h2: 'Human',
        resources: List  # just a list of objects with .x and .y
    ) -> None:
        """
        Record a meeting:
        - always increment total meetings
        - only increment success if giver.memory_spot is not None
          AND there is still a resource at that spot.
        """
        for receiver, giver in ((h1, h2), (h2, h1)):
            self.init_human(receiver.id)
            stats = self.hints[receiver.id]["pair_stats"]  # type: ignore

            succ, total = stats.get(giver.id, (0, 0))
            total += 1  # count this meeting

            spot = giver.memory_spot
            if spot is not None:
                x_spot, y_spot = spot
                if any((r.x == x_spot and r.y == y_spot) for r in resources):
                    succ += 1

            stats[giver.id] = (succ, total)
            self._refresh_trust_lists(receiver.id)
