from human import Human
import csv
from typing import List, Optional, Dict, Set

"""
confiance.py: Implements a TrustSystem for simulated Human agents.
Manages continuous trust values in [-1.0, 1.0], tracks interaction stats,
and exports a trust matrix as CSV.
"""

class TrustSystem:
    """
    Maintains trust relationships between Human agents based on shared memory spots.

    Data structures:
        hints: Dict[int, Dict[str, object]]
            Maps human_id -> {
                "favorite_person": Optional[int],  # ID of top-trusted agent
                "trusted": Set[int],             # Agents with positive trust score
                "not_trusted": Set[int],         # Agents with negative trust score
                "pair_stats": Dict[int, Tuple[int,int]]
                    # other_id -> (success_count, total_interactions)
            }
    Trust score calculation:
        successes / total_interactions -> ratio in [0,1]
        mapped to [-1,1] by: 2*ratio - 1
    """

    def __init__(self) -> None:
        """
        Initialize the hints dictionary empty.
        """
        self.hints: Dict[int, Dict[str, object]] = {}

    def init_human(self, h_id: int) -> None:
        """
        Ensure the trust record for human h_id exists.
        If not, create default entries with empty sets and stats.
        """
        if h_id not in self.hints:
            self.hints[h_id] = {
                "favorite_person": None,
                "trusted": set(),
                "not_trusted": set(),
                "pair_stats": {}  # type: Dict[int, Tuple[int,int]]
            }

    def update_on_meeting(
        self,
        h1: Human,
        h2: Human,
        resources   # just a list of objects with .x and .y
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

            # fetch or init stats
            succ, total = stats.get(giver.id, (0, 0))
            total += 1  # count this meeting

            # only count as success if they shared AND resource still exists
            spot = giver.memory_spot
            if spot is not None:
                x_spot, y_spot = spot
                # duck-type: no isinstance check needed
                if any((r.x == x_spot and r.y == y_spot) for r in resources):
                    succ += 1

            # store back
            stats[giver.id] = (succ, total)
            self._refresh_trust_lists(receiver.id)


    def trust_score(self, h_id: int, other_id: int) -> float:
        """
        Compute continuous trust score for h_id towards other_id.
        Formula:
            ratio = successes / total_interactions
            return 2 * ratio - 1  # maps [0,1] to [-1,1]
        Returns 0.0 if no interactions.
        """
        self.init_human(h_id)
        stats: Dict[int, Tuple[int,int]] = self.hints[h_id]["pair_stats"]  # type: ignore  , "pair_stats":other_id -> (success_count, total_interactions)
        if other_id not in stats:
            return 0.0
        succ, total = stats[other_id]
        if total == 0:
            return 0.0
        ratio = succ / total
        return 2 * ratio - 1

    def _refresh_trust_lists(self, h_id: int) -> None:
        """
        Update the "trusted", "not_trusted", and "favorite_person" entries for h_id:
            - trusted: other_id with score > 0
            - not_trusted: other_id with score < 0
            - favorite_person: other_id with highest score
        """
        data = self.hints[h_id]
        stats: Dict[int, Tuple[int,int]] = data["pair_stats"]  # type: ignore

        # Prepare new sets and favorite placeholder
        trusted: Set[int] = set()
        not_trusted: Set[int] = set()
        favorite: Optional[int] = None
        init_score = -2.0  # lower than any possible trust score

        # Evaluate each pair
        for other_id, (succ, total) in stats.items():
            # Compute continuous score
            score = 0.0 if total == 0 else 2 * (succ / total) - 1
            # Classify
            if score > 0:
                trusted.add(other_id)
            elif score < 0:
                not_trusted.add(other_id)
            # Track highest for favorite
            if score > init_score:
                init_score = score
                favorite = other_id

        # Save back the computed sets and favorite
        data["trusted"] = trusted
        data["not_trusted"] = not_trusted
        data["favorite_person"] = favorite

    def nbre_contacted(self, h_id: int) -> int:
        """Return total number of interactions for h_id."""
        self.init_human(h_id)
        stats: Dict[int, Tuple[int,int]] = self.hints[h_id]["pair_stats"]  # type: ignore , stats={giver_id:int, pair_stats:(success_count, total_interactions)}
        return sum(total for succ, total in stats.values())

    def display_trust_summary(self, h_id: int) -> None:
        """
        Print a summary for h_id:
          - Total interactions
          - IDs trusted (>0)
          - IDs distrusted (<0)
          - Favorite person
          - Continuous scores listing
        """
        self.init_human(h_id)
        data = self.hints[h_id]
        print(f"Human {h_id} summary:")
        print(f"  Interactions: {self.nbre_contacted(h_id)}")
        print(f"  Trusted (>0): {sorted(data['trusted'])}")
        print(f"  Distrusted (<0): {sorted(data['not_trusted'])}")
        print(f"  Favorite: {data['favorite_person']}")
        # Show raw scores
        scores = {other: self.trust_score(h_id, other) for other in data['pair_stats']}
        for other, score in scores.items():
            print(f"    -> {other}: {score:.3f}")

    def export_trust_matrix(self, human_list: List[Human], filename: str = "trust_matrix.csv") -> None:
        """
        Write a CSV with row/column headers as human IDs and
        cells containing continuous trust scores in [-1,1].
        """
        ids = sorted(h.id for h in human_list)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([""] + ids)
            for row_id in ids:
                row = [row_id]
                for col_id in ids:
                    row.append(f"{self.trust_score(row_id, col_id):.3f}")
                writer.writerow(row)
