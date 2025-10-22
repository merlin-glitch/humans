"""
caracteristics.py - Trust system for tracking relationships between agents

Part of the Human Society Simulation project.

Implements a numpy-based trust scoring system with O(1) lookups,
caching of trusted/untrusted sets, and efficient batch updates.
"""
from typing import Dict, Set, List
import numpy as np

__all__ = ['TrustSystem']


class TrustSystem:
    """
    Trust system for tracking relationships between agents.
    
    Maintains trust scores between 0.0 and 1.0 for each (trustor, trustee) pair.
    Uses numpy arrays for efficient storage and O(1) lookups. Supports batch
    updates and caching of trusted/untrusted sets for performance.
    
    Attributes:
        hints: Dictionary mapping trustor_id -> data structure containing
               trustee relationships and cached trust lists
        _dirty: Set of trustor IDs that need cache refresh
    
    Example:
        >>> trust = TrustSystem()
        >>> trust.init_human(1)
        >>> trust.increase_trust(1, 2, 0.1)
        >>> score = trust.trust_score(1, 2)
    """
    __slots__ = ("hints", "_dirty")

    def __init__(self) -> None:
        """Initialize empty trust system."""
        self.hints: Dict[int, Dict[str, object]] = {}
        self._dirty: Set[int] = set()

    # ---------- storage ----------

    def init_human(self, h_id: int) -> None:
        """
        Initialize data structures for a new human agent.
        
        Creates numpy arrays and dictionaries to track this human's trust
        relationships with other agents. Sets up cached trust lists for
        efficient queries.
        
        Args:
            h_id: Unique identifier for the human agent
            
        Note:
            This method is idempotent - safe to call multiple times for
            the same human ID.
        """
        # Input validation
        if not isinstance(h_id, int):
            raise TypeError(f"Human ID must be an integer, got {type(h_id)}")
        
        if h_id < 0:
            raise ValueError(f"Human ID must be non-negative, got {h_id}")
        
        if h_id not in self.hints:
            self.hints[h_id] = {
                "ids":   np.empty(8, dtype=np.int64),
                "succ":  np.zeros(8, dtype=np.float64),
                "total": np.ones(8, dtype=np.int32),  # keep ratio defined (1)
                "index": {},                # trustee_id -> row
                "size":  0,                 # number of valid rows
                # cached views
                "trusted": set(),
                "not_trusted": set(),
                "favorite_people": [],
            }

    def _grow(self, d: Dict[str, object], new_cap: int) -> None:
        size = d["size"]  # type: ignore
        for key in ("ids", "succ", "total"):
            arr = d[key]  # type: ignore
            new_arr = np.empty(new_cap, dtype=arr.dtype)
            new_arr[:size] = arr[:size]
            if key == "succ":
                new_arr[size:] = 0.0
            elif key == "total":
                new_arr[size:] = 1
            d[key] = new_arr  # type: ignore

    def _ensure_slot(self, h_id: int, other_id: int) -> int:
        d = self.hints[h_id]
        idx = d["index"].get(other_id)
        if idx is not None:
            return idx

        n = d["ids"].shape[0]
        d["ids"]   = np.append(d["ids"],   np.int64(other_id))
        d["succ"]  = np.append(d["succ"],  0.5)  # ← start neutral
        d["total"] = np.append(d["total"], 1)    # score = succ/total = 0.5
        d["index"][other_id] = n
        return n

    # ---------- queries ----------

    def trust_score(self, h_id: int, other_id: int) -> float:
        """
        Get the trust score between two agents.
        
        Returns the trust score from h_id to other_id, where 0.0 means
        complete distrust and 1.0 means complete trust. Returns 0.5
        (neutral) for agents who have never interacted.
        
        Args:
            h_id: The trustor (agent whose trust we're measuring)
            other_id: The trustee (agent being trusted)
            
        Returns:
            Trust score between 0.0 and 1.0, or 0.5 if no prior interaction
            
        Example:
            >>> trust.trust_score(1, 2)
            0.5  # neutral trust for new relationship
        """
        # Input validation
        if not isinstance(h_id, int) or not isinstance(other_id, int):
            raise TypeError(f"Both IDs must be integers, got {type(h_id)} and {type(other_id)}")
        
        if h_id < 0 or other_id < 0:
            raise ValueError(f"Both IDs must be non-negative, got {h_id} and {other_id}")
        
        if h_id == other_id:
            raise ValueError("Agent cannot have trust score with itself")
        
        self.init_human(h_id)
        d = self.hints[h_id]
        idx = d["index"].get(other_id)  # type: ignore
        if idx is None:
            return 0.5
        tot = float(d["total"][idx])    # type: ignore
        return 0.0 if tot <= 0.0 else float(d["succ"][idx] / tot)  # type: ignore

    def nbre_contacted(self, h_id: int) -> int:
        self.init_human(h_id)
        d = self.hints[h_id]
        size = d["size"]                # type: ignore
        return int(d["total"][:size].sum())  # type: ignore

    # ---------- updates ----------

    def increase_trust(
        self,
        trustor_id: int,
        trustee_id: int,
        increment: float = 0.01,
        refresh: bool = True,
    ) -> None:
        self.init_human(trustor_id); self.init_human(trustee_id)
        d = self.hints[trustor_id]
        idx = self._ensure_slot(trustor_id, trustee_id)
        tot = float(d["total"][idx])      # type: ignore
        old = 0.0 if tot <= 0.0 else float(d["succ"][idx] / tot)  # type: ignore
        new = min(1.0, max(0.0, old + increment))  # Clamp to [0, 1] range
        d["succ"][idx] = new * tot        # type: ignore
        if refresh:
            self._refresh_trust_lists(trustor_id)
        else:
            self._dirty.add(trustor_id)

    def update_on_meeting(self, h1: 'Human', h2: 'Human', resources: list) -> None:
        for receiver, giver in ((h1, h2), (h2, h1)):
            self.init_human(receiver.id)
            d = self.hints[receiver.id]
            idx = self._ensure_slot(receiver.id, giver.id)
            d["total"][idx] = int(d["total"][idx]) + 1  # type: ignore
            spot = getattr(giver, "memory_spot", None)
            if spot is not None:
                gx, gy = spot
                if any((r.x == gx and r.y == gy) for r in resources):
                    d["succ"][idx] = float(d["succ"][idx]) + 1.0  # type: ignore
            self._dirty.add(receiver.id)

    # ---------- refresh / caching ----------

    def _refresh_trust_lists(self, h_id: int) -> None:
        d = self.hints[h_id]
        size = d["size"]   # type: ignore
        if size == 0:
            d["trusted"].clear(); d["not_trusted"].clear(); d["favorite_people"].clear()  # type: ignore
            return
        ids   = d["ids"][:size]    # type: ignore
        succ  = d["succ"][:size]   # type: ignore
        total = d["total"][:size]  # type: ignore
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.where(total > 0, succ / total, 0.0)
        # Trust threshold: 0.5 separates trusted from untrusted
        d["trusted"]     = set(map(int, ids[scores > 0.5]))    # type: ignore
        d["not_trusted"] = set(map(int, ids[scores < 0.5]))    # type: ignore
        k = min(5, scores.size)  # Top 5 favorite people (limit prevents memory bloat)
        if k:
            topk_idx = np.argpartition(scores, -k)[-k:]
            order = np.argsort(scores[topk_idx])[::-1]
            d["favorite_people"] = ids[topk_idx][order].astype(int).tolist()  # type: ignore
        else:
            d["favorite_people"] = []  # type: ignore

    def refresh_one(self, h_id: int) -> None:
        if h_id in self.hints:
            self._refresh_trust_lists(h_id)

    def refresh_many(self, ids: List[int]) -> None:
        for tid in ids:
            if tid in self.hints:
                self._refresh_trust_lists(tid)

    def flush(self) -> None:
        if not self._dirty:
            return
        self.refresh_many(list(self._dirty))
        self._dirty.clear()

    def refresh_all(self) -> None:
        self.refresh_many(list(self.hints.keys()))
