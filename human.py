"""
human.py - Human and House agent classes for the simulation

Part of the Human Society Simulation project.

Defines the behavior of individual humans (movement, foraging, social
interaction) and houses (resource storage, family membership).
"""

import random
from typing import Optional, Tuple, List
import numpy as np
from config import HOUSE_SIZE

__all__ = ['House', 'Human', 'draw_human']


class House:
    """House represents a dwelling where humans live and store resources."""
    
    def __init__(self, x: int, y: int, color: Tuple[int, int, int]):
        # Input validation
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(f"House position (x, y) must be integers, got ({x}, {y})")
        
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(f"House color must be a 3-element tuple/list (R,G,B), got {color}")
        
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            raise ValueError(f"House color values must be integers 0-255, got {color}")
        
        if x < 0 or y < 0:
            raise ValueError(f"House position must be non-negative, got ({x}, {y})")
        
        # Initialize validated attributes
        self.x = x
        self.y = y
        self.color = color
        self.storage = 0

    def deposit(self, amount: float) -> None:
        """Add `amount` units into this house's storage, capped at 5,000."""
        self.storage = min(5000, self.storage + round(amount))


class Human:
    """Human agent representing an individual in the simulation."""
    
    def __init__(
        self,
        human_id: int,
        sex: str,
        x: int,
        y: int,
        home: House,
        codes,                     # static terrain codes array
        initial_energy: float = 10.0,
        exploration_factor: int = 2,
        bag_capacity: int = 10,
    ):
        # Input validation
        if not isinstance(human_id, int) or human_id < 0:
            raise ValueError(f"human_id must be a non-negative integer, got {human_id}")
        
        if sex not in ['M', 'F', 'homme', 'femme']:
            raise ValueError(f"sex must be 'M', 'F', 'homme', or 'femme', got {sex}")
        
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(f"Position (x, y) must be integers, got ({x}, {y})")
        
        if not isinstance(home, House):
            raise TypeError(f"home must be a House instance, got {type(home)}")
        
        if codes is None or not hasattr(codes, 'shape'):
            raise ValueError("codes must be a numpy array with shape attribute")
        
        # Validate position is within map bounds
        h, w = codes.shape
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"Position ({x}, {y}) is outside map bounds ({w}, {h})")
        
        if initial_energy <= 0:
            raise ValueError(f"initial_energy must be positive, got {initial_energy}")
        
        if bag_capacity <= 0:
            raise ValueError(f"bag_capacity must be positive, got {bag_capacity}")
        
        # Initialize validated attributes
        self.id = human_id
        self.sex = sex
        self.home = home
        self.home_x = home.x + HOUSE_SIZE // 2
        self.home_y = home.y + HOUSE_SIZE // 2
        self.x = x
        self.y = y
        self.codes = codes
        self.obstacles = set()
        
        # Energy and survival
        self.energy = initial_energy
        self.max_energy = initial_energy
        self.alive = True
        
        # Movement and exploration
        self.dir_x, self.dir_y = random.choice([
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),         (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ])
        self.exploration_factor = exploration_factor
        
        # Resource management
        self.bag = 0
        self.bag_capacity = bag_capacity
        self.known_food = []
        self.last_collected = None
        self.memory_spot = None
        self.contributed = 0
        
        # Optional fields used elsewhere
        self.sleep_count = 0.0
        self.max_sleep_count = 0.0

    def perform_action(self, cost: float = 0.01) -> None:
        """
        Spend energy on an action and handle death if energy is depleted.
        
        This is called whenever a human performs any action (movement, foraging, etc.).
        If energy drops to zero or below, the human dies and is marked as not alive.
        
        Args:
            cost: Amount of energy to deduct (default 0.01)
            
        Example:
            >>> human.perform_action(0.05)  # Spend 0.05 energy on movement
            >>> if human.energy <= 0:
            ...     print("Human died!")
        """
        self.energy = max(0.0, self.energy - cost)
        if self.energy <= 0.0:
            self.alive = False

    def decay_energy(self, rate: float = 0.005) -> None:
        """Passive energy decay even when still."""
        self.energy = max(0.0, self.energy - rate)
        if self.energy <= 0.0:
            self.alive = False

    def eat(self, gain: float = 1.0, spot: Optional[Tuple[int, int]] = None) -> None:
        """
        Consume food to replenish energy and remember the location.
        
        When a human eats food, they gain energy and can remember the location
        for future foraging. The location is added to their known food spots
        and becomes their memory spot for sharing with others.
        
        Args:
            gain: Amount of energy to gain (default 1.0)
            spot: Optional (x, y) coordinates where food was found
            
        Example:
            >>> human.eat(2.0, (10, 15))  # Eat 2 energy worth of food at (10,15)
            >>> print(f"Energy: {human.energy}, Last spot: {human.memory_spot}")
        """
        # Input validation
        if not isinstance(gain, (int, float)) or gain <= 0:
            raise ValueError(f"Energy gain must be a positive number, got {gain}")
        
        if spot is not None:
            if not isinstance(spot, (tuple, list)) or len(spot) != 2:
                raise ValueError(f"Spot must be a 2-element tuple/list (x, y), got {spot}")
            x, y = spot
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError(f"Spot coordinates must be integers, got ({x}, {y})")
            # Validate spot is within map bounds
            h, w = self.codes.shape
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"Spot ({x}, {y}) is outside map bounds ({w}, {h})")
        
        self.energy = min(self.max_energy, self.energy + gain)
        if spot is not None:
            self.last_collected = spot
            self.memory_spot = spot
            self.known_food.append(spot)
            if len(self.known_food) > 5:
                self.known_food.pop(0)
        if self.energy <= 0.0:
            self.alive = False

    def store_in_bag(self, spot: Optional[Tuple[int, int]] = None) -> None:
        """If energy ≥9 and bag not full, store one unit."""
        # Energy threshold of 9 ensures humans don't store food when they need it themselves
        if self.energy >= 9 and self.bag < self.bag_capacity:
            self.bag += 1
            self.last_collected = spot
            self.memory_spot = spot
            self.known_food.append(spot)
            # Keep only the 20 most recent food locations to prevent memory overflow
            if len(self.known_food) > 20:
                self.known_food.pop(0)

    def _can_move(self, newx: int, newy: int) -> bool:
        """Check map bounds, obstacles, and terrain != -1."""
        h, w = self.codes.shape
        if not (0 <= newx < w and 0 <= newy < h):
            return False
        if (newx, newy) in self.obstacles:
            return False
        # Terrain code -1 represents walls/obstacles that block movement
        return self.codes[newy][newx] != -1

    def stay_in_house(self) -> None:
        """Force human to remain in its house."""
        self.x, self.y = self.home_x, self.home_y

    def move_towards(self, tx: int, ty: int, cost: float = 0.01) -> None:
        """Step one cell toward (tx,ty), with obstacle detour."""
        dx = tx - self.x
        dy = ty - self.y
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        newx, newy = self.x + step_x, self.y + step_y

        if self._can_move(newx, newy):
            self.perform_action(cost)
            self.x, self.y = newx, newy
            self.dir_x, self.dir_y = step_x, step_y
        else:
            # try X-only or Y-only, else random detour
            if step_x and self._can_move(self.x + step_x, self.y):
                self.perform_action(cost)
                self.x += step_x
                self.dir_x, self.dir_y = step_x, 0
            elif step_y and self._can_move(self.x, self.y + step_y):
                self.perform_action(cost)
                self.y += step_y
                self.dir_x, self.dir_y = 0, step_y
            else:
                self.obstacles.add((newx, newy))
                self.random_move(cost)

    def random_move(self, cost: float = 0.01) -> None:
        """Pick one of four angled moves, biased toward forward (branchless-ish)."""
        fx, fy = self.dir_x, self.dir_y
        r = random.random()
        # Movement probability distribution: 70% forward, 15% left, 14.5% right, 0.5% back
        # This creates natural-looking movement patterns with forward bias
        if r < 0.70:
            dx, dy = fx, fy               # forward
        elif r < 0.85:
            dx, dy = -fy, fx              # left
        elif r < 0.995:
            dx, dy =  fy, -fx             # right
        else:
            dx, dy = -fx, -fy             # back
        newx, newy = self.x + dx, self.y + dy
        if self._can_move(newx, newy):
            self.perform_action(cost)
            self.x, self.y = newx, newy
            self.dir_x, self.dir_y = dx, dy

    def deposit_food(self) -> None:
        """Deposit all bag contents into house storage."""
        if self.bag > 0:
            self.home.deposit(self.bag)  # Use deposit() to apply 10,000 cap
            self.contributed += self.bag
            self.bag = 0

    def share_food(self, humans: List['Human'], trust_system=None) -> None:
        """Share food with nearby humans, prioritizing trusted allies."""
        if self.bag <= 0:
            return
        
        # Find nearby humans within 2 cells
        nearby_humans = []
        for other in humans:
            if other is not self and other.alive:
                dx, dy = other.x - self.x, other.y - self.y
                if dx*dx + dy*dy <= 4:  # within 2 cells
                    nearby_humans.append(other)
        
        if not nearby_humans:
            return
        
        # Sort by trust score (highest first) if trust system available
        if trust_system:
            nearby_humans.sort(key=lambda h: trust_system.trust_score(self.id, h.id), reverse=True)
        
        # Share with the most trusted (or first if no trust system)
        for other in nearby_humans:
            # Share food only if recipient is low energy (<5) and donor has surplus (>8)
            if other.energy < 5 and self.energy > 8:
                # Trust-based sharing: higher trust = more generous sharing
                trust_bonus = 1.0
                if trust_system:
                    trust_score = trust_system.trust_score(self.id, other.id)
                    trust_bonus = 1.0 + (trust_score - 0.5) * 2.0  # 0.5-1.5x multiplier based on trust
                
                # Share one unit: recipient gets energy based on trust, donor loses 1 bag unit
                energy_given = int(2.0 * trust_bonus)
                self.bag -= 1
                other.energy += energy_given
                break

    def find_food(self, resources: np.ndarray, humans: List['Human']=None, trust_system=None) -> Optional[Tuple[int, int]]:
        """Look for food in current cell and nearby cells, with cooperative foraging."""
        # Check current cell first
        if resources[self.y, self.x, 1] > 0:
            return (self.x, self.y)
        
        # Check nearby cells in a 5x5 grid around current position (-2 to +2)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = self.x + dx, self.y + dy
                if self._can_move(nx, ny) and resources[ny, nx, 1] > 0:
                    return (nx, ny)
        
        # Cooperative foraging: follow trusted allies to food locations
        if humans and trust_system:
            # Look for trusted allies who know about food locations
            trusted_allies = []
            for other in humans:
                if other is not self and other.alive and other.memory_spot:
                    trust_score = trust_system.trust_score(self.id, other.id)
                    if trust_score > 0.6:  # Only follow highly trusted allies
                        trusted_allies.append((other, trust_score))
            
            # Sort by trust score and follow the most trusted ally
            if trusted_allies:
                trusted_allies.sort(key=lambda x: x[1], reverse=True)
                most_trusted = trusted_allies[0][0]
                food_spot = most_trusted.memory_spot
                if food_spot and resources[food_spot[1], food_spot[0], 1] > 0:
                    return food_spot
        
        return None

    def share_knowledge(self, humans: List['Human'], trust_system=None) -> None:
        """Share food location knowledge with nearby humans, prioritizing trusted allies."""
        if not self.memory_spot:
            return
        
        # Find nearby humans within 2 cells
        nearby_humans = []
        for other in humans:
            if other is not self and other.alive:
                dx, dy = other.x - self.x, other.y - self.y
                if dx*dx + dy*dy <= 4:  # within 2 cells
                    nearby_humans.append(other)
        
        if not nearby_humans:
            return
        
        # Sort by trust score (highest first) if trust system available
        if trust_system:
            nearby_humans.sort(key=lambda h: trust_system.trust_score(self.id, h.id), reverse=True)
        
        # Share with trusted allies first (or all if no trust system)
        for other in nearby_humans:
            # Trust-based knowledge sharing: only share with trusted allies
            should_share = True
            if trust_system:
                trust_score = trust_system.trust_score(self.id, other.id)
                # Only share with humans we trust above neutral (0.5)
                should_share = trust_score > 0.5
            
            if should_share:
                other.memory_spot = self.memory_spot
                # Limit sharing to prevent spam - only share with top 3 trusted allies
                if trust_system and len(nearby_humans) > 3:
                    break

    def assist_trusted_allies(self, humans: List['Human'], trust_system=None) -> None:
        """Provide survival assistance to trusted allies in danger."""
        if not trust_system or self.energy <= 5:
            return
        
        # Find nearby trusted allies who are in danger (low energy)
        for other in humans:
            if other is not self and other.alive:
                dx, dy = other.x - self.x, other.y - self.y
                if dx*dx + dy*dy <= 4:  # within 2 cells
                    trust_score = trust_system.trust_score(self.id, other.id)
                    
                    # Help highly trusted allies who are in danger
                    if trust_score > 0.7 and other.energy <= 3 and self.energy > 8:
                        # Emergency assistance: give energy to trusted ally
                        assistance = min(3, self.energy - 5)  # Don't risk own survival
                        other.energy += assistance
                        self.energy -= assistance
                        
                        # Increase trust when providing assistance
                        trust_system.increase_trust(
                            trustor_id=other.id,
                            trustee_id=self.id,
                            increment=0.01,  # Small trust boost for helping
                            refresh=False
                        )
                        break  # Only help one ally per tick to prevent spam

    def step(
        self,
        resources,
        houses,
        humans,                    # now expected to be *nearby peers*, not everyone
        trust_system,
        is_day: bool,
        action_cost: float = 0.01,
        food_gain: float = 2.0,
        decay_rate: float = 0.005,
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """
        Execute one simulation tick of human behavior.
        
        This is the main behavior function called each tick. It handles:
        - Night behavior: returning home and depositing food
        - Day behavior: foraging, eating, sharing knowledge and food
        - Energy management and survival checks
        - Social interactions with nearby humans
        
        Args:
            resources: 3D numpy array of resource data [y, x, (lifetime, food_left)]
            houses: List of house objects in the simulation
            humans: List of nearby human agents (not all humans)
            trust_system: Trust relationship management system
            is_day: Whether it's currently day time (True) or night (False)
            action_cost: Energy cost for movement actions (default 0.01)
            food_gain: Energy gained from consuming food (default 2.0)
            decay_rate: Passive energy decay rate per tick (default 0.005)
            
        Returns:
            Tuple of (picked_coord, shared_bool):
            - picked_coord: (x, y) coordinates where food was collected, or None
            - shared_bool: True if food/knowledge was shared, False otherwise
            
        Example:
            >>> coord, shared = human.step(resources, houses, nearby_humans, 
            ...                          trust_system, is_day=True)
            >>> if coord:
            ...     print(f"Collected food at {coord}")
        """

        # ── obey leader at start of day ────────────────────────────
        if is_day and getattr(self, "next_day_target", None):
            tx, ty = self.next_day_target
            if (self.x, self.y) != (tx, ty):
                self.move_towards(tx, ty, action_cost)
                return None, False
            del self.next_day_target
            return None, False

        # ── NIGHT BEHAVIOR ───────────────────────────────
        if not is_day:
            if (self.x, self.y) != (self.home_x, self.home_y):
                self.move_towards(self.home_x, self.home_y, action_cost)
                if self.bag > 0:
                    self.deposit_food()
                return None, False

        # ── DAY BEHAVIOR ───────────────────────────────────
        hx, hy = self.home.x, self.home.y
        in_house = (hx <= self.x < hx + HOUSE_SIZE) and (hy <= self.y < hy + HOUSE_SIZE)

        self.decay_energy(decay_rate)
        if not self.alive:
            return None, False

        # Priority 1: If critically low energy, eat from bag first
        if self.energy < 5 and self.bag > 0:
            self.eat(food_gain)
            self.bag -= 1
            return None, False
        
        # Priority 2: If still hungry and bag empty, go home to eat from storage
        if self.energy < 5 and self.bag == 0:
            if (self.x, self.y) != (self.home_x, self.home_y):
                self.move_towards(self.home_x, self.home_y, action_cost)
                return None, False
            # At home, eat from storage if available
            if self.home.storage > 0 and self.energy < self.max_energy:
                self.eat(food_gain)
                self.home.storage -= 1
            return None, False

        # Forage for food
        picked, shared = self.forage(resources, houses, humans, trust_system, is_day)
        
        # Share knowledge (now trust-aware)
        self.share_knowledge(humans, trust_system)
        
        # Share food (now trust-aware)
        self.share_food(humans, trust_system)
        
        # Trust-based survival assistance
        self.assist_trusted_allies(humans, trust_system)

        return picked, shared

    def forage(self, resources, houses, humans, trust_system, is_day: bool) -> Tuple[Optional[Tuple[int, int]], bool]:
        """Main foraging behavior during day time."""
        if not is_day:
            return None, False
            
        # Look for food (now with cooperative foraging)
        food_spot = self.find_food(resources, humans, trust_system)
        if food_spot:
            fx, fy = food_spot
            if (fx, fy) != (self.x, self.y):
                # Move towards food
                self.move_towards(fx, fy)
                return None, False
            else:
                # Collect food
                if self.energy < 8:  # Eat if low energy
                    self.eat(2.0, (fx, fy))
                    self.last_pick_pos = (fx, fy)  # Track for zone consumption analytics
                    resources[fy, fx, 1] -= 1
                    return (fx, fy), False
                elif self.bag < self.bag_capacity:  # Store if bag not full
                    self.store_in_bag((fx, fy))
                    self.last_pick_pos = (fx, fy)  # Track for zone consumption analytics
                    resources[fy, fx, 1] -= 1
                    return (fx, fy), False
                else:
                    # Bag full, go home to deposit
                    self.move_towards(self.home_x, self.home_y)
                    return None, False
        
        # No food found, explore or go home
        if self.bag > 0:
            self.move_towards(self.home_x, self.home_y)
        else:
            self.random_move()
        return None, False


def draw_human(surface, human, cell_size: int, font):
    """Draw a human agent on the pygame surface."""
    import pygame
    
    if not human.alive:
        return
        
    # Position
    x = human.x * cell_size + cell_size // 2
    y = human.y * cell_size + cell_size // 2
    
    # Color based on house
    color = human.home.color
    
    # Draw human as circle
    pygame.draw.circle(surface, color, (x, y), cell_size // 3)
    
    # Draw energy indicator
    energy_ratio = human.energy / human.max_energy
    energy_color = (int(255 * (1 - energy_ratio)), int(255 * energy_ratio), 0)
    pygame.draw.circle(surface, energy_color, (x, y), cell_size // 6)
    
    # Draw ID
    if font:
        text = font.render(str(human.id), True, (255, 255, 255))
        text_rect = text.get_rect(center=(x, y))
        surface.blit(text, text_rect)
