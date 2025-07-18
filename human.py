

import random
import pygame
from typing import Tuple, List
from config import *
from caracteristics import *


class House:
    def __init__(self, x: int, y: int, color: Tuple[int, int, int]):
        """
        A home tile at (x,y) that also holds a color and a food storage.
        """
        self.x, self.y = x, y
        self.color     = color      # each house now carries its own color
        self.storage   = 0          # total food units stored here

    def store_food(self, amount: int) -> None:
        """Add `amount` units into this house’s storage."""
        self.storage += amount


class Human:
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
        initial_sleep: float = 200.0
    ):
        self.id                 = human_id
        self.sex                = sex
        self.home               = home
        self.home_x             = home.x + HOUSE_SIZE // 2
        self.home_y             = home.y + HOUSE_SIZE // 2
        self.x                  = x
        self.y                  = y
        self.max_energy         = max(0.0, initial_energy)
        self.energy             = self.max_energy
        self.bag                = 0
        self.bag_capacity       = bag_capacity
        self.exploration_factor = exploration_factor
        self.dir_x, self.dir_y  = random.choice([
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),         (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ])
        self.known_food        = []      # list of (x,y) spots
        self.last_collected    = None
        self.memory_spot       = None    # last food pixel remembered
        self.max_sleep_count   = initial_sleep
        self.sleep_count       = initial_sleep
        self.alive             = True
        self.codes             = codes   # static map array to block movement
        self.obstacles         = set()   # remembered blocked tiles
        # how much I’ve personally deposited into my house
        self.contributed: int = 0

    def perform_action(self, cost: float = 0.01) -> None:
        """Spend `cost` energy; die if energy hits zero."""
        self.energy = max(0.0, self.energy - cost)
        if self.energy <= 0.0:
            self.alive = False

    def decay_energy(self, rate: float = 0.005) -> None:
        """Passive energy decay even when still."""
        self.energy = max(0.0, self.energy - rate)
        if self.energy <= 0.0:
            self.alive = False

    def eat(self, gain: float = 1.0, spot=None) -> None:
        """Replenish energy by `gain`, remember spot if given."""
        self.energy = min(self.max_energy, self.energy + gain)
        if spot is not None:
            self.last_collected = spot
            self.memory_spot    = spot
            self.known_food.append(spot)
            if len(self.known_food) > 5:
                self.known_food.pop(0)
        if self.energy <= 0.0:
            self.alive = False

    def store_in_bag(self, spot=None) -> None:
        """If energy ≥9 and bag not full, store one unit."""
        if self.energy >= 9 and self.bag < self.bag_capacity:
            self.bag += 1
            self.last_collected = spot
            self.memory_spot    = spot
            self.known_food.append(spot)
            if len(self.known_food) > 20:
                self.known_food.pop(0)

    def _can_move(self, newx: int, newy: int) -> bool:
        """Check map bounds, obstacles, and terrain != -1."""
        h, w = self.codes.shape
        if not (0 <= newx < w and 0 <= newy < h):
            return False
        if (newx, newy) in self.obstacles:
            return False
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
        """Pick one of four angled moves, biased toward forward."""
        fx, fy = self.dir_x, self.dir_y
        choices = [(fx, fy), (-fy, fx), (fy, -fx), (-fx, -fy)]
        weights = [0.7, 0.15, 0.15, 0.005]
        dx, dy = random.choices(choices, weights)[0]
        newx, newy = self.x + dx, self.y + dy
        if self._can_move(newx, newy):
            self.perform_action(cost)
            self.x, self.y = newx, newy
            self.dir_x, self.dir_y = dx, dy

    def sleep(self, in_house: bool) -> None:
        """Regenerate sleep in-house, lose sleep outside, die at 0."""
        if in_house:
            self.sleep_count = min(self.max_sleep_count, self.sleep_count + 10)
            if self.sleep_count != self.max_sleep_count:
                self.stay_in_house()
        else:
            self.sleep_count = max(0.0, self.sleep_count - 0.5)
        if self.sleep_count <= 0.0:
            self.alive = False

    def deposit_food(self) -> None:
        """If at home, move all bagged food into house storage."""
        if (self.x, self.y) == (self.home_x, self.home_y) and self.bag > 0:
            self.contributed += self.bag
            self.home.store_food(self.bag)
            self.bag = 0


  
    def share_food_with(
            self,
            other: "Human",
            trust_system: TrustSystem,
            amount: int = 1,
            trust_inc: float = 0.01
        ) -> bool:
            """
            Donne `amount` unité(s) de mon sac à `other` si adjacent,
            et augmente de `trust_inc` la confiance que other a en moi.
            """
            dx = abs(self.x - other.x)
            dy = abs(self.y - other.y)
            if max(dx, dy) <= 1 and self.bag > 0:
                give = min(self.bag, amount)
                self.bag -= give
                other.bag += give
                
                # **Ici** on augmente la confiance que `other` a en `self`
                trust_system.increase_trust(
                    trustor_id=other.id,
                    trustee_id=self.id,
                    increment=trust_inc
                )
                return True
            return False    
    def step(
        self,
        resources,
        houses,
        humans,
        trust_system,
        action_cost: float = 0.01,
        food_gain: float = 1.0,
        decay_rate: float = 0.005
    ) -> Tuple[Optional[Tuple[int,int]], bool]:
        """
        One tick of behavior: movement, foraging, depositing, sharing.

        Returns:
        - picked: (x,y) if this human picked up a resource this tick, else None
        - shared: True if this human gave food to a neighbor this tick, else False

        We return a tuple `(picked, shared)` on every code path so that:
        1. Callers can always unpack two values without a crash.
        2. We avoid `TypeError: cannot unpack non-iterable NoneType`.
        """
        # ─── initialize outputs ────────────────────────────────────────────────
        picked = None
        shared = False

        # ─── 0) Am I inside my house? ───────────────────────────────────────────
        in_house = any(
            h.x <= self.x < h.x + HOUSE_SIZE and
            h.y <= self.y < h.y + HOUSE_SIZE
            for h in houses
        )

        # ─── 1) Sleep & energy decay ────────────────────────────────────────────
        self.sleep(in_house)
        if not self.alive:
            return None, False       # died in sleep
        self.decay_energy(decay_rate)
        if not self.alive:
            return None, False       # died of exhaustion

        # ─── 2) Tired? go home ─────────────────────────────────────────────────
        if self.sleep_count < 75:
            if (self.x, self.y) != (self.home_x, self.home_y):
                self.move_towards(self.home_x, self.home_y, action_cost)
            return None, False       # no picking/sharing this tick

        # ─── 2a) Starving? fetch from home ─────────────────────────────────────
        if self.energy <= 2 and self.bag == 0:
            if (self.x, self.y) != (self.home_x, self.home_y):
                self.move_towards(self.home_x, self.home_y, action_cost)
                return None, False
            if self.home.storage > 0 and self.energy < self.max_energy:
                self.eat(food_gain * 10)
                self.home.storage -= 1
            return None, False

        # ─── 3) Out of energy entirely ─────────────────────────────────────────
        if self.energy <= 0.0:
            if self.bag > 0:
                self.bag -= 1
                self.energy = self.max_energy
            else:
                self.alive = False
            return None, False       # either refueled or died

        # ─── 3a) Bag full? deposit and rest ────────────────────────────────────
        if self.bag >= self.bag_capacity:
            if not in_house:
                self.move_towards(self.home_x, self.home_y, action_cost)
                return None, False
            self.deposit_food()
            return None, False       # after deposit, nothing else

        # ─── 4) Move toward known food spot ────────────────────────────────────
        if self.memory_spot and self.bag < self.bag_capacity:
            mx, my = self.memory_spot
            if (self.x, self.y) != (mx, my):
                self.move_towards(mx, my, action_cost)
                return None, False
            # forget if gone
            if not any((r.x, r.y) == (mx, my) for r in resources):
                self.memory_spot = None

        # ─── 5) Prune stale known_food ─────────────────────────────────────────
        self.known_food = [
            spot for spot in self.known_food
            if any((r.x, r.y) == spot for r in resources)
        ]

        # ─── 6) Explore or head to closest ─────────────────────────────────────
        if self.known_food:
            tx, ty = min(
                self.known_food,
                key=lambda pos: (pos[0] - self.x)**2 + (pos[1] - self.y)**2
            )
            self.move_towards(tx, ty, action_cost)
        else:
            for _ in range(self.exploration_factor):
                if self.energy <= 0 or self.bag >= self.bag_capacity:
                    break
                self.random_move(action_cost)

        # ─── 7) Pick up or eat resource ────────────────────────────────────────
        if not in_house and self.bag < self.bag_capacity:
            for resource in list(resources):
                dx, dy = resource.x - self.x, resource.y - self.y
                if dx*dx + dy*dy <= 1.5**2:
                    if self.energy >= 9:
                        self.store_in_bag(spot=(resource.x, resource.y))
                    else:
                        self.eat(food_gain, spot=(resource.x, resource.y))
                    resources.remove(resource)
                    picked = (resource.x, resource.y)
                    break

        # ─── 8) Share memory spots ─────────────────────────────────────────────
        SHARE_RADIUS = 2
        R2 = SHARE_RADIUS * SHARE_RADIUS
        for other in humans:
            if other is not self:
                dx, dy = other.x - self.x, other.y - self.y
                if dx*dx + dy*dy <= R2 and self.memory_spot:
                    other.memory_spot = self.memory_spot

        # ─── 9) Share food with adjacent human ─────────────────────────────────
        for other in humans:
            if other is not self and self.bag > 0 and other.bag < other.bag_capacity:
                if self.share_food_with(other, trust_system):
                    shared = True
                    break  # only one share per tick

        # ─── final unified return ───────────────────────────────────────────────
        return picked, shared


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
    bg2 = pygame.Rect(bx, y2, bar_w, bar_h)
    bg3 = pygame.Rect(bx, y3, bar_w, bar_h)
    pygame.draw.rect(screen, (50, 50, 50), bg1)
    pygame.draw.rect(screen, (50, 50, 50), bg2)
    pygame.draw.rect(screen, (50, 50, 50), bg3)
    # fills
    e_frac = human.energy / human.max_energy if human.max_energy>0 else 0
    s_frac = human.sleep_count / human.max_sleep_count if human.max_sleep_count>0 else 0
    b_frac = human.bag / human.bag_capacity if human.bag_capacity>0 else 0
    pygame.draw.rect(screen, (0,255,0), (bx, y1, int(bar_w*e_frac), bar_h))
    pygame.draw.rect(screen, (0,128,255), (bx, y2, int(bar_w*s_frac), bar_h))
    pygame.draw.rect(screen, (255,0,0), (bx, y3, int(bar_w*b_frac), bar_h))
    # borders
    pygame.draw.rect(screen, (0,0,0), bg1, 1)
    pygame.draw.rect(screen, (0,0,0), bg2, 1)
    pygame.draw.rect(screen, (0,0,0), bg3, 1)





