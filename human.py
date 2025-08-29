

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
        self.storage +=round(amount) 


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
    
    def share_memory_spot(self, humans):
            """
            Share memory spot with nearby humans within SHARE_RADIUS.
            """
            SHARE_RADIUS = 2
            R2 = SHARE_RADIUS * SHARE_RADIUS
            for other in humans:
                if other is not self:
                    dx, dy = other.x - self.x, other.y - self.y
                    if dx*dx + dy*dy <= R2 and self.memory_spot:
                        other.memory_spot = self.memory_spot

                        
    def step(
            self,
            resources,
            houses,
            humans,
            trust_system,
            is_day: bool,
            action_cost: float = 0.01,
            food_gain: float = 2.0,
            decay_rate: float = 0.005,
        ) -> Tuple[Optional[Tuple[int,int]], bool]:
        """
        One tick of behavior: movement, foraging, depositing, sharing.
        Returns (picked_coord, shared_bool).
        """
        # ── obéir au leader en tout début de journée ────────────────────────────
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
        in_house = any(
            h.x <= self.x < h.x + HOUSE_SIZE and
            h.y <= self.y < h.y + HOUSE_SIZE
            for h in houses
        )

        self.decay_energy(decay_rate)
        if not self.alive:
            return None, False

        if self.energy <= 3 and self.bag == 0:
            if (self.x, self.y) != (self.home_x, self.home_y):
                self.move_towards(self.home_x, self.home_y, action_cost)
                return None, False
            if self.home.storage > 0 and self.energy < self.max_energy:
                self.eat(food_gain * 10)
                self.home.storage -= 1
            return None, False

        if self.energy <= 1.0:
            if self.bag > 0:
                self.bag -= 1
                self.energy += food_gain
            else:
                self.alive = False
            return None, False

        if self.bag >= self.bag_capacity:
            if not in_house:
                self.move_towards(self.home_x, self.home_y, action_cost)
                return None, False
            self.deposit_food()
            return None, False

        # 6) Move toward known food or explore
        if self.memory_spot and self.bag < self.bag_capacity:
            mx, my = self.memory_spot
            if (self.x, self.y) != (mx, my):
                self.move_towards(mx, my, action_cost)
                return None, False
            if resources[my, mx, 1] <= 0:   # ressource épuisée
                self.memory_spot = None

        # prune stale memory
        self.known_food = [
            spot for spot in self.known_food
            if resources[spot[1], spot[0], 1] > 0
        ]

        if self.known_food:
            tx, ty = min(
                self.known_food,
                key=lambda pos: (pos[0]-self.x)**2 + (pos[1]-self.y)**2
            )
            self.move_towards(tx, ty, action_cost)
        else:
            for _ in range(self.exploration_factor):
                if self.energy <= 0 or self.bag >= self.bag_capacity:
                    break
                self.random_move(action_cost)

        # 7) Pick up / eat resource
        picked = None
        if not in_house and self.bag < self.bag_capacity:
            # vérifier uniquement autour (x,y) ± 1
            for nx in range(self.x-1, self.x+2):
                for ny in range(self.y-1, self.y+2):
                    if 0 <= nx < resources.shape[1] and 0 <= ny < resources.shape[0]:
                        if resources[ny, nx, 1] > 0:  # nourriture dispo
                            if self.energy >= 9:
                                self.store_in_bag(spot=(nx, ny))
                            else:
                                self.eat(food_gain, spot=(nx, ny))
                            resources[ny, nx] = [0, 0]  # remove_resource
                            picked = (nx, ny)
                            break
                if picked: break

        # 9) Trust-based share
        shared = False
        for other in humans:
            if other is self:
                continue
            if max(abs(self.x-other.x), abs(self.y-other.y)) > 1:
                continue
            t = trust_system.trust_score(self.id, other.id)
            if t > 0.5 and self.bag > 0:
                self.bag    -= 1
                other.bag   += 1
                other.memory_spot = self.memory_spot
                trust_system.increase_trust(
                    trustor_id=other.id,
                    trustee_id=self.id,
                    increment=t+0.01
                )
                shared = True
                break
            elif t < 0.5:
                bogus = random.choice(self.known_food or [(0,0)])
                other.memory_spot = bogus
                trust_system.increase_trust(
                    trustor_id=other.id,
                    trustee_id=self.id,
                    increment=t-0.01
                )
                shared = True
                break

        return picked, shared
    

    # def step(
    #     self,
    #     resources,
    #     houses,
    #     humans,
    #     trust_system,
    #     is_day: bool,
    #     action_cost: float = 0.01,
    #     food_gain: float = 2.0,
    #     decay_rate: float = 0.005,
    # ) -> Tuple[Optional[Tuple[int,int]], bool]:
    #     """
    #     One tick of behavior: movement, foraging, depositing, sharing.
    #     Returns (picked_coord, shared_bool).
    #     """
        

    #         # ── obéir au leader en tout début de journée ────────────────────────────
    #     if is_day and getattr(self, "next_day_target", None):
    #         tx, ty = self.next_day_target
    #         # si on n’est pas encore arrivé, on y va
    #         if (self.x, self.y) != (tx, ty):
    #             self.move_towards(tx, ty, action_cost)   # <<< TX, TY ici, pas home_x/home_y !
    #             return None, False
    #         # une fois arrivé, on supprime la cible pour reprendre le foraging
    #         del self.next_day_target
    #         return None, False
        
        


    #     # ── NIGHT BEHAVIOR ───────────────────────────────
    #     if not is_day:
    #         # head home
    #         if (self.x, self.y) != (self.home_x, self.home_y):
    #             self.move_towards(self.home_x, self.home_y, action_cost)
    #             if self.bag > 0:
    #                 self.deposit_food()
    #             return None, False
            
    #     # ── DAY BEHAVIOR ───────────────────────────────────
    #     # # 0) If leader told us where to go, obey


    #     # 1) Sleep & energy decay
    #     in_house = any(
    #         h.x <= self.x < h.x + HOUSE_SIZE and
    #         h.y <= self.y < h.y + HOUSE_SIZE
    #         for h in houses
    #     )
    #     # self.sleep(in_house)
    #     # if not self.alive:
    #     #     return None, False
    #     self.decay_energy(decay_rate)
    #     if not self.alive:
    #         return None, False

    #     # # 2) If tired go home
    #     # if self.sleep_count < 75:
    #     #     if (self.x, self.y) != (self.home_x, self.home_y):
    #     #         self.move_towards(self.home_x, self.home_y, action_cost)
    #     #     return None, False

    #     # 3) If starving fetch from home
    #     if self.energy <= 3 and self.bag == 0:
    #         if (self.x, self.y) != (self.home_x, self.home_y):
    #             self.move_towards(self.home_x, self.home_y, action_cost)
    #             return None, False
    #         if self.home.storage > 0 and self.energy < self.max_energy:
    #             self.eat(food_gain * 10)
    #             self.home.storage -= 1
    #         return None, False

    #     # 4) If out of energy entirely
    #     if self.energy <= 1.0:
    #         if self.bag > 0:
    #             self.bag -= 1
    #             self.energy += food_gain
    #         else:
    #             self.alive = False
    #         return None, False

    #     # 5) If bag full, deposit
    #     if self.bag >= self.bag_capacity:
    #         if not in_house:
    #             self.move_towards(self.home_x, self.home_y, action_cost)
    #             return None, False
    #         self.deposit_food()
            
    #         return None, False
        
         
    #     # 6) Move toward known food or explore
    #     if self.memory_spot and self.bag < self.bag_capacity:
    #         mx, my = self.memory_spot
    #         if (self.x, self.y) != (mx, my):
    #             self.move_towards(mx, my, action_cost)
    #             return None, False
    #         if not any((r.x, r.y) == (mx, my) for r in resources):
    #             self.memory_spot = None

    #     # prune stale
    #     self.known_food = [
    #         spot for spot in self.known_food
    #         if any((r.x, r.y) == spot for r in resources)
    #     ]

    #     # explore
    #     if self.known_food:
    #         tx, ty = min(
    #             self.known_food,
    #             key=lambda pos: (pos[0]-self.x)**2 + (pos[1]-self.y)**2
    #         )
    #         self.move_towards(tx, ty, action_cost)
    #     else:
    #         for _ in range(self.exploration_factor):
    #             if self.energy <= 0 or self.bag >= self.bag_capacity:
    #                 break
    #             self.random_move(action_cost)

    #     # 7) Pick up / eat resource
    #     picked = None
    #     if not in_house and self.bag < self.bag_capacity:
    #         for resource in list(resources):
    #             dx, dy = resource.x - self.x, resource.y - self.y
    #             if dx*dx + dy*dy <= 1**2:
    #                 if self.energy >= 9:
    #                     self.store_in_bag(spot=(resource.x, resource.y))
    #                 else:
    #                     self.eat(food_gain, spot=(resource.x, resource.y))
    #                 resources.remove(resource)
    #                 picked = (resource.x, resource.y)
    #                 break




    #     # # Call the function in step()
    #     # self.share_memory_spot(humans) 

    #      # 9) Trust‑based share
    #     shared = False
    #     for other in humans:
    #         if other is self:
    #             continue
    #         if max(abs(self.x-other.x), abs(self.y-other.y)) > 1:
    #             continue
    #         t = trust_system.trust_score(self.id, other.id)
    #         if t > 0.5 and self.bag > 0 :
    #             # real share
    #             self.bag    -= 1
    #             other.bag   += 1
    #             other.memory_spot = self.memory_spot
    #             trust_system.increase_trust(
    #                 trustor_id=other.id,
    #                 trustee_id=self.id,
    #                 increment=t+0.01
    #             )
    #             shared = True
    #             break
    #         elif t < 0.5 :
    #             # sabotage
    #             bogus = random.choice(self.known_food or ( 0,0))
    #             other.memory_spot = bogus
    #             trust_system.increase_trust(
    #                 trustor_id=other.id,
    #                 trustee_id=self.id,
    #                 increment=t-0.01
    #             )
    #             shared = True
    #             break

    #     return picked, shared


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





