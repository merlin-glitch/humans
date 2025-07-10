# human.py
import random
import pygame
from config import MAP_WIDTH, MAP_HEIGHT, HOUSE_SIZE

class Human:
    def __init__(
        self,
        human_id,
        sex,
        x,
        y,
        home,
        codes,                     # static terrain codes array
        initial_energy: float = 10.0,
        exploration_factor: int= 1,
        bag_capacity: int      = 10,
        initial_sleep: float   = 200.0

    ):
        self.id                = human_id
        self.sex               = sex
        self.home              = home
        self.home_x            = home.x + HOUSE_SIZE // 2
        self.home_y            = home.y + HOUSE_SIZE // 2
        self.x                 = x
        self.y                 = y
        self.max_energy        = max(0.0, initial_energy)
        self.energy            = self.max_energy
        #self.energy_reserve    = initial_reserve
        self.bag               = 0
        self.bag_capacity      = bag_capacity
        self.exploration_factor= exploration_factor
        self.dir_x, self.dir_y = random.choice([(-1, -1),( 0, -1),( 1, -1),(-1,  0),( 1,  0),(-1,  1), ( 0,  1), ( 1,  1)])
        self.known_food        = []
        self.last_collected    = None
        self.memory_spot       = None  #remember last food pixel
        self.max_sleep_count   = initial_sleep
        self.sleep_count       = initial_sleep
        self.alive             = True
        # static map codes, used to block any tile == -1
        self.codes = codes
        self.obstacles         = set()   # : remember blocked tiles

    def perform_action(self, cost=0.01):# each action has a cost taken from the human's self energy
        self.energy = max(0.0, self.energy - cost)

    def decay_energy(self, rate=0.005):# even when staying still the human consumes energy
        self.energy = max(0.0, self.energy - rate)

    def eat(self, gain=1.0, spot=None):
        #Replenish energy and remember the food spot.
        self.energy = min(self.max_energy, self.energy + gain)  # prevent overfilling energy

        if spot is not None:
            self.last_collected = spot
            self.memory_spot = spot
            self.known_food.append(spot)
            if len(self.known_food) > 5:
                self.known_food.pop(0)

    def store_in_bag(self,spot=None):
        #Store food in bag if energy is sufficient.
        if self.energy >= 9 and self.bag < self.bag_capacity:
            self.bag += 1
            self.last_collected = spot
            self.memory_spot = spot
            self.known_food.append(spot)
            if len(self.known_food) > 20:
                self.known_food.pop(0)


    def _can_move(self, newx, newy): # defines where the humans can move
        h, w = self.codes.shape # h, w of the map
        if not (0 <= newx < w and 0 <= newy < h):# no mouvements outside the borders
            return False
        
        if (newx, newy) in self.obstacles:   # ← new obstacle check , no mvt on obstacle tiles
           return False
        
        return self.codes[newy][newx] != -1
    
    def stay_in_house(self):
    #Force the human to remain inside its house for this tick.
        self.x, self.y = self.home_x, self.home_y


    def move_towards(self, tx, ty, cost=0.01):
        # Compute direction vector
        dx = tx - self.x
        dy = ty - self.y

        # Normalize direction to step (-1, 0, or 1)
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)

        # Try to move diagonally first (8-direction movement)
        newx = self.x + step_x
        newy = self.y + step_y

        if self._can_move(newx, newy):
            self.perform_action(cost)
            self.x, self.y = newx, newy
            self.dir_x, self.dir_y = step_x, step_y
        else:
            # Try to move in x direction only
            if step_x != 0 and self._can_move(self.x + step_x, self.y):
                self.perform_action(cost)
                self.x += step_x
                self.dir_x, self.dir_y = step_x, 0
            # Try to move in y direction only
            elif step_y != 0 and self._can_move(self.x, self.y + step_y):
                self.perform_action(cost)
                self.y += step_y
                self.dir_x, self.dir_y = 0, step_y
            else:
                # All directions blocked: mark obstacle and detour randomly
                self.obstacles.add((newx, newy))
                self.random_move(cost)

    def deposit_food(self):
        """
        If I'm standing at home, move everything from my bag into my house storage.
        """
        # check position
        if (self.x, self.y) == (self.home_x, self.home_y) and self.bag > 0:
            self.home.store_food(self.bag)
            self.bag = 0    


# redefine it with angles 
    def random_move(self, cost=0.01):
        fx, fy = self.dir_x, self.dir_y #previous direction
        choices = [(fx, fy), (-fy, fx), (fy, -fx), (-fx, -fy)] # possible choices
        weights = [0.7, 0.15, 0.15, 0.005] # weights that favor going forward more , less chances of going backward, done to encourage exploration away from safe spaces
        dx, dy = random.choices(choices, weights)[0]# funtion choices in python returns a list , so we only take the first item
        newx = self.x + dx
        newy = self.y + dy
        if self._can_move(newx, newy):
            self.perform_action(cost)
            self.x, self.y = newx, newy
            self.dir_x, self.dir_y = dx, dy

    def sleep(self, in_house: bool):
        if in_house:
            self.sleep_count = min(self.max_sleep_count, self.sleep_count + 10) # 10 is the rate by which sleep is generated 10 sleep_count per ticks
            if self.sleep_count!= self.max_sleep_count:
                self.stay_in_house()
        else:
            self.sleep_count = max(0.0, self.sleep_count - 0.5)# 0.5 is the rate by which sleep diminishs
        if self.sleep_count <= 0.0:# human is dead if sleep reaches 0
            self.alive = False

#function to define human actions
    def step(self, resources, houses, humans,
             action_cost=0.01,
             food_gain=1.0,
             decay_rate=0.005):
        # 0) in house? ( check if the human is in the house)
        in_house = any(
            h.x <= self.x < h.x + HOUSE_SIZE and
            h.y <= self.y < h.y + HOUSE_SIZE
            for h in houses
        )

        # 1) sleep update
        self.sleep(in_house)
        if not self.alive:# checks if the agent is still alive in the house
            return
        
        # 2) if tired, go home
        if self.sleep_count < 75:
            if (self.x, self.y) != (self.home_x, self.home_y):
                        self.move_towards(self.home_x, self.home_y, action_cost)
            return
        # 3) fully rested => start foraging
        # 3a) passive energy decay                       
        self.decay_energy(decay_rate)
    
        if self.energy <= 0.0:
            if self.bag > 0:
                self.bag -= 1
                # gain one unit (or food_gain) without exceeding max
                self.energy = min(self.max_energy, self.energy + food_gain)
            # elif self.energy_reserve > 0:
            #     self.energy_reserve -= 1
            #     self.energy = min(self.max_energy, self.energy + food_gain)
            else:
                self.alive = False
                return

        # 3) passive energy decay
        self.decay_energy(decay_rate)
        if self.energy <= 0.0:
            # first try stored bag food
            if self.bag > 0:
                self.bag -= 1
                self.energy = self.max_energy
            # then try energy reserve
            # elif self.energy_reserve > 0:
            #     self.energy_reserve -= 1
            #     self.energy = self.max_energy
            else:
                # no energy sources → death
                self.alive = False
                return

        # # 3b) deposit if bag full
        if self.bag >= self.bag_capacity:
            if not in_house:
                self.move_towards(self.home_x, self.home_y, action_cost)
                return
            
            # → deposit if you’ve got food and you’re home
            if in_house :
                self.deposit_food()
                return
        



        # 4) if we have a memory of a food spot, go there first
        if self.memory_spot and self.bag < self.bag_capacity:
            mx, my = self.memory_spot
            if (self.x, self.y) != (mx, my):
                self.move_towards(mx, my, action_cost)
                return
            else:
                # reached: clear memory if no resource remains
                if not any(r.x == mx and r.y == my for r in resources):
                    self.memory_spot = None
        # 5) clean known-food
        self.known_food = [
            (px, py)
            for item in self.known_food # in case there is None in the list
            if item is not None
            for (px, py) in [item]
            if any((res.x - px) ** 2 + (res.y - py) ** 2 <= 1.5**2 for res in resources)
        ]

        

        # 6) normal move: known spot or random
        if self.known_food:
            tx, ty = min(
                self.known_food,
                key=lambda pos: (pos[0]-self.x)**2 + (pos[1]-self.y)**2 #calculate the closest distance to the human
            )
            self.move_towards(tx, ty, action_cost)
        else:
            for _ in range(self.exploration_factor):
                if self.energy <= 0.0 or self.bag >= self.bag_capacity:
                    break
                self.random_move(action_cost)

        # 7) pick up nearby ant remove ressource from map either eaten or in bag
        picked = None
        if not in_house and self.bag < self.bag_capacity:
            for resource in list(resources):
                dx, dy = resource.x - self.x, resource.y - self.y
                if dx*dx + dy*dy <= 1.5**2:
                    if self.energy >= 9:
                        self.store_in_bag()
                    else:
                        self.eat(food_gain, spot=(resource.x, resource.y))
                    resources.remove(resource)
                    picked = (resource.x, resource.y) #pour supprimer l'effet fade , utilier dans map_manage dans main.py

                    break


        # share persistent memory_spot when two humans meet
        # 8) partage si dans un rayon R
        SHARE_RADIUS = 5  # en cellules
        R2 = SHARE_RADIUS * SHARE_RADIUS
        for other in humans:
            if other is not self:
                dx = other.x - self.x
                dy = other.y - self.y
                # distance au carré ≤ R2 ?
                if dx*dx + dy*dy <= R2:
                    # j'échange nos mémoires de spot
                    if self.memory_spot is not None:
                        other.memory_spot = self.memory_spot
                    if other.memory_spot is not None:
                        self.memory_spot = other.memory_spot

        #9)partage de memoire
        return picked



# def draw_human(screen, human, cell_size, reference_house, font):
#     """
#     Draws a human as a circle with:
#       • bag count above head
#       • sleep count above bag
#       • an energy bar (green) and reserve bar (blue) below the circle
#     """
#     # Body color
#     color = (255, 0, 255) if human.home is reference_house else (128, 0, 128)
#     cx = human.x * cell_size + cell_size // 2
#     cy = human.y * cell_size + cell_size // 2
#     r  = cell_size * 1

#     # 1) Draw the human
#     pygame.draw.circle(screen, color, (cx, cy), r)

#     # 2) Bag count (white) above head
#     bag_surf = font.render(str(human.bag), True, (255, 255, 255))
#     bag_y    = cy - r - 4
#     screen.blit(bag_surf, bag_surf.get_rect(center=(cx, bag_y)))

#     # 3) Sleep count (yellow) above the bag
#     sleep_surf = font.render(str(int(human.sleep_count)), True, (255, 200, 0))
#     sleep_y    = bag_y - font.get_linesize() - 2
#     screen.blit(sleep_surf, sleep_surf.get_rect(center=(cx, sleep_y)))

#     # 4) Bars parameters
#     bar_width  = cell_size * 2
#     bar_height = max(4, cell_size // 4)
#     bar_x      = cx - bar_width // 2
#     first_bar_y  = cy + r + 4
#     #second_bar_y = first_bar_y + bar_height + 2

#     # 5) Draw background for both bars (grey)
#     bg1 = pygame.Rect(bar_x, first_bar_y, bar_width, bar_height)
#     #bg2 = pygame.Rect(bar_x, second_bar_y, bar_width, bar_height)
#     pygame.draw.rect(screen, (50, 50, 50), bg1)
#     #pygame.draw.rect(screen, (50, 50, 50), bg2)

#     # 6) Draw green energy bar
#     if human.max_energy > 0:
#         e_frac = human.energy / human.max_energy
#     else:
#         e_frac = 0
#     e_w = int(bar_width * e_frac)
#     pygame.draw.rect(screen, (0, 255, 0), (bar_x, first_bar_y, e_w, bar_height))

#     # 7) Draw blue reserve bar (assume reserve max ~ bag_capacity or choose a constant)
#     #    Here we normalize by bag_capacity so reserve ≥0 always fits.
#     # if human.bag_capacity > 0:
#     #     r_frac = human.energy_reserve / human.bag_capacity
#     # else:
#     # r_frac = 0
#     # r_w = int(bar_width * r_frac)
#     # pygame.draw.rect(screen, (0, 128, 255), (bar_x, second_bar_y, r_w, bar_height))

#     # 8) Borders
#     pygame.draw.rect(screen, (0, 0, 0), bg1, 1)
#     # pygame.draw.rect(screen, (0, 0, 0), bg2, 1)
def draw_human(screen, human, cell_size, reference_house, font):
    """
    Draws a human as a circle with:
      • bag count above head
      • sleep count above bag (replaced by a blue bar labeled 'slp')
      • an energy bar (green) labeled 'energy'
      • a bag-capacity bar (red) labeled 'bag_cap'
    """
    # Body color
    color = (255, 0, 255) if human.home is reference_house else (128, 0, 128)
    cx = human.x * cell_size + cell_size // 2
    cy = human.y * cell_size + cell_size // 2
    r  = cell_size  # radius

    # Draw the human
    pygame.draw.circle(screen, color, (cx, cy), r)

    # ---------------- Bar parameters ----------------
    bar_width   = cell_size * 2
    bar_height  = max(4, cell_size // 4)
    bar_x       = cx - bar_width // 2
    # Bars stacked below the circle
    energy_y    = cy + r + 2
    sleep_y     = energy_y + bar_height + 2
    bag_y       = sleep_y  + bar_height + 2

    # ---------------- Backgrounds ----------------
    bg_energy = pygame.Rect(bar_x, energy_y, bar_width, bar_height)
    bg_sleep  = pygame.Rect(bar_x, sleep_y,  bar_width, bar_height)
    bg_bag    = pygame.Rect(bar_x, bag_y,    bar_width, bar_height)
    pygame.draw.rect(screen, (50, 50, 50), bg_energy)
    pygame.draw.rect(screen, (50, 50, 50), bg_sleep)
    pygame.draw.rect(screen, (50, 50, 50), bg_bag)

    # ---------------- Filled bars ----------------
    # Energy (green)
    e_frac = human.energy / human.max_energy if human.max_energy > 0 else 0
    pygame.draw.rect(screen, (0, 255, 0), (bar_x, energy_y, int(bar_width * e_frac), bar_height))
    # Sleep (blue)
    s_frac = human.sleep_count / human.max_sleep_count if human.max_sleep_count > 0 else 0
    pygame.draw.rect(screen, (0, 128, 255), (bar_x, sleep_y, int(bar_width * s_frac), bar_height))
    # Bag capacity (red)
    b_frac = human.bag / human.bag_capacity if human.bag_capacity > 0 else 0
    pygame.draw.rect(screen, (255, 0, 0), (bar_x, bag_y, int(bar_width * b_frac), bar_height))

    # ---------------- Borders ----------------
    pygame.draw.rect(screen, (0, 0, 0), bg_energy, 1)
    pygame.draw.rect(screen, (0, 0, 0), bg_sleep,  1)
    pygame.draw.rect(screen, (0, 0, 0), bg_bag,    1)

    # ---------------- Labels ----------------


    # 9) Small labels: create a smaller font, e.g. half a cell high (but at least 8px)
    label_size = max(12, cell_size // 2)
    label_font = pygame.font.Font(None, label_size)

    # # Energy label
    # en_surf = label_font.render("energy", True, (255,255,255))
    # screen.blit(en_surf, (bar_x + bar_width + 4, energy_y))
    # # Sleep label
    # slp_surf = label_font.render("slp", True, (255,255,255))
    # screen.blit(slp_surf, (bar_x + bar_width + 4, sleep_y))
    # # Bag label
    # bag_surf = label_font.render("bag_cap", True, (255,255,255))
    # screen.blit(bag_surf, (bar_x + bar_width + 4, bag_y))
