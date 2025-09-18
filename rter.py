import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------------
# Paramètres
# ---------------------
WIDTH, HEIGHT = 100, 60
NB_RUNS = 100
DAYS = 200
INIT_BLUE = 50
INIT_RED = 50
INIT_HP = 3
FOOD_RESPAWN = 10  # tous les 10 jours
FOOD_ZONES = [(10, 10), (50, 20), (70, 40)]  # coins sup-gauche des zones 20x20
ZONE_SIZE = 20

# Maisons
HOUSE_BLUE = (0, 0)
HOUSE_RED = (WIDTH-1, HEIGHT-1)

# ---------------------
# Fonctions utilitaires
# ---------------------
def init_food_map():
    food = np.zeros((WIDTH, HEIGHT), dtype=bool)
    for zx, zy in FOOD_ZONES:
        for x in range(zx, zx+ZONE_SIZE):
            for y in range(zy, zy+ZONE_SIZE):
                food[x, y] = True
    return food

def random_move(x, y):
    moves = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]  # reste possible
    dx, dy = random.choice(moves)
    nx, ny = max(0, min(WIDTH-1, x+dx)), max(0, min(HEIGHT-1, y+dy))
    return nx, ny

class Individual:
    def __init__(self, x, y, hp=INIT_HP, color="blue"):
        self.x, self.y, self.hp, self.color = x, y, hp, color
        self.food_today = 0

    def step(self, food_map):
        self.food_today = 0
        self.x, self.y = random_move(self.x, self.y)
        if food_map[self.x, self.y]:
            self.hp += 1
            self.food_today += 1
        else:
            self.hp -= 1
        return self.hp > 0

# ---------------------
# Simulation d’un run
# ---------------------
def simulate_run():
    food_map = init_food_map()
    blue = [Individual(*HOUSE_BLUE, color="blue") for _ in range(INIT_BLUE)]
    red = [Individual(*HOUSE_RED, color="red") for _ in range(INIT_RED)]

    blue_counts, red_counts = [], []

    for day in range(DAYS):
        # régénération nourriture tous les FOOD_RESPAWN jours
        if day % FOOD_RESPAWN == 0:
            food_map = init_food_map()

        new_blue, new_red = [], []

        # update bleus
        for ind in blue:
            if ind.step(food_map):
                new_blue.append(ind)
                if ind.food_today >= 6:
                    new_blue.append(Individual(*HOUSE_BLUE, color="blue"))
            # sinon mort
        blue = new_blue

        # update rouges
        for ind in red:
            if ind.step(food_map):
                new_red.append(ind)
                if ind.food_today >= 6:
                    new_red.append(Individual(*HOUSE_RED, color="red"))
        red = new_red

        blue_counts.append(len(blue))
        red_counts.append(len(red))

    return np.array(blue_counts), np.array(red_counts)

# ---------------------
# Runs multiples
# ---------------------
all_blue = []
all_red = []
for r in range(NB_RUNS):
    b, rpop = simulate_run()
    all_blue.append(b)
    all_red.append(rpop)

all_blue = np.array(all_blue)
all_red = np.array(all_red)

# ---------------------
# Analyse (moyenne et quantiles)
# ---------------------
mean_blue = all_blue.mean(axis=0)
mean_red = all_red.mean(axis=0)
q10_blue, q90_blue = np.percentile(all_blue, [10, 90], axis=0)
q10_red, q90_red = np.percentile(all_red, [10, 90], axis=0)

# ---------------------
# Graphe
# ---------------------
plt.figure(figsize=(10,6))

# courbes individuelles (transparence)
for i in range(NB_RUNS):
    plt.plot(all_blue[i], color="blue", alpha=0.05)
    plt.plot(all_red[i], color="red", alpha=0.05)

# moyenne
plt.plot(mean_blue, color="blue", linewidth=2, label="Bleus (moyenne)")
plt.plot(mean_red, color="red", linewidth=2, label="Rouges (moyenne)")

# bandes 10–90 %
plt.fill_between(range(DAYS), q10_blue, q90_blue, color="blue", alpha=0.2)
plt.fill_between(range(DAYS), q10_red, q90_red, color="red", alpha=0.2)

plt.xlabel("Jour")
plt.ylabel("Population")
plt.title("Simulation populations bleues et rouges ({} runs, {} jours)".format(NB_RUNS, DAYS))
plt.legend()
plt.grid(True)
plt.show()
