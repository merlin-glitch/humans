#----------------------menu.py----------------
# Taille d'une cellule en pixels
CELL_SIZE = 6  # Ajustable pour agrandir la fenêtre
# Nombre de points de nourriture à générer\
FOOD_COUNT = 30

# Durée de vie maximale d'une ressource (en ticks)
MAX_LIFE = 300
# Nombre de maisons à placer
HOUSE_COUNT = 2
# Taille des maisons en cellules
HOUSE_SIZE = 10
# Distance minimale entre deux maisons (en cellules)
MIN_HOUSE_DISTANCE = 30
#--------------------------
# config.py


Nbre_HUMANS = 10
# ── Taille de la carte (en cellules) ────────────────────────────────────────
# Remplacez par les dimensions réelles de votre .map
MAP_WIDTH  = 100
MAP_HEIGHT = 60

# ── Taille d’une cellule (en pixels) ───────────────────────────────────────
CELL_SIZE = 16

# ── Palette terrain : mapping RGB → code entier ─────────────────────────────
# (correspond à votre legend : 0=beach,1=ocean,-1=obstacle,2=safe1,3=safe2)
TERRAIN_PALETTE = {
    (255, 255,   56):  0,   # gold  → plage
    (  0,   0, 255):  1,   # blue  → océan
    (  0,   0,   0): -1,   # black → obstacle
    ( 255, 0,   0):  2,   # green → safe space 1
    (0,   0,   128):  3,   # darkblue   → safe space 2
    ( 54, 109,   70):  4,   # purple → food zone 1
    (255,   0,   255):  5,   # pink   → food zone 2
    (94,  185,   30):  6,   # green   → grass

}

NEW_PALETTE = {
    (255, 255,   56):  0,   # gold  → plage
    (  0,   0, 255):  1,   # blue  → océan
    (  0,   0,   0): -1,   # black → obstacle
    ( 255, 0,   0):  2,   # red → safe space 1
    (0,   0,   128):  3,   # darkblue   → safe space 2
    (94,  200,   30):  4,   # green → food zone 1
    (255,   0,   255):  5,   # pink   → food zone 2
    (94,  185,   30):  6,   # green   → grass

}


# ── Paramètres de la nourriture ────────────────────────────────────────────


# Couleur de départ (fraîche → dark green) et d’arrivée (disparition → dark red)
FOOD_START_RGB = (  0, 100,   0)
FOOD_END_RGB   = (100,   0,   0)


# how many resources to seed at start
INITIAL_FOOD_COUNT = Nbre_HUMANS # 1 unité pour chaque human        

# every SPAWN_INTERVAL ticks, spawn FOOD_SPAWN_COUNT new food items
SPAWN_INTERVAL     = 100      
FOOD_SPAWN_COUNT   = 100        

# how long (in ticks) a piece of food remains alive
FOOD_LIFETIME      = 900      
# how many items per tile max
FOOD_STACK        = 20

#energy cost to mate
ENERGY_COST       = 5.0
# DAY length and mating cooldown
DAY_LENGTH        = 350
MATING_COOLDOWN   = 1 * DAY_LENGTH  # 30 days