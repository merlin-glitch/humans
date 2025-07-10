# # map.py
# import pygame
# import random
# import math
# from config import MAP_WIDTH, MAP_HEIGHT, CELL_SIZE, MAX_LIFE , HOUSE_POSITIONS


# # Centre fixe : à 80% de la largeur, 50% de la hauteur
# CLUSTER_CENTER = (int(MAP_WIDTH * 0.8), MAP_HEIGHT // 2)
# CLUSTER_STD    = 5  # écart-type en cellules


# class Resource:
#     def __init__(self, x, y, life=None):
#         self.x = x
#         self.y = y
#         self.life = MAX_LIFE if life is None else life

#     def update(self):
#         self.life -= 1

#     def is_alive(self):
#         return self.life > 0

#     def get_color(self):
#         ratio = max(0.0, min(1.0, self.life / MAX_LIFE))
#         dark_green = (0, 250, 0)
#         dark_red   = (250, 0, 0)
#         r = int(dark_red[0] * (1 - ratio) + dark_green[0] * ratio)
#         g = int(dark_red[1] * (1 - ratio) + dark_green[1] * ratio)
#         b = int(dark_red[2] * (1 - ratio) + dark_green[2] * ratio)
#         return (r, g, b)

# class House:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

# from config import HOUSE_SIZE

# class House:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def draw(self, screen):
#         pygame.draw.rect(
#             screen,
#             (150,75,0),
#             pygame.Rect(self.x*CELL_SIZE,
#                         self.y*CELL_SIZE,
#                         HOUSE_SIZE*CELL_SIZE,
#                         HOUSE_SIZE*CELL_SIZE)
#         )

# def generate_houses(params=None):
#     # ignore params; return fixed positions
#     return [House(x, y) for x, y in HOUSE_POSITIONS]

# def is_in_house(x, y, houses):
#     for h in houses:
#         if h.x <= x < h.x + HOUSE_SIZE and h.y <= y < h.y + HOUSE_SIZE:
#             return True
#     return False


# def generate_cluster_centers(cluster_count, margin, houses):
#     """
#     Retourne une liste de centres (x,y) pour les clusters,
#     en évitant de tomber dans une maison et en laissant une marge.
#     """
#     centers = []
#     attempts = 0
#     while len(centers) < cluster_count and attempts < cluster_count * 1000:
#         x = random.randint(margin, MAP_WIDTH - margin - 1)
#         y = random.randint(margin, MAP_HEIGHT - margin - 1)
#         # on évite les maisons
#         if any(h.x <= x < h.x + HOUSE_SIZE and h.y <= y < h.y + HOUSE_SIZE for h in houses):
#             attempts += 1
#             continue
#         centers.append((x, y))
#         attempts += 1
#     return centers

# def generate_resources(params, houses):
#     """
#     Génère toutes les ressources autour d'un seul cluster
#     fixé à CLUSTER_CENTER avec distribution normale.
#     """
#     total = params['food_count']
#     cx, cy = CLUSTER_CENTER
#     std = CLUSTER_STD
#     resources = []
#     while len(resources) < total:
#         # tirage gaussien autour du centre droite
#         x = int(random.gauss(cx, std))
#         y = int(random.gauss(cy, std))
#         # bornes dans la carte
#         x = max(0, min(MAP_WIDTH - 1, x))
#         y = max(0, min(MAP_HEIGHT - 1, y))
#         # pas sous une maison
#         if any(h.x <= x < h.x + HOUSE_SIZE and
#                h.y <= y < h.y + HOUSE_SIZE
#                for h in houses):
#             continue
#         life = random.randint(0, MAX_LIFE)
#         resources.append(Resource(x, y, life=life))
#     return resources


# def draw_map(screen, resources, houses, params):
#     """
#     Dessine la carte en fonction de params:
#       params['cell_size'], params['house_size']
#     """
#     cs = params['cell_size']
#     screen.fill((30, 150, 50))

#     # maisons
#     for h in houses:
#         pygame.draw.rect(
#             screen,
#             (150, 75, 0),
#             pygame.Rect(h.x * cs, h.y * cs, params['house_size'] * cs, params['house_size'] * cs)
#         )

#     # ressources
#     for r in resources:
#         color = r.get_color()
#         pygame.draw.rect(
#             screen,
#             color,
#             pygame.Rect(r.x * cs, r.y * cs, cs, cs)
#         )

#     #pygame.display.flip()
