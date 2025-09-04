# import os
# import numpy as np
# from PIL import Image
# from config import *
# import cv2

# # resources[x, y, 0] = lifetime
# # resources[x, y, 1] = food_left
# resources = np.zeros((MAP_HEIGHT, MAP_WIDTH, 2), dtype=np.int32)


# FOOD_COLOR_BGR = (54, 109, 70)  # from config

# def extract_resource_coords_from_zones(zone_map, food_zone_id=4):
#     """
#     Extract resource coordinates directly from zone_map.
#     Only pixels with zone_id == food_zone_id are considered food.
#     """
#     food_mask = (zone_map == food_zone_id)

#     # coords in (x, y) order
#     coords = np.column_stack(np.where(food_mask))

#     return coords, food_mask.astype(np.uint8) * 255  # binary mask for visualization

# def add_resource(x: int, y: int, life: int = FOOD_LIFETIME, food: int = 30):
#     if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH:
#         resources[x, y, 0] = life      # lifetime
#         resources[x, y, 1] = food      # food left

# def remove_resource(x: int, y: int) -> None:
#     """Remove the resource stack at (x, y).""" 
#     if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH: # test a enlever
#         resources[x, y] = [0, 0]  # clear lifetime and food


# def life_span_ressource():
#     """Decrease lifetime of all resources by 1, remove dead ones."""
#     # decrease all lifetimes
#     resources[:, :, 0] -= 1
#     dead_ressources = resources[:, :, 0] <= 0 # find dead resources
#     resources[dead_ressources] = 0  # remove dead resources

# def total_food():
#     """Return the total amount of food left in the world."""
#     return resources[:, :, 1].sum()

# def identify_zones(map_image_path, min_size=10):
#     """
#     Identifie toutes les zones, fusionne les petites avec la plus proche.
#     Retourne zone_map + dictionnaire des IDs nourriture.
#     """
#     img = cv2.imread(map_image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     img = cv2.resize(img, (MAP_WIDTH, MAP_HEIGHT), interpolation=cv2.INTER_NEAREST)

#     zone_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int32)

#     # attribution initiale via la palette
#     for rgb, zone_id in NEW_PALETTE.items():
#         mask = np.all(img == rgb, axis=-1)
#         zone_map[mask] = zone_id

#     # --- étape spéciale : séparer les zones de nourriture ---
#     food_mask = (zone_map == 4).astype(np.uint8)  # ancien id pour "food"
#     num_labels, labels = cv2.connectedComponents(food_mask)

#     # Calculer la taille et le centre de chaque zone
#     sizes, centroids = {}, {}
#     for i in range(1, num_labels):
#         ys, xs = np.where(labels == i)
#         if len(xs) == 0:
#             continue
#         sizes[i] = len(xs)
#         centroids[i] = (np.mean(xs), np.mean(ys))

#     # Séparer grandes zones et petites zones
#     large_zones = {i for i, s in sizes.items() if s >= min_size}
#     small_zones = {i for i, s in sizes.items() if s < min_size}

#     # ID de base pour les nouvelles zones nourriture
#     food_ids = {}
#     base_id = 40
#     mapping = {}

#     # Assigner IDs aux grandes zones
#     for j, i in enumerate(sorted(large_zones), start=1):
#         new_id = base_id + j
#         mapping[i] = new_id
#         food_ids[f"food_zone_{j}"] = new_id

#     # Réattribuer petites zones au plus proche centroid de grande zone
#     for i in small_zones:
#         cx, cy = centroids[i]
#         nearest = min(
#             large_zones,
#             key=lambda j: (centroids[j][0] - cx) ** 2 + (centroids[j][1] - cy) ** 2
#         )
#         mapping[i] = mapping[nearest]

#     # Construire le nouveau zone_map
#     for i in range(1, num_labels):
#         zone_map[labels == i] = mapping[i]

#     print("Zones nourriture identifiées :", food_ids)
#     return zone_map, food_ids



# from typing import Optional

# # état global : cooldown restant par zone
# cooldown_state = {}

# def resource_spawn_interval_inverse(
#     f_avg,
#     zone_id,
#     zone_map,
#     I_max=200,
#     k=0.5,
#     I_min=10,
#     stock_today=None,
#     spawn_baseline=None,
#     reset=False,
#     cooldown_days=5
# ):
#     global cooldown_state

#     # reset total au démarrage
#     if reset:
#         cooldown_state = {}
#         return I_min

#     # calcul du stock si pas donné
#     if stock_today is None:
#         stock_today = resources[:, :, 1][zone_map == zone_id].sum()

#     # mise à jour quotidienne
#     remaining = cooldown_state.get(zone_id, 0)
#     if stock_today is not None:
#         if remaining > 0:
#             cooldown_state[zone_id] = max(0, remaining - 1)
#         else:
#             baseline = spawn_baseline if spawn_baseline is not None else FOOD_SPAWN_COUNT
#             threshold = max(1, int(0.10 * baseline))
#             if stock_today < threshold:
#                 cooldown_state[zone_id] = cooldown_days

#     # si cooldown actif → désactivé
#     if cooldown_state.get(zone_id, 0) > 0:
#         return None

#     # sinon calcule l’intervalle normal
#     interval = I_max / (1.0 + k * max(0.0, f_avg))
#     return max(I_min, int(interval))






# # img_path = os.path.join(os.path.dirname(__file__), "images", "5_spots.png")

# # # run zone detection once → returns zone_map + food_ids
# # zone_map, food_ids = identify_zones(img_path)

# # all_coords = []
# # mask_debug = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)

# # for name, zid in food_ids.items():
# #     coords, zone_mask = extract_resource_coords_from_zones(zone_map, food_zone_id=zid)
# #     print(f"{name}: {len(coords)} pixels")

# #     for (x, y) in coords:
# #         add_resource(x, y)

# #     all_coords.append(coords)
# #     mask_debug |= zone_mask  # accumulate for visualization

# # if all_coords:
# #     all_coords = np.vstack(all_coords)
# #     print("Total resource pixels:", len(all_coords))

# # print("Total food units:", total_food())

# # cv2.imwrite("mask.png", mask_debug)
# # print("Mask saved as mask.png")

# # import csv

# # import csv

# # if all_coords is not None and len(all_coords) > 0:
# #     with open("resources_coords.csv", mode="w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["x", "y", "zone_id", "lifetime", "food_left"])  # en-tête

# #         for (x, y) in all_coords:
# #             zid = zone_map[x, y]   # ⚠️ attention à l’ordre (x,y) vs (row,col)
# #             writer.writerow([x, y, zid, resources[x, y, 0], resources[x, y, 1]])

# #     print("Coordonnées des ressources sauvegardées dans resources_coords.csv")

# # import cv2
# # import numpy as np

# # def visualize_food_ids(zone_map, food_ids, out_path="food_zones_ids.png"):
# #     # copie en couleur pour afficher le texte
# #     vis = np.zeros((zone_map.shape[0], zone_map.shape[1], 3), dtype=np.uint8)
# #     vis[:] = (200, 200, 200)  # fond gris clair

# #     for name, zid in food_ids.items():
# #         # coordonnées de la zone
# #         ys, xs = np.where(zone_map == zid)
# #         if len(xs) == 0:
# #             continue

# #         # centroïde (position moyenne pour écrire le texte)
# #         cx, cy = int(xs.mean()), int(ys.mean())

# #         # écrire l'ID de zone
# #         cv2.putText(
# #             vis, str(zid), (cx, cy),
# #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
# #         )

# #     cv2.imwrite(out_path, vis)
# #     print(f"Image des IDs sauvegardée sous {out_path}")
# # zone_map, food_ids = identify_zones(img_path)
# # visualize_food_ids(zone_map, food_ids)

import os
import numpy as np
import cv2
import pygame
from config import *
from human import draw_human

# resources[y, x, 0] = lifetime
# resources[y, x, 1] = food_left
resources = np.zeros((MAP_HEIGHT, MAP_WIDTH, 2), dtype=np.int32)

FOOD_COLOR_BGR = (54, 109, 70)  # from config

# ─────────────── Resource utils ───────────────
def extract_resource_coords_from_zones(zone_map, food_zone_id=4):
    """Return coordinates of cells belonging to given food zone."""
    food_mask = (zone_map == food_zone_id)
    coords = np.column_stack(np.where(food_mask))
    return coords, food_mask.astype(np.uint8) * 255



def life_span_ressource():
    """Decrease all lifetimes by 1 and clear dead resources."""
    resources[:, :, 0] -= 1
    dead = resources[:, :, 0] <= 0
    resources[dead] = 0

def total_food():
    """Total remaining food across the map."""
    return resources[:, :, 1].sum()

# ─────────────── Zone detection ───────────────
def identify_zones(map_image_path, min_size=10):
    """
    Identify zones from map image. Food zones are split into connected components,
    small components are merged into nearest large zone.
    Returns zone_map + dict of food zone IDs.
    """
    img = cv2.imread(map_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (MAP_WIDTH, MAP_HEIGHT), interpolation=cv2.INTER_NEAREST)

    zone_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int32)
    for rgb, zone_id in NEW_PALETTE.items():
        mask = np.all(img == rgb, axis=-1)
        zone_map[mask] = zone_id

    # Connected components just for food
    food_mask = (zone_map == 4).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(food_mask)

    sizes, centroids = {}, {}
    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        sizes[i] = len(xs)
        centroids[i] = (np.mean(xs), np.mean(ys))

    large_zones = {i for i, s in sizes.items() if s >= min_size}
    small_zones = {i for i, s in sizes.items() if s < min_size}

    food_ids, mapping = {}, {}
    base_id = 40
    for j, i in enumerate(sorted(large_zones), start=1):
        new_id = base_id + j
        mapping[i] = new_id
        food_ids[f"food_zone_{j}"] = new_id

    for i in small_zones:
        cx, cy = centroids[i]
        nearest = min(
            large_zones,
            key=lambda j: (centroids[j][0] - cx) ** 2 + (centroids[j][1] - cy) ** 2
        )
        mapping[i] = mapping[nearest]

    for i in range(1, num_labels):
        zone_map[labels == i] = mapping[i]

    print("Zones nourriture identifiées :", food_ids)
    return zone_map, food_ids

# ─────────────── Resource respawn dynamics ───────────────
cooldown_state = {}

def resource_spawn_interval_inverse(
    f_avg: float,
    zone_id: int = 0,
    zone_map=None,
    I_max: int = 200,
    k: float = 0.5,
    I_min: int = 10,
    stock_today=None,
    spawn_baseline: int = FOOD_SPAWN_COUNT,
    reset: bool = False,
    cooldown_days: int = 5
):
    """
    Compute inverse interval for resource spawning.
    Includes cooldown based on zone depletion.
    """
    global cooldown_state

    if reset:
        cooldown_state = {}
        return I_min

    if stock_today is None and zone_map is not None:
        stock_today = resources[:, :, 1][zone_map == zone_id].sum()

    remaining = cooldown_state.get(zone_id, 0)
    if stock_today is not None:
        if remaining > 0:
            cooldown_state[zone_id] = max(0, remaining - 1)
        else:
            threshold = max(1, int(0.10 * spawn_baseline))
            if stock_today < threshold:
                cooldown_state[zone_id] = cooldown_days

    if cooldown_state.get(zone_id, 0) > 0:
        return None

    interval = I_max / (1.0 + k * max(0.0, f_avg))
    return max(I_min, int(interval))

# ─────────────── Drawing / Pygame helpers ───────────────
def map_draw(image_path: str):
    """Return zone_map array (OpenCV-based)."""
    zone_map, _ = identify_zones(image_path)
    return zone_map

def map_manage(zone_map):
    """Draw the static terrain (houses, borders, empty ground)."""
    surf = pygame.Surface((MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
    rev_color = {v: k for k, v in NEW_PALETTE.items()}

    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            zone_id = zone_map[y, x]
            if zone_id >= 40:   
                color = (0, 150, 0)  # very light green (background of food zone)
            else:
                color = rev_color.get(zone_id, (50, 50, 50))
            pygame.draw.rect(
                surf, color,
                pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )
    return surf



def pixel_update(screen, static_layer, humans, font):
    """Draw static terrain layer + live food + humans."""
    screen.blit(static_layer, (0, 0))

    # draw food stacks dynamically
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if resources[y, x, 1] > 0:
                frac = resources[y, x, 1] / FOOD_STACK
                color = (int(0 + 200*frac), int(255*frac), int(0 + 50*frac))
                pygame.draw.circle(
                    screen, color,
                    (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2),
                    CELL_SIZE//2
                )

    # draw humans
    for h in humans:
        if h.alive:
            draw_human(screen, h, CELL_SIZE, font)

def display_house_storage(screen, houses, cell_size, font):
    """Draw storage info for each house."""
    for i, house in enumerate(houses):
        txt = f"House {i} storage: {house.storage}"
        surf = font.render(txt, True, (255,255,0))
        screen.blit(surf, (10, 30 + i*20))
