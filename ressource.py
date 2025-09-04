# import os
# import numpy as np
# import pygame
# from PIL import Image
# from config import *
# from human import draw_human
# from typing import Optional

# from config import FOOD_SPAWN_COUNT as _SPAWN_BASELINE


# # ─── 0) Resource store (NumPy only, no lists) ─────────────────────────────────
# # Columns per row: [x, y, life]
# resource_array = np.zeros((MAP_HEIGHT * MAP_WIDTH, 3), dtype=np.int32)
# resource_count = 0  # number of active rows in resource_array

# def active_resources() -> np.ndarray:
#     """View of active rows only (shape: [resource_count, 3])."""
#     return resource_array[:resource_count]

# def add_resource(x: int, y: int, life: int = FOOD_LIFETIME) -> None:
#     """Append one resource at (x, y). O(1)."""
#     global resource_count
#     if resource_count >= resource_array.shape[0]:
#         return # Prevents out-of-bounds access   
#     resource_array[resource_count, 0] = x
#     resource_array[resource_count, 1] = y
#     resource_array[resource_count, 2] = life
#     resource_count += 1

# def number_of_resources(x: int, y: int) -> int:
#     """Count resources at (x, y) among the active slice."""
#     if resource_count == 0:
#         return 0
#     A = active_resources()
#     # Count how many active resources have coordinates (x, y)
#     matches = (A[:, 0] == x) & (A[:, 1] == y)
#     count = np.sum(matches)
#     return int(count)

# def remove_resource(x: int, y: int) -> None:
#     """
#     Remove ONE resource at (x, y) by swapping the found row with the last
#     active row, then shrinking the active count. O(1).
#     """
#     global resource_count
#     if resource_count == 0:
#         return
#     A = active_resources()
#     idxs = np.flatnonzero((A[:, 0] == x) & (A[:, 1] == y))
#     if idxs.size == 0:
#         return
#     i = int(idxs[0])
#     last = resource_count - 1
#     if i != last:
#         resource_array[i] = resource_array[last]
#     resource_count -= 1

# def tick_resources_and_compact() -> None:
#     """
#     life -= 1 for all active; remove dead (life <= 0) by compacting survivors
#     to the front. Vectorized.
#     """
#     global resource_count
#     if resource_count == 0:
#         return
#     A = active_resources()
#     A[:, 2] -= 1
#     alive_mask = A[:, 2] > 0
#     if np.all(alive_mask):
#         return
#     survivors = A[alive_mask]
#     k = survivors.shape[0]
#     resource_array[:k] = survivors
#     resource_count = k


# # ─── 1) Resource class ────────
# class Resource:
#     def __init__(self, x: int, y: int, life: int = FOOD_LIFETIME):
#         self.x, self.y = x, y
#         self.life = life

#     def update(self) -> None:
#         self.life -= 1

#     def is_alive(self) -> bool:
#         return self.life > 0








# def resource_spawn_interval_inverse(
#     f_avg: float,
#     I_max: int = 200,
#     k: float = 0.5,
#     I_min: int = 10,
#     *,
#     zone_id: Optional[int] = None,        # which zone (0/1/2)
#     stock_today: Optional[int] = None,    # stock at end-of-day for that zone
#     spawn_baseline: Optional[int] = None, # threshold baseline (defaults to FOOD_SPAWN_COUNT)
#     reset: bool = False,                  # call once at sim start
#     cooldown_days: int = 5               # how long a disabled zone stays off
# ) -> Optional[int]:
#     """
#     Inverse spawn interval with a finite disable cooldown.

#     Usage:
#       • Per tick:  interval = resource_spawn_interval_inverse(f_avg, zone_id=z)
#                     -> returns int interval for active zones, None if in cooldown.

#       • End of day: resource_spawn_interval_inverse(0.0, zone_id=z, stock_today=present[z],
#                                                     spawn_baseline=...)
#                     -> if stock_today < 10% of baseline, start/renew a cooldown.
#                        While cooling down, the zone is disabled for 'cooldown_days' days.
#                        After it reaches 0, the zone becomes active again.

#       • Start of run: resource_spawn_interval_inverse(0.0, reset=True)
#     """
#     # one-time state
#     if not hasattr(resource_spawn_interval_inverse, "_cooldown"):
#         resource_spawn_interval_inverse._cooldown = {}  # type: ignore[attr-defined]
#     cooldown = resource_spawn_interval_inverse._cooldown  # type: ignore[attr-defined]

#     # reset state between runs
#     if reset:
#         cooldown.clear()
#         return I_min

#     # daily update: adjust cooldown based on today's stock
#     if zone_id is not None and stock_today is not None:
#         remaining = cooldown.get(zone_id, 0)

#         if remaining > 0:
#             # tick down the existing cooldown by one day
#             cooldown[zone_id] = max(0, remaining - 1)
#         else:
#             # no cooldown active → check threshold
#             baseline = spawn_baseline if spawn_baseline is not None else (_SPAWN_BASELINE or 1)
#             threshold = max(1, int(0.10 * baseline))  # 10% of baseline, at least 1
#             if stock_today < threshold:
#                 cooldown[zone_id] = max(1, cooldown_days)  # start cooldown

#     # per-tick query: if cooling down, zone is disabled (no spawns)
#     if zone_id is not None and cooldown.get(zone_id, 0) > 0:
#         return None

#     # standard inverse-interval response based on recent consumption
#     interval = I_max / (1.0 + k * max(0.0, f_avg))
#     return max(I_min, int(interval))


# # ─── helper ───────────────────────────────────────────────────────────────────
# def is_in_house(x: int, y: int, houses) -> bool:
#     return any(
#         h.x <= x < h.x + 1 and h.y <= y < h.y + 1
#         for h in houses
#     )


# # ─── 2) PNG → terrain codes text ──────────────────────────────────────────────
# def map_draw(image_filename: str) -> str:
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     img_path = os.path.join(base_dir, 'images', image_filename)
#     img = Image.open(img_path).convert('RGB')
#     arr = np.array(img)
#     H, W, _ = arr.shape
#     rgb_to_palette = {rgb: code for rgb, code in TERRAIN_PALETTE.items()}

#     codes = np.zeros((H, W), dtype=int)
#     for y in range(H):
#         for x in range(W):
#             # default to code 6 if color not in palette
#             codes[y, x] = rgb_to_palette.get(tuple(arr[y, x]), 6)

#     out_name = os.path.splitext(image_filename)[0] + '.txt'
#     out_path = os.path.join(base_dir, out_name)
#     with open(out_path, 'w') as f:
#         for row in codes:
#             f.write(' '.join(map(str, row)) + '\n')
#     return out_path


# # ─── 3) Dynamic food layer from NumPy resources ───────────────────────────────
# def map_manage(codes: np.ndarray) -> np.ndarray:
#     """
#     Returns an (H, W, 3) int array with -1 where there's no dynamic pixel,
#     and FOOD_START_RGB at cells where a live resource is present.
#     """
#     H, W = codes.shape
#     managed = np.full((H, W, 3), -1, dtype=np.int32)

#     if resource_count == 0:
#         return managed

#     A = active_resources()
#     x_coords = A[:, 0]
#     y_coords = A[:, 1]

#     in_bounds = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
#     if not np.any(in_bounds):
#         return managed

#     x_coords = x_coords[in_bounds]
#     y_coords = y_coords[in_bounds]

#     # vectorized paint (no fade; always FOOD_START_RGB)
#     managed[y_coords, x_coords] = np.int32(FOOD_START_RGB)
#     return managed


# # ─── 4) Draw only dynamic layers ──────────────────────────────────────────────
# def pixel_update(screen, managed_map: np.ndarray, humans, font) -> None:
#     height, width, _ = managed_map.shape

#     # Draw food: find all dynamic cells once, then draw
#     has_food_mask = managed_map[:, :, 0] >= 0
#     if np.any(has_food_mask):
#         ys, xs = np.nonzero(has_food_mask)
#         colors = managed_map[ys, xs].astype(int)
#         for (y, x), (r, g, b) in zip(zip(ys, xs), colors):
#             rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#             pygame.draw.rect(screen, (r, g, b), rect)

#     # Draw humans
#     for human in humans:
#         draw_human(screen, human, CELL_SIZE, font)


# def display_house_storage(screen, houses, cell_size, font) -> None:
#     """Draw each house’s stored-food count just above the house rectangle."""
#     for house in houses:
#         label = str(house.storage)
#         text_surf = font.render(label, True, (255, 255, 0))  # yellow

#         px = house.x * cell_size + cell_size // 2
#         py = house.y * cell_size
#         offset = font.get_linesize() // 2
#         text_pos = (px, py - offset)

#         rect = text_surf.get_rect(center=text_pos)
#         screen.blit(text_surf, rect)


# resource_utils.py
import numpy as np
from config import MAP_HEIGHT, MAP_WIDTH

resources = np.zeros((MAP_HEIGHT, MAP_WIDTH, 2), dtype=np.int32)

def add_resource(x, y, life, food=30):
    if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH:
        resources[y, x, 0] = life
        resources[y, x, 1] = food

def remove_resource(x, y):
    if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH:
        resources[y, x] = [0, 0]