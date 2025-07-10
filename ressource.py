import os
import numpy as np
import pygame
from PIL import Image
from config import *
from human import draw_human


# ─── 1) Resource & House ─────────────────────────────────────────────────────
class Resource:
    def __init__(self, x, y, life=FOOD_LIFETIME):
        self.x, self.y = x, y
        self.life      = life
    # durée de la nourriture
    def update(self):
        self.life -= 1
    # update de la nourriture

    def is_alive(self):
        return self.life > 0
    # nourriture en vie
#
class House:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.storage      = 0       # ← total food units stored here

    def store_food(self, amount: int) -> None:
        """Add `amount` units into this house’s storage."""
        self.storage += amount


# ─── helper ───────────────────────────────────────────────────────────────────
#human dans  house 
def is_in_house(x, y, houses):
    return any(
        h.x <= x < h.x + 1 and h.y <= y < h.y + 1
        for h in houses
    )


# ─── 2) PNG → terrain codes text ──────────────────────────────────────────────
def map_draw(image_filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, 'images', image_filename)
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    H, W, _ = arr.shape
    rgb_to_palette = {rgb: code for rgb, code in TERRAIN_PALETTE.items()}

    codes = np.zeros((H, W), dtype=int)
    for y in range(H):
        for x in range(W):
            # default to 0 (land) if color not in palette
            #codes[y, x] = rev.get(tuple(arr[y, x]), 0)
            codes[y, x] = rgb_to_palette.get(tuple(arr[y, x]), 6)

    out_name = os.path.splitext(image_filename)[0] + '.txt'
    out_path = os.path.join(base_dir, out_name)
    with open(out_path, 'w') as f:
        for row in codes:
            f.write(' '.join(map(str, row)) + '\n')
    return out_path

# ─── 3) Fade food colors over time ────────────────────────────────────────────
    #codes           : 2D np.array de codes terrain
    #food_birth_map  : dict[(x,y)→birth_tick]
    #tick            : tick actuel

    #Retourne un tableau (h, w, 3) de valeurs RGB où :chaque (x,y) dans food_birth_map subit une interpolation de FOOD_START_RGB → FOOD_END_RGB sur FOOD_LIFETIME ticks
    
    

def map_manage(codes: np.ndarray, food_birth: dict, tick: int) -> np.ndarray:
    H, W = codes.shape #using the "codes" already generated we define the H and W of the map
    managed = np.full((H, W, 3), -1, dtype=int)# managed(H,W,rgb values) , -1 is their default value
    for (x, y), born in food_birth.items(): #food_birth[(coord_x,coord_y),tick]
        age  = tick - born #each SPAWN_INTERVAL a new food is born
        frac = min(max(age / FOOD_LIFETIME, 0.0), 1.0)
        r = int(FOOD_START_RGB[0] * (1 - frac) + FOOD_END_RGB[0] * frac)  # FOOD_START_RGB is dark green rgb , FOOD_END_RGB is dark_red rgb , check config.py
        g = int(FOOD_START_RGB[1] * (1 - frac) + FOOD_END_RGB[1] * frac)
        b = int(FOOD_START_RGB[2] * (1 - frac) + FOOD_END_RGB[2] * frac)
        if 0 <= y < H and 0 <= x < W:
            managed[y, x] = (r, g, b)
    return managed

# ─── 4) Draw only dynamic layers ──────────────────────────────────────────────
    
   # Dessine sur `screen` :
    # • food cells : pour chaque pixel où managed_map[y,x] != -1, dessine un rectangle CELL_SIZE×CELL_SIZE de la couleur RGB
    # • humains   : dessin via human.draw_human (cercles + compteurs)
  
def pixel_update(screen, managed_map, humans, font):
    H, W, _ = managed_map.shape
    # draw food
    for y in range(H):
        for x in range(W):
            r, g, b = managed_map[y, x]
            if r >= 0:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (r, g, b), rect)
    # draw humans
    for h in humans:
        draw_human(screen, h, CELL_SIZE, humans[0].home, font)

def display_house_storage(screen, houses, cell_size, font):
    """
    Draws each house’s stored‐food count just above the house rectangle.
    
    • screen: your pygame display surface  
    • houses: list of House instances (each must have .x, .y, .storage)  
    • cell_size: size of one map cell in pixels  
    • font: pygame.font.Font for rendering the text  
    """
    for house in houses:
        # Prepare the text
        label = str(house.storage)
        text_surf = font.render(label, True, (255, 255, 0))  # yellow
        
        # Compute pixel‐coordinates: center of the house, but a bit above
        px = house.x * cell_size + cell_size // 2
        py = house.y * cell_size            # top edge of the house
        offset = font.get_linesize() // 2   # half a line above
        text_pos = (px, py - offset)
        
        # Blit centered
        rect = text_surf.get_rect(center=text_pos)
        screen.blit(text_surf, rect)
