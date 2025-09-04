# generate_map.py
import os
import cv2
import numpy as np
import random

MAP_WIDTH, MAP_HEIGHT = 1000, 600
CELL_SIZE = 8  # only for visualization scaling

NEW_PALETTE = {
    (0, 0, 0): 0,           # border/walls
    (0, 200, 0): 1,         # grass background
    (0, 0, 255): 2,         # blue house
    (255, 0, 0): 3,         # red house
    (54, 109, 70): 4,       # food zones (dark green)
}
def generate_map(out_path="imag/5_spots.png"):
# make background green instead of black
    BACKGROUND_COLOR = (0, 200, 0)  # light grass green

    img = np.full((MAP_HEIGHT, MAP_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

    # draw houses in fixed positions
    HOUSE_SIZE = 100

    # red house (top-right corner)
    img[50:50+HOUSE_SIZE, MAP_WIDTH-(50+HOUSE_SIZE):MAP_WIDTH-50] = (255, 0, 0)

    # blue house (bottom-right corner)
    img[MAP_HEIGHT-(50+HOUSE_SIZE):MAP_HEIGHT-50, MAP_WIDTH-(50+HOUSE_SIZE):MAP_WIDTH-50] = (0, 0, 255)



    # scatter 5 food zones (random but well-placed)
    ZONE_RADIUS = 30   # radius of circular zone
    rng = random.Random(45)
    zones = []

    for _ in range(5):
        while True:
            x = rng.randint(ZONE_RADIUS+5, MAP_WIDTH - ZONE_RADIUS - 5)
            y = rng.randint(ZONE_RADIUS+5, MAP_HEIGHT - ZONE_RADIUS - 5)
            if all((x-x0)**2 + (y-y0)**2 > (2*ZONE_RADIUS)**2 for x0,y0 in zones):
                zones.append((x,y))
                break

        # draw circular mask
        for dy in range(-ZONE_RADIUS, ZONE_RADIUS+1):
            for dx in range(-ZONE_RADIUS, ZONE_RADIUS+1):
                if dx*dx + dy*dy <= ZONE_RADIUS*ZONE_RADIUS:  # inside circle
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < MAP_HEIGHT and 0 <= xx < MAP_WIDTH:
                        img[yy, xx] = (54, 109, 70)  # dark green

    # ensure folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # save
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✅ Map saved to {out_path}")

if __name__ == "__main__":
    generate_map()
