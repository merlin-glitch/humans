"""
test_share.py - Testing and validation utilities for the simulation

Part of the Human Society Simulation project.

Contains test functions and validation utilities for ensuring
simulation correctness and performance benchmarks.
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# === 1. Charger les données ===
data_path = "batch_results/all_combined.csv"   # CSV conso par zone
coords_path = "resources_coords.csv"                # CSV positions initiales
map_path = "humans/humans/images/5_spots.png"       # image de la carte

df = pd.read_csv(data_path)
coords_df = pd.read_csv(coords_path)

# On prend uniquement run 0
df0 = df[df["run"] == 7]

# === 2. Détecter automatiquement les zones ===
zones = coords_df["zone_id"].unique()
print("Zones détectées :", zones)

# Consommation Blue et Red (41→z0, 42→z1, etc.)
blue_cons = {z: df0[f"z{z-41}_blue_cons"].sum() for z in zones}
red_cons  = {z: df0[f"z{z-41}_red_cons"].sum()  for z in zones}

print("Blue consumption:", blue_cons)
print("Red consumption:", red_cons)

# === 3. Calculer le centroïde de chaque zone (avec scale et offset calibrés) ===
MAP_WIDTH = coords_df["x"].max() + 1
MAP_HEIGHT = coords_df["y"].max() + 1

base_map = np.array(Image.open(map_path).convert("RGB"))
img_height, img_width = base_map.shape[0], base_map.shape[1]

scale_x, scale_y = 0.55, 1.0   # tes valeurs ajustées
offset_x, offset_y = 1, 0      # tes offsets trouvés

zone_centroids = {}
for z in zones:
    subset = coords_df[coords_df["zone_id"] == z]
    if len(subset) > 0:
        cy, cx = subset["x"].mean(), subset["y"].mean()  # inversion x/y

        cx = int(cx / MAP_WIDTH  * img_width  * scale_x) + offset_x
        cy = int(cy / MAP_HEIGHT * img_height * scale_y) + offset_y

        zone_centroids[z] = (cx, cy)

print("Zone centroids (remapped):", zone_centroids)

# === 4. Créer calques vides pour Blue et Red ===
heat_blue = np.zeros((img_height, img_width), dtype=np.float32)
heat_red  = np.zeros((img_height, img_width), dtype=np.float32)

# === 5. Remplir les heatmaps (petits spots gaussiens, normalisation locale) ===
for z in zones:
    if z in zone_centroids:
        cx, cy = zone_centroids[z]
        b_val = blue_cons[z]
        r_val = red_cons[z]

        radius = 12   # taille de la tache
        sigma = 5     # étalement

        # Normalisation locale : chaque zone max = 1
        max_val = max(b_val, r_val)
        if max_val == 0:
            continue
        b_norm = b_val / max_val
        r_norm = r_val / max_val

        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                x, y = cx + dx, cy + dy
                if 0 <= x < img_width and 0 <= y < img_height:
                    weight = np.exp(-(dx**2 + dy**2) / (2*sigma**2))
                    heat_blue[y, x] += b_norm * weight
                    heat_red[y, x]  += r_norm * weight

# === 6. Combiner en RGB ===
heat_rgb = np.zeros_like(base_map, dtype=np.float32)
heat_rgb[...,0] = heat_red   # rouge
heat_rgb[...,2] = heat_blue  # bleu
heat_rgb = np.clip(heat_rgb,0,1)

# Superposer sur la carte
alpha = 0.6
overlay = (1-alpha)*base_map/255.0 + alpha*heat_rgb


# === 7. Affichage ===
plt.figure(figsize=(10,10))
plt.imshow(overlay)
plt.title("Exploitation Run 0 (Bleu=Blue, Rouge=Red, Violet=Both)")
plt.axis("off")

# Ajouter les pourcentages à chaque zone (centrés)
for z in zones:
    if z in zone_centroids:
        cx, cy = zone_centroids[z]
        b_val = blue_cons[z]
        r_val = red_cons[z]
        total = b_val + r_val
        if total > 0:
            b_pct = 100 * b_val / total
            r_pct = 100 * r_val / total
            text = f"B:{b_pct:.1f}%\nR:{r_pct:.1f}%"
            plt.text(cx, cy-5, text,
                     color="white", fontsize=9, ha="center", va="center",
                     bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"))

plt.show()
