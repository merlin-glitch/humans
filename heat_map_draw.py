# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # import cv2

# # # === 1. Charger les données ===
# # data_path = "batch_results/all_runs_combined.csv"   # CSV conso par zone
# # coords_path = "resources_coords.csv"                # CSV positions initiales
# # map_path = "humans/humans/images/5_spots.png"       # image de la carte

# # df = pd.read_csv(data_path)
# # coords_df = pd.read_csv(coords_path)

# # # On prend uniquement run 0
# # df0 = df[df["run"] == 0]

# # # === 2. Détecter automatiquement les zones ===
# # zones = coords_df["zone_id"].unique()
# # print("Zones détectées :", zones)

# # # Consommation Blue et Red (on mappe 41→z0, 42→z1, etc.)
# # blue_cons = {z: df0[f"z{z-41}_blue_cons"].sum() for z in zones}
# # red_cons  = {z: df0[f"z{z-41}_red_cons"].sum()  for z in zones}

# # print("Blue consumption:", blue_cons)
# # print("Red consumption:", red_cons)

# # # === 3. Calculer le centroïde de chaque zone avec rescaling ===
# # MAP_WIDTH = coords_df["x"].max() + 1
# # MAP_HEIGHT = coords_df["y"].max() + 1

# # base_map = np.array(Image.open(map_path).convert("RGB"))
# # img_height, img_width = base_map.shape[0], base_map.shape[1]

# # zone_centroids = {}
# # for z in zones:
# #     subset = coords_df[coords_df["zone_id"] == z]
# #     if len(subset) > 0:
# #         # inversion x/y
# #         cy, cx = subset["x"].mean(), subset["y"].mean()

# #         # Remap des coords (grille simulation → taille image)
# #         cx = int(cx / MAP_WIDTH  * img_width)
# #         cy = int(cy / MAP_HEIGHT * img_height)

# #         zone_centroids[z] = (cx, cy)

# # print("Zone centroids (remapped):", zone_centroids)

# # # === 4. Créer calques vides pour Blue et Red ===
# # heat_blue = np.zeros((img_height, img_width), dtype=np.float32)
# # heat_red  = np.zeros((img_height, img_width), dtype=np.float32)

# # # === 5. Remplir les heatmaps (cercles gaussiens localisés) ===
# # for z in zones:
# #     if z in zone_centroids:
# #         cx, cy = zone_centroids[z]
# #         b_val = blue_cons[z]
# #         r_val = red_cons[z]

# #         radius = 6   # rayon en px
# #         sigma = 3     # étalement

# #         for dx in range(-radius, radius+1):
# #             for dy in range(-radius, radius+1):
# #                 x, y = cx + dx, cy + dy
# #                 if 0 <= x < img_width and 0 <= y < img_height:
# #                     weight = np.exp(-(dx**2 + dy**2) / (2*sigma**2))
# #                     heat_blue[y, x] += b_val * weight
# #                     heat_red[y, x]  += r_val * weight

# # # Normalisation
# # heat_blue /= heat_blue.max() if heat_blue.max() > 0 else 1
# # heat_red  /= heat_red.max() if heat_red.max() > 0 else 1

# # # === 6. Combiner en RGB ===
# # heat_rgb = np.zeros_like(base_map, dtype=np.float32)
# # heat_rgb[...,0] = heat_red   # canal rouge
# # heat_rgb[...,2] = heat_blue  # canal bleu
# # heat_rgb = np.clip(heat_rgb,0,1)

# # # Superposer sur la carte
# # alpha = 0.6
# # overlay = (1-alpha)*base_map/255.0 + alpha*heat_rgb

# # # === 7. Affichage ===
# # plt.figure(figsize=(10,10))
# # plt.imshow(overlay)
# # plt.title("Exploitation Run 0 (Bleu=Blue, Rouge=Red, Violet=Both)")
# # plt.axis("off")
# # plt.show()

# # # Sauvegarde optionnelle
# # Image.fromarray((overlay*255).astype(np.uint8)).save("run0_heatmap.png")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# from PIL import Image
# import cv2

# # === 1. Charger les données ===
# coords_path = "resources_coords.csv"
# map_path = "humans/humans/images/5_spots.png"

# coords_df = pd.read_csv(coords_path)
# zones = coords_df["zone_id"].unique()

# # Dimensions simulation

# MAP_WIDTH = coords_df["x"].max() + 1
# MAP_HEIGHT = coords_df["y"].max() + 1

# # Charger carte
# base_map = np.array(Image.open(map_path).convert("RGB"))
# img_height, img_width = base_map.shape[0], base_map.shape[1]

# # === 2. Fonction pour calculer centroïdes avec offset ===
# def compute_centroids(offset_x=0, offset_y=0):
#     zone_centroids = {}
#     for z in zones:
#         subset = coords_df[coords_df["zone_id"] == z]
#         if len(subset) > 0:
#             cy, cx = subset["x"].mean(), subset["y"].mean()  # inversion x/y

#             scale_x = 0.55   # réduit la distance horizontale
#             scale_y = 1   # réduit la distance verticale

#             cx = int(cx / MAP_WIDTH  * img_width  * scale_x) + offset_x
#             cy = int(cy / MAP_HEIGHT * img_height * scale_y) + offset_y

#             zone_centroids[z] = (cx, cy)
#     return zone_centroids

# # === 3. Fonction pour afficher les centroïdes ===
# def plot_debug(offset_x=0, offset_y=0):
#     debug_map = base_map.copy()
#     zone_centroids = compute_centroids(offset_x, offset_y)

#     for (cx, cy) in zone_centroids.values():
#         cv2.circle(debug_map, (cx, cy), 10, (255,255,255), -1)  # cercles blancs

#     plt.imshow(debug_map)
#     plt.title(f"Offsets: x={offset_x}, y={offset_y}")
#     plt.axis("off")
#     plt.show()

# # === 4. Interface interactive avec sliders ===
# def interactive_debug():
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(left=0.25, bottom=0.25)

#     # Image initiale
#     debug_map = base_map.copy()
#     im = ax.imshow(debug_map)
#     ax.set_title("Adjust offsets")
#     ax.axis("off")

#     # Sliders
#     ax_offset_x = plt.axes([0.25, 0.1, 0.65, 0.03])
#     ax_offset_y = plt.axes([0.25, 0.05, 0.65, 0.03])

#     slider_x = Slider(ax_offset_x, 'Offset X', -100, 100, valinit=0, valstep=1)
#     slider_y = Slider(ax_offset_y, 'Offset Y', -100, 100, valinit=0, valstep=1)

#     def update(val):
#         ox = int(slider_x.val)
#         oy = int(slider_y.val)
#         debug_map = base_map.copy()
#         zone_centroids = compute_centroids(ox, oy)
#         for (cx, cy) in zone_centroids.values():
#             cv2.circle(debug_map, (cx, cy), 3, (255,255,255), -1)
#         im.set_data(debug_map)
#         fig.canvas.draw_idle()

#     slider_x.on_changed(update)
#     slider_y.on_changed(update)

#     plt.show()

# # Lance le mode interactif
# interactive_debug()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2

# # === 1. Charger les données ===
# data_path = "batch_results/all_combined.csv"   # CSV conso par zone
# coords_path = "resources_coords.csv"                # CSV positions initiales
# map_path = "humans/humans/images/5_spots.png"       # image de la carte

# df = pd.read_csv(data_path)
# coords_df = pd.read_csv(coords_path)

# # On prend uniquement run 0
# df0 = df[df["run"] == 0]

# # === 2. Détecter automatiquement les zones ===
# zones = coords_df["zone_id"].unique()
# print("Zones détectées :", zones)

# # Consommation Blue et Red (41→z0, 42→z1, etc.)
# blue_cons = {z: df0[f"z{z-41}_blue_cons"].sum() for z in zones}
# red_cons  = {z: df0[f"z{z-41}_red_cons"].sum()  for z in zones}

# print("Blue consumption:", blue_cons)
# print("Red consumption:", red_cons)

# # === 3. Calculer le centroïde de chaque zone (avec scale et offset calibrés) ===
# MAP_WIDTH = coords_df["x"].max() + 1
# MAP_HEIGHT = coords_df["y"].max() + 1

# base_map = np.array(Image.open(map_path).convert("RGB"))
# img_height, img_width = base_map.shape[0], base_map.shape[1]

# scale_x, scale_y = 0.55, 1.0   # tes valeurs ajustées
# offset_x, offset_y = 1, 0      # tes offsets trouvés

# zone_centroids = {}
# for z in zones:
#     subset = coords_df[coords_df["zone_id"] == z]
#     if len(subset) > 0:
#         cy, cx = subset["x"].mean(), subset["y"].mean()  # inversion x/y

#         cx = int(cx / MAP_WIDTH  * img_width  * scale_x) + offset_x
#         cy = int(cy / MAP_HEIGHT * img_height * scale_y) + offset_y

#         zone_centroids[z] = (cx, cy)

# print("Zone centroids (remapped):", zone_centroids)

# # === 4. Créer calques vides pour Blue et Red ===
# heat_blue = np.zeros((img_height, img_width), dtype=np.float32)
# heat_red  = np.zeros((img_height, img_width), dtype=np.float32)

# # === 5. Remplir les heatmaps (petits spots gaussiens, normalisation locale) ===
# for z in zones:
#     if z in zone_centroids:
#         cx, cy = zone_centroids[z]
#         b_val = blue_cons[z]
#         r_val = red_cons[z]

#         radius = 12   # taille de la tache
#         sigma = 5     # étalement

#         # Normalisation locale : chaque zone max = 1
#         max_val = max(b_val, r_val)
#         if max_val == 0:
#             continue
#         b_norm = b_val / max_val
#         r_norm = r_val / max_val

#         for dx in range(-radius, radius+1):
#             for dy in range(-radius, radius+1):
#                 x, y = cx + dx, cy + dy
#                 if 0 <= x < img_width and 0 <= y < img_height:
#                     weight = np.exp(-(dx**2 + dy**2) / (2*sigma**2))
#                     heat_blue[y, x] += b_norm * weight
#                     heat_red[y, x]  += r_norm * weight

# # === 6. Combiner en RGB ===
# heat_rgb = np.zeros_like(base_map, dtype=np.float32)
# heat_rgb[...,0] = heat_red   # rouge
# heat_rgb[...,2] = heat_blue  # bleu
# heat_rgb = np.clip(heat_rgb,0,1)

# # Superposer sur la carte
# alpha = 0.6
# overlay = (1-alpha)*base_map/255.0 + alpha*heat_rgb


# # === 7. Affichage ===
# plt.figure(figsize=(10,10))
# plt.imshow(overlay)
# plt.title("Exploitation Run 0 (Bleu=Blue, Rouge=Red, Violet=Both)")
# plt.axis("off")

# # Ajouter les pourcentages à chaque zone (centrés)
# for z in zones:
#     if z in zone_centroids:
#         cx, cy = zone_centroids[z]
#         b_val = blue_cons[z]
#         r_val = red_cons[z]
#         total = b_val + r_val
#         if total > 0:
#             b_pct = 100 * b_val / total
#             r_pct = 100 * r_val / total
#             text = f"B:{b_pct:.1f}%\nR:{r_pct:.1f}%"
#             plt.text(cx, cy-5, text,
#                      color="white", fontsize=9, ha="center", va="center",
#                      bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"))

# plt.show()


# # visuals/heat_map_draw.py
# # Heatmap overlay of Blue vs Red consumption on the base map.

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image


# def plot_heatmap_for_run(
#     data_path: str,
#     coords_path: str,
#     map_path: str,
#     run: int = 0,
#     scale_x: float = 0.55,
#     scale_y: float = 1.0,
#     offset_x: int = 1,
#     offset_y: int = 0,
#     radius: int = 12,
#     sigma: float = 5.0,
#     alpha: float = 0.6,
#     save_path: str = None,
# ):
#     """
#     Plot a heatmap of Blue vs Red consumption for a given run.

#     Args:
#         data_path: CSV with aggregated zone consumption (batch_results/all_combined.csv)
#         coords_path: CSV with resource cell coordinates (resources_coords.csv)
#         map_path: Path to base map PNG
#         run: which run number to display (default=0)
#         scale_x, scale_y: rescaling factors for coordinates
#         offset_x, offset_y: pixel offsets to align coords with map
#         radius: radius of Gaussian spots
#         sigma: spread of Gaussian spots
#         alpha: blending factor for overlay
#         save_path: if provided, save the figure instead of showing
#     """
#     # 1. Load data
#     df = pd.read_csv(data_path)
#     coords_df = pd.read_csv(coords_path)
#     base_map = np.array(Image.open(map_path).convert("RGB"))

#     # Select run
#     df_run = df[df["run"] == run]

#     # Detect zones
#     zones = coords_df["zone_id"].unique()

#     # Blue/Red consumption totals per zone
#     blue_cons = {z: df_run[f"z{z-41}_blue_cons"].sum() for z in zones}
#     red_cons  = {z: df_run[f"z{z-41}_red_cons"].sum()  for z in zones}

#     # 2. Compute centroids (remapped to image space)
#     MAP_WIDTH = coords_df["x"].max() + 1
#     MAP_HEIGHT = coords_df["y"].max() + 1
#     img_h, img_w = base_map.shape[0], base_map.shape[1]

#     centroids = {}
#     for z in zones:
#         subset = coords_df[coords_df["zone_id"] == z]
#         if len(subset) == 0:
#             continue
#         cy, cx = subset["x"].mean(), subset["y"].mean()  # swap
#         cx = int(cx / MAP_WIDTH  * img_w * scale_x) + offset_x
#         cy = int(cy / MAP_HEIGHT * img_h * scale_y) + offset_y
#         centroids[z] = (cx, cy)

#     # 3. Heatmap layers
#     heat_blue = np.zeros((img_h, img_w), dtype=np.float32)
#     heat_red  = np.zeros((img_h, img_w), dtype=np.float32)

#     for z in zones:
#         if z not in centroids:
#             continue
#         cx, cy = centroids[z]
#         b_val = blue_cons[z]
#         r_val = red_cons[z]
#         max_val = max(b_val, r_val)
#         if max_val == 0:
#             continue
#         b_norm = b_val / max_val
#         r_norm = r_val / max_val

#         for dx in range(-radius, radius + 1):
#             for dy in range(-radius, radius + 1):
#                 x, y = cx + dx, cy + dy
#                 if 0 <= x < img_w and 0 <= y < img_h:
#                     w = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
#                     heat_blue[y, x] += b_norm * w
#                     heat_red[y, x]  += r_norm * w

#     # 4. RGB heatmap overlay
#     heat_rgb = np.zeros_like(base_map, dtype=np.float32)
#     heat_rgb[..., 0] = heat_red
#     heat_rgb[..., 2] = heat_blue
#     heat_rgb = np.clip(heat_rgb, 0, 1)

#     overlay = (1 - alpha) * base_map / 255.0 + alpha * heat_rgb

#     # 5. Plot
#     plt.figure(figsize=(10, 10))
#     plt.imshow(overlay)
#     plt.title(f"Consumption Run {run} (Blue=Blue, Red=Red, Violet=Both)")
#     plt.axis("off")

#     # Percentages per zone
#     for z in zones:
#         if z in centroids:
#             cx, cy = centroids[z]
#             b_val, r_val = blue_cons[z], red_cons[z]
#             total = b_val + r_val
#             if total > 0:
#                 b_pct = 100 * b_val / total
#                 r_pct = 100 * r_val / total
#                 text = f"B:{b_pct:.1f}%\nR:{r_pct:.1f}%"
#                 plt.text(cx, cy - 5, text,
#                          color="white", fontsize=9, ha="center", va="center",
#                          bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"))

#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()

# visuals/heat_map_draw.py
# Heatmap overlay of Blue vs Red consumption on the base map.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_heatmap_for_run(
    data_path: str,
    coords_path: str,
    map_path: str,
    run: int = 0,
    scale_x: float = 0.55,
    scale_y: float = 1.0,
    offset_x: int = 1,
    offset_y: int = 0,
    radius: int = 12,
    sigma: float = 5.0,
    alpha: float = 0.6,
    save_path: str = None,
):
    print(f"[INFO] Loading data from {data_path}, coords from {coords_path}, map from {map_path}")

    # 1. Load data
    df = pd.read_csv(data_path)
    coords_df = pd.read_csv(coords_path)
    base_map = np.array(Image.open(map_path).convert("RGB"))
    print(f"[INFO] Data loaded: {len(df)} rows, {len(coords_df)} coords, map shape={base_map.shape}")

    # Select run
    df_run = df[df["run"] == run]
    print(f"[INFO] Selected run={run}, {len(df_run)} rows")

    if df_run.empty:
        print("[WARNING] No rows found for this run! Exiting.")
        return

    # Detect zones
    zones = coords_df["zone_id"].unique()
    print(f"[INFO] Found {len(zones)} unique zones in coords")

    # Blue/Red consumption totals per zone
    blue_cons = {}
    red_cons = {}
    for z in zones:
        blue_col = f"z{z-41}_blue_cons"
        red_col  = f"z{z-41}_red_cons"
        if blue_col not in df_run or red_col not in df_run:
            print(f"[WARNING] Missing columns for zone {z}: {blue_col}, {red_col}")
            continue
        blue_cons[z] = df_run[blue_col].sum()
        red_cons[z]  = df_run[red_col].sum()
    print(f"[INFO] Consumption dicts built: {len(blue_cons)} zones with data")

    # 2. Compute centroids
    MAP_WIDTH = coords_df["x"].max() + 1
    MAP_HEIGHT = coords_df["y"].max() + 1
    img_h, img_w = base_map.shape[0], base_map.shape[1]
    print(f"[INFO] Coord bounds: width={MAP_WIDTH}, height={MAP_HEIGHT}, image={img_w}x{img_h}")

    centroids = {}
    for z in zones:
        subset = coords_df[coords_df["zone_id"] == z]
        if len(subset) == 0:
            print(f"[WARNING] No coords for zone {z}")
            continue
        cy, cx = subset["x"].mean(), subset["y"].mean()
        cx = int(cx / MAP_WIDTH  * img_w * scale_x) + offset_x
        cy = int(cy / MAP_HEIGHT * img_h * scale_y) + offset_y
        centroids[z] = (cx, cy)
    print(f"[INFO] Centroids computed for {len(centroids)} zones")

    # 3. Heatmap layers
    heat_blue = np.zeros((img_h, img_w), dtype=np.float32)
    heat_red  = np.zeros((img_h, img_w), dtype=np.float32)

    for z in zones:
        if z not in centroids or z not in blue_cons:
            continue
        cx, cy = centroids[z]
        b_val = blue_cons[z]
        r_val = red_cons[z]
        max_val = max(b_val, r_val)
        if max_val == 0:
            continue
        b_norm = b_val / max_val
        r_norm = r_val / max_val
        # Debug values
        print(f"[DEBUG] Zone {z}: centroid=({cx},{cy}), blue={b_val}, red={r_val}, norm=({b_norm:.2f},{r_norm:.2f})")

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < img_w and 0 <= y < img_h:
                    w = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    heat_blue[y, x] += b_norm * w
                    heat_red[y, x]  += r_norm * w

    # 4. RGB heatmap overlay
    heat_rgb = np.zeros_like(base_map, dtype=np.float32)
    heat_rgb[..., 0] = heat_red
    heat_rgb[..., 2] = heat_blue
    heat_rgb = np.clip(heat_rgb, 0, 1)

    overlay = (1 - alpha) * base_map / 255.0 + alpha * heat_rgb

    # 5. Plot
    print(f"[INFO] Plotting overlay, alpha={alpha}, save_path={save_path}")
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Consumption Run {run} (Blue=Blue, Red=Red, Violet=Both)")
    plt.axis("off")

    # Percentages per zone
    for z in zones:
        if z in centroids and z in blue_cons:
            cx, cy = centroids[z]
            b_val, r_val = blue_cons[z], red_cons[z]
            total = b_val + r_val
            if total > 0:
                b_pct = 100 * b_val / total
                r_pct = 100 * r_val / total
                text = f"B:{b_pct:.1f}%\nR:{r_pct:.1f}%"
                plt.text(cx, cy - 5, text,
                         color="white", fontsize=9, ha="center", va="center",
                         bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Figure saved to {save_path}")
    else:
        plt.show()
        print("[INFO] Figure displayed")


plot_heatmap_for_run(
    data_path="batch_results/all_combined.csv",
    coords_path="resources_coords.csv",
    map_path="images/5_spots_fixed.png",
    run=0
)