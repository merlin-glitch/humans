# menu.py
import pygame
import pygame_menu
from config import (
    MAP_WIDTH, MAP_HEIGHT, CELL_SIZE, FOOD_COUNT, MAX_LIFE,
    SPAWN_INTERVAL, HOUSE_COUNT, HOUSE_SIZE, MIN_HOUSE_DISTANCE
)

# runtime parameters (will be mutated by the sliders)
params = {
    'map_width': MAP_WIDTH,
    'map_height': MAP_HEIGHT,
    'cell_size': CELL_SIZE,
    'food_count': FOOD_COUNT,
    'max_life': MAX_LIFE,
    'spawn_interval': SPAWN_INTERVAL,
    'house_count': HOUSE_COUNT,
    'house_size': HOUSE_SIZE,
    'min_house_distance': MIN_HOUSE_DISTANCE
}

def show_menu():
    from main import start_simulation

    """Affiche le menu de paramètres avec des curseurs interactifs."""
    pygame.init()
    surface = pygame.display.set_mode((800, 550))
    menu = pygame_menu.Menu('Paramètres', 800, 550, theme=pygame_menu.themes.THEME_DARK)

    # sliders via menu.add.range_slider(...)
    menu.add.range_slider(
        'Taille cellule :', params['cell_size'],
        (1, 20), 1,
        onchange=lambda v, **_: params.update({'cell_size': int(v)})
    )
    menu.add.range_slider(
        'Nb ressources :', params['food_count'],
        (10, 1000), 10,
        onchange=lambda v, **_: params.update({'food_count': int(v)})
    )
    menu.add.range_slider(
        'Vie max res :', params['max_life'],
        (10, 10000), 10,
        onchange=lambda v, **_: params.update({'max_life': int(v)})
    )
    menu.add.range_slider(
        'Interval spawn :', params['spawn_interval'],
        (1, 1000), 1,
        onchange=lambda v, **_: params.update({'spawn_interval': int(v)})
    )
    menu.add.range_slider(
        'Nb maisons :', params['house_count'],
        (1, 10), 1,
        onchange=lambda v, **_: params.update({'house_count': int(v)})
    )
    menu.add.range_slider(
        'Taille maison :', params['house_size'],
        (1, 50), 1,
        onchange=lambda v, **_: params.update({'house_size': int(v)})
    )
    menu.add.range_slider(
        'Dist min maisons :', params['min_house_distance'],
        (0, 100), 1,
        onchange=lambda v, **_: params.update({'min_house_distance': int(v)})
    )

    # boutons
    menu.add.button('Start', lambda: start_simulation(params))
    menu.add.button('Quit', pygame_menu.events.EXIT)

    menu.mainloop(surface)
