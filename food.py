# food.py

class FoodZone:
    def __init__(self, zone_id, position, units, color_rgb):
        """
        zone_id    : ex. 'V1', 'P3'
        position   : (x, y) float
        units      : 1 (mauve) ou 10 (rose)
        color_rgb  : tuple (R,G,B)
        """
        self.zone_id = zone_id
        self.position = position
        self.units = units
        self.color = color_rgb

    def harvest(self):
        """Récolte 1 unité et la consomme (+1 énergie)."""
        if self.units <= 0:
            return 0
        self.units -= 1
        return 1
