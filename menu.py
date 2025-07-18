# # menu.py
# import pygame
# import pygame

# class Slider:
#     def __init__(self, rect, min_val, max_val, initial, orientation='vertical'):
#         """
#         A slider widget that can be vertical or horizontal.

#         rect: (x, y, w, h) area of the track
#         min_val, max_val: numeric range
#         initial: starting value
#         orientation: 'vertical' or 'horizontal'
#         """
#         self.rect        = pygame.Rect(rect)
#         self.min, self.max = float(min_val), float(max_val)
#         self.value       = float(initial)
#         self.radius      = 12
#         self.dragging    = False
#         assert orientation in ('vertical','horizontal')
#         self.orientation = orientation

#     def knob_rect(self):
#         """Returns a pygame.Rect for the knob based on current value & orientation."""
#         frac = (self.value - self.min) / (self.max - self.min)
#         if self.orientation == 'vertical':
#             x = self.rect.centerx
#             y = self.rect.y + frac * self.rect.height
#         else:  # horizontal
#             x = self.rect.x + frac * self.rect.width
#             y = self.rect.centery

#         return pygame.Rect(
#             int(x - self.radius),
#             int(y - self.radius),
#             2 * self.radius,
#             2 * self.radius
#         )

#     def handle_event(self, event):
#         """Call on every pygame event to update dragging/value."""
#         kr = self.knob_rect()
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             if kr.collidepoint(event.pos):
#                 self.dragging = True

#         elif event.type == pygame.MOUSEBUTTONUP:
#             self.dragging = False

#         elif event.type == pygame.MOUSEMOTION and self.dragging:
#             if self.orientation == 'vertical':
#                 # clamp Y and convert back to value, then round to integer
#                 y    = max(self.rect.y, min(event.pos[1], self.rect.bottom))
#                 frac = (y - self.rect.y) / self.rect.height
#                 raw  = self.min + frac * (self.max - self.min)
#                 self.value = float(int(round(raw)))
#             else:
#                 # clamp X and convert back to continuous value
#                 x    = max(self.rect.x, min(event.pos[0], self.rect.right))
#                 frac = (x - self.rect.x) / self.rect.width
#                 self.value = self.min + frac * (self.max - self.min)

#             # ensure value stays in bounds
#             self.value = max(self.min, min(self.value, self.max))

#     def draw(self, screen, font=None):
#         """Draw the track, knob, and an optional label to the screen."""
#         # draw track
#         pygame.draw.rect(screen, (180, 180, 180), self.rect)

#         # draw knob
#         kr = self.knob_rect()
#         pygame.draw.circle(screen, (100, 100, 250), kr.center, self.radius)
#         pygame.draw.circle(screen, (0,   0,   0  ), kr.center, self.radius, 1)

#         # draw label if font provided
#         if font:
#             if self.orientation == 'vertical':
#                 # integer human count
#                 label_text = f"H: {int(self.value)}"
#             else:
#                 # float speed
#                 label_text = f"Speed: {self.value:.2f}"
#             lbl = font.render(label_text, True, (255, 255, 255))

#             if self.orientation == 'vertical':
#                 lx = self.rect.centerx - lbl.get_width() // 2
#                 ly = self.rect.y - lbl.get_height() - 4
#             else:
#                 lx = kr.centerx - lbl.get_width() // 2
#                 ly = self.rect.y - lbl.get_height() - 4

#             screen.blit(lbl, (lx, ly))




# def handle_pause_event(event, paused: bool) -> bool:
#     """
#     Toggle the `paused` flag when the user presses P.
#     """
#     if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
#         return not paused
#     return paused

# def draw_paused_overlay(screen: pygame.Surface, font: pygame.font.Font) -> None:
#     """
#     Draw a semi-transparent PAUSED overlay at the center of the screen.
#     """
#     sw, sh = screen.get_size()
#     # Render the text
#     pause_surf = font.render("PAUSED", True, (255, 0, 0))
#     rect = pause_surf.get_rect(center=(sw // 2, sh // 2))
#     # Optionally: dim the background
#     overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
#     overlay.fill((0, 0, 0, 128))  # black at 50% alpha
#     screen.blit(overlay, (0, 0))
#     screen.blit(pause_surf, rect)

# menu.py
import pygame

class Slider:
    def __init__(self, rect, min_val, max_val, initial, orientation='vertical'):
        """
        A slider widget that can be vertical or horizontal.

        rect: (x, y, w, h) area of the track
        min_val, max_val: numeric range
        initial: starting value
        orientation: 'vertical' or 'horizontal'
        """
        self.rect        = pygame.Rect(rect)
        self.min, self.max = float(min_val), float(max_val)
        self.value       = float(initial)
        self.radius      = 12
        self.dragging    = False
        assert orientation in ('vertical','horizontal')
        self.orientation = orientation

    def knob_rect(self):
        """Returns a pygame.Rect for the knob based on current value & orientation."""
        frac = (self.value - self.min) / (self.max - self.min)
        if self.orientation == 'vertical':
            x = self.rect.centerx
            y = self.rect.y + frac * self.rect.height
        else:  # horizontal
            x = self.rect.x + frac * self.rect.width
            y = self.rect.centery

        return pygame.Rect(
            int(x - self.radius),
            int(y - self.radius),
            2 * self.radius,
            2 * self.radius
        )

    def handle_event(self, event):
        """Call on every pygame event to update dragging/value."""
        kr = self.knob_rect()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if kr.collidepoint(event.pos):
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            if self.orientation == 'vertical':
                # clamp Y and convert back to integer value
                y    = max(self.rect.y, min(event.pos[1], self.rect.bottom))
                frac = (y - self.rect.y) / self.rect.height
                raw  = self.min + frac * (self.max - self.min)
                self.value = float(int(round(raw)))
            else:
                # clamp X and convert back to continuous value
                x    = max(self.rect.x, min(event.pos[0], self.rect.right))
                frac = (x - self.rect.x) / self.rect.width
                self.value = self.min + frac * (self.max - self.min)

            # ensure within bounds
            self.value = max(self.min, min(self.value, self.max))

    def draw(self, screen, font=None):
        """Draw track, knob, and optional label."""
        # track
        pygame.draw.rect(screen, (180, 180, 180), self.rect)
        # knob
        kr = self.knob_rect()
        pygame.draw.circle(screen, (100,100,250), kr.center, self.radius)
        pygame.draw.circle(screen, (0,0,0),     kr.center, self.radius, 1)

        # label
        if font:
            if self.orientation == 'vertical':
                text = f"H: {int(self.value)}"
            else:
                text = f"Speed: {self.value:.2f}"
            lbl = font.render(text, True, (255,255,255))

            if self.orientation == 'vertical':
                lx = self.rect.centerx - lbl.get_width()//2
                ly = self.rect.y - lbl.get_height() - 4
            else:
                lx = kr.centerx - lbl.get_width()//2
                ly = self.rect.y - lbl.get_height() - 4

            screen.blit(lbl, (lx, ly))


def handle_pause_event(event, paused: bool) -> bool:
    """
    Toggle the `paused` flag when the user presses P.
    """
    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
        return not paused
    return paused


def draw_paused_overlay(screen: pygame.Surface, font: pygame.font.Font) -> None:
    """
    Draw a semi-transparent PAUSED overlay at the center of the screen.
    """
    sw, sh = screen.get_size()
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,128))
    screen.blit(overlay, (0, 0))
    pause_surf = font.render("PAUSED", True, (255,0,0))
    screen.blit(pause_surf, pause_surf.get_rect(center=(sw//2, sh//2)))


# ─── new functions for Reset & Export Trust ──────────────────────────────────

def create_action_buttons(slider: Slider,
                          spacing: int = 8,
                          btn_h: int = 24):
    """
    Given a horizontal slider, return two pygame.Rects:
      (reset_rect, export_rect)
    sitting just below its track.
    """
    btn_w = (slider.rect.width - spacing) // 2
    y    = slider.rect.bottom + spacing
    x1   = slider.rect.x
    x2   = x1 + btn_w + spacing
    reset_rect  = pygame.Rect(x1, y, btn_w, btn_h)
    export_rect = pygame.Rect(x2, y, btn_w, btn_h)
    return reset_rect, export_rect


def handle_action_buttons(event,
                          rects,
                          on_reset,
                          on_export):
    """
    Call from your event loop.  If the user clicks inside
    rects[0], calls on_reset().  If inside rects[1], calls on_export().
    """
    reset_rect, export_rect = rects
    if event.type == pygame.MOUSEBUTTONDOWN:
        if reset_rect.collidepoint(event.pos):
            on_reset()
        elif export_rect.collidepoint(event.pos):
            on_export()


def draw_action_buttons(screen,
                        rects,
                        font: pygame.font.Font):
    """
    Draw two grey buttons with black border and white text.
    """
    reset_rect, export_rect = rects

    for rect, text in ((reset_rect, "Reset"),
                       (export_rect, "Export Trust")):
        pygame.draw.rect(screen, (80,80,80), rect)
        pygame.draw.rect(screen, (0,0,0),     rect, 1)
        # split on newline for multi-line label
        lines = text.split("\n")
        for i, line in enumerate(lines):
            lbl = font.render(line, True, (255,255,255))
            pos = (
                rect.centerx - lbl.get_width()//2,
                rect.y + (i * lbl.get_height()) + (rect.height - len(lines)*lbl.get_height())//2
            )
            screen.blit(lbl, pos)
