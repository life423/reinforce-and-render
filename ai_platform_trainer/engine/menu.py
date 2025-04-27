# ai_platform_trainer/engine/menu.py
import pygame
import textwrap
from ai_platform_trainer.core.color_manager import get_color

class MainMenu:
    OPTIONS = [
        "Play RL Game",
        "Train RL Model (live)",
        "Train RL Model (headless)",
        "Train Supervised Model (live)",
        "Train Supervised Model (headless)",
        "Help / Controls",
        "Toggle Fullscreen",
        "Quit",
    ]

    def __init__(self, screen: pygame.Surface):
        self.screen  = screen
        self.full    = False
        self.width, self.height = screen.get_size()
        self.font    = pygame.font.SysFont("consolas", 46, bold=True)
        self.small   = pygame.font.SysFont("consolas", 24)
        self.active  = 0
        self.done    = False
        self.action  = None
        self.help_on = False

    # ---------------- MAIN LOOP ---------------- #
    def run(self):
        clock = pygame.time.Clock()
        while not self.done:
            self._handle_events()
            self._draw()
            pygame.display.flip()
            clock.tick(60)
        return self.action

    # ------------- EVENT HANDLING -------------- #
    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.done = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    if self.help_on:
                        self.help_on = False
                    else:
                        self.done = True
                elif not self.help_on:
                    if e.key in (pygame.K_UP, pygame.K_w):
                        self.active = (self.active - 1) % len(self.OPTIONS)
                    elif e.key in (pygame.K_DOWN, pygame.K_s):
                        self.active = (self.active + 1) % len(self.OPTIONS)
                    elif e.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._confirm()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and not self.help_on:
                self._mouse_pick(e.pos)

    def _confirm(self):
        label = self.OPTIONS[self.active]
        if label == "Toggle Fullscreen":
            self._toggle_fullscreen()
        elif label == "Help / Controls":
            self.help_on = True
        elif label == "Quit":
            self.done = True
        else:
            self.action = label
            self.done   = True

    def _mouse_pick(self, pos):
        for idx in range(len(self.OPTIONS)):
            _, rect = self._render_option(idx, preview=True)
            if rect.collidepoint(pos):
                self.active = idx
                self._confirm()
                break

    # ------------------ DRAW ------------------- #
    def _draw(self):
        self.screen.fill(get_color("bg"))

        if self.help_on:
            self._draw_help()
            return

        title = self.font.render("Reinforce-and-Render", True, get_color("primary"))
        self.screen.blit(title, title.get_rect(center=(self.width//2, self.height//3)))

        for i in range(len(self.OPTIONS)):
            surf, rect = self._render_option(i)
            self.screen.blit(surf, rect)

        foot = self.small.render("↑/↓ to select · ↵ to confirm · H for help", True, get_color("gray500"))
        self.screen.blit(foot, foot.get_rect(center=(self.width//2, self.height - 40)))

    def _draw_help(self):
        lines = textwrap.wrap(
            "Controls:\n"
            "Player  –  WASD / Arrow Keys\n"
            "Quit    –  Esc / Q\n\n"
            "Menu navigation uses the same keys.\n\n"
            "Live-training modes open a new window so you can observe rewards updating.\n"
            "Headless modes run in the background (faster).",
            width=50)
        y = 80
        for ln in lines:
            surf = self.small.render(ln, True, get_color("gray300"))
            self.screen.blit(surf, surf.get_rect(center=(self.width//2, y)))
            y += 30
        esc = self.small.render("Press Esc to return", True, get_color("accent"))
        self.screen.blit(esc, esc.get_rect(center=(self.width//2, self.height - 60)))

    def _render_option(self, idx, preview=False):
        clr = get_color("accent") if idx == self.active else get_color("gray300")
        surf = self.font.render(self.OPTIONS[idx], True, clr)
        rect = surf.get_rect(center=(self.width//2, self.height//2 + idx*55))
        return (surf, rect) if preview else (surf, rect)

    # ---------------- FULLSCREEN --------------- #
    def _toggle_fullscreen(self):
        self.full = not self.full
        flags = pygame.FULLSCREEN | pygame.SCALED | pygame.DOUBLEBUF if self.full else pygame.SCALED | pygame.DOUBLEBUF
        pygame.display.set_mode((0, 0) if self.full else (800, 600), flags, vsync=1)
        self.screen = pygame.display.get_surface()
        self.width, self.height = self.screen.get_size()
