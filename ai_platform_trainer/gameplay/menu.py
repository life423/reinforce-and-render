import pygame


class Menu:
    def __init__(self, screen_width, screen_height):
        # Menu options
        # self.menu_options = ["Play", "Train", "Exit"]
        self.menu_options = ["Play", "Train", "Help", "Exit"]
        self.selected_option = 0
        
        self.show_help = False  

        # Screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts and colors
        self.font_title = pygame.font.Font(None, 100)  # Font for the title
        self.font_option = pygame.font.Font(None, 60)  # Font for menu options
        self.color_background = (135, 206, 235)  # Light blue background
        self.color_title = (0, 51, 102)  # Dark blue title color
        # Light gray for unselected options
        self.color_option = (245, 245, 245)
        self.color_selected = (255, 223, 0)  # Yellow for selected option
        self.option_rects = {}  # Store rects for click detection

    def handle_menu_events(self, event):
        if self.show_help:
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_RETURN]:
                self.show_help = False
            return None
        elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
            chosen = self.menu_options[self.selected_option]
            if chosen == "Help":
                self.show_help = True
                return None  # Don’t return "help" if you handle it internally
            else:
                return chosen.lower()
        # Handle menu navigation using arrow keys or WASD
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_option = (self.selected_option - 1) % len(
                    self.menu_options
                )
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_option = (self.selected_option + 1) % len(
                    self.menu_options
                )
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self.menu_options[self.selected_option].lower()
            # Escape key handling is removed to let the game handle exiting
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_option = index
                    return self.menu_options[self.selected_option].lower()
        return None

    def draw(self, screen):
        # Fill screen with background color
        screen.fill(self.color_background)

        # Render the title
        title_surface = self.font_title.render("Pixel Pursuit", True, self.color_title)
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 5)
        )
        screen.blit(title_surface, title_rect)

        # Render menu options with consistent spacing
        mouse_x, mouse_y = pygame.mouse.get_pos()  # Get mouse pos for highlighting
        for index, option in enumerate(self.menu_options):
            color = (
                self.color_selected
                if index == self.selected_option
                else self.color_option
            )
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 80)
            )
            self.option_rects[index] = option_rect  # Store rect for click detection
            screen.blit(option_surface, option_rect)
