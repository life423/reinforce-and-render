import pygame

class Menu:
    def __init__(self, screen_width: int, screen_height: int) -> None:
        self.menu_active = True
        self.menu_options = ["Play", "Training", "Exit"]
        self.selected_option = 0
        self.screen_width = screen_width
        self.screen_height = screen_height

    def handle_menu_events(self, event: pygame.event.Event) -> str:
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_option = (
                    self.selected_option - 1) % len(self.menu_options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_option = (
                    self.selected_option + 1) % len(self.menu_options)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                return self.menu_options[self.selected_option].lower()
        return ""

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the menu on the screen.
        """
        # Colors
        title_color = (0, 51, 102)  # Dark blue for the title
        option_color = (245, 245, 245)  # Off-white for unselected options
        selected_color = (255, 215, 0)  # Gold for the selected option

        # Fonts
        title_font = pygame.font.Font(None, 100)
        # Adjusted font size for better spacing
        option_font = pygame.font.Font(None, 60)

        # Render the title
        title_text = title_font.render("Pixel Pursuit", True, title_color)
        title_rect = title_text.get_rect(
            center=(self.screen_width // 2, self.screen_height // 5)
        )

        # Calculate spacing
        spacing = 30  # Spacing between options

        # Start position for the first menu option
        options_start_y = title_rect.bottom + 50  # 50 pixels below the title

        # Render the menu options
        option_surfaces = []
        option_rects = []
        for index, option in enumerate(self.menu_options):
            if index == self.selected_option:
                color = selected_color
            else:
                color = option_color
            option_surface = option_font.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(
                    self.screen_width // 2,
                    options_start_y + index *
                    (option_font.get_height() + spacing),
                )
            )
            option_surfaces.append(option_surface)
            option_rects.append(option_rect)
            # Blit the title and options onto the screen
        screen.blit(title_text, title_rect)
        for surface, rect in zip(option_surfaces, option_rects):
            screen.blit(surface, rect)

    def get_selected_mode(self) -> str:
        print('chosen here ' + self.menu_options[self.selected_option])
        return self.menu_options[self.selected_option]

    
