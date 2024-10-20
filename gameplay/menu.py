import pygame


class Menu:
    def __init__(self, screen_width, screen_height):
        self.menu_options = ["Training", "Play", "Exit"]
        self.selected_option = 0
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.selected_action = None

    def handle_menu_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_option = (
                    self.selected_option - 1) % len(self.menu_options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_option = (
                    self.selected_option + 1) % len(self.menu_options)
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                # Update selected_action when the user presses Enter
                self.selected_action = self.menu_options[self.selected_option].lower(
                )

    def draw(self, screen):
        # Draw the menu here
        font = pygame.font.Font(None, 74)
        for index, option in enumerate(self.menu_options):
            color = (255, 215, 0) if index == self.selected_option else (
                245, 245, 245)
            text_surface = font.render(option, True, color)
            screen.blit(text_surface, (self.screen_width // 3,
                        self.screen_height // 3 + index * 100))


    def get_selected_mode(self) -> str:
        print('chosen here ' + self.menu_options[self.selected_option])
        return self.menu_options[self.selected_option]

    
