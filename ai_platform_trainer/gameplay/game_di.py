# file: ai_platform_trainer/gameplay/game_di.py
"""
Game class with dependency injection for the AI Platform Trainer.
"""
import logging
import os
import pygame
import torch
from typing import Optional

# Import service locator
from ai_platform_trainer.core.service_locator import ServiceLocator

# We'll use the config_manager from ServiceLocator

# Import state machine components
from ai_platform_trainer.gameplay.state_machine import (
    MenuState,
    PlayState,
    TrainingState,
    PausedState,
    GameOverState,
)

# Import missile model
from ai_platform_trainer.ai_model.simple_missile_model import SimpleMissileModel

# Import display manager
from ai_platform_trainer.gameplay.display_manager import toggle_fullscreen_display

# Import spawner
from ai_platform_trainer.gameplay.spawner import respawn_enemy_with_fade_in


class Game:
    """
    Main class to run the Pixel Pursuit game.
    Uses dependency injection for services and state management for game flow.
    """

    def __init__(self) -> None:
        """Initialize the game with injected dependencies."""
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None
        self.paused: bool = False  # Track pause state

        # Get dependencies from the service locator
        self.config_manager = ServiceLocator.get("config_manager")
        # Legacy config for backward compatibility
        self.config = ServiceLocator.get("config")
        self.renderer = ServiceLocator.get("renderer")
        self.input_handler = ServiceLocator.get("input_handler")
        self.menu = ServiceLocator.get("menu")
        self.clock = ServiceLocator.get("clock")
        self.play_entity_factory = ServiceLocator.get("play_entity_factory")
        self.training_entity_factory = ServiceLocator.get("training_entity_factory")
        self.screen = ServiceLocator.get("screen")
        
        # Get screen dimensions from the display service
        display_service = ServiceLocator.get("display")
        self.screen_width, self.screen_height = display_service.get_dimensions()

        # Entities and managers
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.training_mode_manager = None

        # Additional logic
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        # Load missile model once
        self.missile_model = None
        self._load_missile_model_once()
        
        # Reusable tensor for missile AI input
        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        # Initialize state machine - create states directly instead of getting from ServiceLocator
        self.states = {
            "menu": MenuState(self),
            "play": PlayState(self),
            "train": TrainingState(self),
            "paused": PausedState(self),
            "game_over": GameOverState(self),
        }
        self.current_state = self.states["menu"]
        self.current_state.enter()

        logging.info("Game initialized with dependency injection.")

    def _load_missile_model_once(self) -> None:
        """
        Load the missile AI model once during initialization.
        """
        missile_model_path = "models/missile_model.pth"
        if os.path.isfile(missile_model_path):
            logging.info(
                f"Found missile model at '{missile_model_path}'. Loading once..."
            )
            try:
                model = SimpleMissileModel()
                model.load_state_dict(torch.load(missile_model_path, map_location="cpu"))
                model.eval()
                self.missile_model = model
            except Exception as e:
                logging.error(f"Failed to load missile model: {e}")
                self.missile_model = None
        else:
            logging.warning(
                f"No missile model found at '{missile_model_path}'. "
                "Skipping missile AI."
            )

    def run(self) -> None:
        """Main game loop using state machine architecture."""
        while self.running:
            delta_time = self.clock.tick(self.config_manager.get("display.frame_rate", 60)) / 1000.0
            
            # Handle input through the input handler
            continue_game, events = self.input_handler.handle_input()
            if not continue_game:
                self.running = False
                break
            
            # Let the current state handle events
            for event in events:
                next_state = self.current_state.handle_event(event)
                if next_state:
                    self.transition_to(next_state)
                    break
            
            # Update and render the current state
            next_state = self.current_state.update(delta_time)
            if next_state:
                self.transition_to(next_state)
                
            self.current_state.render(self.renderer)
            pygame.display.flip()

        # Save data if we were training
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def transition_to(self, state_name: str) -> None:
        """
        Transition from the current state to a new state.
        
        Args:
            state_name: The name of the state to transition to
        """
        if state_name in self.states:
            logging.info(f"Transitioning from {type(self.current_state).__name__} to {state_name}")
            self.current_state.exit()
            self.current_state = self.states[state_name]
            self.current_state.enter()
        else:
            logging.error(f"Attempted to transition to unknown state: {state_name}")

    def _toggle_fullscreen(self) -> None:
        """
        Helper that toggles between windowed and fullscreen, 
        updating the display.
        """
        # Get the current fullscreen state from configuration
        was_fullscreen = self.config_manager.get("display.fullscreen", False)
        
        # Get screen dimensions from configuration
        screen_size = (
            self.config_manager.get("display.width", 800),
            self.config_manager.get("display.height", 600)
        )
        
        # Toggle fullscreen
        new_display, w, h = toggle_fullscreen_display(
            not was_fullscreen,
            screen_size
        )
        
        # Update configuration
        self.config_manager.set("display.fullscreen", not was_fullscreen)
        self.config_manager.save()
        
        # Also update legacy settings for backward compatibility
        settings_service = ServiceLocator.get("settings")
        settings = settings_service.get_settings()
        settings["fullscreen"] = not was_fullscreen
        settings_service.save_settings(settings)
        
        # Update services
        display_service = ServiceLocator.get("display")
        display_service.set_screen(new_display, w, h)
        
        # Update local references
        self.screen = new_display
        self.screen_width, self.screen_height = w, h
        
        # Reinitialize menu for the new dimensions
        menu_service = ServiceLocator.get("menu_factory")
        self.menu = menu_service.create_menu(w, h)
        ServiceLocator.register("menu", self.menu)
        
        # If in game, restart the current state
        if self.mode and self.mode != "menu":
            current_mode = self.mode
            self.reset_game_state()
            self.transition_to(current_mode)

    def check_collision(self) -> bool:
        """Check for collision between player and enemy."""
        if not (self.player and self.enemy):
            return False
            
        player_rect = pygame.Rect(
            self.player.position["x"],
            self.player.position["y"],
            self.player.size,
            self.player.size,
        )
        
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"],
            self.enemy.pos["y"],
            self.enemy.size,
            self.enemy.size
        )
        
        return player_rect.colliderect(enemy_rect)

    def check_missile_collisions(self) -> None:
        """Check for collisions between missiles and enemy."""
        if not self.enemy or not self.player:
            return

        collision_service = ServiceLocator.get("collision_service")
        collision_service.check_missile_collisions(
            self.player, 
            self.enemy, 
            self._respawn_callback
        )
    
    def _respawn_callback(self) -> None:
        """Callback for when the enemy needs to respawn."""
        self.is_respawning = True
        self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
        logging.info("Missile-Enemy collision in play mode, enemy will respawn.")

    def handle_respawn(self, current_time: int) -> None:
        """Handle respawning the enemy after a delay."""
        if (
            self.is_respawning
            and current_time >= self.respawn_timer
            and self.enemy
            and self.player
        ):
            respawn_enemy_with_fade_in(self, current_time)

    def reset_game_state(self) -> None:
        """Reset game state, typically when returning to menu."""
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.is_respawning = False
        self.respawn_timer = 0
        logging.info("Game state reset.")
