import os
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")  # Shut up pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"  # Shut up pygame

import pygame
from sb3_contrib import MaskablePPO

from render.menu import MenuScreen
from render.game_mode import GameMode
from render.game_state import PVP_MODE, WATCH_MODE


class BuckshotRouletteApp:
    """Main application orchestrator."""

    def __init__(self, model_path: str = "agent/models/champion.zip"):
        """Initialize the application.

        Args:
            model_path: Path to the trained AI model
        """
        pygame.init()

        self.WIDTH = 1000
        self.HEIGHT = 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Buckshot Roulette")

        self.clock = pygame.time.Clock()
        self.FPS = 60

        # Load model once at startup
        print("Loading AI model...")
        self.model = MaskablePPO.load(model_path, device="cpu")
        print(f"Model loaded from: {model_path}")

        # Components
        self.menu = MenuScreen(self.screen, self.WIDTH, self.HEIGHT)
        self.current_mode = None

        # State
        self.current_screen = "menu"  # "menu" or "game"

    def _on_back_to_menu(self):
        """Callback to return to menu from game mode."""
        self.current_screen = "menu"
        self.current_mode = None
        pygame.display.set_caption("Buckshot Roulette")

    def _start_play_mode(self):
        """Start Player vs AI mode."""
        pygame.display.set_caption("Buckshot Roulette - vs Champion")
        self.current_mode = GameMode(
            self.screen,
            self.model,
            PVP_MODE,
            self._on_back_to_menu
        )
        self.current_screen = "game"

    def _start_watch_mode(self):
        """Start AI vs AI watch mode."""
        pygame.display.set_caption("Buckshot Roulette - AI Self-Play")
        self.current_mode = GameMode(
            self.screen,
            self.model,
            WATCH_MODE,
            self._on_back_to_menu
        )
        self.current_screen = "game"

    def run(self):
        """Main application loop."""
        running = True

        while running:
            if self.current_screen == "menu":
                # Menu screen
                play_rect, watch_rect, quit_rect = self.menu.draw()
                choice = self.menu.handle_events(play_rect, watch_rect, quit_rect)

                if choice == "play":
                    self._start_play_mode()
                elif choice == "watch":
                    self._start_watch_mode()
                elif choice == "quit":
                    running = False

            elif self.current_screen == "game":
                # Game mode
                if self.current_mode:
                    running = self.current_mode.handle_events()
                    if self.current_screen == "game":  # Check if still in game after event handling
                        self.current_mode.update()
                        self.current_mode.draw()
                else:
                    # Fallback to menu if mode is None
                    self.current_screen = "menu"

            self.clock.tick(self.FPS)

        pygame.quit()


def main():
    """Main entry point."""
    model_path = "agent/models/champion.zip"

    if not os.path.exists(model_path):
        print(f"No champion model found at {model_path}")
        print("Please train a model first using train.py")
        return

    app = BuckshotRouletteApp(model_path)
    app.run()


if __name__ == "__main__":
    main()
