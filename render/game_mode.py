"""Unified game mode controller for both PvP and Watch modes."""

import pygame
import numpy as np
from typing import Callable, Optional
from sb3_contrib import MaskablePPO

from core.game import BuckshotRouletteGame
from core.constants import GameAction, Turn
from render.game_ui import GameUIComponents
from render.game_logic import GameLogic
from render.game_state import GameModeConfig, GameStats
from render.colors import *


class GameMode:
    """Unified game mode that handles both Player vs AI and AI vs AI."""

    def __init__(self, screen: pygame.Surface, model: MaskablePPO,
                 config: GameModeConfig, on_back_to_menu: Callable[[], None]):
        """Initialize game mode.

        Args:
            screen: Pygame screen surface
            model: Trained AI model
            config: Game mode configuration
            on_back_to_menu: Callback to return to menu
        """
        self.screen = screen
        self.model = model
        self.config = config
        self.on_back_to_menu = on_back_to_menu

        self.WIDTH = 1000
        self.HEIGHT = 700

        # Components
        self.ui = GameUIComponents(screen, self.WIDTH, self.HEIGHT)
        self.stats = GameStats(config.is_pvp)

        # Game state
        self.game = BuckshotRouletteGame(np.random.seed())
        self.game.start_new_round()

        # UI state
        self.message = ""
        self.message_timer = 0
        self.game_over = False

        # PvP-specific state
        self.show_action_menu = False
        self.available_actions = []

        # AI state
        self.waiting_for_ai = False
        self.ai_action_timer = 0

        # UI rects
        self.action_button_rect = None
        self.play_again_rect = None
        self.menu_rect = None
        self.quit_rect = None
        self.back_to_menu_rect = None

        # Initialize AI turn if needed
        self._check_ai_turn()

    def _check_ai_turn(self):
        """Check if it's AI's turn and set up timer."""
        # In watch mode, both are AI
        # In PvP mode, only dealer is AI
        if self.config.is_pvp:
            if self.game.turn == Turn.DEALER:
                self.waiting_for_ai = True
                self.ai_action_timer = 120
        else:
            # Watch mode - always AI turn
            self.waiting_for_ai = True
            self.ai_action_timer = 120

    def _execute_action(self, action: GameAction, is_player_turn: bool):
        """Execute an action and update UI.

        Args:
            action: Action to execute
            is_player_turn: True if player's turn, False if dealer's
        """
        # Determine actor name
        if is_player_turn:
            actor_name = self.config.player_name.upper() if self.config.is_pvp else self.config.player_name
            target_name = self.config.dealer_name.split()[0]  # Get first word (CHAMPION or DEALER)
        else:
            actor_name = self.config.dealer_name.split()[0]  # CHAMPION or DEALER
            target_name = self.config.player_name.split()[0] if self.config.is_pvp else "PLAYER"

        # Track ejected bullet
        ejected_bullet = None
        if action == GameAction.USE_BEER and len(self.game.bullet_sequence) > 0:
            ejected_bullet = self.game.bullet_sequence[0]

        step_result = self.game.step(action)

        # Format message
        msg = GameLogic.format_action_message(actor_name, target_name, action, ejected_bullet)

        # Add damage info
        if action in [GameAction.SHOOT_SELF, GameAction.SHOOT_TARGET]:
            if step_result.new_bot_hp < step_result.prev_bot_hp:
                damage = step_result.prev_bot_hp - step_result.new_bot_hp
                msg += f" -{damage} HP (LIVE)"
            elif step_result.new_target_hp < step_result.prev_target_hp:
                damage = step_result.prev_target_hp - step_result.new_target_hp
                msg += f" -{damage} HP (LIVE)"
            else:
                msg += " (BLANK)"

        self._show_message(msg, 180)

        # Check if game ended
        if self.game.player.hp <= 0 or self.game.dealer.hp <= 0:
            self.game_over = True
            self.stats.record_game_end(self.game.player.hp, self.game.dealer.hp)

            # Auto-restart in watch mode
            if not self.config.is_pvp:
                self.ai_action_timer = 60  # 1 second before restart

    def _show_message(self, msg: str, duration: int = 120):
        """Show a temporary message.

        Args:
            msg: Message to display
            duration: Duration in frames
        """
        self.message = msg
        self.message_timer = duration

    def _restart_game(self):
        """Restart the game."""
        self.game = BuckshotRouletteGame(np.random.seed())
        self.game.start_new_round()
        self.game_over = False
        self.message = ""
        self.show_action_menu = False
        self.waiting_for_ai = False
        self._check_ai_turn()

    def _draw_action_button(self):
        """Draw action button for player in PvP mode."""
        if not self.config.allow_player_input:
            return None
        if self.game.turn != Turn.PLAYER or self.game_over:
            return None

        button_rect = pygame.Rect(550, 450, 400, 80)
        self.ui.draw_button(button_rect, "TAKE ACTION", DARK_GRAY, BLUE)
        return button_rect

    def _draw_back_to_menu_button(self):
        """Draw back to menu button in top right corner."""
        if self.game_over:
            return None

        button_rect = pygame.Rect(self.WIDTH - 150, 50, 130, 40)
        self.ui.draw_button(button_rect, "Menu", DARK_GRAY, BLUE, WHITE, WHITE, 2)
        return button_rect

    def _draw_action_menu(self):
        """Draw action selection menu."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        # Menu box
        menu_width = 500
        menu_height = min(400, 100 + len(self.available_actions) * 60)
        menu_x = (self.WIDTH - menu_width) // 2
        menu_y = (self.HEIGHT - menu_height) // 2

        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(self.screen, DARK_GRAY, menu_rect)
        pygame.draw.rect(self.screen, GOLD, menu_rect, 4)

        # Title
        title = self.ui.header_font.render("Choose Action", True, GOLD)
        title_rect = title.get_rect(centerx=menu_rect.centerx, top=menu_y + 20)
        self.screen.blit(title, title_rect)

        # Action buttons
        button_y = menu_y + 80
        mouse_pos = pygame.mouse.get_pos()

        action_rects = []
        for action in self.available_actions:
            button_rect = pygame.Rect(menu_x + 20, button_y, menu_width - 40, 50)

            # Hover effect
            color = BLUE if button_rect.collidepoint(mouse_pos) else GRAY
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, WHITE, button_rect, 2)

            action_text = self.ui.normal_font.render(
                GameLogic.get_action_description(action, True), True, WHITE
            )
            text_rect = action_text.get_rect(center=button_rect.center)
            self.screen.blit(action_text, text_rect)

            action_rects.append(button_rect)
            button_y += 60

        return menu_rect, action_rects

    def _draw_game_over(self):
        """Draw game over screen."""
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        # Result box
        box_rect = pygame.Rect(200, 150, 600, 400)
        pygame.draw.rect(self.screen, DARK_GRAY, box_rect)
        pygame.draw.rect(self.screen, GOLD, box_rect, 5)

        # Title
        player_hp = self.game.player.hp
        dealer_hp = self.game.dealer.hp

        if player_hp <= 0 and dealer_hp <= 0:
            title_text = "DRAW!"
            title_color = YELLOW
        elif player_hp > 0:
            title_text = "YOU WIN!" if self.config.is_pvp else "PLAYER WINS!"
            title_color = GREEN
        else:
            title_text = "YOU LOSE!" if self.config.is_pvp else "DEALER WINS!"
            title_color = RED

        title = self.ui.title_font.render(title_text, True, title_color)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 220))
        self.screen.blit(title, title_rect)

        # Scores
        score_y = 290
        player_label = "Your HP" if self.config.is_pvp else "Player HP"
        dealer_label = "Champion HP" if self.config.is_pvp else "Dealer HP"

        player_score = self.ui.normal_font.render(f"{player_label}: {player_hp}", True, WHITE)
        self.screen.blit(player_score, (self.WIDTH // 2 - 100, score_y))

        dealer_score = self.ui.normal_font.render(f"{dealer_label}: {dealer_hp}", True, WHITE)
        self.screen.blit(dealer_score, (self.WIDTH // 2 - 100, score_y + 40))

        # Buttons
        if not self.config.is_pvp:
            # Watch mode - auto restart message
            restart_text = self.ui.small_font.render("Restarting in 1...", True, LIGHT_GRAY)
            self.screen.blit(restart_text, (self.WIDTH // 2 - 80, 380))

        # Play again button
        self.play_again_rect = pygame.Rect(250, 380, 200, 50)
        self.ui.draw_button(self.play_again_rect, "Play Again", GREEN, BLUE)

        # Back to menu button
        self.menu_rect = pygame.Rect(250, 450, 200, 50)
        self.ui.draw_button(self.menu_rect, "Menu", DARK_GRAY, BLUE)

        # Quit button
        self.quit_rect = pygame.Rect(470, 380, 180, 50)
        self.ui.draw_button(self.quit_rect, "Quit", DARK_RED, BLUE)

    def draw(self):
        """Main draw function."""
        # Background
        self.ui.draw_background()

        # Stats header
        self.ui.draw_stats_header(
            self.stats.get_stats_text(),
            self.game.round,
            self.game.sub_round
        )

        # Entity sections
        self.ui.draw_entity_section(
            self.config.player_name,
            self.game.player,
            100,
            GREEN,
            show_items=True,
            bullet_sequence=self.game.bullet_sequence
        )

        self.ui.draw_entity_section(
            self.config.dealer_name,
            self.game.dealer,
            350,
            RED,
            show_items=self.config.show_dealer_items,
            bullet_sequence=self.game.bullet_sequence
        )

        # Bullet info and turn indicator
        self.ui.draw_bullet_info(self.game.bullet_sequence, self.game.saw_active)

        if self.game.turn == Turn.PLAYER:
            turn_label = "YOUR TURN" if self.config.is_pvp else "PLAYER'S TURN"
            self.ui.draw_turn_indicator(turn_label, True)
        else:
            turn_label = "CHAMPION'S TURN" if self.config.is_pvp else "DEALER'S TURN"
            self.ui.draw_turn_indicator(turn_label, False)

        # Action button (PvP only)
        self.action_button_rect = self._draw_action_button()

        # Back to menu button
        self.back_to_menu_rect = self._draw_back_to_menu_button()

        # Message
        self.message_timer = self.ui.draw_message(self.message, self.message_timer)

        # Menus
        if self.show_action_menu:
            self._draw_action_menu()

        if self.game_over:
            self._draw_game_over()

        pygame.display.flip()

    def handle_events(self) -> bool:
        """Handle pygame events.

        Returns:
            False to quit, True to continue
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                # Back to menu button (works anytime except game over)
                if not self.game_over and self.back_to_menu_rect and self.back_to_menu_rect.collidepoint(mouse_pos):
                    self.on_back_to_menu()
                    return True

                if self.game_over:
                    # Game over buttons
                    if self.play_again_rect and self.play_again_rect.collidepoint(mouse_pos):
                        self._restart_game()
                    elif self.menu_rect and self.menu_rect.collidepoint(mouse_pos):
                        self.on_back_to_menu()
                    elif self.quit_rect and self.quit_rect.collidepoint(mouse_pos):
                        return False

                elif self.show_action_menu:
                    # Action menu
                    menu_rect, action_rects = self._draw_action_menu()

                    for i, button_rect in enumerate(action_rects):
                        if button_rect.collidepoint(mouse_pos):
                            action = self.available_actions[i]
                            self._execute_action(action, is_player_turn=True)
                            self.show_action_menu = False

                            # Set up AI turn if needed
                            if not self.game_over:
                                self._check_ai_turn()
                            break

                    # Click outside to close
                    if not menu_rect.collidepoint(mouse_pos):
                        self.show_action_menu = False

                elif self.action_button_rect and self.action_button_rect.collidepoint(mouse_pos):
                    # Open action menu
                    if self.game.turn == Turn.PLAYER and not self.game_over:
                        self.available_actions = GameLogic.get_available_actions(self.game)
                        self.show_action_menu = True

        return True

    def update(self):
        """Update game state."""
        if self.game_over and not self.config.is_pvp:
            # Watch mode auto-restart
            self.ai_action_timer -= 1
            if self.ai_action_timer <= 0:
                self._restart_game()
            return

        # AI turn handling
        if self.waiting_for_ai and not self.game_over:
            self.ai_action_timer -= 1
            if self.ai_action_timer <= 0:
                # Determine whose turn it is
                is_player_turn = (self.game.turn == Turn.PLAYER)

                # Get AI action
                action = GameLogic.get_ai_action(
                    self.game,
                    self.model,
                    is_player_perspective=is_player_turn
                )

                # Execute action
                self._execute_action(action, is_player_turn)

                # Check if still AI turn (in watch mode or if AI continues)
                if not self.game_over:
                    if self.config.is_pvp and self.game.turn == Turn.DEALER:
                        # AI continues
                        self.ai_action_timer = 120
                    elif not self.config.is_pvp:
                        # Watch mode - always continue
                        self.ai_action_timer = 120
                    else:
                        # Player's turn now
                        self.waiting_for_ai = False
