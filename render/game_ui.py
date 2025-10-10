"""Common UI drawing functions for the game."""

import pygame
from typing import Optional, Dict
from core.constants import Item
from render.colors import *


class GameUIComponents:
    """Shared UI drawing components for game modes."""

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """Initialize UI components.

        Args:
            screen: Pygame screen surface to draw on
            width: Screen width
            height: Screen height
        """
        self.screen = screen
        self.WIDTH = width
        self.HEIGHT = height

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.header_font = pygame.font.Font(None, 36)
        self.normal_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        # Item names mapping
        self.item_names = {
            Item.GLASS: "Glass",
            Item.CIGARETTES: "Cigs",
            Item.HANDCUFFS: "Cuffs",
            Item.SAW: "Saw",
            Item.BEER: "Beer"
        }

        # Background
        self.background = self._load_background()

    def _load_background(self) -> Optional[pygame.Surface]:
        """Load background image if available."""
        try:
            bg = pygame.image.load("core/images/general_background.png")
            return pygame.transform.scale(bg, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            print(f"Could not load background image: {e}")
            return None

    def draw_background(self):
        """Draw background or fill with black."""
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(BLACK)

    def draw_entity_section(self, name: str, entity, y_offset: int, color: tuple,
                           show_items: bool = True, bullet_sequence: list = None):
        """Draw entity stats section.

        Args:
            name: Display name for the entity
            entity: Entity object with hp, items, etc.
            y_offset: Vertical offset for drawing
            color: Color for the header text
            show_items: Whether to show item details (False for opponent in PvP)
            bullet_sequence: Current bullet sequence for known_next display
        """
        # Header
        header_text = self.header_font.render(name, True, color)
        self.screen.blit(header_text, (50, y_offset))

        # HP
        hp_text = self.normal_font.render(f"HP: {entity.hp}", True, WHITE)
        self.screen.blit(hp_text, (50, y_offset + 40))

        # HP hearts
        for i in range(entity.hp):
            heart_rect = pygame.Rect(180 + i * 30, y_offset + 40, 25, 25)
            pygame.draw.rect(self.screen, RED, heart_rect)

        # Items
        items_text = self.small_font.render("Items:", True, WHITE)
        self.screen.blit(items_text, (50, y_offset + 80))

        if entity.items:
            if show_items:
                # Show detailed item list
                item_counts = {}
                for item in entity.items:
                    item_counts[item] = item_counts.get(item, 0) + 1

                x_pos = 50
                y_pos = y_offset + 110
                for item, count in item_counts.items():
                    item_name = self.item_names.get(item, str(item))
                    if count > 1:
                        item_name += f" x{count}"

                    # Draw item box
                    item_rect = pygame.Rect(x_pos, y_pos, 100, 30)
                    item_color = BLUE if show_items else DARK_GRAY
                    pygame.draw.rect(self.screen, item_color, item_rect)
                    pygame.draw.rect(self.screen, WHITE, item_rect, 2)

                    item_text = self.small_font.render(item_name, True, WHITE)
                    text_rect = item_text.get_rect(center=item_rect.center)
                    self.screen.blit(item_text, text_rect)

                    x_pos += 110
                    if x_pos > 400:
                        x_pos = 50
                        y_pos += 40
            else:
                # Just show count
                count_text = self.small_font.render(f"{len(entity.items)} items", True, WHITE)
                self.screen.blit(count_text, (50, y_offset + 110))
        else:
            no_items = self.small_font.render("None", True, GRAY)
            self.screen.blit(no_items, (50, y_offset + 110))

        # Status effects
        status_y = y_offset + 170
        if entity.handcuff_strength > 0:
            status = self.small_font.render(
                f"HANDCUFFED (str {entity.handcuff_strength})", True, YELLOW
            )
            self.screen.blit(status, (50, status_y))
            status_y += 25

        if entity.known_next and bullet_sequence and len(bullet_sequence) > 0:
            next_bullet = "LIVE" if bullet_sequence[0] == 1 else "BLANK"
            known = self.small_font.render(f"Known: Next is {next_bullet}", True, YELLOW)
            self.screen.blit(known, (50, status_y))

    def draw_bullet_info(self, bullet_sequence: list, saw_active: bool, x_offset: int = 550, y_offset: int = 50):
        """Draw bullet/shotgun information.

        Args:
            bullet_sequence: List of bullets (0=blank, 1=live)
            saw_active: Whether saw is currently active
            x_offset: Horizontal position
            y_offset: Vertical position
        """
        # Header
        header_text = self.header_font.render("SHOTGUN", True, YELLOW)
        self.screen.blit(header_text, (x_offset, y_offset))

        # Bullet counts
        lives = bullet_sequence.count(1)
        blanks = bullet_sequence.count(0)
        total = len(bullet_sequence)

        info_y = y_offset + 50
        total_text = self.normal_font.render(f"Bullets: {total}", True, WHITE)
        self.screen.blit(total_text, (x_offset, info_y))

        info_y += 35
        lives_text = self.normal_font.render(f"Live: {lives}", True, RED)
        self.screen.blit(lives_text, (x_offset, info_y))

        info_y += 35
        blanks_text = self.normal_font.render(f"Blank: {blanks}", True, WHITE)
        self.screen.blit(blanks_text, (x_offset, info_y))

        # Saw active
        if saw_active:
            info_y += 40
            saw_text = self.normal_font.render("SAW ACTIVE!", True, YELLOW)
            self.screen.blit(saw_text, (x_offset, info_y))
            saw_info = self.small_font.render("Next shot: 2x damage", True, YELLOW)
            self.screen.blit(saw_info, (x_offset, info_y + 30))

    def draw_turn_indicator(self, turn_text: str, is_player_turn: bool, x_offset: int = 550, y_offset: int = 350):
        """Draw turn indicator.

        Args:
            turn_text: Text to display
            is_player_turn: True if it's the player's turn (green), False for opponent (red)
            x_offset: Horizontal position
            y_offset: Vertical position
        """
        color = GREEN if is_player_turn else RED
        text = self.header_font.render(turn_text, True, color)
        self.screen.blit(text, (x_offset, y_offset))

    def draw_message(self, message: str, timer: int, y_pos: int = 50) -> int:
        """Draw temporary message with background.

        Args:
            message: Message text to display
            timer: Current timer value (decrements each frame)
            y_pos: Vertical position

        Returns:
            Updated timer value
        """
        if timer > 0:
            msg_surface = self.header_font.render(message, True, YELLOW)
            msg_rect = msg_surface.get_rect(center=(self.WIDTH // 2, y_pos))

            # Background
            bg_rect = msg_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, BLACK, bg_rect)
            pygame.draw.rect(self.screen, YELLOW, bg_rect, 3)

            self.screen.blit(msg_surface, msg_rect)
            return timer - 1
        return 0

    def draw_button(self, rect: pygame.Rect, text: str, base_color: tuple,
                   hover_color: tuple, text_color: tuple = WHITE,
                   border_color: tuple = WHITE, border_width: int = 3) -> bool:
        """Draw a button and return if it's hovered.

        Args:
            rect: Button rectangle
            text: Button text
            base_color: Normal button color
            hover_color: Color when hovered
            text_color: Text color
            border_color: Border color
            border_width: Border width in pixels

        Returns:
            True if mouse is hovering over button
        """
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)

        color = hover_color if is_hovered else base_color
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, border_color, rect, border_width)

        button_text = self.header_font.render(text, True, text_color)
        text_rect = button_text.get_rect(center=rect.center)
        self.screen.blit(button_text, text_rect)

        return is_hovered

    def draw_stats_header(self, stats_text: str, round_num: int, sub_round: int):
        """Draw stats header and round info.

        Args:
            stats_text: Statistics text to display on left
            round_num: Current round number
            sub_round: Current sub-round number
        """
        stats = self.normal_font.render(stats_text, True, GOLD)
        self.screen.blit(stats, (20, 10))

        round_text = self.small_font.render(f"Round {round_num} | Subround {sub_round}", True, WHITE)
        self.screen.blit(round_text, (self.WIDTH - 250, 10))
