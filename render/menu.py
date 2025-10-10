"""Main menu selection screen."""

import pygame
from typing import Optional, Callable
from render.colors import *


class MenuScreen:
    """Main menu for selecting game mode."""

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """Initialize menu screen.

        Args:
            screen: Pygame screen surface
            width: Screen width
            height: Screen height
        """
        self.screen = screen
        self.WIDTH = width
        self.HEIGHT = height

        # Fonts
        self.title_font = pygame.font.Font(None, 72)
        self.header_font = pygame.font.Font(None, 42)
        self.normal_font = pygame.font.Font(None, 28)

        # Load background
        self.background = self._load_background()

        # Load character images
        self.player_image = self._load_image("core/images/player_pixel.png", (120, 120))
        self.dealer_image = self._load_image("core/images/dealer_pixel.png", (120, 120))

    def _load_background(self) -> Optional[pygame.Surface]:
        """Load background image if available."""
        try:
            bg = pygame.image.load("core/background/image.png")
            return pygame.transform.scale(bg, (self.WIDTH, self.HEIGHT))
        except Exception:
            return None

    def _load_image(self, path: str, size: tuple) -> Optional[pygame.Surface]:
        """Load and scale an image.

        Args:
            path: Path to image file
            size: Target size (width, height)

        Returns:
            Loaded and scaled surface, or None if not found
        """
        try:
            img = pygame.image.load(path)
            return pygame.transform.scale(img, size)
        except Exception:
            return None

    def _create_placeholder_image(self, size: tuple, color: tuple, text: str) -> pygame.Surface:
        """Create a placeholder image with text.

        Args:
            size: Image size (width, height)
            color: Background color
            text: Text to display

        Returns:
            Surface with placeholder
        """
        surface = pygame.Surface(size)
        surface.fill(color)

        # Draw border
        pygame.draw.rect(surface, WHITE, surface.get_rect(), 3)

        # Draw text
        font = pygame.font.Font(None, 36)
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect(center=(size[0] // 2, size[1] // 2))
        surface.blit(text_surface, text_rect)

        return surface

    def draw(self) -> tuple:
        """Draw menu screen.

        Returns:
            Tuple of (play_vs_ai_rect, ai_vs_ai_rect, quit_rect)
        """
        # Background
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(BLACK)

        # Title
        title = self.title_font.render("BUCKSHOT ROULETTE", True, GOLD)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 80))
        self.screen.blit(title, title_rect)

        # Subtitle
        subtitle = self.normal_font.render("Select Game Mode", True, WHITE)
        subtitle_rect = subtitle.get_rect(center=(self.WIDTH // 2, 140))
        self.screen.blit(subtitle, subtitle_rect)

        # Mode selection boxes
        box_width = 320
        box_height = 420
        box_y = 200
        spacing = 60
        left_box_x = (self.WIDTH - box_width * 2 - spacing) // 2
        right_box_x = left_box_x + box_width + spacing

        mouse_pos = pygame.mouse.get_pos()

        # Play vs AI box (player vs dealer)
        play_rect = pygame.Rect(left_box_x, box_y, box_width, box_height)
        self._draw_mode_box(
            play_rect,
            "PLAY VS AI",
            "Challenge the champion AI",
            left_image=self.player_image,
            right_image=self.dealer_image,
            mouse_pos=mouse_pos
        )

        # AI vs AI box (dealer vs dealer)
        watch_rect = pygame.Rect(right_box_x, box_y, box_width, box_height)
        self._draw_mode_box(
            watch_rect,
            "AI VS AI",
            "Watch AI agents battle",
            left_image=self.dealer_image,
            right_image=self.dealer_image,
            mouse_pos=mouse_pos
        )

        # Quit button
        quit_rect = pygame.Rect(self.WIDTH // 2 - 100, 650, 200, 40)
        self._draw_quit_button(quit_rect, mouse_pos)

        pygame.display.flip()
        return play_rect, watch_rect, quit_rect

    def _draw_mode_box(self, rect: pygame.Rect, title: str, description: str,
                      left_image: Optional[pygame.Surface], right_image: Optional[pygame.Surface],
                      mouse_pos: tuple):
        """Draw a game mode selection box with character images and VS.

        Args:
            rect: Box rectangle
            title: Mode title
            description: Mode description
            left_image: Left character image
            right_image: Right character image
            mouse_pos: Current mouse position
        """
        is_hovered = rect.collidepoint(mouse_pos)

        # Box background - always BLACK
        pygame.draw.rect(self.screen, BLACK, rect)
        pygame.draw.rect(self.screen, GOLD if is_hovered else WHITE, rect, 4)

        # Title
        title_surface = self.header_font.render(title, True, GOLD if is_hovered else WHITE)
        title_rect = title_surface.get_rect(centerx=rect.centerx, top=rect.top + 15)
        self.screen.blit(title_surface, title_rect)

        # Character images with VS in the middle
        content_y = rect.top + 80
        content_height = 280

        # Calculate positions for left image, VS text, and right image
        vs_font = pygame.font.Font(None, 72)
        vs_text = vs_font.render("VS", True, GOLD if is_hovered else WHITE)
        vs_rect = vs_text.get_rect(center=(rect.centerx, content_y + content_height // 2))

        # Left character image
        if left_image:
            left_x = rect.centerx - 100  # Position left of center
            left_rect = left_image.get_rect(center=(left_x, content_y + content_height // 2))
            self.screen.blit(left_image, left_rect)
        else:
            # Placeholder for left
            placeholder = self._create_placeholder_image((100, 100), DARK_GRAY, "?")
            left_rect = placeholder.get_rect(center=(rect.centerx - 100, content_y + content_height // 2))
            self.screen.blit(placeholder, left_rect)

        # VS text in the middle
        self.screen.blit(vs_text, vs_rect)

        # Right character image
        if right_image:
            right_x = rect.centerx + 100  # Position right of center
            right_rect = right_image.get_rect(center=(right_x, content_y + content_height // 2))
            self.screen.blit(right_image, right_rect)
        else:
            # Placeholder for right
            placeholder = self._create_placeholder_image((100, 100), DARK_GRAY, "?")
            right_rect = placeholder.get_rect(center=(rect.centerx + 100, content_y + content_height // 2))
            self.screen.blit(placeholder, right_rect)

        # Description
        desc_surface = self.normal_font.render(description, True, WHITE)
        desc_rect = desc_surface.get_rect(centerx=rect.centerx, bottom=rect.bottom - 15)
        self.screen.blit(desc_surface, desc_rect)

    def _draw_quit_button(self, rect: pygame.Rect, mouse_pos: tuple):
        """Draw quit button.

        Args:
            rect: Button rectangle
            mouse_pos: Current mouse position
        """
        is_hovered = rect.collidepoint(mouse_pos)

        color = BLUE if is_hovered else DARK_RED
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, WHITE, rect, 2)

        text = self.normal_font.render("Quit", True, WHITE)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def handle_events(self, play_rect: pygame.Rect, watch_rect: pygame.Rect,
                     quit_rect: pygame.Rect) -> Optional[str]:
        """Handle menu events.

        Args:
            play_rect: Play vs AI button rectangle
            watch_rect: AI vs AI button rectangle
            quit_rect: Quit button rectangle

        Returns:
            "play", "watch", "quit", or None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if play_rect.collidepoint(mouse_pos):
                    return "play"
                elif watch_rect.collidepoint(mouse_pos):
                    return "watch"
                elif quit_rect.collidepoint(mouse_pos):
                    return "quit"

        return None
