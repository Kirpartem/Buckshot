import os
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*") #Shut up pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1" #Shut up pygame
import pygame
from sb3_contrib import MaskablePPO
from core.game import BuckshotRouletteGame
from core.constants import GameAction, Item, Turn
import numpy as np


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)
DARK_GRAY = (50, 50, 50)
RED = (220, 50, 50)
GREEN = (50, 220, 50)
BLUE = (50, 150, 220)
YELLOW = (255, 215, 0)
DARK_RED = (139, 0, 0)
GOLD = (255, 215, 0)


class BuckshotGUI:
    def __init__(self, champion_path: str = "agent/models/champion.zip"):
        """Initialize pygame GUI."""
        pygame.init()

        self.WIDTH = 1000
        self.HEIGHT = 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Buckshot Roulette - vs Champion")

        self.clock = pygame.time.Clock()
        self.FPS = 60

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.header_font = pygame.font.Font(None, 36)
        self.normal_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        # Load champion
        print("Loading champion model...")
        self.champion = MaskablePPO.load(champion_path, device="cpu")
        print(f"Champion loaded from: {champion_path}")

        # Load background image
        try:
            self.background = pygame.image.load("core/background/image.png")
            self.background = pygame.transform.scale(self.background, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            print(f"Could not load background image: {e}")
            self.background = None

        # Game state
        self.game = BuckshotRouletteGame(np.random.seed())
        self.game.start_new_round()

        # Winrate tracking
        self.wins = 0
        self.losses = 0

        # UI state
        self.message = ""
        self.message_timer = 0
        self.show_action_menu = False
        self.available_actions = []
        self.selected_action_idx = 0
        self.waiting_for_ai = False
        self.ai_action_timer = 0
        self.game_over = False

        # Check if AI goes first
        if self.game.turn == Turn.DEALER:
            self.waiting_for_ai = True
            self.ai_action_timer = 120  # 2 seconds delay

        # Item icons mapping
        self.item_names = {
            Item.GLASS: "Glass",
            Item.CIGARETTES: "Cigs",
            Item.HANDCUFFS: "Cuffs",
            Item.SAW: "Saw",
            Item.BEER: "Beer"
        }

    def get_obs_for_agent(self, is_player: bool):
        """Build observation for the AI agent."""
        if is_player:
            bot = self.game.player
            target = self.game.dealer
        else:
            bot = self.game.dealer
            target = self.game.player

        obs = np.zeros(19, dtype=np.float32)
        obs[0] = bot.hp / 10.0
        obs[1] = bot.items.count(Item.GLASS) / 8.0
        obs[2] = bot.items.count(Item.CIGARETTES) / 8.0
        obs[3] = bot.items.count(Item.HANDCUFFS) / 8.0
        obs[4] = bot.items.count(Item.SAW) / 8.0
        obs[5] = bot.items.count(Item.BEER) / 8.0
        obs[6] = target.hp / 10.0
        obs[7] = target.items.count(Item.GLASS) / 8.0
        obs[8] = target.items.count(Item.CIGARETTES) / 8.0
        obs[9] = target.items.count(Item.HANDCUFFS) / 8.0
        obs[10] = target.items.count(Item.SAW) / 8.0
        obs[11] = target.items.count(Item.BEER) / 8.0
        obs[12] = target.handcuff_strength / 2.0

        bullet_seq = self.game.bullet_sequence
        blanks_left = bullet_seq.count(0)
        lives_left = len(bullet_seq) - blanks_left
        obs[13] = blanks_left / 6.0
        obs[14] = lives_left / 6.0

        if bot.known_next and len(bullet_seq) > 0:
            obs[15] = 1.0 if bullet_seq[0] == 1 else 0.0
            obs[16] = 1.0 if bullet_seq[0] == 0 else 0.0
            obs[17] = 0.0
        else:
            obs[15] = 0.0
            obs[16] = 0.0
            obs[17] = 1.0

        obs[18] = 1.0 if self.game.saw_active else 0.0
        return obs

    def get_ai_action(self):
        """Get action from champion AI."""
        obs = self.get_obs_for_agent(is_player=False)
        action_mask = self.game.get_valid_actions_mask()
        action_idx, _ = self.champion.predict(obs, action_masks=action_mask, deterministic=False)
        action = list(GameAction)[int(action_idx)]
        return action

    def format_action_message(self, actor_name: str, action: GameAction, ejected_bullet=None) -> str:
        """Format action as message."""
        actor = actor_name
        target = "Champion" if actor == "YOU" else "You"

        if action == GameAction.SHOOT_SELF:
            return f"{actor} shot themselves!"
        elif action == GameAction.SHOOT_TARGET:
            return f"{actor} shot {target}!"
        elif action == GameAction.USE_GLASS:
            return f"{actor} used Magnifying Glass!"
        elif action == GameAction.USE_CIGARETTES:
            return f"{actor} smoked and healed 1 HP!"
        elif action == GameAction.USE_HANDCUFFS:
            return f"{actor} handcuffed {target}!"
        elif action == GameAction.USE_SAW:
            return f"{actor} used Saw - 2x damage!"
        elif action == GameAction.USE_BEER:
            if ejected_bullet is not None:
                bullet_type = "LIVE" if ejected_bullet == 1 else "BLANK"
                return f"{actor} ejected a {bullet_type}!"
            return f"{actor} used Beer!"
        return ""

    def show_message(self, msg: str, duration: int = 120):
        """Show a temporary message."""
        self.message = msg
        self.message_timer = duration

    def get_available_actions(self):
        """Get list of available actions for human player."""
        action_mask = self.game.get_valid_actions_mask()
        actions = []
        for i, action in enumerate(GameAction):
            if action_mask[i] == 1:
                actions.append(action)
        return actions

    def action_description(self, action: GameAction) -> str:
        """Get readable action description."""
        descriptions = {
            GameAction.SHOOT_SELF: "Shoot Yourself",
            GameAction.SHOOT_TARGET: "Shoot Dealer",
            GameAction.USE_GLASS: "Use Glass",
            GameAction.USE_CIGARETTES: "Use Cigarettes",
            GameAction.USE_HANDCUFFS: "Use Handcuffs",
            GameAction.USE_SAW: "Use Saw",
            GameAction.USE_BEER: "Use Beer"
        }
        return descriptions.get(action, str(action))

    def execute_action(self, action: GameAction, is_player: bool):
        """Execute an action and update UI."""
        actor_name = "YOU" if is_player else "CHAMPION"

        # Track ejected bullet
        ejected_bullet = None
        if action == GameAction.USE_BEER and len(self.game.bullet_sequence) > 0:
            ejected_bullet = self.game.bullet_sequence[0]

        step_result = self.game.step(action)

        # Show action message
        msg = self.format_action_message(actor_name, action, ejected_bullet)

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

        self.show_message(msg, 180)

        # Check if game ended
        if self.game.player.hp <= 0 or self.game.dealer.hp <= 0:
            self.game_over = True
            # Update winrate
            if self.game.player.hp > 0:
                self.wins += 1
            elif self.game.dealer.hp > 0:
                self.losses += 1
            # Don't count draws

    def draw_player_section(self, y_offset: int):
        """Draw player stats section."""
        # Header
        header_text = self.header_font.render("You", True, GREEN)
        self.screen.blit(header_text, (50, y_offset))

        # HP
        hp_text = self.normal_font.render(f"HP: {self.game.player.hp}", True, WHITE)
        self.screen.blit(hp_text, (50, y_offset + 40))

        # HP hearts
        for i in range(self.game.player.hp):
            heart_rect = pygame.Rect(180 + i * 30, y_offset + 40, 25, 25)
            pygame.draw.rect(self.screen, RED, heart_rect)

        # Items
        items_text = self.small_font.render("Items:", True, WHITE)
        self.screen.blit(items_text, (50, y_offset + 80))

        if self.game.player.items:
            item_counts = {}
            for item in self.game.player.items:
                item_counts[item] = item_counts.get(item, 0) + 1

            x_pos = 50
            y_pos = y_offset + 110
            for item, count in item_counts.items():
                item_name = self.item_names.get(item, str(item))
                if count > 1:
                    item_name += f" x{count}"

                # Draw item box
                item_rect = pygame.Rect(x_pos, y_pos, 100, 30)
                pygame.draw.rect(self.screen, BLUE, item_rect)
                pygame.draw.rect(self.screen, WHITE, item_rect, 2)

                item_text = self.small_font.render(item_name, True, WHITE)
                text_rect = item_text.get_rect(center=item_rect.center)
                self.screen.blit(item_text, text_rect)

                x_pos += 110
                if x_pos > 400:
                    x_pos = 50
                    y_pos += 40
        else:
            no_items = self.small_font.render("None", True, GRAY)
            self.screen.blit(no_items, (50, y_offset + 110))

        # Status effects
        status_y = y_offset + 170
        if self.game.player.handcuff_strength > 0:
            status = self.small_font.render(f"HANDCUFFED (str {self.game.player.handcuff_strength})", True, YELLOW)
            self.screen.blit(status, (50, status_y))
            status_y += 25

        if self.game.player.known_next and len(self.game.bullet_sequence) > 0:
            next_bullet = "LIVE" if self.game.bullet_sequence[0] == 1 else "BLANK"
            known = self.small_font.render(f"Known: Next is {next_bullet}", True, YELLOW)
            self.screen.blit(known, (50, status_y))

    def draw_dealer_section(self, y_offset: int):
        """Draw dealer stats section."""
        # Header
        header_text = self.header_font.render("CHAMPION (Dealer)", True, RED)
        self.screen.blit(header_text, (50, y_offset))

        # HP
        hp_text = self.normal_font.render(f"HP: {self.game.dealer.hp}", True, WHITE)
        self.screen.blit(hp_text, (50, y_offset + 40))

        # HP hearts
        for i in range(self.game.dealer.hp):
            heart_rect = pygame.Rect(180 + i * 30, y_offset + 40, 25, 25)
            pygame.draw.rect(self.screen, RED, heart_rect)

        # Items count
        items_text = self.small_font.render(f"Items: {len(self.game.dealer.items)}", True, WHITE)
        self.screen.blit(items_text, (50, y_offset + 80))

        # Show dealer items
        if self.game.dealer.items:
            item_counts = {}
            for item in self.game.dealer.items:
                item_counts[item] = item_counts.get(item, 0) + 1

            x_pos = 50
            y_pos = y_offset + 110
            for item, count in item_counts.items():
                item_name = self.item_names.get(item, str(item))
                if count > 1:
                    item_name += f" x{count}"

                # Draw item box
                item_rect = pygame.Rect(x_pos, y_pos, 100, 30)
                pygame.draw.rect(self.screen, DARK_GRAY, item_rect)
                pygame.draw.rect(self.screen, WHITE, item_rect, 2)

                item_text = self.small_font.render(item_name, True, WHITE)
                text_rect = item_text.get_rect(center=item_rect.center)
                self.screen.blit(item_text, text_rect)

                x_pos += 110
                if x_pos > 400:
                    x_pos = 50
                    y_pos += 40

        # Status effects
        if self.game.dealer.handcuff_strength > 0:
            status = self.small_font.render(f"HANDCUFFED (str {self.game.dealer.handcuff_strength})", True, YELLOW)
            self.screen.blit(status, (50, y_offset + 170))

    def draw_bullet_info(self):
        """Draw bullet/shotgun info."""
        x_offset = 550
        y_offset = 50

        # Header
        header_text = self.header_font.render("SHOTGUN", True, YELLOW)
        self.screen.blit(header_text, (x_offset, y_offset))

        # Bullet counts
        bullet_seq = self.game.bullet_sequence
        lives = bullet_seq.count(1)
        blanks = bullet_seq.count(0)
        total = len(bullet_seq)

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
        if self.game.saw_active:
            info_y += 40
            saw_text = self.normal_font.render("SAW ACTIVE!", True, YELLOW)
            self.screen.blit(saw_text, (x_offset, info_y))
            saw_info = self.small_font.render("Next shot: 2x damage", True, YELLOW)
            self.screen.blit(saw_info, (x_offset, info_y + 30))

    def draw_turn_indicator(self):
        """Draw whose turn it is."""
        x_offset = 550
        y_offset = 350

        if self.game.turn == Turn.PLAYER:
            turn_text = self.header_font.render("YOUR TURN", True, GREEN)
        else:
            turn_text = self.header_font.render("CHAMPION'S TURN", True, RED)

        self.screen.blit(turn_text, (x_offset, y_offset))

    def draw_action_button(self):
        """Draw the action button."""
        if self.game.turn != Turn.PLAYER or self.game_over:
            return

        button_rect = pygame.Rect(550, 450, 400, 80)

        # Hover effect
        mouse_pos = pygame.mouse.get_pos()
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, BLUE, button_rect)
        else:
            pygame.draw.rect(self.screen, DARK_GRAY, button_rect)

        pygame.draw.rect(self.screen, WHITE, button_rect, 3)

        button_text = self.header_font.render("TAKE ACTION", True, WHITE)
        text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, text_rect)

        return button_rect

    def draw_action_menu(self):
        """Draw popup action menu."""
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
        title = self.header_font.render("Choose Action", True, GOLD)
        title_rect = title.get_rect(centerx=menu_rect.centerx, top=menu_y + 20)
        self.screen.blit(title, title_rect)

        # Action buttons
        button_y = menu_y + 80
        mouse_pos = pygame.mouse.get_pos()

        for i, action in enumerate(self.available_actions):
            button_rect = pygame.Rect(menu_x + 20, button_y, menu_width - 40, 50)

            # Hover effect
            if button_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, BLUE, button_rect)
                self.selected_action_idx = i
            else:
                pygame.draw.rect(self.screen, GRAY, button_rect)

            pygame.draw.rect(self.screen, WHITE, button_rect, 2)

            action_text = self.normal_font.render(self.action_description(action), True, WHITE)
            text_rect = action_text.get_rect(center=button_rect.center)
            self.screen.blit(action_text, text_rect)

            button_y += 60

    def draw_game_over(self):
        """Draw game over screen."""
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        # Result box
        box_rect = pygame.Rect(200, 200, 600, 300)
        pygame.draw.rect(self.screen, DARK_GRAY, box_rect)
        pygame.draw.rect(self.screen, GOLD, box_rect, 5)

        # Title
        player_hp = self.game.player.hp
        dealer_hp = self.game.dealer.hp

        if player_hp <= 0 and dealer_hp <= 0:
            title = self.title_font.render("DRAW!", True, YELLOW)
        elif player_hp > 0:
            title = self.title_font.render("YOU WIN!", True, GREEN)
        else:
            title = self.title_font.render("YOU LOSE!", True, RED)

        title_rect = title.get_rect(center=(self.WIDTH // 2, 270))
        self.screen.blit(title, title_rect)

        # Scores
        score_y = 340
        your_score = self.normal_font.render(f"Your HP: {player_hp}", True, WHITE)
        self.screen.blit(your_score, (self.WIDTH // 2 - 100, score_y))

        champ_score = self.normal_font.render(f"Champion HP: {dealer_hp}", True, WHITE)
        self.screen.blit(champ_score, (self.WIDTH // 2 - 100, score_y + 40))

        # Play again button
        play_again_rect = pygame.Rect(300, 420, 200, 50)
        mouse_pos = pygame.mouse.get_pos()

        if play_again_rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, BLUE, play_again_rect)
        else:
            pygame.draw.rect(self.screen, GREEN, play_again_rect)

        pygame.draw.rect(self.screen, WHITE, play_again_rect, 3)

        again_text = self.normal_font.render("Play Again", True, WHITE)
        text_rect = again_text.get_rect(center=play_again_rect.center)
        self.screen.blit(again_text, text_rect)

        # Quit button
        quit_rect = pygame.Rect(520, 420, 180, 50)

        if quit_rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, BLUE, quit_rect)
        else:
            pygame.draw.rect(self.screen, DARK_RED, quit_rect)

        pygame.draw.rect(self.screen, WHITE, quit_rect, 3)

        quit_text = self.normal_font.render("Quit", True, WHITE)
        text_rect = quit_text.get_rect(center=quit_rect.center)
        self.screen.blit(quit_text, text_rect)

        return play_again_rect, quit_rect

    def draw_message(self):
        """Draw temporary message."""
        if self.message_timer > 0:
            msg_surface = self.header_font.render(self.message, True, YELLOW)
            msg_rect = msg_surface.get_rect(center=(self.WIDTH // 2, 50))

            # Background
            bg_rect = msg_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, BLACK, bg_rect)
            pygame.draw.rect(self.screen, YELLOW, bg_rect, 3)

            self.screen.blit(msg_surface, msg_rect)
            self.message_timer -= 1

    def draw(self):
        """Main draw function."""
        # Draw background image or fill with black
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(BLACK)

        # Draw winrate counter at top
        total_games = self.wins + self.losses
        if total_games > 0:
            win_pct = (self.wins / total_games) * 100
            winrate_text = self.normal_font.render(
                f"Wins: {self.wins} | Losses: {self.losses} | Win%: {win_pct:.1f}%",
                True, GOLD
            )
        else:
            winrate_text = self.normal_font.render(
                "Wins: 0 | Losses: 0 | Win%: 0.0%",
                True, GOLD
            )
        self.screen.blit(winrate_text, (20, 10))

        # Draw round info
        round_text = self.small_font.render(f"Round {self.game.round} | Subround {self.game.sub_round}", True, WHITE)
        self.screen.blit(round_text, (self.WIDTH - 250, 10))

        # Draw sections
        self.draw_player_section(100)
        self.draw_dealer_section(350)
        self.draw_bullet_info()
        self.draw_turn_indicator()

        # Draw action button
        self.action_button_rect = self.draw_action_button()

        # Draw message
        self.draw_message()

        # Draw menus
        if self.show_action_menu:
            self.draw_action_menu()

        if self.game_over:
            self.play_again_rect, self.quit_rect = self.draw_game_over()

        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if self.game_over:
                    # Game over buttons
                    if self.play_again_rect.collidepoint(mouse_pos):
                        self.game = BuckshotRouletteGame(np.random.seed())
                        self.game.start_new_round()
                        self.game_over = False
                        self.message = ""
                        self.show_action_menu = False
                        self.waiting_for_ai = False

                        # Check if AI goes first in new game
                        if self.game.turn == Turn.DEALER:
                            self.waiting_for_ai = True
                            self.ai_action_timer = 120  # 2 seconds delay
                    elif self.quit_rect.collidepoint(mouse_pos):
                        return False

                elif self.show_action_menu:
                    # Action menu
                    menu_width = 500
                    menu_height = min(400, 100 + len(self.available_actions) * 60)
                    menu_x = (self.WIDTH - menu_width) // 2
                    menu_y = (self.HEIGHT - menu_height) // 2

                    button_y = menu_y + 80

                    for i, action in enumerate(self.available_actions):
                        button_rect = pygame.Rect(menu_x + 20, button_y, menu_width - 40, 50)
                        if button_rect.collidepoint(mouse_pos):
                            # Execute action
                            self.execute_action(action, is_player=True)
                            self.show_action_menu = False

                            # If game not over and not player's turn anymore, set up AI turn
                            if not self.game_over and self.game.turn != Turn.PLAYER:
                                self.waiting_for_ai = True
                                self.ai_action_timer = 120  # 2 seconds delay
                            break
                        button_y += 60

                    # Click outside to close
                    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
                    if not menu_rect.collidepoint(mouse_pos):
                        self.show_action_menu = False

                elif self.action_button_rect and self.action_button_rect.collidepoint(mouse_pos):
                    # Open action menu
                    if self.game.turn == Turn.PLAYER and not self.game_over:
                        self.available_actions = self.get_available_actions()
                        self.show_action_menu = True
                        self.selected_action_idx = 0

        return True

    def update(self):
        """Update game state."""
        # Check if it's AI turn but not waiting yet (shouldn't happen, but safety check)
        if not self.waiting_for_ai and not self.game_over and self.game.turn == Turn.DEALER and not self.show_action_menu:
            self.waiting_for_ai = True
            self.ai_action_timer = 120  # 2 seconds delay

        # AI turn
        if self.waiting_for_ai and not self.game_over:
            self.ai_action_timer -= 1
            if self.ai_action_timer <= 0:
                # Execute AI action
                action = self.get_ai_action()
                self.execute_action(action, is_player=False)

                # Check if AI gets another turn
                if not self.game_over and self.game.turn == Turn.DEALER:
                    self.ai_action_timer = 120  # 2 seconds delay between actions
                else:
                    self.waiting_for_ai = False

    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.FPS)

        pygame.quit()


def main():
    """Main entry point."""
    champion_path = "agent/models/champion.zip"

    if not os.path.exists(champion_path):
        print(f"No champion model found at {champion_path}")
        print("Please train a model first using: python -m agent.train")
        return

    game = BuckshotGUI(champion_path)
    game.run()


if __name__ == "__main__":
    main()
