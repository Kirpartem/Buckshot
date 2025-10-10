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
PURPLE = (147, 51, 234)


class SelfPlayGUI:
    def __init__(self, model_path: str = "agent/models/champion.zip"):
        """Initialize pygame GUI for self-play."""
        pygame.init()

        self.WIDTH = 1000
        self.HEIGHT = 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Buckshot Roulette - AI Self-Play")

        self.clock = pygame.time.Clock()
        self.FPS = 60

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.header_font = pygame.font.Font(None, 36)
        self.normal_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        # Load model
        print("Loading model...")
        self.model = MaskablePPO.load(model_path, device="cpu")
        print(f"Model loaded from: {model_path}")

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

        # Stats tracking
        self.player_wins = 0
        self.dealer_wins = 0
        self.draws = 0

        # UI state
        self.message = ""
        self.message_timer = 0
        self.ai_action_timer = 120  # 1 second delay between actions
        self.game_over = False
        self.auto_restart = True
        self.restart_timer = 0

        # Item icons mapping
        self.item_names = {
            Item.GLASS: "Glass",
            Item.CIGARETTES: "Cigs",
            Item.HANDCUFFS: "Cuffs",
            Item.SAW: "Saw",
            Item.BEER: "Beer"
        }

    def get_obs_for_current_turn(self):
        """Build observation for whichever AI's turn it is.
        The model always expects observation from the perspective of the acting player (bot vs target)."""
        # Determine bot and target based on whose turn it is
        if self.game.turn == Turn.PLAYER:
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
        """Get action from AI for the current turn."""
        obs = self.get_obs_for_current_turn()
        action_mask = self.game.get_valid_actions_mask()
        action_idx, _ = self.model.predict(obs, action_masks=action_mask, deterministic=False)
        action = list(GameAction)[int(action_idx)]
        return action

    def format_action_message(self, actor_name: str, action: GameAction, ejected_bullet=None) -> str:
        """Format action as message."""
        actor = actor_name
        target = "DEALER" if actor == "PLAYER" else "PLAYER"

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

    def execute_action(self, action: GameAction, is_player: bool):
        """Execute an action and update UI."""
        actor_name = "PLAYER" if is_player else "DEALER"

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
            self.restart_timer = 60  # 1 seconds before auto-restart

            # Update stats
            if self.game.player.hp > 0 and self.game.dealer.hp <= 0:
                self.player_wins += 1
            elif self.game.dealer.hp > 0 and self.game.player.hp <= 0:
                self.dealer_wins += 1
            elif self.game.player.hp <= 0 and self.game.dealer.hp <= 0:
                self.draws += 1

    def draw_entity_section(self, name: str, entity, y_offset: int, color):
        """Draw entity stats section."""
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
        if entity.handcuff_strength > 0:
            status = self.small_font.render(f"HANDCUFFED (str {entity.handcuff_strength})", True, YELLOW)
            self.screen.blit(status, (50, status_y))
            status_y += 25

        if entity.known_next and len(self.game.bullet_sequence) > 0:
            next_bullet = "LIVE" if self.game.bullet_sequence[0] == 1 else "BLANK"
            known = self.small_font.render(f"Known: Next is {next_bullet}", True, YELLOW)
            self.screen.blit(known, (50, status_y))

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
            turn_text = self.header_font.render("PLAYER'S TURN", True, GREEN)
        else:
            turn_text = self.header_font.render("DEALER'S TURN", True, RED)

        self.screen.blit(turn_text, (x_offset, y_offset))

    def draw_game_over(self):
        """Draw game over screen."""
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(150)
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
            title = self.title_font.render("PLAYER WINS!", True, GREEN)
        else:
            title = self.title_font.render("DEALER WINS!", True, RED)

        title_rect = title.get_rect(center=(self.WIDTH // 2, 270))
        self.screen.blit(title, title_rect)

        # Scores
        score_y = 340
        player_score = self.normal_font.render(f"Player HP: {player_hp}", True, WHITE)
        self.screen.blit(player_score, (self.WIDTH // 2 - 100, score_y))

        dealer_score = self.normal_font.render(f"Dealer HP: {dealer_hp}", True, WHITE)
        self.screen.blit(dealer_score, (self.WIDTH // 2 - 100, score_y + 40))

        # Auto-restart message
        restart_sec = self.restart_timer // 60
        restart_msg = self.small_font.render(f"Restarting in {restart_sec + 1}...", True, LIGHT_GRAY)
        self.screen.blit(restart_msg, (self.WIDTH // 2 - 100, score_y + 90))

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

        # Draw stats counter at top
        total_games = self.player_wins + self.dealer_wins + self.draws
        if total_games > 0:
            player_pct = (self.player_wins / total_games) * 100
            dealer_pct = (self.dealer_wins / total_games) * 100
            stats_text = self.normal_font.render(
                f"Player: {self.player_wins} ({player_pct:.1f}%) | Dealer: {self.dealer_wins} ({dealer_pct:.1f}%) | Draws: {self.draws}",
                True, GOLD
            )
        else:
            stats_text = self.normal_font.render(
                "Player: 0 | Dealer: 0 | Draws: 0",
                True, GOLD
            )
        self.screen.blit(stats_text, (20, 10))

        # Draw round info
        round_text = self.small_font.render(f"Round {self.game.round} | Subround {self.game.sub_round}", True, WHITE)
        self.screen.blit(round_text, (self.WIDTH - 250, 10))

        # Draw sections
        self.draw_entity_section("PLAYER (AI)", self.game.player, 100, GREEN)
        self.draw_entity_section("DEALER (AI)", self.game.dealer, 350, RED)
        self.draw_bullet_info()
        self.draw_turn_indicator()

        # Draw message
        self.draw_message()

        if self.game_over:
            self.draw_game_over()

        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def update(self):
        """Update game state."""
        if self.game_over:
            # Handle auto-restart
            if self.auto_restart:
                self.restart_timer -= 1
                if self.restart_timer <= 0:
                    self.game = BuckshotRouletteGame(np.random.seed())
                    self.game.start_new_round()
                    self.game_over = False
                    self.message = ""
                    self.ai_action_timer = 60
            return

        # AI turn
        self.ai_action_timer -= 1
        if self.ai_action_timer <= 0:
            # Determine which entity is acting (for display purposes)
            is_player = (self.game.turn == Turn.PLAYER)

            # Get AI action (observation is built from current turn's perspective)
            action = self.get_ai_action()
            self.execute_action(action, is_player)

            # Reset timer for next action
            self.ai_action_timer = 120  # 1 second delay

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
    model_path = "agent/models/champion.zip"

    if not os.path.exists(model_path):
        print(f"No champion model found at {model_path}")
        print("Please train a model first using train.py")
        return

    game = SelfPlayGUI(model_path)
    game.run()


if __name__ == "__main__":
    main()
