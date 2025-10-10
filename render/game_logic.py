"""Game logic helpers - observation building, AI actions, action formatting."""

import numpy as np
from typing import Optional
from sb3_contrib import MaskablePPO

from core.game import BuckshotRouletteGame
from core.constants import GameAction, Item, Turn


class GameLogic:
    """Handles game logic operations like building observations and getting AI actions."""

    @staticmethod
    def build_observation(game: BuckshotRouletteGame, is_player_perspective: bool) -> np.ndarray:
        """Build observation array for AI model.

        Args:
            game: Current game instance
            is_player_perspective: True for player perspective, False for dealer

        Returns:
            Observation array for the model
        """
        if is_player_perspective:
            bot = game.player
            target = game.dealer
        else:
            bot = game.dealer
            target = game.player

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

        bullet_seq = game.bullet_sequence
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

        obs[18] = 1.0 if game.saw_active else 0.0
        return obs

    @staticmethod
    def get_ai_action(game: BuckshotRouletteGame, model: MaskablePPO,
                     is_player_perspective: bool) -> GameAction:
        """Get action from AI model.

        Args:
            game: Current game instance
            model: Trained PPO model
            is_player_perspective: True for player, False for dealer

        Returns:
            Selected action
        """
        obs = GameLogic.build_observation(game, is_player_perspective)
        action_mask = game.get_valid_actions_mask()
        action_idx, _ = model.predict(obs, action_masks=action_mask, deterministic=False)
        return list(GameAction)[int(action_idx)]

    @staticmethod
    def format_action_message(actor_name: str, target_name: str, action: GameAction,
                             ejected_bullet: Optional[int] = None) -> str:
        """Format action as a message string.

        Args:
            actor_name: Name of the actor performing the action
            target_name: Name of the target
            action: Action performed
            ejected_bullet: Bullet ejected if using beer (0=blank, 1=live)

        Returns:
            Formatted message string
        """
        if action == GameAction.SHOOT_SELF:
            return f"{actor_name} shot themselves!"
        elif action == GameAction.SHOOT_TARGET:
            return f"{actor_name} shot {target_name}!"
        elif action == GameAction.USE_GLASS:
            return f"{actor_name} used Magnifying Glass!"
        elif action == GameAction.USE_CIGARETTES:
            return f"{actor_name} smoked and healed 1 HP!"
        elif action == GameAction.USE_HANDCUFFS:
            return f"{actor_name} handcuffed {target_name}!"
        elif action == GameAction.USE_SAW:
            return f"{actor_name} used Saw - 2x damage!"
        elif action == GameAction.USE_BEER:
            if ejected_bullet is not None:
                bullet_type = "LIVE" if ejected_bullet == 1 else "BLANK"
                return f"{actor_name} ejected a {bullet_type}!"
            return f"{actor_name} used Beer!"
        return ""

    @staticmethod
    def get_action_description(action: GameAction, is_shooting_dealer: bool = True) -> str:
        """Get human-readable action description.

        Args:
            action: GameAction enum value
            is_shooting_dealer: True if shooting dealer, False if opponent is different

        Returns:
            Human-readable description
        """
        descriptions = {
            GameAction.SHOOT_SELF: "Shoot Yourself",
            GameAction.SHOOT_TARGET: "Shoot Dealer" if is_shooting_dealer else "Shoot Opponent",
            GameAction.USE_GLASS: "Use Glass",
            GameAction.USE_CIGARETTES: "Use Cigarettes",
            GameAction.USE_HANDCUFFS: "Use Handcuffs",
            GameAction.USE_SAW: "Use Saw",
            GameAction.USE_BEER: "Use Beer"
        }
        return descriptions.get(action, str(action))

    @staticmethod
    def get_available_actions(game: BuckshotRouletteGame):
        """Get list of available actions.

        Args:
            game: Current game instance

        Returns:
            List of valid GameAction values
        """
        action_mask = game.get_valid_actions_mask()
        return [action for i, action in enumerate(GameAction) if action_mask[i] == 1]
