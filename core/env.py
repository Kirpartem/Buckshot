import gymnasium as gym
import numpy as np
from typing import Optional, Callable, Dict, Any
from core.game import BuckshotRouletteGame, Player
from core.constants import (
    GameAction, Item, Turn,
    ACTION_MAP, GAME_ACTIONS,
    OBS_SIZE, MAX_HP, MAX_ITEM_COUNT, MAX_CYLINDER
)


class BuckshotRouletteEnv(gym.Env):
    """
    Gym environment for Buckshot Roulette with action masking.
    Supports flexible opponent policies for self-play training.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_policy: Optional[Callable] = None):
        """
        Args:
            opponent_policy: Callable that takes (obs, action_mask) and returns an action index.
                           If None, opponent takes random valid actions.
        """
        super().__init__()

        self.opponent_policy = opponent_policy
        self.game = BuckshotRouletteGame()

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(GAME_ACTIONS))

        # For MaskablePPO: observation space is just the Box, mask returned separately
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

        # Track whose turn it is
        self._agent_is_player = True
        self._agent_went_first = False  # Track if agent took the first action of the episode
        self._episode_steps = 0
        self._max_episode_steps = 1000

        # Pre-allocate arrays to reduce allocations
        self._obs_array = np.zeros(OBS_SIZE, dtype=np.float32)
        self._action_mask = np.zeros(len(GAME_ACTIONS), dtype=np.int8)

        # Cache constants for faster access
        self._max_hp_inv = 1.0 / MAX_HP
        self._max_item_inv = 1.0 / MAX_ITEM_COUNT
        self._max_cylinder_inv = 1.0 / MAX_CYLINDER
        self._handcuff_inv = 1.0 / 2.0

    def _get_obs(self, for_opponent: bool = False) -> np.ndarray:
        """
        Build observation vector for the current player.
        Returns from the specified player's perspective (bot vs target).
        Optimized to reuse pre-allocated array and minimize allocations.

        Args:
            for_opponent: If True, returns from opponent's perspective. If False, returns from agent's perspective.
        """
        # Determine whose perspective to use
        if for_opponent:
            # Opponent's perspective (opposite of agent)
            if self._agent_is_player:
                bot = self.game.dealer
                target = self.game.player
            else:
                bot = self.game.player
                target = self.game.dealer
        else:
            # Agent's perspective
            if self._agent_is_player:
                bot = self.game.player
                target = self.game.dealer
            else:
                bot = self.game.dealer
                target = self.game.player

        # Count items inline (faster than function call)
        bot_items = bot.items
        target_items = target.items

        # Reuse pre-allocated array
        obs = self._obs_array

        # Bot stats
        obs[0] = bot.hp * self._max_hp_inv
        obs[1] = bot_items.count(Item.GLASS) * self._max_item_inv
        obs[2] = bot_items.count(Item.CIGARETTES) * self._max_item_inv
        obs[3] = bot_items.count(Item.HANDCUFFS) * self._max_item_inv
        obs[4] = bot_items.count(Item.SAW) * self._max_item_inv
        obs[5] = bot_items.count(Item.BEER) * self._max_item_inv

        # Target stats
        obs[6] = target.hp * self._max_hp_inv
        obs[7] = target_items.count(Item.GLASS) * self._max_item_inv
        obs[8] = target_items.count(Item.CIGARETTES) * self._max_item_inv
        obs[9] = target_items.count(Item.HANDCUFFS) * self._max_item_inv
        obs[10] = target_items.count(Item.SAW) * self._max_item_inv
        obs[11] = target_items.count(Item.BEER) * self._max_item_inv
        obs[12] = target.handcuff_strength * self._handcuff_inv

        # Bullet info - optimized checking
        bullet_seq = self.game.bullet_sequence
        blanks_left = bullet_seq.count(0)
        lives_left = len(bullet_seq) - blanks_left

        obs[13] = blanks_left * self._max_cylinder_inv
        obs[14] = lives_left * self._max_cylinder_inv

        # Knowledge about next bullet
        if bot.known_next and len(bullet_seq) > 0:
            obs[15] = 1.0 if bullet_seq[0] == 1 else 0.0  # next_live
            obs[16] = 1.0 if bullet_seq[0] == 0 else 0.0  # next_blank
            obs[17] = 0.0  # next_unknown
        else:
            obs[15] = 0.0
            obs[16] = 0.0
            obs[17] = 1.0

        obs[18] = 1.0 if self.game.saw_active else 0.0

        return obs

    def _get_action_mask(self) -> np.ndarray:
        """
        Generate action mask by calling the single, efficient method in the core game.
        This eliminates the Python loop bottleneck by delegating to the game's optimized method.
        """
        return self.game.get_valid_actions_mask()

    def action_masks(self) -> np.ndarray:
        """
        Required by MaskablePPO - returns the action mask.
        This method is called by MaskablePPO during rollout collection.
        """
        return self._get_action_mask()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple:
        """Reset the environment."""
        super().reset(seed=seed)

        # Use seed if provided, otherwise generate a random one
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)

        # Set numpy random seed for this environment instance
        np.random.seed(seed)

        # Create RNG for this episode
        rng = np.random.default_rng(seed)

        # Create game with seeded RNG
        self.game = BuckshotRouletteGame(rng_seed=seed)
        self.game.start_new_round()

        # Randomly assign agent to player or dealer using RNG
        self._agent_is_player = rng.choice([True, False])

        # Randomly decide who goes first (independent of role assignment)
        agent_goes_first = rng.choice([True, False])
        self._agent_went_first = agent_goes_first

        # Set the turn based on who should go first
        if self._agent_is_player:
            # Agent is player
            if agent_goes_first:
                self.game.turn = Turn.PLAYER
            else:
                self.game.turn = Turn.DEALER
                self._opponent_turn()
        else:
            # Agent is dealer
            if agent_goes_first:
                self.game.turn = Turn.DEALER
            else:
                self.game.turn = Turn.PLAYER
                self._opponent_turn()

        self._episode_steps = 0

        return self._get_obs(), {}

    def _opponent_turn(self):
        """Execute opponent's turn(s) until it's the agent's turn."""
        max_opponent_steps = 500
        opponent_steps = 0

        while not self._is_agent_turn() and not self._is_terminal() and opponent_steps < max_opponent_steps:
            
            # Calculate the mask ONCE at the start of the decision.
            action_mask = self._get_action_mask() 

            if self.opponent_policy is not None:
                obs = self._get_obs(for_opponent=True)
                # Use the single, fresh mask to make a decision
                action_idx = self.opponent_policy(obs, action_mask)
            else:
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) == 0:
                    break
                action_idx = np.random.choice(valid_actions)

            action = ACTION_MAP[action_idx]
            self.game.step(action)
            opponent_steps += 1  
            # The mask will be recalculated on the next iteration of the while loop,
            # but only if the opponent gets another move in the same turn.

    def _is_agent_turn(self) -> bool:
        if self._agent_is_player:
            return self.game.turn == Turn.PLAYER
        else:
            return self.game.turn == Turn.DEALER

    def _is_terminal(self) -> bool:
        """Check if game is over."""
        return self.game.player.hp <= 0 or self.game.dealer.hp <= 0

    def step(self, action_idx: int) -> tuple:
        """
        Execute one step in the environment.

        Args:
            action_idx: Integer index of the action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        self._episode_steps += 1

        action = ACTION_MAP[action_idx]
        step_result = self.game.step(action)

        # Calculate reward from agent's perspective
        agent_hp_change = step_result.new_bot_hp - step_result.prev_bot_hp
        opponent_hp_change = step_result.new_target_hp - step_result.prev_target_hp

        # Reward structure itself
        if not step_result.valid:
            reward = -10.0
        else:
            reward = 0.0
            # Damage rewards
            if opponent_hp_change < 0:
                reward += -opponent_hp_change  # Reward for damaging opponent
            if agent_hp_change < 0:
                reward += agent_hp_change  # Penalty for taking damage

        # Terminal rewards
        terminated = step_result.terminated
        if terminated:
            # Optimized terminal reward calculation
            if self._agent_is_player:
                if step_result.dealer_dead and not step_result.player_dead:
                    reward += 100.0  # Win
                elif step_result.player_dead:
                    reward -= 100.0  # Loss
            else:
                if step_result.player_dead and not step_result.dealer_dead:
                    reward += 100.0  # Win
                elif step_result.dealer_dead:
                    reward -= 100.0  # Loss
        else:
            # Let opponent take their turn(s) if game continues
            self._opponent_turn()

            # Check if opponent ended the game
            if self._is_terminal():
                terminated = True
                # Cache HP values to avoid repeated attribute access
                player_hp = self.game.player.hp
                dealer_hp = self.game.dealer.hp

                if self._agent_is_player:
                    reward += 100.0 if dealer_hp <= 0 else -100.0
                else:
                    reward += 100.0 if player_hp <= 0 else -100.0

        truncated = self._episode_steps >= self._max_episode_steps

        # Minimal info dict
        info = {
            'invalid_action': not step_result.valid,
            'episode_steps': self._episode_steps,
        }

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the current game state."""
        print(f"\n=== Round {self.game.round}, Subround {self.game.sub_round} ===")
        print(f"Player HP: {self.game.player.hp} | Dealer HP: {self.game.dealer.hp}")
        print(f"Turn: {self.game.turn.name}")
        print(f"Bullets: {len(self.game.bullet_sequence)} remaining ({self.game.bullet_sequence.count(1)} live, {self.game.bullet_sequence.count(0)} blank)")
        print(f"Player items: {[item.name for item in self.game.player.items]}")
        print(f"Dealer items: {[item.name for item in self.game.dealer.items]}")
        print(f"Saw active: {self.game.saw_active}")
        print(f"Player handcuffed: {self.game.player.handcuff_strength}")
        print(f"Dealer handcuffed: {self.game.dealer.handcuff_strength}")
