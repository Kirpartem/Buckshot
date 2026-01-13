import time
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from tqdm import tqdm
from sb3_contrib import MaskablePPO

from core.env import BuckshotRouletteEnv
from agent.config import TrainingConfig

# Global cache to prevent redundant model loading during evaluation
_policy_cache: Dict[str, any] = {}  # type: ignore


def load_policy_for_env(
    model_path: str, use_cache: bool = True, deterministic: bool = False
):
    """
    Load a policy to act as an opponent.

    Args:
        model_path: Path to the saved model .zip file.
        use_cache: If True, reuses the loaded model from memory.
        deterministic: If True, the policy will not use stochastic sampling.
    """
    if use_cache and model_path in _policy_cache:
        model = _policy_cache[model_path]
    else:
        # Load onto CPU to avoid CUDA multiprocessing issues
        model = MaskablePPO.load(model_path, device="cpu")
        if use_cache:
            _policy_cache[model_path] = model

    def policy(obs, action_mask):
        action, _ = model.predict(
            obs, action_masks=action_mask, deterministic=deterministic
        )
        return int(action)

    return policy


class OpponentPool:
    """Manages a historical pool of 'Champion' policies for training diversity."""

    def __init__(self, pool_size: int, champions_dir: str):
        self.pool_size = pool_size
        self.champions_dir = Path(champions_dir)
        self.champions_dir.mkdir(parents=True, exist_ok=True)
        self.pool: List[Path] = []
        self._load_existing_pool()

    def _load_existing_pool(self):
        champion_files = sorted(self.champions_dir.glob("champion_gen_*.zip"))
        self.pool = [f for f in champion_files[-self.pool_size :] if f.exists()]
        print(f"Loaded {len(self.pool)} champions into opponent pool.")

    def add_champion(self, model_path: Path, generation: int):
        new_champion_path = self.champions_dir / f"champion_gen_{generation}.zip"
        shutil.copy(model_path, new_champion_path)

        self.pool.append(new_champion_path)

        # Maintain pool size
        if len(self.pool) > self.pool_size:
            oldest = self.pool.pop(0)
            if oldest.exists():
                oldest.unlink()
                if str(oldest) in _policy_cache:
                    del _policy_cache[str(oldest)]
                print(f"Removed oldest champion: {oldest.name}")

        print(f"Added champion to pool: {new_champion_path.name}")

    def sample_opponent_policy(self, rng: Optional[np.random.Generator] = None):
        if not self.pool:
            return None

        champion_path = rng.choice(self.pool) if rng else np.random.choice(self.pool)  # type: ignore
        return load_policy_for_env(str(champion_path))

    def get_latest_champion_path(self) -> Optional[Path]:
        return self.pool[-1] if self.pool else None


def evaluate_model(
    model: MaskablePPO,
    opponent_policy=None,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 0,
    use_paired_games: bool = False,
) -> dict:
    """
    Evaluate a model against an opponent.

    Supports Common Random Numbers (CRN) via `use_paired_games` to reduce variance
    by playing the same seed twice (swapping roles).
    """
    wins = 0
    losses = 0
    draws = 0

    # 1. Paired Evaluation (CRN)
    if use_paired_games:
        env_as_player = BuckshotRouletteEnv(
            opponent_policy=opponent_policy, force_agent_as_player=True
        )
        env_as_dealer = BuckshotRouletteEnv(
            opponent_policy=opponent_policy, force_agent_as_player=False
        )

        n_pairs = n_episodes
        total_games = n_pairs * 2

        for i in tqdm(
            range(n_pairs), desc="Evaluating (Paired)", leave=False, mininterval=0.5
        ):
            pair_seed = seed + i

            # Run both perspectives with the same seed
            obs1, _ = env_as_player.reset(seed=pair_seed)
            obs2, _ = env_as_dealer.reset(seed=pair_seed)

            for env, obs in [(env_as_player, obs1), (env_as_dealer, obs2)]:
                _run_eval_episode(env, obs, model, deterministic)

                # Result aggregation
                if env.game.player.hp <= 0 and env.game.dealer.hp <= 0:
                    draws += 1
                elif env._agent_is_player:
                    if env.game.dealer.hp <= 0:
                        wins += 1
                    else:
                        losses += 1
                else:
                    if env.game.player.hp <= 0:
                        wins += 1
                    else:
                        losses += 1

        win_rate = wins / total_games if total_games > 0 else 0.0
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "total_episodes": total_games,
        }

    # 2. Standard Evaluation
    else:
        env = BuckshotRouletteEnv(opponent_policy=opponent_policy)
        for i in tqdm(
            range(n_episodes), desc="Evaluating", leave=False, mininterval=0.5
        ):
            obs, _ = env.reset(seed=seed + i)
            _run_eval_episode(env, obs, model, deterministic)

            # Result aggregation
            if env.game.player.hp <= 0 and env.game.dealer.hp <= 0:
                draws += 1
            elif env._agent_is_player:
                if env.game.dealer.hp <= 0:
                    wins += 1
                else:
                    losses += 1
            else:
                if env.game.player.hp <= 0:
                    wins += 1
                else:
                    losses += 1

        win_rate = wins / n_episodes if n_episodes > 0 else 0.0
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "total_episodes": n_episodes,
        }


def _run_eval_episode(env, obs, model, deterministic):
    """Helper to run a single episode until completion."""
    done = False
    step_count = 0
    max_steps = 1000

    while not done and step_count < max_steps:
        action_masks = env.action_masks()
        action, _ = model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        step_count += 1


def evaluate_challenger(
    challenger: MaskablePPO,
    champion_path: Optional[Path],
    config: TrainingConfig,
    generation: int,
) -> bool:
    """
    Orchestrates the 'Arena' logic: Challenger vs Random, then Challenger vs Champion.
    Returns True if the challenger should be promoted.
    """
    print(f"\n{'=' * 60}\nGENERATION {generation}: Evaluation Arena\n{'=' * 60}")
    eval_seed = config.seed + generation * 10000

    # Match 1: Baseline Verification
    print("\n Match 1: Challenger vs Random Opponent")
    t0 = time.time()
    random_results = evaluate_model(
        challenger,
        opponent_policy=None,
        n_episodes=config.eval_random_episodes,
        deterministic=True,
        seed=eval_seed,
        use_paired_games=config.use_paired_evaluation,
    )
    print(
        f"  Wins: {random_results['wins']}/{random_results['total_episodes']} "
        f"({random_results['win_rate']:.2%}) in {time.time() - t0:.2f}s"
    )

    if random_results["win_rate"] < 0.45:
        print("  Failed baseline check (win rate < 45%).")
        return False

    # Match 2: Championship Fight
    if champion_path is None:
        print("\n  No existing champion - Challenger promoted by default!")
        return True

    print("\n Match 2: Challenger vs Champion")
    t0 = time.time()
    champion_policy = load_policy_for_env(str(champion_path), deterministic=True)
    champion_results = evaluate_model(
        challenger,
        opponent_policy=champion_policy,
        n_episodes=config.eval_champion_episodes,
        deterministic=True,
        seed=eval_seed + 100000,
        use_paired_games=config.use_paired_evaluation,
    )
    print(
        f"  Wins: {champion_results['wins']}/{champion_results['total_episodes']} "
        f"({champion_results['win_rate']:.2%}) in {time.time() - t0:.2f}s"
    )

    if champion_results["win_rate"] >= config.win_threshold:
        print(
            f" Down with the king type shit. Challenger wins! ({champion_results['win_rate']:.2%} >= {config.win_threshold:.2%})"
        )
        return True
    else:
        print("Challenger loses. What a bummer")
        return False
