import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from core.env import BuckshotRouletteEnv
from functools import lru_cache


def set_global_seed(seed: int, configure_cudnn: bool = False):
    """
    Set seed for all random number generators for reproducibility.

    Args:
        seed: Random seed value
        configure_cudnn: Whether to configure CUDNN settings. Should be False when
                        using multiprocessing to avoid pickle errors with CUDNN modules.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Only configure CUDNN when not using multiprocessing
    # CUDNN modules cannot be pickled, causing errors with SubprocVecEnv
    if configure_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    # Training - optimized for speed
    total_timesteps_per_generation: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 4096  # Increased from 2048 - collect more steps before update (less overhead)
    batch_size: int = 256  # Increased from 64 - larger batches are more efficient
    n_epochs: int = 4  # Reduced from 10 - fewer epochs per update (faster)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Environment
    n_envs: int = 16  # Parallel environments

    # Evaluation
    eval_random_episodes: int = 1000
    eval_champion_episodes: int = 10000
    win_threshold: float = 0.5+1e-5  # Challenger must win 50%+ to dethrone

    # Opponent pool
    pool_size: int = 10  # Keep last N champions

    # Paths
    models_dir: str = "agent/models"
    champions_dir: str = "agent/models/champions"

    # Seed for reproducibility
    seed: int = 42


class GenerationCallback(BaseCallback):
    """Callback to log generation-specific metrics."""

    def __init__(self, generation: int, verbose=0):
        super().__init__(verbose)
        self.generation = generation

    def _on_step(self) -> bool:
        return True


def make_env(opponent_model_path: Optional[str] = None, rank: int = 0, seed: int = 0):
    """
    Utility function for multiprocessed env.

    This function creates an environment factory that will be pickled and sent
    to subprocesses. To avoid pickle errors with CUDA modules, we pass the model
    path as a string rather than the loaded policy object.

    Args:
        opponent_model_path: Path to opponent model file (None for random opponent)
        rank: Index of the subprocess
        seed: Random seed

    Returns:
        Callable that initializes and returns a BuckshotRouletteEnv
    """
    def _init():
        # Set seed for this subprocess
        # Note: configure_cudnn=False to avoid pickle errors in multiprocessing
        env_seed = seed + rank
        set_global_seed(env_seed, configure_cudnn=False)

        # Load opponent policy inside subprocess to avoid pickle issues
        opponent_policy = None
        if opponent_model_path is not None:
            opponent_policy = load_policy_for_env(
                opponent_model_path,
                use_cache=False,  # Each subprocess has its own cache
                deterministic=False
            )

        env = BuckshotRouletteEnv(opponent_policy=opponent_policy)
        env.reset(seed=env_seed)
        return env
    return _init


def create_vec_env(n_envs: int, opponent_model_path: Optional[str] = None, seed: int = 0):
    """
    Create a vectorized environment with n parallel environments.

    Uses 'fork' start method to avoid pickle errors with PyTorch CUDNN modules.
    The 'spawn' method requires pickling the entire module state, which fails with
    CUDNN objects. The 'fork' method copies the process memory directly, avoiding
    the pickle requirement.

    Args:
        n_envs: Number of parallel environments
        opponent_model_path: Path to opponent model file (None for random opponent)
        seed: Base random seed

    Returns:
        SubprocVecEnv with n parallel environments
    """
    env_fns = [make_env(opponent_model_path, i, seed) for i in range(n_envs)]
    # Use 'fork' instead of 'spawn' to avoid pickle errors with PyTorch/CUDNN
    # Fork is safe here since we're not using CUDA in subprocesses (device="cpu")
    return SubprocVecEnv(env_fns, start_method='fork')


# Global cache for loaded policies to avoid redundant loading
_policy_cache: Dict[str, any] = {}

def load_policy_for_env(model_path: str, use_cache: bool = True, deterministic: bool = False):
    """
    Load a policy that can be used as an opponent in the environment.
    Uses caching to avoid redundant model loads.

    Args:
        model_path: Path to the saved model
        use_cache: Whether to use cached models
        deterministic: Whether the policy should act deterministically

    Returns:
        Callable that takes (obs, action_mask) and returns action index
    """
    if use_cache and model_path in _policy_cache:
        model = _policy_cache[model_path]
    else:
        model = MaskablePPO.load(model_path, device="cpu")
        if use_cache:
            _policy_cache[model_path] = model

    def policy(obs, action_mask):
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=deterministic)
        return int(action)

    return policy


def evaluate_model(model: MaskablePPO, opponent_policy=None, n_episodes: int = 100, deterministic: bool = True, seed: int = 0) -> dict:
    """
    Evaluate a model against an opponent.

    Args:
        model: The model to evaluate
        opponent_policy: Opponent policy (None for random)
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        seed: Random seed for evaluation

    Returns:
        Dict with wins, losses, draws, win_rate
    """
    env = BuckshotRouletteEnv(opponent_policy=opponent_policy)

    wins = 0
    losses = 0
    draws = 0

    # Pre-allocate for faster checking
    for i in tqdm(range(n_episodes), desc="Evaluating", leave=False, mininterval=0.5):
        obs, _ = env.reset(seed=seed + i)
        done = False
        step_count = 0
        max_steps = 1000  # Safety limit

        while not done and step_count < max_steps:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        # Optimized result checking
        player_hp = env.game.player.hp
        dealer_hp = env.game.dealer.hp

        if player_hp <= 0 and dealer_hp <= 0:
            draws += 1
        elif env._agent_is_player:
            if dealer_hp <= 0:
                wins += 1
            else:
                losses += 1
        else:  # Agent is dealer
            if player_hp <= 0:
                wins += 1
            else:
                losses += 1

    win_rate = wins / n_episodes if n_episodes > 0 else 0.0

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'total_episodes': n_episodes
    }


class OpponentPool:
    """Manages a pool of opponent policies for training diversity."""

    def __init__(self, pool_size: int, champions_dir: str):
        self.pool_size = pool_size
        self.champions_dir = Path(champions_dir)
        self.champions_dir.mkdir(parents=True, exist_ok=True)
        self.pool: List[Path] = []

        # Load existing champions
        self._load_existing_pool()

    def _load_existing_pool(self):
        """Load existing champion models from disk."""
        champion_files = sorted(self.champions_dir.glob("champion_gen_*.zip"))
        # Only keep files that actually exist
        self.pool = [f for f in champion_files[-self.pool_size:] if f.exists()]
        print(f"Loaded {len(self.pool)} champions into opponent pool")

    def add_champion(self, model_path: Path, generation: int):
        """
        Add a new champion to the pool.

        Args:
            model_path: Path to the champion model
            generation: Generation number
        """
        new_champion_path = self.champions_dir / f"champion_gen_{generation}.zip"
        shutil.copy(model_path, new_champion_path)

        self.pool.append(new_champion_path)

        # Remove oldest if pool is full
        if len(self.pool) > self.pool_size:
            oldest = self.pool.pop(0)
            if oldest.exists():
                oldest.unlink()
                # Also remove from policy cache if it's there
                if str(oldest) in _policy_cache:
                    del _policy_cache[str(oldest)]
                print(f"Removed oldest champion: {oldest.name}")

        print(f"Added champion to pool: {new_champion_path.name}")

    def sample_opponent_policy(self, rng: np.random.Generator = None):
        """
        Sample a random opponent policy from the pool.

        Args:
            rng: NumPy random generator (if None, uses global np.random)

        Returns:
            Opponent policy callable or None for random
        """
        if not self.pool:
            return None

        if rng is not None:
            champion_path = rng.choice(self.pool)
        else:
            champion_path = np.random.choice(self.pool)
        return load_policy_for_env(str(champion_path))

    def get_latest_champion_path(self) -> Optional[Path]:
        """Get the path to the most recent champion."""
        return self.pool[-1] if self.pool else None


def train_generation(
    challenger: MaskablePPO,
    opponent_pool: OpponentPool,
    config: TrainingConfig,
    generation: int,
    rng: np.random.Generator
):
    """
    Train a challenger for one generation.

    Args:
        challenger: The model to train
        opponent_pool: Pool of opponent policies
        config: Training configuration
        generation: Current generation number
        rng: NumPy random generator for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"GENERATION {generation}: Training Challenger")
    print(f"{'='*60}")

    # Sample one opponent model path to use across all training environments
    # Pass the path (not the loaded model) to avoid pickle errors with CUDA modules
    opponent_model_path = None
    champion_paths = opponent_pool.pool
    if champion_paths:
        # Only use champions that still exist on disk
        existing_champions = [p for p in champion_paths if p.exists()]
        if existing_champions:
            opponent_model_path = str(rng.choice(existing_champions))
            print(f"Training against: {Path(opponent_model_path).name}")
        else:
            print("No valid champions found - training against random opponent")
    else:
        print("Empty opponent pool - training against random opponent")

    # Use deterministic seed based on generation and config seed
    train_seed = config.seed + generation * 1000
    train_env = create_vec_env(config.n_envs, opponent_model_path, train_seed)
    
    # Train the challenger
    callback = GenerationCallback(generation=generation)
    challenger.set_env(train_env)

    # Track timesteps at start of this generation
    initial_timesteps = challenger.num_timesteps

    with tqdm(total=config.total_timesteps_per_generation, desc=f"Gen {generation}",
              smoothing=0.1, unit=" steps", mininterval=0.5) as pbar:
        class ProgressCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.last_reported = 0

            def _on_step(self):
                # Calculate progress within this generation only
                current_progress = self.model.num_timesteps - initial_timesteps
                delta = current_progress - self.last_reported

                if delta > 0:
                    pbar.update(delta)
                    self.last_reported = current_progress

                return True

        progress_callback = ProgressCallback()
        challenger.learn(
            total_timesteps=config.total_timesteps_per_generation,
            callback=[callback, progress_callback],
            reset_num_timesteps=False  # Continue from previous timesteps
        )

        # Ensure progress bar reaches 100%
        final_progress = challenger.num_timesteps - initial_timesteps
        remaining = final_progress - progress_callback.last_reported
        if remaining > 0:
            pbar.update(remaining)

    # Close environment cleanly
    try:
        train_env.close()
    except Exception:
        pass  # Ignore cleanup errors

    print(f"Generation {generation} training complete!")


def evaluate_challenger(
    challenger: MaskablePPO,
    champion_path: Optional[Path],
    config: TrainingConfig,
    generation: int
) -> bool:
    """
    Evaluate challenger against baseline and champion.

    Args:
        challenger: The challenger model
        champion_path: Path to current champion (None if first generation)
        config: Training configuration
        generation: Current generation number

    Returns:
        True if challenger should be promoted, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"GENERATION {generation}: Evaluation Arena")
    print(f"{'='*60}")

    # Use deterministic seed for evaluation
    eval_seed = config.seed + generation * 10000

    # Match 1: Baseline against random
    print(f"\nMatch 1: Challenger vs Random Opponent ({config.eval_random_episodes} games)")
    t0 = time.time()
    random_results = evaluate_model(
        challenger,
        opponent_policy=None,
        n_episodes=config.eval_random_episodes,
        deterministic=True,
        seed=eval_seed
    )
    eval_time = time.time() - t0
    print(f"  Wins: {random_results['wins']}/{random_results['total_episodes']} "
          f"(Win rate: {random_results['win_rate']:.2%}, took {eval_time:.2f}s)")

    # Baseline check: should beat random opponent reasonably well
    if random_results['win_rate'] < 0.45:
        print(f"  ❌ Failed baseline check (win rate < 45%)")
        return False

    print(f"  ✓ Passed baseline check")

    # Match 2: Title fight against champion
    if champion_path is None:
        print(f"\n  No existing champion - Challenger promoted by default!")
        return True

    print(f"\nMatch 2: Challenger vs Champion ({config.eval_champion_episodes} games)")
    t0 = time.time()
    champion_policy = load_policy_for_env(str(champion_path), deterministic=True)
    champion_results = evaluate_model(
        challenger,
        opponent_policy=champion_policy,
        n_episodes=config.eval_champion_episodes,
        deterministic=True,
        seed=eval_seed + 100000
    )
    eval_time = time.time() - t0
    print(f"  Wins: {champion_results['wins']}/{champion_results['total_episodes']} "
          f"(Win rate: {champion_results['win_rate']:.2%}, took {eval_time:.2f}s)")

    # Check if challenger wins
    if champion_results['win_rate'] >= config.win_threshold:
        print(f"  🏆 Challenger wins! ({champion_results['win_rate']:.2%} >= {config.win_threshold:.2%})")
        return True
    else:
        print(f"  ❌ Challenger loses ({champion_results['win_rate']:.2%} < {config.win_threshold:.2%})")
        return False


def main():
    """Main training loop."""
    config = TrainingConfig()

    # Set global seed for reproducibility
    # Note: configure_cudnn=False to avoid pickle errors when spawning subprocesses
    # CUDNN state cannot be pickled and causes "cannot pickle 'CudnnModule' object" errors
    print(f"Setting global seed: {config.seed}")
    set_global_seed(config.seed, configure_cudnn=False)

    # Create RNG for training decisions
    rng = np.random.default_rng(config.seed)

    # Create directories
    Path(config.models_dir).mkdir(exist_ok=True)
    Path(config.champions_dir).mkdir(exist_ok=True)

    # Initialize opponent pool
    opponent_pool = OpponentPool(config.pool_size, config.champions_dir)

    # Track current champion
    champion_path = Path(config.models_dir) / "champion.zip"
    current_champion_path = opponent_pool.get_latest_champion_path()

    # If we have a champion on disk but not in memory, load it
    if current_champion_path is None and champion_path.exists():
        current_champion_path = champion_path

    generation = len(opponent_pool.pool)
    print(f"\n{'='*60}")
    print(f"Starting training at generation {generation}")
    print(f"{'='*60}")

    while True:
        generation += 1

        # Step A: Create challenger
        print(f"\n{'='*60}")
        print(f"GENERATION {generation}: Creating Challenger")
        print(f"{'='*60}")

        # Create initial training environment (just for initialization)
        init_seed = config.seed + generation * 100000
        init_env = create_vec_env(config.n_envs, opponent_model_path=None, seed=init_seed)

        if current_champion_path is not None and current_champion_path.exists():
            print(f"Loading champion weights from: {current_champion_path.name}")
            challenger = MaskablePPO.load(
                str(current_champion_path),
                env=init_env,
                custom_objects={'learning_rate': config.learning_rate},
                device="cpu",
                seed=config.seed,
                tensorboard_log=None
            )
            # Reset optimizer to avoid momentum from previous training
            challenger.policy.optimizer = challenger.policy.optimizer.__class__(
                challenger.policy.parameters(),
                lr=config.learning_rate
            )
        else:
            print("No existing champion - creating fresh model")
            challenger = MaskablePPO(
                "MlpPolicy",
                init_env,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm,
                verbose=0,
                seed=config.seed,
                tensorboard_log=None
            )

        init_env.close()

        # Step B: Train challenger
        train_generation(challenger, opponent_pool, config, generation, rng)

        # Step C: Evaluate in arena
        promote = evaluate_challenger(
            challenger,
            current_champion_path,
            config,
            generation
        )

        # Step D: Promote or discard
        if promote:
            print(f"\n{'='*60}")
            print(f"🎉 PROMOTION: Challenger becomes Champion!")
            print(f"{'='*60}")

            # Save as champion
            challenger.save(champion_path)
            print(f"Saved new champion: {champion_path}")

            # Add to pool
            opponent_pool.add_champion(champion_path, generation)
            current_champion_path = champion_path

        else:
            print(f"\n{'='*60}")
            print(f"Champion retains the title")
            print(f"{'='*60}")

        print(f"\nGeneration {generation} complete. Starting next generation...\n")


if __name__ == "__main__":
    main()
