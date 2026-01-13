from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

from core.env import BuckshotRouletteEnv
from agent.config import TrainingConfig, set_global_seed
from agent.callbacks import GenerationCallback, ProgressCallback
import agent.arena as arena


def make_env(opponent_model_path: Optional[str] = None, rank: int = 0, seed: int = 0):
    """
    Factory function for multiprocessing.
    Notes:
        - Opponent path is passed as string to avoid pickling the policy object.
        - CUDNN configuration is disabled inside subprocesses to avoid errors.
    """

    def _init():
        env_seed = seed + rank
        set_global_seed(env_seed, configure_cudnn=False)

        opponent_policy = None
        if opponent_model_path is not None:
            # Each subprocess loads its own copy of the opponent
            opponent_policy = arena.load_policy_for_env(
                opponent_model_path,
                use_cache=False,
                deterministic=False,
            )

        env = BuckshotRouletteEnv(opponent_policy=opponent_policy)
        env.reset(seed=env_seed)
        return env

    return _init


def create_vec_env(
    n_envs: int, opponent_model_path: Optional[str] = None, seed: int = 0
):
    """
    Creates the vectorized environment using 'fork' start method.
    'fork' is used to avoid pickling errors with PyTorch CUDNN modules.
    """
    env_fns = [make_env(opponent_model_path, i, seed) for i in range(n_envs)]
    return SubprocVecEnv(env_fns, start_method="fork")  # type: ignore


def train_generation(
    challenger: MaskablePPO,
    opponent_pool: arena.OpponentPool,
    config: TrainingConfig,
    generation: int,
    rng: np.random.Generator,
):
    """Executes the training phase for a single generation."""
    print(f"\n{'=' * 60}\nGENERATION {generation}: Training Challenger\n{'=' * 60}")

    # Select opponent for this generation
    opponent_model_path = None
    if opponent_pool.pool:
        # Sample path from pool (str) to pass to subprocesses
        valid_paths = [p for p in opponent_pool.pool if p.exists()]
        if valid_paths:
            opponent_model_path = str(rng.choice(valid_paths))  # type: ignore
            print(f"Training against: {Path(opponent_model_path).name}")
        else:
            print("No valid champions found - training against random opponent.")
    else:
        print("Empty opponent pool - training against random opponent.")

    train_seed = config.seed + generation * 1000
    train_env = create_vec_env(config.n_envs, opponent_model_path, train_seed)

    # Setup Training
    gen_callback = GenerationCallback(generation=generation)
    challenger.set_env(train_env)
    initial_timesteps = challenger.num_timesteps

    with tqdm(
        total=config.total_timesteps_per_generation,
        desc=f"Gen {generation}",
        unit=" steps",
    ) as pbar:
        progress_callback = ProgressCallback(pbar, initial_timesteps)

        challenger.learn(
            total_timesteps=config.total_timesteps_per_generation,
            callback=[gen_callback, progress_callback],
            reset_num_timesteps=False,
        )

        # Ensure progress bar completes
        remaining = (
            challenger.num_timesteps - initial_timesteps
        ) - progress_callback.last_reported
        if remaining > 0:
            pbar.update(remaining)

    train_env.close()
    print(f"Generation {generation} training complete.")


def main():
    config = TrainingConfig()

    # Configure Global Seed
    print(f"Setting global seed: {config.seed}")
    set_global_seed(config.seed, configure_cudnn=False)
    rng = np.random.default_rng(config.seed)

    # File System Setup
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.champions_dir).mkdir(parents=True, exist_ok=True)

    # Initialize State
    opponent_pool = arena.OpponentPool(config.pool_size, config.champions_dir)
    champion_path = Path(config.models_dir) / "champion.zip"
    current_champion_path = opponent_pool.get_latest_champion_path()

    if current_champion_path is None and champion_path.exists():
        current_champion_path = champion_path

    generation = len(opponent_pool.pool)
    print(f"\n{'=' * 60}\nStarting training at generation {generation}\n{'=' * 60}")

    while True:
        generation += 1

        # 1. Initialize Challenger
        init_seed = config.seed + generation * 100000
        # Create a temp env just for model initialization
        init_env = create_vec_env(
            config.n_envs, opponent_model_path=None, seed=init_seed
        )

        if current_champion_path is not None and current_champion_path.exists():
            print(f"Loading weights from: {current_champion_path.name}")
            challenger = MaskablePPO.load(
                str(current_champion_path),
                env=init_env,
                custom_objects={"learning_rate": config.learning_rate},
                device="cpu",  # Training will move to GPU automatically if available
                seed=config.seed,
            )
            # Reset optimizer
            challenger.policy.optimizer = challenger.policy.optimizer.__class__(
                challenger.policy.parameters(),
                lr=config.learning_rate,  # type: ignore
            )
        else:
            print("No existing champion - creating fresh model.")
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
            )
        init_env.close()

        # 2. Train
        train_generation(challenger, opponent_pool, config, generation, rng)

        # 3. Evaluate
        promote = arena.evaluate_challenger(
            challenger, current_champion_path, config, generation
        )

        # 4. Promote
        if promote:
            print(f"\n{'=' * 60}\nPROMOTION: Challenger becomes Champion!\n{'=' * 60}")
            challenger.save(champion_path)
            opponent_pool.add_champion(champion_path, generation)
            current_champion_path = champion_path
        else:
            print(f"\n{'=' * 60}\nChampion retains the title.\n{'=' * 60}")


if __name__ == "__main__":
    main()
