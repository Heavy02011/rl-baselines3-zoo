"""
Side-by-side training comparison between SAC, GRPO, and TQC 
(from Stable-Baselines3 and sb3-contrib) on MountainCarContinuous-v0.

The script trains each agent until it reaches the specified reward
threshold (default: 90) or hits the maximum timesteps budget.
It records evaluation rewards along the way and produces a small
comparison plot so you can see which learner solves the task faster.
Results are logged to wandb.ai for experiment tracking.

Usage:
    python examples/sac_vs_grpo_vs_tqc.py --max-timesteps 400000 --threshold 90 \
        --wandb-project-name rl-baselines3-zoo --wandb-entity <your-entity>

See examples/sac_vs_grpo_vs_tqc.md for the tuned hyperparameters and context.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import GRPO, TQC
except ImportError:
    raise ImportError(
        "sb3-contrib is required for GRPO and TQC. "
        "Install it with: pip install sb3-contrib"
    )

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


ENV_ID = "MountainCarContinuous-v0"


@dataclass
class RunStats:
    algo: str
    rewards: list[float]
    timesteps: list[int]
    wallclock_s: float
    solved: bool


def format_stats(stats: RunStats) -> str:
    status = "✅" if stats.solved else "⚠️"
    best = f"{max(stats.rewards):.1f}" if stats.rewards else "N/A"
    steps = stats.timesteps[-1] if stats.timesteps else 0
    return f"{status} {stats.algo.upper()}: best={best}, steps={steps}, wallclock={stats.wallclock_s:.1f}s"


def make_env(seed: int) -> gym.Env:
    env = Monitor(gym.make(ENV_ID))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def create_train_env(algo: str, seed: int, n_envs: int = 1):
    """Create training environment based on algorithm type.
    
    Off-policy algorithms (SAC, TQC) use single env with replay buffer.
    On-policy algorithms (GRPO) use vectorized env for parallel collection.
    """
    if algo in ("sac", "tqc"):
        return make_env(seed)
    else:  # On-policy algorithms like GRPO
        return make_vec_env(ENV_ID, n_envs=n_envs, seed=seed)


def train_until_solved(
    algo: str,
    max_timesteps: int,
    threshold: float,
    eval_episodes: int,
    seed: int,
    eval_every: int = 10_000,
    n_envs: int = 1,
    wandb_run=None,
) -> tuple[RunStats, SAC | GRPO | TQC]:
    train_env = create_train_env(algo, seed, n_envs)
    eval_env = make_env(seed)
    
    if algo == "sac":
        # SAC configuration from sac_vs_grpo.py
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=0.0003,
            gamma=0.9999,
            buffer_size=50_000,
            batch_size=512,
            train_freq=32,
            gradient_steps=32,
            tau=0.01,
            ent_coef=0.1,
            use_sde=False,
            verbose=1,
            seed=seed,
        )
    elif algo == "grpo":
        # Optimized GRPO hyperparameters from sac_vs_grpo.py
        # Solves MountainCarContinuous-v0 (mean reward > 90)
        # with 8 parallel environments in ~60k-80k timesteps (5 deterministic evals).
        model = GRPO(
            "MlpPolicy",
            train_env,
            learning_rate=4e-4,
            n_steps=512,
            batch_size=512,
            n_epochs=20,
            gamma=0.999,
            gae_lambda=0.95,
            group_size=4,
            kl_coef=0.02,
            clip_range=0.25,
            ent_coef=0.0,
            vf_coef=0.5,
            clip_range_vf=0.2,
            max_grad_norm=0.5,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=seed,
            verbose=1,
        )
    elif algo == "tqc":
        # TQC configuration from hyperparams/tqc.yml
        model = TQC(
            "MlpPolicy",
            train_env,
            learning_rate=0.0003,
            gamma=0.9999,
            buffer_size=50_000,
            batch_size=512,
            train_freq=32,
            gradient_steps=32,
            tau=0.01,
            ent_coef=0.1,
            learning_starts=0,
            use_sde=True,
            policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]),
            seed=seed,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    rewards, steps = [], []
    solved = False
    start = time.time()

    while model.num_timesteps < max_timesteps:
        chunk = min(eval_every, max_timesteps - model.num_timesteps)
        if chunk <= 0:
            break
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
        rewards.append(mean_reward)
        steps.append(model.num_timesteps)
        print(f"[{algo.upper()}] {model.num_timesteps} steps -> mean_reward={mean_reward:.2f}")
        
        # Log to wandb if available
        if wandb_run is not None:
            wandb_run.log({
                f"{algo}/timesteps": model.num_timesteps,
                f"{algo}/mean_reward": mean_reward,
            })
        
        if mean_reward >= threshold:
            solved = True
            break

    wallclock_s = time.time() - start
    
    # Log final stats to wandb
    if wandb_run is not None:
        wandb_run.log({
            f"{algo}/solved": solved,
            f"{algo}/final_timesteps": model.num_timesteps,
            f"{algo}/wallclock_s": wallclock_s,
            f"{algo}/best_reward": max(rewards) if rewards else 0,
        })
    
    train_env.close()
    eval_env.close()
    return RunStats(algo=algo, rewards=rewards, timesteps=steps, wallclock_s=wallclock_s, solved=solved), model


def plot_progress(
    results: list[RunStats], output_path: Path, threshold: float, eval_episodes: int | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot creation.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for stats in results:
        ax.plot(stats.timesteps, stats.rewards, marker="o", label=f"{stats.algo.upper()}")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"target reward ({threshold})")
    ax.set_xlabel("Timesteps")
    episode_label = f"{eval_episodes} eval episodes" if eval_episodes is not None else "evaluation rollouts"
    ax.set_ylabel(f"Mean reward ({episode_label})")
    ax.set_title("MountainCarContinuous-v0: SAC vs GRPO vs TQC")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=90.0, help="Reward threshold for success.")
    parser.add_argument("--max-timesteps", type=int, default=400_000, help="Per-agent training budget.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes used for evaluation rollouts.")
    parser.add_argument("--eval-every", type=int, default=20_000, help="Train this many timesteps between evals.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments for on-policy (GRPO).")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path(__file__).with_name("sac_vs_grpo_vs_tqc.png"),
        help="Where to store the comparison plot.",
    )
    parser.add_argument("--wandb-project-name", type=str, default="rl-baselines3-zoo", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity (team) name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Initialize wandb if available and not disabled
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"SAC-GRPO-TQC-{ENV_ID}-seed{args.seed}",
            config={
                "env_id": ENV_ID,
                "threshold": args.threshold,
                "max_timesteps": args.max_timesteps,
                "eval_episodes": args.eval_episodes,
                "eval_every": args.eval_every,
                "seed": args.seed,
                "n_envs": args.n_envs,
            },
            sync_tensorboard=False,
        )
        print(f"Wandb run initialized: {wandb_run.url}")

    print(f"Training SAC, GRPO, and TQC on {ENV_ID} until reward >= {args.threshold}")
    results: list[RunStats] = []
    for algo_name in ("sac", "grpo", "tqc"):
        print(f"\n{'='*60}")
        print(f"Starting training for {algo_name.upper()}")
        print(f"{'='*60}\n")
        stats, _ = train_until_solved(
            algo_name,
            max_timesteps=args.max_timesteps,
            threshold=args.threshold,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            eval_every=args.eval_every,
            n_envs=args.n_envs,
            wandb_run=wandb_run,
        )
        results.append(stats)

    sac_stats, grpo_stats, tqc_stats = results
    plot_progress(results, args.plot_path, args.threshold, args.eval_episodes)

    # Log plot to wandb if available
    if wandb_run is not None:
        try:
            wandb_run.log({"comparison_plot": wandb.Image(str(args.plot_path))})
        except Exception as e:
            print(f"Warning: Could not log plot to wandb: {e}")

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("  " + format_stats(sac_stats))
    print("  " + format_stats(grpo_stats))
    print("  " + format_stats(tqc_stats))
    
    # Determine winner(s)
    solved_algos = [s for s in results if s.solved]
    if len(solved_algos) == 0:
        print("=> None of the algorithms solved the task within the timestep budget.")
    elif len(solved_algos) < 3:
        winners = ", ".join([s.algo.upper() for s in solved_algos])
        print(f"=> {winners} solved the task while others did not.")
    else:
        # All solved, compare timesteps to solve
        def get_final_timesteps(s):
            return s.timesteps[-1] if s.timesteps and len(s.timesteps) > 0 else float('inf')
        fastest = min(solved_algos, key=get_final_timesteps)
        if fastest.timesteps and len(fastest.timesteps) > 0:
            print(f"=> All algorithms solved the task. {fastest.algo.upper()} was fastest with {fastest.timesteps[-1]} steps.")
    
    # Compare peak rewards
    if all(s.rewards for s in results):
        best_algo = max(results, key=lambda s: max(s.rewards))
        print(f"=> {best_algo.algo.upper()} reached the highest peak reward: {max(best_algo.rewards):.2f}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
