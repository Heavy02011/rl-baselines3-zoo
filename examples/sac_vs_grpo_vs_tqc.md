# SAC vs. GRPO vs. TQC on MountainCarContinuous-v0

This walkthrough trains **SAC**, **GRPO**, and **TQC** (from Stable-Baselines3 and sb3-contrib) side by side on `MountainCarContinuous-v0` until they exceed a reward of **90**. The helper script logs which algorithm solves the task faster and produces a compact plot with results tracked in wandb.ai.

> This comparison example is based on the maintained SAC vs GRPO comparison from stable-baselines3-contrib, extended to include TQC and integrated with wandb.ai for experiment tracking.

## How to run

```bash
python examples/sac_vs_grpo_vs_tqc.py \
  --threshold 90 \
  --max-timesteps 400000 \
  --eval-every 20000 \
  --n-envs 8 \
  --eval-episodes 5 \
  --seed 0 \
  --wandb-project-name rl-baselines3-zoo \
  --wandb-entity <your-wandb-entity>
```

- The script will print intermediate evaluation rewards for all three agents.
- Training stops early as soon as the current agent crosses the reward threshold or the budget is spent.
- A plot named `sac_vs_grpo_vs_tqc.png` is written next to the script (requires `matplotlib`).
- Results are logged to wandb.ai for tracking and comparison.

## Hyperparameters

### SAC Configuration
- learning_rate: 3e-4
- gamma: 0.9999
- buffer_size: 50000
- batch_size: 512
- train_freq: 32
- gradient_steps: 32
- tau: 0.01
- ent_coef: 0.1
- use_sde: False
- policy_kwargs: dict(net_arch=[64, 64])

### GRPO Configuration (Optimized)
- learning_rate: 4e-4
- n_steps: 512
- batch_size: 512
- n_epochs: 20
- gamma: 0.999
- gae_lambda: 0.95
- group_size: 4
- kl_coef: 0.02
- clip_range: 0.25
- ent_coef: 0.0
- vf_coef: 0.5
- clip_range_vf: 0.2
- max_grad_norm: 0.5
- use_sde: True
- sde_sample_freq: 4
- policy_kwargs: dict(net_arch=[256, 256])

### TQC Configuration
- learning_rate: 3e-4
- gamma: 0.9999
- buffer_size: 50000
- batch_size: 512
- train_freq: 32
- gradient_steps: 32
- tau: 0.01
- ent_coef: 0.1
- use_sde: True
- policy_kwargs: dict(log_std_init=-3.67, net_arch=[64, 64])

## Notes on current results

- GRPO uses the faster MountainCar configuration with 8 parallel environments and optimized hyperparameters (lr=4e-4, n_steps=512, batch_size=512, n_epochs=20, group_size=4, kl_coef=0.02, clip_range=0.25, vf_coef=0.5, gSDE on, net_arch=[256, 256]).
- With these hyperparameters, GRPO typically crosses 90 reward after ~60k-80k steps (5 eval episodes).
- SAC and TQC maintain configurations from the hyperparams folder and also solve the task efficiently.
- All results are tracked in wandb.ai for easy comparison and analysis.

## Wandb Integration

The script automatically logs:
- Training progress for each algorithm
- Evaluation rewards over time
- Final statistics (best reward, timesteps to solve, wallclock time)
- Comparison plot

To use wandb tracking, ensure you have:
1. Installed wandb: `pip install wandb`
2. Logged in: `wandb login`
3. Provided the project name and entity (optional) as command-line arguments
