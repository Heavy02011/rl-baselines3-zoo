# Examples

This folder contains example scripts demonstrating how to use RL Baselines3 Zoo for various tasks, including algorithm comparisons and custom training setups.

## Available Examples

### SAC vs GRPO vs TQC Comparison

Compare three state-of-the-art reinforcement learning algorithms on MountainCarContinuous-v0:
- **SAC** (Soft Actor-Critic) - Off-policy RL algorithm
- **GRPO** (Group Relative Policy Optimization) - On-policy RL algorithm with group-based advantages
- **TQC** (Truncated Quantile Critics) - Off-policy RL algorithm with distributional critics

#### Files
- `sac_vs_grpo_vs_tqc.md` - Documentation with hyperparameters and usage instructions
- `train_mountain_car_all.sh` - Shell script to train all three algorithms using rl-zoo3

#### Quick Start

```bash
# Train all three algorithms sequentially with wandb tracking
bash examples/train_mountain_car_all.sh <your-wandb-entity>

# Or train individually using rl-zoo3
python train.py --algo sac --env MountainCarContinuous-v0 --track --wandb-project-name rl-baselines3-zoo
python train.py --algo grpo --env MountainCarContinuous-v0 --track --wandb-project-name rl-baselines3-zoo
python train.py --algo tqc --env MountainCarContinuous-v0 --track --wandb-project-name rl-baselines3-zoo
```

#### Features
- Training uses optimized hyperparameters from `hyperparams/` folder
- WandB integration for experiment tracking and comparison
- All rl-zoo3 features available (evaluation, checkpoints, logging, etc.)
- Easy comparison of multiple algorithms on the same environment

#### Hyperparameters

The hyperparameters for each algorithm are tuned for MountainCarContinuous-v0:

**SAC** (from `hyperparams/sac.yml`):
- Fast convergence with off-policy learning
- Uses entropy regularization for exploration
- Network: 2-layer [64, 64] MLP

**GRPO** (optimized, from `hyperparams/grpo.yml`):
- Solves task in ~60k-80k timesteps with 8 parallel environments
- Uses group-based advantage estimation
- Network: 2-layer [256, 256] MLP with gSDE exploration

**TQC** (from `hyperparams/tqc.yml`):
- Distributional RL with quantile regression
- Robust to hyperparameter choices
- Network: 2-layer [64, 64] MLP

## Contributing

To add new examples:
1. Create a well-documented Python script
2. Add a corresponding markdown file with usage instructions
3. Update this README with a link to your example
4. Ensure the example follows the existing code style

## References

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Original SAC vs GRPO comparison](https://github.com/Heavy02011/stable-baselines3-contrib/blob/master/examples/sac_vs_grpo.md)
