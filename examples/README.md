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
- `sac_vs_grpo_vs_tqc.py` - Standalone Python script for side-by-side comparison
- `train_mountain_car_all.sh` - Shell script to train all three algorithms using rl-zoo3

#### Quick Start

**Option 1: Standalone comparison script with wandb**
```bash
# Install dependencies
pip install stable-baselines3 sb3-contrib gymnasium wandb matplotlib

# Run comparison
python examples/sac_vs_grpo_vs_tqc.py \
  --threshold 90 \
  --max-timesteps 400000 \
  --wandb-project-name rl-baselines3-zoo \
  --wandb-entity <your-entity>
```

**Option 2: Using rl-zoo3 framework**
```bash
# Train all three algorithms sequentially
bash examples/train_mountain_car_all.sh <your-wandb-entity>

# Or train individually
python train.py --algo sac --env MountainCarContinuous-v0 --track --wandb-project-name rl-baselines3-zoo
python train.py --algo grpo --env MountainCarContinuous-v0 --track --wandb-project-name rl-baselines3-zoo
python train.py --algo tqc --env MountainCarContinuous-v0 --track --wandb-project-name rl-baselines3-zoo
```

#### Features
- Side-by-side training comparison
- Evaluation at regular intervals
- Automatic threshold detection (stops when reward > 90)
- WandB integration for experiment tracking
- Comparison plot generation
- Performance statistics (timesteps to solve, wallclock time, peak reward)

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
