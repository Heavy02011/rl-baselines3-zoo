# SAC vs. GRPO vs. TQC on MountainCarContinuous-v0

This example demonstrates training **SAC**, **GRPO**, and **TQC** (from Stable-Baselines3 and sb3-contrib) on `MountainCarContinuous-v0` using the rl-zoo3 framework with optimized hyperparameters. Results are tracked in wandb.ai for easy comparison.

> This comparison example is based on the maintained SAC vs GRPO comparison from stable-baselines3-contrib, extended to include TQC and using the rl-zoo3 training framework.

## How to run

**Train all three algorithms sequentially:**
```bash
bash examples/train_mountain_car_all.sh <your-wandb-entity>
```

**Or train individually:**
```bash
# Train SAC
python train.py --algo sac --env MountainCarContinuous-v0 \
  --seed 0 --track \
  --wandb-project-name rl-baselines3-zoo \
  --wandb-entity <your-wandb-entity>

# Train GRPO  
python train.py --algo grpo --env MountainCarContinuous-v0 \
  --seed 0 --track \
  --wandb-project-name rl-baselines3-zoo \
  --wandb-entity <your-wandb-entity>

# Train TQC
python train.py --algo tqc --env MountainCarContinuous-v0 \
  --seed 0 --track \
  --wandb-project-name rl-baselines3-zoo \
  --wandb-entity <your-wandb-entity>
```

The hyperparameters are loaded from the `hyperparams/` folder and results are logged to wandb.ai for tracking and comparison.

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

The rl-zoo3 framework automatically logs to wandb when using the `--track` flag:
- Training progress and rewards
- Evaluation metrics
- Hyperparameters
- System information

To use wandb tracking:
1. Install wandb: `pip install wandb`
2. Log in: `wandb login`
3. Use the `--track` flag with `train.py` and provide project name and entity as shown above
