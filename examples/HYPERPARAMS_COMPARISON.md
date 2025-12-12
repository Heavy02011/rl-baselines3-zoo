# Hyperparameters Comparison: Repository vs. sac_vs_grpo.py

This document compares the hyperparameters from the [sac_vs_grpo.py](https://github.com/Heavy02011/stable-baselines3-contrib/blob/master/examples/sac_vs_grpo.py) reference script with those stored in the `hyperparams/` folder.

## Summary of Changes

### SAC (Soft Actor-Critic)

**Source:** `hyperparams/sac.yml` - MountainCarContinuous-v0

| Parameter | Repository (before) | sac_vs_grpo.py | Updated Value | Changed? |
|-----------|---------------------|----------------|---------------|----------|
| learning_rate | 3e-4 | 0.0003 | 3e-4 | ✓ Same |
| gamma | 0.9999 | 0.9999 | 0.9999 | ✓ Same |
| buffer_size | 50000 | 50000 | 50000 | ✓ Same |
| batch_size | 512 | 512 | 512 | ✓ Same |
| train_freq | 32 | 32 | 32 | ✓ Same |
| gradient_steps | 32 | 32 | 32 | ✓ Same |
| tau | 0.01 | 0.01 | 0.01 | ✓ Same |
| ent_coef | 0.1 | 0.1 | 0.1 | ✓ Same |
| learning_starts | 0 | (default) | 0 | ✓ Same |
| **use_sde** | **True** | **False** | **False** | **✗ Changed** |
| policy_kwargs.net_arch | [64, 64] | (default) | [64, 64] | ✓ Same |
| policy_kwargs.log_std_init | -3.67 | (not set) | (removed) | ✗ Changed |

**Changes Made:**
- ✗ `use_sde`: Changed from `True` to `False` to match reference script
- ✗ `policy_kwargs`: Removed `log_std_init` parameter to match reference script

### GRPO (Group Relative Policy Optimization)

**Source:** `hyperparams/grpo.yml` - MountainCarContinuous-v0

| Parameter | Repository (before) | sac_vs_grpo.py | Updated Value | Changed? |
|-----------|---------------------|----------------|---------------|----------|
| **learning_rate** | 3.0e-4 | **4e-4** | **4e-4** | **✗ Changed** |
| **n_steps** | 2048 | **512** | **512** | **✗ Changed** |
| batch_size | 512 | 512 | 512 | ✓ Same |
| **n_epochs** | (not set) | **20** | **20** | **✗ Added** |
| **gamma** | 0.9999 | **0.999** | **0.999** | **✗ Changed** |
| gae_lambda | 0.95 | 0.95 | 0.95 | ✓ Same |
| **group_size** | (commented) | **4** | **4** | **✗ Enabled** |
| **kl_coef** | (not set) | **0.02** | **0.02** | **✗ Added** |
| **clip_range** | 0.2 | **0.25** | **0.25** | **✗ Changed** |
| **ent_coef** | 0.1 | **0.0** | **0.0** | **✗ Changed** |
| vf_coef | 0.5 | 0.5 | 0.5 | ✓ Same |
| **clip_range_vf** | (not set) | **0.2** | **0.2** | **✗ Added** |
| max_grad_norm | 0.5 | 0.5 | 0.5 | ✓ Same |
| **use_sde** | (not set) | **True** | **True** | **✗ Added** |
| **sde_sample_freq** | (not set) | **4** | **4** | **✗ Added** |
| **policy_kwargs** | (not set) | **net_arch=[256,256]** | **[256,256]** | **✗ Added** |
| **n_envs** | 1 | **(implicit 8)** | **8** | **✗ Changed** |

**Changes Made:** 
The GRPO configuration was significantly updated to match the optimized parameters from the reference script:
- ✗ `learning_rate`: 3e-4 → 4e-4 (33% increase for faster learning)
- ✗ `n_steps`: 2048 → 512 (4x reduction for more frequent updates)
- ✗ `n_epochs`: Added with value 20 (more optimization epochs per update)
- ✗ `gamma`: 0.9999 → 0.999 (slightly less far-sighted)
- ✗ `group_size`: Enabled with value 4 (GRPO-specific parameter)
- ✗ `kl_coef`: Added with value 0.02 (KL divergence penalty)
- ✗ `clip_range`: 0.2 → 0.25 (allows larger policy updates)
- ✗ `ent_coef`: 0.1 → 0.0 (no entropy bonus)
- ✗ `clip_range_vf`: Added with value 0.2 (value function clipping)
- ✗ `use_sde`: Enabled (gSDE exploration)
- ✗ `sde_sample_freq`: Added with value 4 (exploration noise sampling frequency)
- ✗ `policy_kwargs`: Added net_arch=[256, 256] (larger network than default)
- ✗ `n_envs`: 1 → 8 (8x parallel environments for faster training)

### TQC (Truncated Quantile Critics)

**Source:** `hyperparams/tqc.yml` - MountainCarContinuous-v0

| Parameter | Repository | Notes |
|-----------|-----------|-------|
| learning_rate | 3e-4 | ✓ Already optimal |
| gamma | 0.9999 | ✓ Already optimal |
| buffer_size | 50000 | ✓ Already optimal |
| batch_size | 512 | ✓ Already optimal |
| train_freq | 32 | ✓ Already optimal |
| gradient_steps | 32 | ✓ Already optimal |
| tau | 0.01 | ✓ Already optimal |
| ent_coef | 0.1 | ✓ Already optimal |
| learning_starts | 0 | ✓ Already optimal |
| use_sde | True | ✓ Already optimal |
| policy_kwargs | log_std_init=-3.67, net_arch=[64,64] | ✓ Already optimal |

**Changes Made:**
- No changes needed - TQC parameters were already correctly configured

## Performance Notes

According to the [sac_vs_grpo.md](https://github.com/Heavy02011/stable-baselines3-contrib/blob/master/examples/sac_vs_grpo.md) documentation:

> GRPO now uses the faster MountainCar configuration (lr=4e-4, n_steps=512, batch_size=512, n_epochs=20, group_size=4, kl_coef=0.02, clip_range=0.25, vf_coef=0.5, gSDE on, net_arch=[256, 256]) with 8 parallel environments.

> With these hyperparameters a recent run crossed 90 reward after ~60k steps (5 eval episodes), an order-of-magnitude reduction compared to the previous ~340k budget.

Note: The "previous ~340k budget" refers to earlier, non-optimized GRPO configurations that required significantly more timesteps to solve the task. The optimized GRPO parameters achieve approximately **5.6x faster convergence** (60k vs 340k steps).

## References

- [sac_vs_grpo.py source](https://github.com/Heavy02011/stable-baselines3-contrib/blob/master/examples/sac_vs_grpo.py)
- [sac_vs_grpo.md documentation](https://github.com/Heavy02011/stable-baselines3-contrib/blob/master/examples/sac_vs_grpo.md)
- [RL Baselines3 Zoo Hyperparameters](../../hyperparams/)
