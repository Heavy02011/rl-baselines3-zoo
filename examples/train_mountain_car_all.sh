#!/bin/bash
# Script to train SAC, GRPO, and TQC on MountainCarContinuous-v0 using rl-zoo3
# Results are logged to wandb.ai for easy comparison
#
# Usage:
#   bash examples/train_mountain_car_all.sh [wandb-entity]
#
# Example:
#   bash examples/train_mountain_car_all.sh myteam

set -e

WANDB_ENTITY="${1:-}"
WANDB_PROJECT="rl-baselines3-zoo"
ENV_ID="MountainCarContinuous-v0"
SEED=0

echo "=================================================="
echo "Training SAC, GRPO, and TQC on $ENV_ID"
echo "Wandb project: $WANDB_PROJECT"
if [ -n "$WANDB_ENTITY" ]; then
    echo "Wandb entity: $WANDB_ENTITY"
fi
echo "=================================================="

# Train SAC
echo ""
echo "Training SAC..."
if [ -n "$WANDB_ENTITY" ]; then
    python train.py --algo sac --env "$ENV_ID" \
        --seed $SEED \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --wandb-tags sac mountain_car comparison
else
    python train.py --algo sac --env "$ENV_ID" \
        --seed $SEED \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-tags sac mountain_car comparison
fi

# Train GRPO
echo ""
echo "Training GRPO..."
if [ -n "$WANDB_ENTITY" ]; then
    python train.py --algo grpo --env "$ENV_ID" \
        --seed $SEED \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --wandb-tags grpo mountain_car comparison
else
    python train.py --algo grpo --env "$ENV_ID" \
        --seed $SEED \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-tags grpo mountain_car comparison
fi

# Train TQC
echo ""
echo "Training TQC..."
if [ -n "$WANDB_ENTITY" ]; then
    python train.py --algo tqc --env "$ENV_ID" \
        --seed $SEED \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --wandb-tags tqc mountain_car comparison
else
    python train.py --algo tqc --env "$ENV_ID" \
        --seed $SEED \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-tags tqc mountain_car comparison
fi

echo ""
echo "=================================================="
echo "Training complete! Check wandb.ai for results."
echo "Project: $WANDB_PROJECT"
echo "=================================================="
