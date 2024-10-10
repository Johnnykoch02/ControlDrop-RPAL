#!/bin/bash

source scripts/setup_env.sh

# export SIM_GUI=true

python "$CONTROL_DROP_DIR/control_dropping/src/training/validation.py" \
  --checkpoint rl_logs/rl_control_drop_24/checkpoints/checkpoint_20992_steps.zip \
  --num_episodes 100 \
  --experiment_path ./rl_logs/rl_control_drop_24/
