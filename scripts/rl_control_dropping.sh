#!/bin.bash

source scripts/setup_env.sh


python "$CONTROL_DROP_DIR/control_dropping/src/training/rl_control_dropping.py" \
  --embedding_chkpoint "/media/rpal/Drive_10TB/John/ControlDrop-RPAL/extracted_encoder_state_dict.pth" \
  --rl_chkpoint "rl_logs/rl_control_drop_23/checkpoints/checkpoint_25088_steps.zip" \
  --unlock_encoder \
  --experiment_name "rl_control_drop" \
  --total_timesteps 200000 \
  --save_freq 512 \
  --learning_rate 1e-4 \
  --n_steps 512 \
  --batch_size 128 \
  --gamma 0.995 \
  --clip_range 0.1 \
  --total_timesteps 2000000 \
  --num_envs 1 \
  --normalize_advantage