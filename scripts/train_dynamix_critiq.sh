#!/bin/bash

source scripts/setup_env.sh

python "$CONTROL_DROP_DIR/control_dropping/src/training/training_dynamix.py" \
  --name "dynamix_critiq_training-9-01-2024" \
  --batch_size 256 \
  --epochs 500 \
  --warmup_epochs 10 \
  --num_workers 12 \
  --lr 0.005 \
  --clip_grad_norm 2.0
  