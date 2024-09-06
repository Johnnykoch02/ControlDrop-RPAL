#!/bin/bash

source scripts/setup_env.sh

python '/home/blazekin/dev/ControlDrop-RPAL/control_dropping/src/training/training_dynamix.py' \
  --name "dynamix_critiq_training-9-01-2024" \
  --batch_size 512 \
  --epochs 500 \
  --warmup_epochs 10 \
  --num_workers 12 \
  --lr 0.0005 \
  --clip_grad_norm 2.0
  