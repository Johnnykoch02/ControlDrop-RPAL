#!/bin.bash

source scripts/setup_env.sh

python "$CONTROL_DROP_DIR/control_dropping/src/training/data_collection_dynamix.py" \
  --num_steps 25000 \
  --num_instances 4 \
  --save_interval 10