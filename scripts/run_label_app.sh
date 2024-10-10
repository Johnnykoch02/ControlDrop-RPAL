#!/bin.bash

source scripts/setup_env.sh

export SIM_GUI=true
# export QT_PLUGIN_PATH=/home/rpal/anaconda3/envs/raylib/lib/python3.10/site-packages/cv2/qt/plugins
# export LD_LIBRARY_PATH=/home/rpal/anaconda3/envs/raylib/lib/python3.10/site-packages/cv2/qt/plugins:$LD_LIBRARY_PATH


python "$CONTROL_DROP_DIR/control_dropping/src/rpal_utils/label_metadata.py"