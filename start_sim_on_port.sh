#!/bin/bash

# Check if a port number was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <port_number>"
    exit 1
fi

port_number=$1
scene_file="./40mm_Sphere_Random_color.ttt" # Replace with the name of your CoppeliaSim scene file


# Start CoppeliaSim with the modified scene
echo "Starting CoppeliaSim with port $port_number"
/home/rpal/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/coppeliaSim.sh "$scene_file" -gREMOTEAPISERVERSERVICE_"$port_number"_FALSE_FALSE -s