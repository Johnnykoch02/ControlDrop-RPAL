#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 <port_number> [--is_gui true|false]"
    echo "  <port_number>: The port number for CoppeliaSim"
    echo "  --is_gui: Optional. Set to 'true' for GUI mode, 'false' for headless mode. Default is 'true'."
}

# Check if at least a port number was provided
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

port_number=$1
is_gui=false
scene_file="${CONTROL_DROP_DIR}/40mm_Sphere_Random_color.ttt" # Replace with the name of your CoppeliaSim scene file

# Parse additional arguments
shift
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --is_gui)
        is_gui="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        print_usage
        exit 1
        ;;
    esac
done

# Validate is_gui argument
if [[ "$is_gui" != "true" && "$is_gui" != "false" ]]; then
    echo "Error: --is_gui must be 'true' or 'false'"
    print_usage
    exit 1
fi

# Set up the command
coppeliasim_cmd="${SIM_DIR}/coppeliaSim.sh"
args=("$scene_file" -gREMOTEAPISERVERSERVICE_"$port_number"_FALSE_FALSE)

# Add headless flag if is_gui is false
if [ "$is_gui" = "false" ]; then
    args+=(-h)
fi

# Add silent mode flag
args+=(-s)

# Start CoppeliaSim
echo "Starting CoppeliaSim with port $port_number in $([ "$is_gui" = "true" ] && echo "GUI" || echo "headless") mode"

if [ "$is_gui" = "true" ]; then
    "$coppeliasim_cmd" "${args[@]}"
else 
    xvfb-run -a "$coppeliasim_cmd" "${args[@]}"
fi