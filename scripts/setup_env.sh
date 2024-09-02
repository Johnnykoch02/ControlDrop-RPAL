#!/bin/bash

# Define the path to the env file
ENV_FILE="rpal.env"

# Function to prompt the user for input
prompt_for_variable() {
    local var_name=$1
    local default_value=$2
    read -p "Please enter the value for $var_name (default: $default_value): " input_value
    echo "${input_value:-$default_value}"
}

# Check if the env file exists
if [ -f "$ENV_FILE" ]; then
    # Source the env file to load variables
    echo "Loading environment variables from $ENV_FILE..."
    source "$ENV_FILE"
else
    # Prompt the user to enter values for the variables
    echo "$ENV_FILE does not exist. Creating it..."
    
    CONTROL_DROP_DIR=$(prompt_for_variable "CONTROL_DROP_DIR" "/path/to/ControlDrop-RPAL")
    SIM_DIR=$(prompt_for_variable "SIM_DIR" "/path/to/CoppeliaSim")
    
    # Write the variables to the env file
    echo "CONTROL_DROP_DIR=\"$CONTROL_DROP_DIR\"" > "$ENV_FILE"
    echo "SIM_DIR=\"$SIM_DIR\"" >> "$ENV_FILE"
    
    echo "$ENV_FILE has been created with the following content:"
    cat "$ENV_FILE"
fi

# Export the variables to the environment
export CONTROL_DROP_DIR
export SIM_DIR

# Display the set environment variables
echo "Environment variables set:"
echo "CONTROL_DROP_DIR=$CONTROL_DROP_DIR"
echo "SIM_DIR=$SIM_DIR"
