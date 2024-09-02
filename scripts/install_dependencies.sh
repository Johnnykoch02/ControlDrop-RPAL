#!/bin/bash

# Script to install project dependencies

# Function to print usage
print_start() {
    echo "Starting installation of project dependencies..."
}

# Function to print completion message
print_completion() {
    echo "All dependencies have been installed successfully."
}

print_start

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install required packages
echo "Installing required packages..."

# RPAL Control Dropping dependencies
sudo apt-get install -y \
        xvfb \
        x11-xserver-utils \
        wget \
        unzip

# Clean up
echo "Cleaning up..."
sudo apt-get clean

print_completion
