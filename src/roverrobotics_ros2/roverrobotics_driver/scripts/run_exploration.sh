#!/bin/bash
# Simple launcher for the exploration script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$HOME/final_robin"

# Source ROS and workspace
source /opt/ros/humble/setup.bash
if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
fi

# Run the exploration script
python3 "$SCRIPT_DIR/explore_wander.py"


