#!/bin/bash

# Script to find and kill ROS2 joystick control processes

echo "=========================================="
echo "ROS2 Joystick Control Killer"
echo "=========================================="
echo ""

# Method 1: Kill by process name patterns
echo "Step 1: Checking for joystick/teleop processes..."

# Find processes related to joystick/teleop
JOYSTICK_PIDS=$(ps aux | grep -iE "joy|joystick|teleop" | grep -v grep | grep -v "$0" | awk '{print $2}')

if [ -z "$JOYSTICK_PIDS" ]; then
    echo "  No joystick processes found in process list."
else
    echo "  Found joystick processes:"
    ps aux | grep -iE "joy|joystick|teleop" | grep -v grep | grep -v "$0" | sed 's/^/    /'
    echo ""
    echo "  Killing processes..."
    for pid in $JOYSTICK_PIDS; do
        echo "    Killing PID: $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "  Processes killed."
fi

# Method 2: Kill by specific executable names
echo ""
echo "Step 2: Killing specific joystick executables..."

pkill -9 -f "joy_linux_node" 2>/dev/null && echo "  Killed joy_linux_node processes" || echo "  No joy_linux_node processes found"
pkill -9 -f "joys_manager" 2>/dev/null && echo "  Killed joys_manager processes" || echo "  No joys_manager processes found"
pkill -9 -f "ros2.*launch.*teleop" 2>/dev/null && echo "  Killed teleop launch processes" || echo "  No teleop launch processes found"
pkill -9 -f "ros2.*launch.*controller" 2>/dev/null && echo "  Killed controller launch processes" || echo "  No controller launch processes found"

# Wait a moment for processes to terminate
sleep 1

# Method 3: Check for ROS2 nodes (if ROS2 is sourced)
if command -v ros2 &> /dev/null; then
    echo ""
    echo "Step 3: Checking for ROS2 joystick nodes..."
    JOYSTICK_NODES=$(ros2 node list 2>/dev/null | grep -iE "joy|teleop" || true)
    
    if [ -z "$JOYSTICK_NODES" ]; then
        echo "  ✓ No joystick nodes found in ROS2."
    else
        echo "  Found ROS2 joystick nodes:"
        echo "$JOYSTICK_NODES" | sed 's/^/    /'
        echo ""
        echo "  Note: If nodes persist after killing processes, they may be in a different"
        echo "        ROS2 domain or the ROS2 daemon may need to be restarted."
        echo "        Try: ros2 daemon stop && ros2 daemon start"
    fi
else
    echo ""
    echo "Step 3: ROS2 not found in PATH (may not be sourced)."
    echo "  Skipping ROS2 node check."
fi

# Final verification
echo ""
echo "Step 4: Final verification..."
REMAINING=$(ps aux | grep -iE "joy|joystick|teleop" | grep -v grep | grep -v "$0" | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "  ✓ All joystick processes appear to be terminated."
else
    echo "  ⚠ Warning: Some processes may still be running:"
    ps aux | grep -iE "joy|joystick|teleop" | grep -v grep | grep -v "$0" | sed 's/^/    /'
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="


