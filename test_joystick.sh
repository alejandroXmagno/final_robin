#!/bin/bash
# Test joystick functionality

echo "=== Testing Joystick ==="
echo ""

# Check device
echo "1. Checking joystick device..."
ls -la /dev/input/js0
echo ""

# Test with jstest
echo "2. Testing with jstest (press Ctrl+C after a few seconds)..."
timeout 3 jstest /dev/input/js0 2>&1 | head -5
echo ""

# Test joy_linux_node
echo "3. Testing joy_linux_node..."
timeout 5 ros2 run joy_linux joy_linux_node --ros-args 2>&1 &
JOY_PID=$!
sleep 2

echo "4. Checking if /joy topic exists..."
if ros2 topic list 2>/dev/null | grep -q "^/joy$"; then
    echo "   ✓ /joy topic exists!"
    echo "5. Checking message rate..."
    timeout 2 ros2 topic hz /joy 2>&1 | head -3
else
    echo "   ✗ /joy topic NOT found!"
fi

kill $JOY_PID 2>/dev/null
wait $JOY_PID 2>/dev/null
echo ""

# Check joys_manager
echo "6. Testing joys_manager (this will fail if /joy doesn't exist)..."
timeout 3 ros2 run roverrobotics_input_manager joys_manager.py --ros-args --params-file /home/stickykeys/final_robin/install/roverrobotics_driver/share/roverrobotics_driver/config/ps4_controller_config.yaml 2>&1 | head -10

echo ""
echo "=== Done ==="


