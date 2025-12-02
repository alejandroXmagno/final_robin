#!/bin/bash
# Clean up ROS 2 shared memory segments

echo "Cleaning up ROS 2 shared memory..."

# Kill any stale ROS processes
pkill -9 -f "ros2\|ekf\|slam\|rplidar\|bno055" 2>/dev/null

# Remove FastRTPS shared memory files
rm -f /dev/shm/fastrtps_* 2>/dev/null

# Clean up shared memory segments
ipcs -m 2>/dev/null | grep $USER | awk '{print $2}' | xargs -r -I {} ipcrm -m {} 2>/dev/null

echo "Done! You can now launch ROS 2 nodes."


