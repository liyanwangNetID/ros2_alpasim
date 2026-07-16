colcon build \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

ros2 run alpasim_bridge ego_state_publisher

ros2 topic echo /alpasim/ego_state

ros2 launch ego_vehicle_description display.launch.py

