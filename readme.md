colcon build \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

ros2 run alpasim_bridge ego_state_publisher

ros2 topic echo /alpasim/ego_state

ros2 launch ego_vehicle_description display.launch.py


青绿色：lane centerlines
白色：lane left/right boundaries
红色：road edges
黄色：STOP wait lines
橙色：YIELD wait lines
灰色：UNKNOWN wait lines
品红色球体：traffic signs
