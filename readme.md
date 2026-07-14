colcon build \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

ros2 run alpasim_bridge ego_state_publisher

ros2 topic echo /alpasim/ego_state




map
└── base_link                       动态 TF，来自 EgoState
    ├── vehicle_body                URDF 固定关节
    ├── wheel_front_left            URDF 固定关节
    ├── wheel_front_right           URDF 固定关节
    ├── wheel_rear_left             URDF 固定关节
    ├── wheel_rear_right            URDF 固定关节
    ├── camera_cross_left_120fov_optical   静态 TF
    ├── camera_front_wide_120fov_optical   静态 TF
    ├── camera_cross_right_120fov_optical  静态 TF
    └── camera_rear_left_70fov_optical     静态 TF