# Simulation timestamp
builtin_interfaces/Time stamp

# Pose of child_frame_id expressed in pose_frame_id
string pose_frame_id
string child_frame_id

# Frame in which velocity and acceleration vectors are expressed
string dynamics_frame_id

geometry_msgs/Point position
geometry_msgs/Quaternion orientation

geometry_msgs/Vector3 linear_velocity
geometry_msgs/Vector3 angular_velocity
geometry_msgs/Vector3 linear_acceleration
geometry_msgs/Vector3 angular_acceleration

float64 speed