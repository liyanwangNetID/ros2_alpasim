# Simulation timestamp
builtin_interfaces/Time stamp

# AlpaSim coordinate frame
string frame_id

# Ego pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation

# Ego dynamics
geometry_msgs/Vector3 linear_velocity
geometry_msgs/Vector3 angular_velocity
geometry_msgs/Vector3 linear_acceleration
geometry_msgs/Vector3 angular_acceleration

# Magnitude of linear velocity
float64 speed