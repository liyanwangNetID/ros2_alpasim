# Absolute AlpaSim simulation timestamp of this point
builtin_interfaces/Time stamp

# Relative time from ActorTrajectoryArray.reference_stamp.
# History points are <= 0.
# Future/prediction points are >= 0.
builtin_interfaces/Duration time_from_reference

# Actor AABB pose at this point
geometry_msgs/Pose pose

# Motion values at this point
geometry_msgs/Vector3 linear_velocity
geometry_msgs/Vector3 linear_acceleration

float64 yaw
float64 yaw_rate
float64 yaw_acceleration
float64 speed