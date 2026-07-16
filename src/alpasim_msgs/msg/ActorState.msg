# Persistent actor ID in the AlpaSim scenario
string track_id

# Native AlpaSim semantic label, for example:
# automobile, heavy_truck, trailer
string label_class

# Whether AlpaSim treats this actor as static
bool is_static

# Pose of the actor AABB center at ActorStateArray.stamp.
# Expressed in ActorStateArray.pose_frame_id.
geometry_msgs/Pose pose

# AABB dimensions along the actor AABB x/y/z axes.
# x: length, y: width, z: height
geometry_msgs/Vector3 dimensions

# Motion quantities at ActorStateArray.stamp.
# Expressed in ActorStateArray.dynamics_frame_id.
geometry_msgs/Vector3 linear_velocity
geometry_msgs/Vector3 linear_acceleration

# Rotation and derivatives around the map z axis
float64 yaw
float64 yaw_rate
float64 yaw_acceleration

# Magnitude of linear_velocity
float64 speed