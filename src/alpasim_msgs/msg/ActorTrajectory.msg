# Persistent actor ID
string track_id

# Native AlpaSim semantic label
string label_class

bool is_static

# AABB dimensions; assumed constant over this trajectory message
geometry_msgs/Vector3 dimensions

# Ordered by increasing timestamp
alpasim_msgs/TrajectoryPoint[] points