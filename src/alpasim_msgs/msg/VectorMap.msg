# Coordinate frame for all geometry
string frame_id

# Source identifiers
string scene_id
string map_id

# Map revision maintained by map_server.
# Starts from 1 and increases when the cached map changes.
uint64 revision

# Map spatial extent
geometry_msgs/Point minimum
geometry_msgs/Point maximum

alpasim_msgs/MapLane[] lanes
alpasim_msgs/MapRoadEdge[] road_edges
alpasim_msgs/MapTrafficSign[] traffic_signs
alpasim_msgs/MapWaitLine[] wait_lines