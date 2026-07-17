uint8 GENERATOR_UNKNOWN=0
uint8 GENERATOR_RECORDED=1
uint8 GENERATOR_MAP=2

# Simulation time at which this route was generated.
builtin_interfaces/Time reference_stamp

# Coordinate frame of points.
# "map" for /alpasim/route/map.
# "base_link" or model-input rig frame for /alpasim/route/model_input.
string frame_id

# Identifies the frame semantics before any client-side conversion.
string source_frame_id

uint8 generator_type

# Example: alpasim_route_generator_map
string producer

# Number that changes each policy cycle.
uint64 sequence

# Configured nominal lookahead, currently 80 meters.
float64 lookahead_distance

# Expected model-input waypoint count, currently 20.
uint32 expected_point_count

alpasim_msgs/RoutePoint[] points