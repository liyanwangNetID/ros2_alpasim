# False when prepare_for_policy padded this waypoint with NaNs.
bool valid

# Route waypoint.
# For model-input route this is expressed in Route.frame_id.
geometry_msgs/Point position

# Cumulative distance from the first valid route point, meters.
float64 longitudinal_distance