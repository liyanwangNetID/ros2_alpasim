uint8 SOURCE_UNKNOWN=0
uint8 SOURCE_RECORDING_GROUND_TRUTH=1
uint8 SOURCE_MODEL_PLANNING=2
uint8 SOURCE_EXECUTED=3
uint8 SOURCE_CONTROLLER_REFERENCE=4

# Reference time t0.
builtin_interfaces/Time reference_stamp

string pose_frame_id
string dynamics_frame_id

uint8 source
string producer

# True only when this trajectory came from an autonomous-driving model.
bool is_model_generated

# True when AlpaSim force-GT mode was active at reference_stamp.
bool force_gt_active

# Requested and actual temporal horizons.
builtin_interfaces/Duration requested_duration
builtin_interfaces/Duration actual_duration

# Ordered by increasing timestamp.
alpasim_msgs/TrajectoryPoint[] points