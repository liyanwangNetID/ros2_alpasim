# Trajectory data source
uint8 SOURCE_UNKNOWN=0
uint8 SOURCE_GROUND_TRUTH_HISTORY=1
uint8 SOURCE_GROUND_TRUTH_FUTURE=2
uint8 SOURCE_MODEL_PREDICTION=3
uint8 SOURCE_GROUND_TRUTH_PLACEHOLDER=4

# Reference simulation time t0
builtin_interfaces/Time reference_stamp

# Pose values are expressed in this frame
string pose_frame_id

# Velocity and acceleration are expressed in this frame
string dynamics_frame_id

# Identifies the semantic origin of this data
uint8 source

# Name/version of the producer
# Examples:
# alpasim_ground_truth
# ground_truth_placeholder
# my_prediction_model_v1
string producer

# True only for genuine model prediction output
bool is_model_generated

# Requested temporal window
builtin_interfaces/Duration requested_duration

# Actual temporal extent available in this message
builtin_interfaces/Duration actual_duration

# Requested sampling interval
builtin_interfaces/Duration sampling_interval

# One trajectory per actor
alpasim_msgs/ActorTrajectory[] trajectories