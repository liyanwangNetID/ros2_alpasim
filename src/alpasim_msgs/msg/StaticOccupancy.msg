# Scene-level non-semantic 2.5D static occupancy in frame_id.
string schema_version
string scene_id
string scene_uuid
string frame_id
string representation
bool semantic_labels
float32 grid_size_xy_m
uint64 revision
string payload_sha256
int32[] grid_i
int32[] grid_j
float32[] min_z
float32[] max_z
uint32[] point_count
uint16[] flags
