# AlpaSim ROS 2 Bridge

将 AlpaSim Runtime 中的自车状态、摄像头、周围交通参与者、矢量地图、导航 Route、模型规划轨迹与真值轨迹转换为 ROS 2 强类型接口，并在 RViz2 中进行统一可视化。

> 当前阶段状态：已完成主要可观测数据的导出、ROS 2 接口设计、Service 缓存与查询、动态轨迹发布，以及 RViz2 联调。

---

## 1. 系统概览

本项目采用分层、分节点设计，避免将所有逻辑集中在单一 ROS node 中。

```text
AlpaSim Runtime
├── Ego state / clock / camera export
├── Actor state export
├── Static VectorMap export
├── Complete recording GT ego trajectory export
└── Dynamic Route and model-plan export
          │
          │ UDP/TCP, non-blocking background transport
          ▼
ROS 2 Bridge
├── Data publishers
├── Static-data servers
├── Query clients
└── Visualization publishers
          │
          ▼
ROS topics / services / TF / RViz2
```

### 1.1 设计原则

1. 数据桥接与可视化分离：算法节点使用强类型 ROS 消息，不需要解析 RViz Marker。
2. 高频动态数据允许丢弃过期帧，始终优先保留最新状态。
3. 地图和完整 Ground Truth 轨迹属于关键静态数据，发送失败时持续重试，直到成功交付。
4. 所有网络传输在后台线程执行，不阻塞 AlpaSim simulation loop。
5. Ground Truth、模型输出与实际执行结果在 topic 名称和 metadata 中严格区分。
6. 动态 Marker 不使用全局 `Marker.DELETEALL`，避免清除其他 Marker publisher 的内容。
7. 仿真时间回退被视为新 rollout，相关 FIFO、sequence 和可视化状态应重置。

---

## 2. AlpaSim Runtime 特征与机制

## 2.1 坐标系

当前桥接中的主要坐标语义：

```text
map
└── base_link
```

- `map`：AlpaSim true local/simulation frame；地图、actors、GT trajectory、model plan 和 executed path 均在该坐标系中表达。
- `base_link`：当前自车 rig frame；Driver 实际接收的 model-input Route 在该坐标系中表达。
- `map -> base_link` TF 由 ego state 数据生成。

消息中显式保留：

```text
pose_frame_id
child_frame_id
dynamics_frame_id
source_frame_id
```

不得仅依靠 topic 名称猜测坐标系。

## 2.2 动态数据与静态数据

### 动态数据

以下数据随 simulation step 或 policy step 更新：

```text
Ego state
Camera images
Actor states
Current processed Route
Driver model-input Route
Model planned trajectory
```

动态 TCP exporter 使用小队列；队列满时丢弃旧 packet，保留最新状态。

### 静态或完整数据

以下数据在一个 rollout 内基本固定：

```text
VectorMap
Complete recording ground-truth ego trajectory
```

它们由 Runtime 可靠发送一次，由 ROS server 缓存，并通过 Service 提供查询。

## 2.3 Actor trajectory 语义

`TrafficObject.trajectory` 是 Runtime 维护的 simulation trajectory。它可能同时包含：

```text
当前时刻之前的历史状态
当前 simulation cycle 已计算出的短期未来状态
```

它不是概率预测模型的输出。因此：

- 当前状态必须按当前 simulation timestamp 取得，不能直接使用 trajectory 最后一个点。
- Ground-truth future 必须明确标记为真值。
- `/alpasim/prediction/actors` 当前只是占位接口，不得把它解释成真实模型预测。

## 2.4 VectorMap 机制

Runtime 地图入口：

```python
state.unbound.vector_map
```

类型：

```python
trajdata.maps.VectorMap
```

地图优先从 ClipGT/map data 加载，必要时回退到 OpenDRIVE。XODR 数据会转换到 AlpaSim simulation/local frame，使地图与 ego 和 actors 对齐。

当前审计场景的示例规模：

```text
Road lanes:       442
Road edges:       359
Traffic signs:     44
Wait lines:        26
Approx. extent:    781 m x 440 m
```

当前可靠暴露的地图内容：

```text
RoadLane
├── centerline: x, y, z, heading
├── left boundary: x, y, z
├── right boundary: x, y, z
├── successor / predecessor IDs
├── left / right adjacent IDs
├── traffic sign IDs
├── wait line IDs
└── road area IDs

RoadEdge
└── polyline: x, y, z, heading

TrafficSign
├── ID
├── position
└── native sign_type string

WaitLine
├── ID
├── polyline
├── wait_line_type: STOP / YIELD / UNKNOWN
└── is_implicit
```

当前版本尚未可靠提供：

```text
Traffic-light state
Lane speed limit
Lane type
Intersection flag
Crosswalk / walkway geometry
Road-area polygons
```

无效 lane 引用 `"-1"` 在 Runtime 导出时被过滤。

## 2.5 AlpaSim processed Route

当前已审计两种 Route generator：

```text
RouteGeneratorRecorded
RouteGeneratorMap
```

当前使用 `RouteGeneratorMap`。其主要流程：

```text
Recorded ego trajectory
→ 匹配候选 lane
→ 求连续 lane sequence
→ 投影到 lane centerline
→ 接近末端时沿 next_lanes 延伸
```

内部完整 Route 位于 local/map frame：

```python
route_generator.route_polyline_in_local
```

每个 PolicyEvent 为 Driver 生成局部 Route：

```text
20 waypoints
80 m nominal lookahead
approximately 4.21 m spacing
frame: base_link / current rig
z = 0
```

`prepare_for_policy()` 保证：

```text
固定 20 点
点间距要求 3.5–4.5 m
异常间距后截断
不足 20 点时使用 NaN padding
```

ROS 侧使用 `RoutePoint.valid` 表达 padding 点，绝不把 NaN 写入 JSON、ROS geometry 或 RViz Marker。

## 2.6 Recording GT Ego Trajectory

Runtime 入口：

```python
state.unbound.gt_ego_trajectory
```

当前审计示例：

```text
202 points
20.0 s duration
approximately 10 Hz
```

完整数据包括：

```text
timestamps
positions
quaternions
velocities
accelerations
yaw
yaw rate
yaw acceleration
```

完整 GT trajectory 在 rollout 内固定，因此一次性可靠发送到 ROS server。客户端按当前时间、未来时长和采样周期请求片段。

## 2.7 Model planned trajectory

Driver 返回的轨迹首先处于 noisy/estimated frame，Runtime 随后将其转换到 true local/map frame。

当前审计示例：

```text
7 points
0.5 s interval
3.0 s horizon
```

消息保留：

```text
source
producer
is_model_generated
force_gt_active
```

如果 force-GT 阶段跳过 Driver，使用 controller reference trajectory，则必须标记为 controller reference，而不能标记为模型输出。

---

## 3. ROS 2 接口

## 3.1 节点职责

### `ego_state_publisher`

负责：

```text
Ego state
/clock
map -> base_link TF
Camera images
CameraInfo
```

### `actor_state_publisher`

负责：

```text
Current actors
Actor history FIFO
Actor ground-truth future
Prediction placeholder
```

### `actor_marker_publisher`

订阅 actor current state，发布 actor boxes、labels 和 velocity arrows。

### `map_server`

监听 Runtime map TCP，缓存当前 `VectorMap`，提供地图查询 Service 和 revision 管理。

### `map_marker_publisher`

请求地图 Service，转换为静态 RViz MarkerArray。缓存完整 MarkerArray，并支持重新发布。

### `ground_truth_trajectory_server`

缓存完整 recording GT ego trajectory，根据请求裁剪和重采样未来轨迹片段。

### `navigation_state_publisher`

接收每个 PolicyEvent 的动态 navigation update，发布：

```text
Route in map
Route model input in base_link
Model/controller planned trajectory in map
```

### `ground_truth_future_publisher`

订阅 `/clock`，按当前 simulation time 请求未来 GT 轨迹，并发布 rolling future window。

### `executed_path_publisher`

订阅 `/alpasim/ego_state`，维护 physics-corrected ego 历史 FIFO，并发布 executed path。

### `navigation_marker_publisher`

将 Route、GT future、model plan 和 executed path 转换成统一的 RViz MarkerArray。

### `robot_state_publisher`

根据 ego URDF 发布固定车辆结构 TF。

### `rviz2`

加载项目 RViz 配置，统一显示车辆、actors、地图、摄像头与导航轨迹。

## 3.2 主要 Topics

### Ego 与 sensors

```text
/alpasim/ego_state
/clock
/camera-related image topics
/camera-related CameraInfo topics
/tf
/tf_static
```

### Actors

```text
/alpasim/actors/current
/alpasim/actors/history
/alpasim/ground_truth/actors/future
/alpasim/prediction/actors
/alpasim/actors/markers
```

### Route 与 ego trajectories

```text
/alpasim/route/map
/alpasim/route/model_input
/alpasim/ground_truth/ego/future_trajectory
/alpasim/planning/ego/trajectory
/alpasim/ego/executed_path
/alpasim/navigation/markers
```

### Map visualization

```text
/alpasim/map/markers
```

地图本体通过 Service 获取，不作为周期 topic 重复发布。

## 3.3 Services

### `/alpasim/map/get_vector_map`

类型：

```text
alpasim_msgs/srv/GetVectorMap
```

功能：

```text
获取当前缓存的 VectorMap
按 scene_id 检查请求
使用 revision / known_revision 避免重复传输
```

### `/alpasim/navigation/get_ground_truth_ego_trajectory`

类型：

```text
alpasim_msgs/srv/GetGroundTruthEgoTrajectory
```

请求参数：

```text
reference_stamp
future_duration
sampling_interval
max_points
known_revision
```

响应：

```text
成功状态和说明
source revision
完整 recording 有效时间范围
裁剪与重采样后的 EgoTrajectory
```

数据不足时只返回 recording 中实际可用部分，不做外推。

## 3.4 自定义消息

### Ego

```text
EgoState.msg
EgoTrajectory.msg
TrajectoryPoint.msg
```

### Actors

```text
ActorState.msg
ActorStateArray.msg
ActorTrajectory.msg
ActorTrajectoryArray.msg
```

### Map

```text
MapPolyline.msg
MapLane.msg
MapRoadEdge.msg
MapTrafficSign.msg
MapWaitLine.msg
VectorMap.msg
```

### Route

```text
RoutePoint.msg
Route.msg
```

## 3.5 关键坐标系

```text
/alpasim/route/model_input
  frame_id: base_link

/alpasim/route/map
  frame_id: map

/alpasim/ground_truth/ego/future_trajectory
  pose_frame_id: map
  dynamics_frame_id: map

/alpasim/planning/ego/trajectory
  pose_frame_id: map
  dynamics_frame_id: map

/alpasim/ego/executed_path
  pose_frame_id: inherited from EgoState, normally map
  dynamics_frame_id: inherited from EgoState, normally map
```

---

## 4. RViz2 可视化

## 4.1 Ego vehicle

Ego 车辆通过 URDF 显示。当前推荐车身颜色：

```xml
<color rgba="1.00 0.80 0.00 1.0"/>
```

即亮黄色，便于与其他车辆和地图元素区分。

## 4.2 Actor markers

```text
automobile       blue
heavy_truck      orange
trailer          purple
pedestrian       red
bicycle          green
motorcycle       yellow
unknown          gray
```

可视元素：

```text
3D bounding box
Track ID and class label
Speed text
Velocity arrow
```

Actor pose 已表示 AABB center，因此 box 不应再次沿 z 方向抬高半个车身。

## 4.3 Map markers

```text
Lane centerline       cyan/teal
Lane boundaries       light gray / white
Road edges            red
STOP wait lines       yellow
YIELD wait lines      orange
UNKNOWN wait lines    gray
Traffic signs         magenta
```

地图 Marker topic 使用：

```text
RELIABLE
TRANSIENT_LOCAL
KEEP_LAST depth=1
```

地图 MarkerArray 会缓存并周期重发，以支持多次 Runtime rollout 和 RViz 状态恢复。

## 4.4 Navigation markers

```text
AlpaSim Route          bright pink solid line
Recording GT future    green solid line
Model planned path     electric blue solid line
Executed path          white thin solid line
```

推荐 RGBA：

```text
Route:          1.00, 0.10, 0.62, 1.00
GT future:      0.10, 1.00, 0.20, 1.00
Model plan:     0.10, 0.30, 1.00, 1.00
Executed path:  1.00, 1.00, 1.00, 0.95
```

不同轨迹还使用不同宽度和 z-offset，减少重叠和 z-fighting。

## 4.5 Marker 隔离规则

多 Marker publisher 同时工作时，不应周期发送全局：

```python
Marker.DELETEALL
```

正确方法：

```text
为每个 Marker 使用稳定 namespace + ID
记录上一帧 marker keys
仅对消失元素发送 Marker.DELETE
```

否则 actor 或 navigation 更新可能清除 map markers。

---

## 5. 网络端口

当前 Runtime 到 ROS Bridge 的端口规划：

```text
Camera / sensor channel       15001
Actor state TCP               15002
VectorMap TCP                 15003
Complete GT ego TCP           15004
Dynamic navigation TCP        15005
```

Ego state 使用项目中已有的独立传输通道。

检查端口占用：

```bash
ss -ltnp | grep -E ':1500[1-5]'
```

如果出现 `Address already in use`，检查临时测试接收器或旧 launch 是否仍在运行。

---

## 6. 配置文件

主要 YAML 配置位于：

```text
alpasim_bridge/config/
```

当前包括：

```text
actor_export.yaml
actor_markers.yaml
map_server.yaml
map_markers.yaml
ground_truth_trajectory_server.yaml
navigation_state.yaml
ground_truth_future.yaml
executed_path.yaml
navigation_markers.yaml
```

常见可配置项：

```text
TCP host and port
Topic and service names
History duration
Future duration
Sampling interval
Maximum points
Publish rate
Marker visibility
Marker width
Marker z-offset
Map marker republish period
```

重要默认值示例：

```text
Actor history:                2.0 s
GT ego future:                6.4 s
GT ego sampling interval:     0.1 s
GT ego max points:            65
Executed path history:        0.0 s = complete rollout
Navigation dynamic TCP:       15005
Map marker republish period:  2.0 s
```

---

## 7. 构建

在 ROS workspace 中：

```bash
cd /home/lab/alpasim_ros2_ws

source /opt/ros/jazzy/setup.zsh

colcon build \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.zsh
```

只重新构建相关 packages：

```bash
colcon build \
  --symlink-install \
  --packages-select \
    alpasim_msgs \
    alpasim_bridge \
    ego_vehicle_description

source install/setup.zsh
```

查看可执行节点：

```bash
ros2 pkg executables alpasim_bridge | sort
```

查看接口：

```bash
ros2 interface show alpasim_msgs/msg/EgoState
ros2 interface show alpasim_msgs/msg/VectorMap
ros2 interface show alpasim_msgs/msg/Route
ros2 interface show alpasim_msgs/msg/EgoTrajectory
ros2 interface show alpasim_msgs/srv/GetVectorMap
ros2 interface show alpasim_msgs/srv/GetGroundTruthEgoTrajectory
```

---

## 8. 运行

## 8.1 启动 ROS 2 Bridge 与 RViz2

```bash
source /opt/ros/jazzy/setup.zsh
source /home/lab/alpasim_ros2_ws/install/setup.zsh

ros2 launch \
  ego_vehicle_description \
  display.launch.py
```

统一 launch 启动：

```text
Ego and camera publisher
Actor state publisher
Map server
GT trajectory server
Navigation state publisher
GT future publisher
Executed path publisher
Robot state publisher
Map marker publisher
RViz2
Actor marker publisher, delayed
Navigation marker publisher, delayed
```

`TimerAction` 只延迟动态 marker nodes，以便地图优先加载和渲染。LaunchDescription 中 Node 的书写顺序不能保证节点完成初始化的顺序。

## 8.2 启动 AlpaSim Runtime

按照项目当前 Runtime 配置启动仿真。建议先启动 ROS launch，再启动 Runtime，使静态数据 exporter 能立即连接 ROS server。

Runtime 完成后可以保持 launch 不关闭，然后再次启动 Runtime。相关节点通过 timestamp rollback、revision 和 marker cache 支持多 rollout。

## 8.3 RViz2 设置

```text
Fixed Frame: map
```

建议分别创建三个独立 MarkerArray Display：

```text
Actor Markers
  /alpasim/actors/markers

Map Markers
  /alpasim/map/markers

Navigation Markers
  /alpasim/navigation/markers
```

不要把已有的 Map MarkerArray Display 直接改成 Navigation topic；应新增独立 Display。

---

## 9. 验证命令

查看节点：

```bash
ros2 node list
```

查看核心 topics：

```bash
ros2 topic list | grep alpasim
```

检查地图 Service：

```bash
ros2 service list | grep alpasim/map
```

检查 Route：

```bash
ros2 topic echo /alpasim/route/map --once
ros2 topic echo /alpasim/route/model_input --once
```

检查模型规划：

```bash
ros2 topic echo \
  /alpasim/planning/ego/trajectory \
  --once
```

检查 GT future：

```bash
ros2 topic echo \
  /alpasim/ground_truth/ego/future_trajectory \
  --once
```

检查 executed path：

```bash
ros2 topic echo \
  /alpasim/ego/executed_path \
  --once
```

检查发布频率：

```bash
ros2 topic hz /alpasim/actors/current
ros2 topic hz /alpasim/route/map
ros2 topic hz /alpasim/planning/ego/trajectory
ros2 topic hz /alpasim/ground_truth/ego/future_trajectory
```

---

## 10. 多 rollout 行为

保持 launch 运行并再次启动 Runtime 时：

```text
Ego timestamp rollback
→ executed path 清空
→ navigation sequence tracking 重置
→ actor history 应清理或重建
→ dynamic markers 更新
→ map server 保留或更新 revision
→ map marker 重发缓存
```

静态地图没有变化时，Service 可能返回：

```text
not_modified = true
```

此时 map marker publisher 仍应重新发布缓存 MarkerArray，以防旧 Marker 曾被 RViz 或其他节点清除。

---

## 11. Ground-Truth 泄漏注意事项

以下数据属于评估真值，不应作为在线驾驶模型输入：

```text
/alpasim/ground_truth/actors/future
/alpasim/ground_truth/ego/future_trajectory
```

当前 actor prediction 占位接口：

```text
/alpasim/prediction/actors
```

必须保留：

```text
source = ground-truth placeholder
is_model_generated = false
```

Route 可以作为模型输入，因为它是经过 AlpaSim 加工的 lane-level navigation guidance，而不是精确的 recorded future trajectory。

---

## 12. 常见问题

### 地图第一次显示，第二次 rollout 消失

检查是否有动态 marker node 发送了全局 `Marker.DELETEALL`。地图节点应缓存并周期重发 MarkerArray，且在 `not_modified=true` 时能够重发缓存。

### Map marker 和 Navigation marker 无法同时显示

在 RViz 中创建两个独立的 MarkerArray Display：

```text
/alpasim/map/markers
/alpasim/navigation/markers
```

### Route model input 中存在无效点

这是 `prepare_for_policy()` 的合法 NaN padding。检查：

```text
RoutePoint.valid
```

无效点的位置字段只是有限占位值。

### 地图或 GT server 无法绑定端口

```bash
ss -ltnp | grep ':15003'
ss -ltnp | grep ':15004'
```

停止临时测试 receiver 或旧进程后重试。

### Executed path 混入上一次 rollout

确认 timestamp 回退时 FIFO 会清空，并检查 `/clock` 和 `/alpasim/ego_state` timestamp 是否正常重置。

### `Polyline.h` 访问报错

部分 polyline 只有三列 `x, y, z`。不要使用 `hasattr(polyline, "h")` 判断，因为 property 访问仍可能触发异常。应先检查：

```python
points.shape[-1] >= 4
```

---

## 13. 当前完成状态

```text
Stage 1A: Ego state, clock and TF                         Complete
Stage 1B: Multi-camera images and calibration             Complete
Stage 1C: Actors, history, future and prediction API      Complete
Stage 1D.1: VectorMap, map service and RViz markers       Complete
Stage 1D.2: Route, GT future, model plan, executed path   Complete
```

当前系统已具备端到端自动驾驶模型所需的主要可观测输入，并可同时提供 Ground Truth、模型规划和实际执行结果用于闭环分析与可视化。

---

## 14. 后续工作

```text
真实 actor prediction model
外部 end-to-end model 接入
规划轨迹回传 Runtime
Steering / throttle / brake control interface
Traffic-light state
Online map update and local-map query
Different-scene marker cleanup
Automated unit and integration tests
```




colcon build \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

ros2 run alpasim_bridge ego_state_publisher

ros2 topic echo /alpasim/ego_state

ros2 launch ego_vehicle_description display.launch.py


青绿色：lane centerlines
白色：lane left/right boundaries
红色：road edges
黄色：STOP wait lines
橙色：YIELD wait lines
灰色：UNKNOWN wait lines
品红色球体：traffic signs





# ROS2 side

## Launch
ros2 launch ego_vehicle_description display.launch.py

## Record Data
ros2 run alpasim_dataset_tools data_batch_recorder

## Start a Planner
ros2 run alpasim_planning synthetic_trajectory_planner

/home/lab/alpasim_ros2_ws/scripts/run_vavam_planner.sh

ros2 run alpasim_planning ground_truth_replay_planner





# Alpasim side

## Bridge to ROS2
uv run --project src/driver python -m alpasim_driver.main \
  --config-path=/home/lab/alpasim/src/driver/configs \
  --config-name=external_trajectory

## For Record Data
/home/lab/alpasim_ros2_ws/scripts/run_gt_dataset_streaming.sh

## Run One Scene
uv run --project src/wizard alpasim_wizard \
  deploy=local \
  driver=manual \
  driver_source=external_static \
  topology=1gpu \
  wizard.log_dir="$PWD/runs/external_control" \
  scenes.scene_ids='["clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"]' \
  wizard.external_services.driver='["172.23.0.1:6789"]' \
  runtime.simulation_config.n_sim_steps=200 \
  runtime.simulation_config.control_timestep_us=100000 \
  runtime.simulation_config.pose_reporting_interval_us=100000 \
  +runtime.simulation_config.realtime_factor=1.0 \
  'runtime.simulation_config.cameras=[{height:480,width:854,logical_id:camera_cross_left_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_wide_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_tele_30fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_cross_right_120fov,frame_interval_us:100000,shutter_duration_us:30000}]'

## Run a List of Scene
uv run --project src/wizard alpasim_wizard \
  deploy=local \
  driver=manual \
  driver_source=external_static \
  topology=1gpu \
  wizard.log_dir="$PWD/runs/gt_replay_dataset" \
  scenes.test_suite_id=public_2601 \
  scenes.suites_csv='["/home/lab/alpasim/data/scenes/my_gt_sim_suites.csv"]' \
  wizard.external_services.driver='["172.23.0.1:6789"]' \
  runtime.endpoints.renderer.n_concurrent_rollouts=1 \
  runtime.endpoints.driver.n_concurrent_rollouts=1 \
  runtime.endpoints.physics.n_concurrent_rollouts=1 \
  runtime.endpoints.controller.n_concurrent_rollouts=1 \
  runtime.simulation_config.n_rollouts=1 \
  runtime.simulation_config.n_sim_steps=200 \
  runtime.simulation_config.control_timestep_us=100000 \
  runtime.simulation_config.pose_reporting_interval_us=100000 \
  +runtime.simulation_config.realtime_factor=1.0 \
  'runtime.simulation_config.cameras=[{height:480,width:854,logical_id:camera_cross_left_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_wide_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_tele_30fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_cross_right_120fov,frame_interval_us:100000,shutter_duration_us:30000}]'

## Change Camera Config
uv run --project src/wizard alpasim_wizard \
  deploy=local \
  driver=manual \
  driver_source=external_static \
  topology=1gpu \
  wizard.log_dir="$PWD/runs/external_vavam_official_camera" \
  scenes.scene_ids='["clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"]' \
  wizard.external_services.driver='["172.23.0.1:6789"]' \
  runtime.simulation_config.n_sim_steps=200 \
  runtime.simulation_config.control_timestep_us=100000 \
  runtime.simulation_config.pose_reporting_interval_us=100000 \
  +runtime.simulation_config.image_format=jpeg \
  +runtime.simulation_config.realtime_factor=1.0 \
  'runtime.simulation_config.cameras=[{height:1080,width:1920,logical_id:camera_cross_left_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:1080,width:1920,logical_id:camera_front_wide_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:1080,width:1920,logical_id:camera_front_tele_30fov,frame_interval_us:100000,shutter_duration_us:30000},{height:1080,width:1920,logical_id:camera_cross_right_120fov,frame_interval_us:100000,shutter_duration_us:30000}]'

## Use CATK
uv run --project src/wizard alpasim_wizard \
  deploy=local \
  driver=manual \
  driver_source=external_static \
  topology=1gpu \
  trafficsim=catk \
  wizard.log_dir="$PWD/runs/catk_reactive" \
  scenes.scene_ids='["clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"]' \
  wizard.external_services.driver='["172.23.0.1:6789"]' \
  runtime.simulation_config.n_sim_steps=200 \
  runtime.simulation_config.control_timestep_us=100000 \
  runtime.simulation_config.pose_reporting_interval_us=100000 \
  +runtime.simulation_config.realtime_factor=1.0 \
  'runtime.simulation_config.cameras=[{height:480,width:854,logical_id:camera_cross_left_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_wide_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_tele_30fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_cross_right_120fov,frame_interval_us:100000,shutter_duration_us:30000}]'














# For vlm backbone training dataset
## 模型输入
    4 cameras × 2 frames
      front_wide
      front_tele
      cross_left
      cross_right

    frame times:
      t0 - 0.5 s
      t0

    ego history:
      过去 M 个 waypoint

    navigation text:
      "Turn right at the upcoming intersection."

## 模型监督输出
    {
      "scene_facts": {
        "road_context": "intersection",
        "traffic_control": "red_light",
        "critical_actor": "vehicle_ahead"
      },
      "decision": {
        "lateral": "keep_lane",
        "longitudinal": "stop"
      },
      "reasoning": "The route continues straight, but the traffic light ahead is red. The ego vehicle should remain in its lane and stop before entering the intersection."
    }

## Step 0：冻结数据格式和版本
正式定义：
  Raw dataset schema v0.1
  Annotation schema v0.1
  Sample schema v0.1
明确：
  clip 目录结构
  时间单位统一为纳秒
  坐标系含义
  camera 名称
  每类 JSON/JSONL 的字段
  navigation、meta-action、scene-fact 的枚举
  annotation version
  generator version
产物:
  schemas/
  ├── clip_schema.md
  ├── meta_action_schema.json
  ├── scene_fact_schema.json
  └── sample_schema.json

## Step 1：Build Clip Manifest
工作内容
  扫描所有clip，汇总：
    clip_id
    validation status
    起止仿真时间
    duration
    每路相机帧数
    每路图像时间范围
    route 是否存在
    executed path 点数
    完整 GT trajectory
    actor 数据
    VectorMap
    文件路径
    文件大小

同时做二次质量检查：
  JSON/JSONL 是否可读
  JPEG 和 timestamps 数量是否一致
  图像时间戳是否递增
  必需文件是否存在
  仿真时长是否正常
  route 是否可用

产物:
  建议先用：
    manifests/clips.jsonl
  稳定后可以转 Parquet, 每行类似：
    {
      "clip_id": "test_clip_001",
      "valid": true,
      "duration_sec": 19.6,
      "camera_counts": {
        "front_wide": 152,
        "front_tele": 155,
        "cross_left": 155,
        "cross_right": 149
      },
      "has_route": true,
      "has_complete_gt": true
    }

## Step 2：统一时间轴和数据读取 API
工作内容:
  实现一个统一的 DrivingClipReader：
    clip = DrivingClipReader(clip_path)

    clip.get_camera_frame(
        camera="front_wide",
        target_ns=t0,
        tolerance_ns=...
    )

    clip.get_ego_history(
        end_ns=t0,
        num_points=M,
    )

    clip.get_route_at(t0)
    clip.get_actors_at(t0)
    clip.get_future_trajectory(t0)
    clip.get_vector_map()

  这是后面所有生成器共享的基础组件。

需要实现：
  最近时间戳查找
  精确同步优先
  时间容差
  轨迹插值
  ego-local 坐标转换
  缺失数据处理
  缓存，避免重复解析大 JSONL

产物:
  dataset/
  ├── clip_reader.py
  ├── temporal_index.py
  ├── coordinate_utils.py
  └── tests/

## Step 3：生成候选 Anchor
工作内容:
  先用固定间隔生成候选 anchor，例如：每 0.5 秒一个候选时刻
  过滤：
    t0 - 0.5 s 找不到完整视觉历史
    t0 前没有足够 ego history
    ego state 不连贯
    t0 后没有所需 future GT
    三路或四路图像时间误差过大
    route 缺失
    clip 开始和结束边界

产物：
  annotations/candidate_anchors.jsonl
每个 clip 可能先产生约 25 至 35 个有效候选点。

## Step 4：Meta-action Generator
工作内容：
  把连续轨迹转换为离散驾驶行为。

输入：
  完整 future ego trajectory
  executed path
  speed
  acceleration
  yaw
  yaw rate
  可选 lane topology
输出：
  {
    "lateral_action": "turn_right",
    "longitudinal_action": "decelerate",
    "motion_state": "moving",
    "confidence": 0.96
  }

Lateral可选：
  keep_direction
  turn_left
  turn_right
  change_lane_left
  change_lane_right

Longitudinal：
  accelerate
  maintain_speed
  decelerate
  stop

主要工作：
  定义速度和加速度阈值
  定义 yaw 和路径曲率阈值
  定义停车判断
  平滑轨迹，避免噪声抖动
  处理动作转换
  可视化轨迹与标签
  人工检查典型样本

产物：
  annotations/meta_actions.jsonl

## Step 5：Keyframe Selector
工作内容：
  从候选 anchor 中选出有价值的时刻。

重点选择：
  动作发生转换
  开始加速或减速
  转弯开始前
  进入路口前
  停车前
  重新起步
  导航方向明显变化
  actor 交互发生变化
同时保留少量正常直行样本，避免数据只包含异常和动作转换。

选择策略：
  固定采样 baseline
  +
  meta-action transition anchors
  +
  类别平衡采样

产物：
  annotations/keyframes.jsonl

## Step 6：Navigation Text Generator
工作内容：
  把局部 route waypoint 转换为模型输入文本。

输入：
  /alpasim/route/model_input
  VectorMap
  当前 ego pose
输出：
  navigation:
    - straight
    - left
    - right
    - unknown
  <!-- Continue straight.
  Turn left ahead.
  Turn right ahead.
  Continue along the current lane.
  Turn right/left at the upcoming intersection.
  Follow the right/left branch ahead.
  Change to the right/left lane when safe. -->

注意不能泄漏：
  未来精确位置
  未来速度
  精确转向时间
  未来控制量

产物：
  annotations/navigation.jsonl

## Step 7：Scene-Fact Generator
工作内容：
  利用仿真真值生成候选场景事实：
    道路环境
    可见交通参与者
    相对位置
    相对运动
    交通标志和 wait line
    与驾驶决策相关的关键对象

暂时只做：
  road_context
  lead_vehicle presence
  left/right nearby vehicle
  relative motion category
  intersection proximity
  stop/yield-line proximity
例如：
  {
    "road_context": "intersection",
    "lead_vehicle": {
      "present": true,
      "relative_distance": "near",
      "relative_motion": "slower"
    },
    "left_vehicle": false,
    "right_vehicle": true
  }

可观察性过滤：
  需要避免把不可见的 simulator 真值写入文本监督。
  第一版可以采用：
    actor 是否位于任一选定相机视锥
    投影框是否在图像范围内
    投影面积是否超过阈值
    距离是否合理
    事实是否与决策相关
  严格遮挡检测可以留到后续版本。

产物:
  annotations/scene_facts.jsonl

## Step 8：结构化 CoC 构造
工作内容:
  把以下内容组合起来：
    Navigation
    Scene facts
    Meta-action
  形成结构化因果链：
    {
      "navigation": "turn_right",
      "critical_components": [
        "approaching intersection",
        "slower lead vehicle"
      ],
      "decision": {
        "lateral": "prepare_right_turn",
        "longitudinal": "decelerate"
      }
    }

## Step 9：Reasoning Generator
工作内容:
  把结构化 CoC 转换成短文本：
    The route requires a right turn at the upcoming intersection.
    A slower vehicle is ahead, so the ego vehicle should reduce speed
    while preparing for the turn.
  考虑模板生成，也可以升级为用教师大模型生成。

## Step 10：Sample Manifest Builder
工作内容：
  将每个 anchor 的所有内容装配为训练索引：
    {
      "sample_id": "test_clip_001_9305000000000",
      "clip_id": "test_clip_001",
      "anchor_ns": 9305000000000,
      "images": {
        "front_wide": ["...", "..."],
        "front_tele": ["...", "..."],
        "cross_left": ["...", "..."],
        "cross_right": ["...", "..."]
      },
      "ego_history": {
        "start_index": 40,
        "end_index": 47
      },
      "navigation_text": "Turn right ahead.",
      "target": {
        "scene_facts": {},
        "decision": {},
        "reasoning": "..."
      }
    }

产物：
  manifests/samples_v0.jsonl

## Step 11：Train/Validation/Test 划分
工作内容：
  必须按 clip 划分，不能随机按 anchor 划分。
  例如：
    train: 80%
    validation: 10%
    test: 10%
  检查各 split 中：
    turn left/right/straight
    accelerate/decelerate/stop
    路口/非路口
    actor 交互类别
  避免同一个 clip 的相邻 anchor 同时出现在训练集和验证集。

产物：
  manifests/train_v0.jsonl
  manifests/validation_v0.jsonl
  manifests/test_v0.jsonl

## Step 12：数据审核与统计
工作内容：
  自动检查：
    图像存在
    时间戳满足容差
    ego history 完整
    route 存在
    target JSON 可解析
    reasoning 与 meta-action 一致
    不可观察事实未进入 reasoning
    样本类别分布合理
  人工抽样：
    人工检查100至200 个 anchor
  重点覆盖：
    直行
    左转
    右转
    减速
    停车
    前车
    路口
    交通灯
    不同相机视角

产物：
  reports/dataset_statistics.json
  reports/manual_review.csv



