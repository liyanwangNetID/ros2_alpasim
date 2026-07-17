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