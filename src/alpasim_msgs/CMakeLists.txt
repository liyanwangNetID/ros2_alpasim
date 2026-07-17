cmake_minimum_required(VERSION 3.8)
project(alpasim_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EgoState.msg"
  "msg/ActorState.msg"
  "msg/ActorStateArray.msg"
  "msg/TrajectoryPoint.msg"
  "msg/ActorTrajectory.msg"
  "msg/ActorTrajectoryArray.msg"

  "msg/MapPolyline.msg"
  "msg/MapLane.msg"
  "msg/MapRoadEdge.msg"
  "msg/MapTrafficSign.msg"
  "msg/MapWaitLine.msg"
  "msg/VectorMap.msg"

  "srv/GetVectorMap.srv"

  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()