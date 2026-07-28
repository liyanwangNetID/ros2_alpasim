#!/usr/bin/env python3

"""Run VaVAM inference from ROS 2 observations and publish an ego trajectory.

VaVAM integration audit
========================

Model implementation
--------------------

The node reuses the official AlpaSim Driver implementation:

    alpasim_driver.models.vam_model.VAMModel

The underlying neural network is:

    vam.action_expert.VideoActionModelInference

Two local model assets are required:

    VAM_width_1024_pretrained_139k.pt
    VQ_ds16_16384_llamagen_encoder.jit

The checkpoint contains the VaVAM model configuration and model weights.
The tokenizer is loaded separately as a TorchScript module.

Camera input
------------

VaVAM requires exactly one camera:

    camera_front_wide_120fov

The official interactive VaVAM configuration uses:

    context_length = 1
    subsample_factor = 1

Therefore this ROS planning node uses the latest front-wide image for each
inference request.

The ROS image topic contains sensor_msgs/CompressedImage. The encoded image is
decoded to an HWC uint8 RGB NumPy array before being passed to VaVAM.

Camera rectification
--------------------

The rendered front-wide camera uses an f-theta camera model. VaVAM was trained
with a NuScenes-style pinhole camera model, so the raw rendered image must not
be sent directly to the model.

The source f-theta intrinsics are obtained from:

    /alpasim/camera/front_wide/calibration

The calibration message contains JSON serialized from the Runtime
AvailableCamera protobuf. The JSON is restored to:

    sensorsim_pb2.AvailableCamerasReturn.AvailableCamera

The node then reuses the official AlpaSim Driver rectifier:

    build_ftheta_rectifier_for_resolution()

The fixed target pinhole parameters are taken from the official VaVAM Driver
configuration:

    resolution:       1920 x 1080
    focal length:     1545.0, 1545.0
    principal point:  960.0, 560.0
    radial:           -0.356123, 0.172545, -0.05231, 0, 0, 0
    tangential:       -0.00213, 0.000464
    thin prism:       0, 0, 0, 0

After rectification, VAMModel performs its own official preprocessing:

    resize and center crop to 1600 x 900
    NeuroNCAPTransform
    VQ tokenizer
    VideoActionModelInference

Navigation command
------------------

The node subscribes to:

    /alpasim/route/model_input

The Route is expressed in the current base_link frame. Positive y is left.

The official command logic selects the first valid waypoint at least 20 metres
ahead and applies the following rule:

    waypoint y >  3.0 m: LEFT
    waypoint y < -3.0 m: RIGHT
    otherwise:           STRAIGHT

If no suitable waypoint is available, STRAIGHT is used.

The canonical DriveCommand values are converted inside VAMModel to the VaVAM
encoding:

    RIGHT:    0
    LEFT:     1
    STRAIGHT: 2

Trajectory output
-----------------

VaVAM produces six ego-relative future positions in the current rig frame.

The VAMModel output frequency is 2 Hz, so the points correspond to:

    0.5 s
    1.0 s
    1.5 s
    2.0 s
    2.5 s
    3.0 s

The total trajectory horizon is 3.0 seconds.

The model returns x/y offsets in the current ego rig frame. Headings are
computed from successive trajectory positions, with the origin treated as the
position preceding the first waypoint.

The ROS output contract is:

    topic:
        /alpasim/planning/ego/trajectory

    message:
        alpasim_msgs/msg/EgoTrajectory

    pose_frame_id:
        base_link

    dynamics_frame_id:
        base_link

The External Driver subscribes to this topic, stores the latest trajectory in
its shared ExternalTrajectoryBuffer, and returns it to the AlpaSim Runtime.
The AlpaSim Controller adds the current ego pose, transforms the relative
trajectory into the simulation local/map frame, and tracks it.

This node publishes only the six VaVAM future points. It does not add a point
at time zero.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from alpasim_driver.models.base import (
    DriveCommand,
    PredictionInput,
)
from alpasim_driver.models.vam_model import VAMModel
from alpasim_driver.rectification import (
    FthetaToPinholeRectifier,
    build_ftheta_rectifier_for_resolution,
)
from alpasim_driver.schema import RectificationTargetConfig
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_msgs.msg import (
    EgoTrajectory,
    Route,
    TrajectoryPoint,
)
from builtin_interfaces.msg import Duration
from google.protobuf.json_format import ParseDict
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


FRONT_WIDE_CAMERA_ID = "camera_front_wide_120fov"


def duration_from_seconds(value: float) -> Duration:
    """Convert non-negative seconds to a ROS Duration."""
    if value < 0.0:
        raise ValueError(
            "Duration value must be non-negative"
        )

    total_nanoseconds = int(
        round(value * 1_000_000_000.0)
    )

    message = Duration()
    message.sec = int(
        total_nanoseconds // 1_000_000_000
    )
    message.nanosec = int(
        total_nanoseconds % 1_000_000_000
    )

    return message


def ros_time_to_microseconds(stamp) -> int:
    """Convert a builtin_interfaces/Time value to microseconds."""
    return (
        int(stamp.sec) * 1_000_000
        + int(stamp.nanosec) // 1000
    )


class VaVAMTrajectoryPlanner(Node):
    """Use ROS camera and Route observations to run VaVAM inference."""

    def __init__(self) -> None:
        super().__init__(
            "vavam_trajectory_planner"
        )

        self.declare_parameter(
            "checkpoint_path",
            (
                "/home/lab/alpasim_ros2_ws/models/vavam/"
                "VAM_width_1024_pretrained_139k.pt"
            ),
        )
        self.declare_parameter(
            "tokenizer_path",
            (
                "/home/lab/alpasim_ros2_ws/models/vavam/"
                "VQ_ds16_16384_llamagen_encoder.jit"
            ),
        )
        self.declare_parameter(
            "device",
            "cuda",
        )
        self.declare_parameter(
            "camera_topic",
            (
                "/alpasim/camera/front_wide/"
                "image/compressed"
            ),
        )
        self.declare_parameter(
            "calibration_topic",
            (
                "/alpasim/camera/front_wide/"
                "calibration"
            ),
        )
        self.declare_parameter(
            "route_topic",
            "/alpasim/route/model_input",
        )
        self.declare_parameter(
            "output_topic",
            "/alpasim/planning/ego/trajectory",
        )
        self.declare_parameter(
            "inference_rate_hz",
            2.0,
        )
        self.declare_parameter(
            "command_distance_threshold_m",
            3.0,
        )
        self.declare_parameter(
            "minimum_lookahead_distance_m",
            20.0,
        )

        checkpoint_path = str(
            self.get_parameter(
                "checkpoint_path"
            ).value
        )
        tokenizer_path = str(
            self.get_parameter(
                "tokenizer_path"
            ).value
        )
        requested_device = str(
            self.get_parameter("device").value
        )
        camera_topic = str(
            self.get_parameter(
                "camera_topic"
            ).value
        )
        calibration_topic = str(
            self.get_parameter(
                "calibration_topic"
            ).value
        )
        route_topic = str(
            self.get_parameter("route_topic").value
        )
        output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )
        inference_rate_hz = float(
            self.get_parameter(
                "inference_rate_hz"
            ).value
        )

        self.command_distance_threshold_m = float(
            self.get_parameter(
                "command_distance_threshold_m"
            ).value
        )
        self.minimum_lookahead_distance_m = float(
            self.get_parameter(
                "minimum_lookahead_distance_m"
            ).value
        )

        if inference_rate_hz <= 0.0:
            raise ValueError(
                "inference_rate_hz must be positive"
            )

        if (
            requested_device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "VaVAM requested CUDA, but CUDA is not available"
            )

        self.device = torch.device(
            requested_device
        )

        self.get_logger().info(
            "Loading VaVAM model; this may take a while"
        )

        self.model = VAMModel(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=self.device,
            camera_ids=[FRONT_WIDE_CAMERA_ID],
            context_length=1,
        )

        self.get_logger().info(
            "VaVAM model and tokenizer loaded successfully"
        )

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        calibration_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            EgoTrajectory,
            output_topic,
            output_qos,
        )

        self.camera_subscription = (
            self.create_subscription(
                CompressedImage,
                camera_topic,
                self.camera_callback,
                image_qos,
            )
        )

        self.calibration_subscription = (
            self.create_subscription(
                String,
                calibration_topic,
                self.calibration_callback,
                calibration_qos,
            )
        )

        self.route_subscription = (
            self.create_subscription(
                Route,
                route_topic,
                self.route_callback,
                route_qos,
            )
        )

        self.latest_image_rgb: Optional[
            np.ndarray
        ] = None
        self.latest_image_timestamp_us: Optional[
            int
        ] = None
        self.latest_route: Optional[Route] = None

        self.camera_proto: Optional[
            sensorsim_pb2
            .AvailableCamerasReturn
            .AvailableCamera
        ] = None

        self.rectifier: Optional[
            FthetaToPinholeRectifier
        ] = None
        self.rectifier_source_shape: Optional[
            tuple[int, int]
        ] = None

        self.inference_in_progress = False
        self.inference_count = 0
        self.last_processed_image_timestamp_us: Optional[
            int
        ] = None

        self.timer = self.create_timer(
            1.0 / inference_rate_hz,
            self.run_inference,
        )

        self.get_logger().info(
            "VaVAM ROS planner initialized: "
            f"camera={camera_topic}, "
            f"calibration={calibration_topic}, "
            f"route={route_topic}, "
            f"output={output_topic}, "
            f"rate={inference_rate_hz:.2f} Hz"
        )

    def calibration_callback(
        self,
        message: String,
    ) -> None:
        """Restore Runtime camera metadata from calibration JSON."""
        if self.camera_proto is not None:
            return

        try:
            calibration = json.loads(
                message.data
            )

            available_camera_dict = calibration[
                "available_camera"
            ]

            camera_proto = (
                sensorsim_pb2
                .AvailableCamerasReturn
                .AvailableCamera()
            )

            ParseDict(
                available_camera_dict,
                camera_proto,
            )
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self.get_logger().error(
                f"Could not parse camera calibration: {exc}"
            )
            return

        if (
            camera_proto.logical_id
            != FRONT_WIDE_CAMERA_ID
        ):
            self.get_logger().error(
                "Unexpected calibration camera: "
                f"{camera_proto.logical_id!r}"
            )
            return

        camera_model = (
            camera_proto.intrinsics.WhichOneof(
                "camera_param"
            )
        )

        if camera_model != "ftheta_param":
            self.get_logger().error(
                "Front-wide calibration does not contain "
                f"f-theta intrinsics: {camera_model!r}"
            )
            return

        self.camera_proto = camera_proto

        self.get_logger().info(
            "Received front-wide calibration: "
            f"native_resolution="
            f"{camera_proto.intrinsics.resolution_w}x"
            f"{camera_proto.intrinsics.resolution_h}, "
            f"camera_model={camera_model}"
        )

    def camera_callback(
        self,
        message: CompressedImage,
    ) -> None:
        """Decode the latest compressed front-wide image as RGB uint8."""
        encoded = np.frombuffer(
            bytes(message.data),
            dtype=np.uint8,
        )

        image_bgr = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            self.get_logger().warning(
                "Could not decode front-wide compressed image"
            )
            return

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        self.latest_image_rgb = (
            np.ascontiguousarray(image_rgb)
        )
        self.latest_image_timestamp_us = (
            ros_time_to_microseconds(
                message.header.stamp
            )
        )

    def route_callback(
        self,
        message: Route,
    ) -> None:
        """Store the latest base_link-frame model-input Route."""
        if message.frame_id != "base_link":
            self.get_logger().warning(
                "Ignoring Route in unexpected frame "
                f"{message.frame_id!r}"
            )
            return

        self.latest_route = message

    def build_rectifier(
        self,
        source_image: np.ndarray,
    ) -> bool:
        """Build the official f-theta to VaVAM pinhole rectifier."""
        if self.camera_proto is None:
            return False

        source_height = int(
            source_image.shape[0]
        )
        source_width = int(
            source_image.shape[1]
        )

        source_shape = (
            source_height,
            source_width,
        )

        if (
            self.rectifier is not None
            and self.rectifier_source_shape
            == source_shape
        ):
            return True

        target_config = RectificationTargetConfig(
            focal_length=(
                1545.0,
                1545.0,
            ),
            principal_point=(
                960.0,
                560.0,
            ),
            resolution_hw=(
                1080,
                1920,
            ),
            radial=(
                -0.356123,
                0.172545,
                -0.05231,
                0.0,
                0.0,
                0.0,
            ),
            tangential=(
                -0.00213,
                0.000464,
            ),
            thin_prism=(
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            max_overscan_scale=2.0,
            safety_margin_px=10,
        )

        try:
            self.rectifier = (
                build_ftheta_rectifier_for_resolution(
                    camera_proto=self.camera_proto,
                    target_cfg=target_config,
                    source_resolution_hw=source_shape,
                )
            )
        except (ValueError, RuntimeError) as exc:
            self.get_logger().error(
                f"Could not build image rectifier: {exc}"
            )
            self.rectifier = None
            return False

        self.rectifier_source_shape = (
            source_shape
        )

        self.get_logger().info(
            "Built front-wide rectifier: "
            f"source={source_width}x{source_height}, "
            "target=1920x1080"
        )

        return True

    def determine_command(self) -> DriveCommand:
        """Derive LEFT, STRAIGHT or RIGHT from the model-input Route."""
        route = self.latest_route

        if route is None:
            return DriveCommand.STRAIGHT

        target_point = None

        for route_point in route.points:
            if not route_point.valid:
                continue

            x = float(
                route_point.position.x
            )
            y = float(
                route_point.position.y
            )

            distance = math.hypot(x, y)

            if (
                distance
                >= self.minimum_lookahead_distance_m
            ):
                target_point = route_point
                break

        if target_point is None:
            return DriveCommand.STRAIGHT

        lateral_displacement = float(
            target_point.position.y
        )

        if (
            lateral_displacement
            > self.command_distance_threshold_m
        ):
            return DriveCommand.LEFT

        if (
            lateral_displacement
            < -self.command_distance_threshold_m
        ):
            return DriveCommand.RIGHT

        return DriveCommand.STRAIGHT

    def run_inference(self) -> None:
        """Run VaVAM once using the latest complete observation set."""
        if self.inference_in_progress:
            return

        image_rgb = self.latest_image_rgb
        image_timestamp_us = (
            self.latest_image_timestamp_us
        )

        if (
            image_rgb is None
            or image_timestamp_us is None
            or self.camera_proto is None
        ):
            return

        if (
            self.last_processed_image_timestamp_us
            == image_timestamp_us
        ):
            return

        if not self.build_rectifier(
            image_rgb
        ):
            return

        assert self.rectifier is not None

        self.inference_in_progress = True

        try:
            rectified_rgb = (
                self.rectifier.rectify(
                    image_rgb
                )
            )

            command = self.determine_command()

            prediction_input = PredictionInput(
                camera_images={
                    FRONT_WIDE_CAMERA_ID: [
                        (
                            image_timestamp_us,
                            rectified_rgb,
                        )
                    ]
                },
                command=command,
                speed=0.0,
                acceleration=0.0,
                ego_pose_history=[],
            )

            prediction = self.model.predict(
                prediction_input
            )

            self.publish_prediction(
                prediction=prediction,
                reference_timestamp_us=(
                    image_timestamp_us
                ),
                command=command,
            )

            self.last_processed_image_timestamp_us = (
                image_timestamp_us
            )

        except Exception as exc:
            self.get_logger().error(
                "VaVAM inference failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            self.inference_in_progress = False

    def publish_prediction(
        self,
        prediction,
        reference_timestamp_us: int,
        command: DriveCommand,
    ) -> None:
        """Convert one VaVAM ModelPrediction to EgoTrajectory."""
        trajectory_xy = np.asarray(
            prediction.trajectory_xy,
            dtype=np.float64,
        )
        headings = np.asarray(
            prediction.headings,
            dtype=np.float64,
        )

        if trajectory_xy.shape != (6, 2):
            raise ValueError(
                "VaVAM trajectory must have shape "
                f"(6, 2), got {trajectory_xy.shape}"
            )

        if headings.shape != (6,):
            raise ValueError(
                "VaVAM headings must have shape "
                f"(6,), got {headings.shape}"
            )

        if not np.all(
            np.isfinite(trajectory_xy)
        ):
            raise ValueError(
                "VaVAM trajectory contains non-finite values"
            )

        if not np.all(
            np.isfinite(headings)
        ):
            raise ValueError(
                "VaVAM headings contain non-finite values"
            )

        message = EgoTrajectory()

        message.reference_stamp.sec = int(
            reference_timestamp_us
            // 1_000_000
        )
        message.reference_stamp.nanosec = int(
            (
                reference_timestamp_us
                % 1_000_000
            )
            * 1000
        )

        message.pose_frame_id = "base_link"
        message.dynamics_frame_id = "base_link"

        message.source = (
            EgoTrajectory.SOURCE_MODEL_PLANNING
        )
        message.producer = "vavam_ros_planner"
        message.is_model_generated = True
        message.force_gt_active = False

        horizon_seconds = 3.0

        message.requested_duration = (
            duration_from_seconds(
                horizon_seconds
            )
        )
        message.actual_duration = (
            duration_from_seconds(
                horizon_seconds
            )
        )

        points: list[TrajectoryPoint] = []

        for index, (
            position_xy,
            heading,
        ) in enumerate(
            zip(
                trajectory_xy,
                headings,
                strict=True,
            ),
            start=1,
        ):
            time_seconds = index * 0.5

            point = TrajectoryPoint()

            point.time_from_reference = (
                duration_from_seconds(
                    time_seconds
                )
            )

            point.pose.position.x = float(
                position_xy[0]
            )
            point.pose.position.y = float(
                position_xy[1]
            )
            point.pose.position.z = 0.0

            point.pose.orientation.x = 0.0
            point.pose.orientation.y = 0.0
            point.pose.orientation.z = math.sin(
                float(heading) / 2.0
            )
            point.pose.orientation.w = math.cos(
                float(heading) / 2.0
            )

            point.yaw = float(heading)

            points.append(point)

        message.points = points

        self.publisher.publish(message)

        self.inference_count += 1

        if self.inference_count == 1:
            self.get_logger().info(
                "Published first VaVAM trajectory: "
                f"command={command.name}, "
                f"points={len(points)}, "
                f"final_x={trajectory_xy[-1, 0]:.3f}, "
                f"final_y={trajectory_xy[-1, 1]:.3f}, "
                "horizon=3.0 s"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VaVAMTrajectoryPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
