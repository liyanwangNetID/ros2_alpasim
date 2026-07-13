#!/usr/bin/env python3

import json
import socket

import rclpy
from rclpy.node import Node

from alpasim_msgs.msg import EgoState


class EgoStatePublisher(Node):
    def __init__(self):
        super().__init__("alpasim_ego_state_publisher")

        self.declare_parameter("bind_host", "127.0.0.1")
        self.declare_parameter("bind_port", 15000)
        self.declare_parameter("topic", "/alpasim/ego_state")

        host = self.get_parameter("bind_host").value
        port = int(self.get_parameter("bind_port").value)
        topic = self.get_parameter("topic").value

        self.publisher = self.create_publisher(EgoState, topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

        # High-frequency polling of local UDP; returns immediately when no data is available, without blocking the ROS executor.
        self.timer = self.create_timer(0.001, self.poll_udp)

        self.received_count = 0
        self.get_logger().info(
            f"Listening on udp://{host}:{port}; publishing {topic}"
        )

    def poll_udp(self):
        while True:
            try:
                payload, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError as exc:
                self.get_logger().error(f"UDP receive error: {exc}")
                break

            try:
                state = json.loads(payload.decode("utf-8"))
                self.publish_state(state)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                self.get_logger().warning(f"Invalid ego-state packet: {exc}")

    def publish_state(self, state):
        msg = EgoState()

        timestamp_us = int(state["timestamp_us"])
        msg.stamp.sec = timestamp_us // 1_000_000
        msg.stamp.nanosec = (timestamp_us % 1_000_000) * 1000

        msg.frame_id = str(state.get("frame_id", "alpasim_local"))
        msg.x = float(state["x"])
        msg.y = float(state["y"])
        msg.z = float(state["z"])
        msg.yaw = float(state["yaw"])
        msg.speed = float(state["speed"])

        self.publisher.publish(msg)
        self.received_count += 1

        if self.received_count % 20 == 0:
            self.get_logger().info(
                f"Published {self.received_count} states: "
                f"t={timestamp_us / 1e6:.3f}s "
                f"xyz=({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}) "
                f"yaw={msg.yaw:.3f} speed={msg.speed:.2f}"
            )

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EgoStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
