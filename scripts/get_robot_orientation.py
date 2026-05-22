#!/usr/bin/env python3
"""获取机器人当前朝向（yaw 角），通过 Gazebo get_entity_state 服务。"""
import math
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState


def quat_to_yaw(q):
    """四元数 → yaw 角（弧度）"""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def main():
    rclpy.init()
    node = Node("get_orientation")
    cli = node.create_client(GetEntityState, "/gazebo/get_entity_state")

    if not cli.wait_for_service(timeout_sec=3.0):
        node.get_logger().error("无法连接 /gazebo/get_entity_state 服务")
        return

    req = GetEntityState.Request()
    req.name = "robot"
    req.reference_frame = "world"

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is None:
        node.get_logger().error("服务调用失败")
        return

    pose = future.result().state.pose
    x, y, z = pose.position.x, pose.position.y, pose.position.z
    q = pose.orientation
    yaw = quat_to_yaw(q)
    yaw_deg = math.degrees(yaw)

    node.get_logger().info(f"位置: x={x:.4f}  y={y:.4f}  z={z:.4f}")
    node.get_logger().info(f"朝向(四元数):  x={q.x:.4f}  y={q.y:.4f}  z={q.z:.4f}  w={q.w:.4f}")
    node.get_logger().info(f"朝向(yaw):  {yaw:.4f} rad  ({yaw_deg:.2f}°)")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
