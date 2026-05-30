#!/usr/bin/env python3
import math
import sys
import time
import subprocess

LCM_PYTHON_PATH = "/home/lcm/build/python"
CONTROL_PATH = "/home/loco_example/loco_hl_example/sequential_motion"

if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)
if CONTROL_PATH not in sys.path:
    sys.path.insert(0, CONTROL_PATH)

import lcm
import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from robot_control_cmd_lcmt import robot_control_cmd_lcmt


class SlopeWalk(Node):
    def __init__(self):
        super().__init__("slope_walk")

        self.declare_parameter("model_name", "robot")
        self.declare_parameter("control_period", 0.1)

        self.declare_parameter("spawn_x", 3.046)
        self.declare_parameter("spawn_y", 12.349)
        self.declare_parameter("spawn_z", 0.228)
        self.declare_parameter("spawn_qx", -0.0013)
        self.declare_parameter("spawn_qy", 0.1013)
        self.declare_parameter("spawn_qz", 0.9948)
        self.declare_parameter("spawn_qw", 0.0082)

        self.declare_parameter("point1_x", -0.088)
        self.declare_parameter("point1_y", 12.435)
        self.declare_parameter("point1_z", 0.211)
        self.declare_parameter("point2_x", -0.333)
        self.declare_parameter("point2_y", 15.096)
        self.declare_parameter("point2_z", 0.207)
        self.declare_parameter("point3_x", 2.982)
        self.declare_parameter("point3_y", 15.478)
        self.declare_parameter("point3_z", 0.201)

        self.declare_parameter("pitch_angle", -0.18)
        self.declare_parameter("turn_angle", 90.0)

        self.declare_parameter("forward_speed_slope", 0.25)
        self.declare_parameter("body_height_slope", 0.25)
        self.declare_parameter("step_height_slope", 0.04)
        self.declare_parameter("roll_angle", 0.40)

        self.model_name = self.get_parameter("model_name").value
        self.control_period = float(self.get_parameter("control_period").value)

        self.spawn_x = float(self.get_parameter("spawn_x").value)
        self.spawn_y = float(self.get_parameter("spawn_y").value)
        self.spawn_z = float(self.get_parameter("spawn_z").value)
        self.spawn_qx = float(self.get_parameter("spawn_qx").value)
        self.spawn_qy = float(self.get_parameter("spawn_qy").value)
        self.spawn_qz = float(self.get_parameter("spawn_qz").value)
        self.spawn_qw = float(self.get_parameter("spawn_qw").value)

        self.points = [
            (float(self.get_parameter("point1_x").value), float(self.get_parameter("point1_y").value), float(self.get_parameter("point1_z").value)),
            (float(self.get_parameter("point2_x").value), float(self.get_parameter("point2_y").value), float(self.get_parameter("point2_z").value)),
            (float(self.get_parameter("point3_x").value), float(self.get_parameter("point3_y").value), float(self.get_parameter("point3_z").value)),
        ]

        self.pitch_angle = float(self.get_parameter("pitch_angle").value)
        self.turn_angle = float(self.get_parameter("turn_angle").value)

        self.forward_speed_slope = float(self.get_parameter("forward_speed_slope").value)
        self.body_height_slope = float(self.get_parameter("body_height_slope").value)
        self.step_height_slope = float(self.get_parameter("step_height_slope").value)
        self.roll_angle = float(self.get_parameter("roll_angle").value)

        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = self.make_cmd()

        self.latest_pose = None
        self.received_pose = False
        self.state = "SPAWN"
        self.final_stopped = False
        self.last_report_time = 0

        self.path_segment = 0
        self.turn_start_yaw = None

        self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self.on_model_states,
            qos_profile_sensor_data,
        )
        self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info("=== SLOPE WALK (Segment 2 Only) ===")
        self.get_logger().info("Spawn: (%.3f, %.3f, %.3f)" % (self.spawn_x, self.spawn_y, self.spawn_z))
        self.get_logger().info("Speed: %.2f | BodyH: %.2f | StepH: %.2f | Roll: %.2f" % (
            self.forward_speed_slope, self.body_height_slope, self.step_height_slope, self.roll_angle))
        for i, p in enumerate(self.points):
            self.get_logger().info("Point%d: (%.3f, %.3f, %.3f)" % (i+1, p[0], p[1], p[2]))

    def make_cmd(self):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = 27
        msg.contact = 15
        msg.duration = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.rpy_des = [0.0, self.pitch_angle, 0.0]
        msg.pos_des = [0.0, 0.0, self.body_height_slope]
        msg.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.ctrl_point = [0.0, 0.0, 0.0]
        msg.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.step_height = [self.step_height_slope, self.step_height_slope]
        return msg

    def publish_cmd_slope(self, vx, vy, yaw):
        self.cmd.mode = 11
        self.cmd.gait_id = 27
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [float(vx), float(vy), float(yaw)]
        self.cmd.rpy_des = [self.roll_angle, self.pitch_angle, 0.0]
        self.cmd.pos_des = [0.0, 0.0, self.body_height_slope]
        self.cmd.step_height = [self.step_height_slope, self.step_height_slope]
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def publish_cmd_flat(self, vx, vy, yaw):
        self.cmd.mode = 11
        self.cmd.gait_id = 27
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [float(vx), float(vy), float(yaw)]
        self.cmd.rpy_des = [0.0, 0.0, 0.0]
        self.cmd.pos_des = [0.0, 0.0, self.body_height_slope]
        self.cmd.step_height = [0.0, 0.0]
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def stand_stay(self):
        self.cmd.mode = 11
        self.cmd.gait_id = 27
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.rpy_des = [0.0, 0.0, 0.0]
        self.cmd.pos_des = [0.0, 0.0, self.body_height_slope]
        self.cmd.step_height = [0.0, 0.0]
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def recovery_stand(self):
        self.get_logger().info("Recovery stand...")
        self.cmd.mode = 12
        self.cmd.gait_id = 0
        self.cmd.contact = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.duration = 5000

        end_time = time.time() + 5.0
        while time.time() < end_time:
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            time.sleep(0.05)

        self.cmd = self.make_cmd()

    def set_entity_state(self):
        self.get_logger().info("Spawning robot to (%.3f, %.3f, %.3f)..." % (self.spawn_x, self.spawn_y, self.spawn_z))

        cmd = [
            "ros2", "service", "call", "/gazebo/set_entity_state",
            "gazebo_msgs/srv/SetEntityState",
            "{state: {name: 'robot', pose: {position: {x: %.3f, y: %.3f, z: %.3f}, orientation: {x: %.4f, y: %.4f, z: %.4f, w: %.4f}}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}, reference_frame: 'world'}}"
            % (self.spawn_x, self.spawn_y, self.spawn_z,
               self.spawn_qx, self.spawn_qy, self.spawn_qz, self.spawn_qw)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if "success" in result.stdout.lower():
                self.get_logger().info("Spawn successful!")
                return True
            else:
                self.get_logger().warn("Spawn may have failed: %s" % result.stdout)
                return False
        except Exception as e:
            self.get_logger().error("Failed to spawn: %s" % str(e))
            return False

    def on_model_states(self, msg):
        try:
            index = list(msg.name).index(self.model_name)
        except ValueError:
            self.received_pose = False
            return

        pose = msg.pose[index]
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.latest_pose = (pose.position.x, pose.position.y, pose.position.z, yaw)
        self.received_pose = True

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def check_point_reached(self, x, y, z, target_idx):
        if target_idx >= len(self.points):
            return False
        tx, ty, tz = self.points[target_idx]
        if tx == 0.0 and ty == 0.0 and tz == 0.0:
            return False
        xy_dist = math.sqrt((x - tx)**2 + (y - ty)**2)
        return xy_dist <= 0.3

    def control_loop(self):
        if self.final_stopped:
            self.stand_stay()
            return

        if self.state == "SPAWN":
            if self.set_entity_state():
                time.sleep(1.0)
                self.state = "RECOVERY"
                self.get_logger().info("State: SPAWN -> RECOVERY")
            else:
                self.get_logger().warn("Spawn failed, retrying...")
                time.sleep(1.0)
            return

        if self.state == "RECOVERY":
            self.recovery_stand()
            self.state = "WALK"
            self.path_segment = 1
            self.get_logger().info("State: RECOVERY -> WALK (seg=1)")
            return

        if not self.received_pose or self.latest_pose is None:
            self.publish_cmd_slope(0.0, 0.0, 0.0)
            return

        x, y, z, yaw = self.latest_pose

        now = time.time()
        if now - self.last_report_time >= 0.5:
            self.get_logger().info("POSITION | x=%.3f | y=%.3f | z=%.3f | yaw=%.3f | seg=%d" % (
                x, y, z, yaw, self.path_segment))
            self.last_report_time = now

        if self.path_segment == 1:
            if self.check_point_reached(x, y, z, 0):
                self.path_segment = 2
                self.turn_start_yaw = yaw
                self.get_logger().info("SEGMENT: 1 -> 2 | Reached point1, turning right...")
            else:
                self.publish_cmd_slope(self.forward_speed_slope, 0.0, 0.0)
                self.get_logger().info("WALKING | seg=1 | toward point1")

        elif self.path_segment == 2:
            yaw_diff = self.normalize_angle(yaw - self.turn_start_yaw)
            target_yaw_diff = math.radians(self.turn_angle)

            if abs(yaw_diff) >= target_yaw_diff:
                self.path_segment = 3
                self.get_logger().info("SEGMENT: 2 -> 3 | Turn complete, going to point2")
            else:
                turn_vel = -0.6
                self.publish_cmd_slope(0.0, 0.0, turn_vel)
                self.get_logger().info("TURNING | seg=2 | yaw=%.3f | diff=%.3f/%.3f" % (yaw, yaw_diff, target_yaw_diff))

        elif self.path_segment == 3:
            if self.check_point_reached(x, y, z, 1):
                self.path_segment = 4
                self.turn_start_yaw = yaw
                self.get_logger().info("SEGMENT: 3 -> 4 | Reached point2, turning right...")
            else:
                self.publish_cmd_slope(self.forward_speed_slope, 0.0, 0.0)
                self.get_logger().info("WALKING | seg=3 | toward point2")

        elif self.path_segment == 4:
            yaw_diff = self.normalize_angle(yaw - self.turn_start_yaw)
            target_yaw_diff = math.radians(self.turn_angle)

            if abs(yaw_diff) >= target_yaw_diff:
                self.path_segment = 5
                self.get_logger().info("SEGMENT: 4 -> 5 | Turn complete, going to point3")
            else:
                turn_vel = -0.6
                self.publish_cmd_slope(0.0, 0.0, turn_vel)
                self.get_logger().info("TURNING | seg=4 | yaw=%.3f | diff=%.3f/%.3f" % (yaw, yaw_diff, target_yaw_diff))

        elif self.path_segment == 5:
            if self.check_point_reached(x, y, z, 2):
                self.path_segment = 6
                self.turn_start_yaw = yaw
                self.get_logger().info("SEGMENT: 5 -> 6 | Reached point3, turning right...")
            else:
                self.publish_cmd_slope(self.forward_speed_slope, 0.0, 0.0)
                self.get_logger().info("WALKING | seg=5 | toward point3")

        elif self.path_segment == 6:
            yaw_diff = self.normalize_angle(yaw - self.turn_start_yaw)
            target_yaw_diff = math.radians(self.turn_angle)

            if abs(yaw_diff) >= target_yaw_diff:
                self.path_segment = 7
                self.get_logger().info("SEGMENT: 6 -> 7 | Turn complete, going to end (FLAT)")
            else:
                turn_vel = -0.6
                self.publish_cmd_flat(0.0, 0.0, turn_vel)
                self.get_logger().info("TURNING | seg=6 | yaw=%.3f | diff=%.3f/%.3f" % (yaw, yaw_diff, target_yaw_diff))

        elif self.path_segment == 7:
            self.publish_cmd_flat(self.forward_speed_slope, 0.0, 0.0)
            self.get_logger().info("WALKING | seg=7 | toward end (FLAT)")

        elif self.path_segment == 8:
            self.publish_cmd_flat(0.0, 0.0, 0.0)
            self.final_stopped = True
            self.get_logger().info("FINISHED | All segments complete!")


def main():
    rclpy.init()
    node = SlopeWalk()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
