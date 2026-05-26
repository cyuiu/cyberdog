#!/usr/bin/env python3
import math
import sys
import time

LCM_PYTHON_PATH = "/home/lcm/build/python"
CONTROL_PATH = "/home/loco_example/sequential_motion"

if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)
if CONTROL_PATH not in sys.path:
    sys.path.insert(0, CONTROL_PATH)

import lcm
import rclpy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from robot_control_cmd_lcmt import robot_control_cmd_lcmt


BALL_GRID = {
    (1, 1): (-0.4, 1.34),
    (1, 2): (-0.4, 2.18),
    (1, 3): (-0.4, 3.02),
    (1, 4): (-0.4, 3.86),
    (2, 1): (0.8, 1.34),
    (2, 2): (0.8, 2.18),
    (2, 3): (0.8, 3.02),
    (2, 4): (0.8, 3.86),
    (3, 1): (2.0, 1.34),
    (3, 2): (2.0, 2.18),
    (3, 3): (2.0, 3.02),
    (3, 4): (2.0, 3.86),
    (4, 1): (3.2, 1.34),
    (4, 2): (3.2, 2.18),
    (4, 3): (3.2, 3.02),
    (4, 4): (3.2, 3.86),
}


class HitOrangeBallsBase(Node):
    def __init__(self):
        super().__init__("hit_orangeballs_base")

        self.declare_parameter("model_states_topic", "/gazebo/model_states")
        self.declare_parameter("model_name", "robot")
        self.declare_parameter("target_cells", "4,3;3,2;2,1;1,4")

        self.declare_parameter("reset_start_pose", True)
        self.declare_parameter("start_x", 3.0985)
        self.declare_parameter("start_y", 0.6795)
        self.declare_parameter("start_z", 0.0542)
        self.declare_parameter("start_yaw", 1.417)
        self.declare_parameter("finish_x", -0.3083)
        self.declare_parameter("finish_y", 4.2618)
        self.declare_parameter("finish_z", 0.0542)
        self.declare_parameter("finish_yaw", 1.558)

        self.declare_parameter("goto_speed", 0.16)
        self.declare_parameter("goto_turn_speed", 0.16)
        self.declare_parameter("goto_pos_deadband", 0.08)
        self.declare_parameter("goto_yaw_deadband", 0.08)

        self.declare_parameter("normal_hit_offset_x", 0.55)
        self.declare_parameter("normal_hit_offset_y", 0.0)
        self.declare_parameter("normal_hit_yaw", 3.14)

        self.declare_parameter("special_column", 4)
        self.declare_parameter("special_hit_offset_x", -0.50)
        self.declare_parameter("special_hit_offset_y", 0.0)
        self.declare_parameter("special_hit_yaw", 3.14)

        self.declare_parameter("impact_speed", 0.60)
        self.declare_parameter("impact_time", 1.4)
        self.declare_parameter("reverse_impact_speed", 0.60)
        self.declare_parameter("reverse_impact_time", 1.5)

        self.declare_parameter("log_period", 5.0)
        self.declare_parameter("control_period", 0.1)
        self.declare_parameter("max_total_time", 250.0)

        self.model_states_topic = self.get_parameter("model_states_topic").value
        self.model_name = self.get_parameter("model_name").value
        self.target_cells_text = self.get_parameter("target_cells").value

        self.reset_start_pose = bool(self.get_parameter("reset_start_pose").value)
        self.start_x = float(self.get_parameter("start_x").value)
        self.start_y = float(self.get_parameter("start_y").value)
        self.start_z = float(self.get_parameter("start_z").value)
        self.start_yaw = float(self.get_parameter("start_yaw").value)
        self.finish_x = float(self.get_parameter("finish_x").value)
        self.finish_y = float(self.get_parameter("finish_y").value)
        self.finish_z = float(self.get_parameter("finish_z").value)
        self.finish_yaw = float(self.get_parameter("finish_yaw").value)

        self.goto_speed = float(self.get_parameter("goto_speed").value)
        self.goto_turn_speed = float(self.get_parameter("goto_turn_speed").value)
        self.goto_pos_deadband = float(self.get_parameter("goto_pos_deadband").value)
        self.goto_yaw_deadband = float(self.get_parameter("goto_yaw_deadband").value)

        self.normal_hit_offset_x = float(
            self.get_parameter("normal_hit_offset_x").value
        )
        self.normal_hit_offset_y = float(
            self.get_parameter("normal_hit_offset_y").value
        )
        self.normal_hit_yaw = float(self.get_parameter("normal_hit_yaw").value)

        self.special_column = int(self.get_parameter("special_column").value)
        self.special_hit_offset_x = float(
            self.get_parameter("special_hit_offset_x").value
        )
        self.special_hit_offset_y = float(
            self.get_parameter("special_hit_offset_y").value
        )
        self.special_hit_yaw = float(self.get_parameter("special_hit_yaw").value)

        self.impact_speed = float(self.get_parameter("impact_speed").value)
        self.impact_time = float(self.get_parameter("impact_time").value)
        self.reverse_impact_speed = float(
            self.get_parameter("reverse_impact_speed").value
        )
        self.reverse_impact_time = float(
            self.get_parameter("reverse_impact_time").value
        )

        self.log_period = float(self.get_parameter("log_period").value)
        self.control_period = float(self.get_parameter("control_period").value)
        self.max_total_time = float(self.get_parameter("max_total_time").value)
        self.post_hit_pause_time = 0.5
        self.recovery_stand_time = 5.0

        self.ball_grid = dict(BALL_GRID)
        self.target_sequence = self.parse_target_cells(self.target_cells_text)

        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = self.make_cmd()

        self.latest_pose = None
        self.received_pose = False

        self.state = "START"
        self.state_start_time = time.time()
        self.total_start_time = self.state_start_time
        self.final_stopped = False
        self.done_reason = ""
        self.hit_count = 0
        self.last_log_times = {}

        self.current_target_index = 0
        self.current_target_cell = None
        self.current_target_ball = None
        self.current_hit_point = None
        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.finish_waypoints = []
        self.finish_waypoint_index = 0
        self.hit_reverse = False

        self.reset_client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        self.reset_future = None
        self.reset_requested = False

        self.create_subscription(
            ModelStates,
            self.model_states_topic,
            self.on_model_states,
            qos_profile_sensor_data,
        )
        self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            "节点启动：model=%s | target_cells=%s | 目标序列=%s | reset_start_pose=%s | 起始位姿=(%.4f, %.4f, %.4f, %.3f) | 终点位姿=(%.4f, %.4f, %.4f, %.3f) | goto=(%.2f, %.2f, %.2f, %.2f) | 普通撞击偏移=(%.3f, %.3f, %.3f) | 特殊列=%d | 特殊撞击偏移=(%.3f, %.3f, %.3f) | 撞击参数=(%.2f, %.2f, %.2f, %.2f) | control_period=%.2f | max_total_time=%.1f"
            % (
                self.model_name,
                self.target_cells_text,
                str(self.target_sequence),
                str(self.reset_start_pose),
                self.start_x,
                self.start_y,
                self.start_z,
                self.start_yaw,
                self.finish_x,
                self.finish_y,
                self.finish_z,
                self.finish_yaw,
                self.goto_speed,
                self.goto_turn_speed,
                self.goto_pos_deadband,
                self.goto_yaw_deadband,
                self.normal_hit_offset_x,
                self.normal_hit_offset_y,
                self.normal_hit_yaw,
                self.special_column,
                self.special_hit_offset_x,
                self.special_hit_offset_y,
                self.special_hit_yaw,
                self.impact_speed,
                self.impact_time,
                self.reverse_impact_speed,
                self.reverse_impact_time,
                self.control_period,
                self.max_total_time,
            )
        )

        if not self.target_sequence:
            self.get_logger().warning("target_cells 为空，恢复站立后将直接结束。")

    @staticmethod
    def parse_target_cells(text):
        if text is None:
            return []

        text = str(text).strip()
        if not text:
            return []

        cells = []
        for index, item in enumerate(text.split(";"), start=1):
            part = item.strip()
            if not part:
                continue

            pieces = [piece.strip() for piece in part.split(",")]
            if len(pieces) != 2:
                raise ValueError(
                    "target_cells 第 %d 项格式错误，应为 column,row：%s"
                    % (index, part)
                )

            try:
                column = int(pieces[0])
                row = int(pieces[1])
            except ValueError as exc:
                raise ValueError(
                    "target_cells 第 %d 项不是整数：%s" % (index, part)
                ) from exc

            if (column, row) not in BALL_GRID:
                raise ValueError(
                    "target_cells 第 %d 项不在 BALL_GRID 中：(%d,%d)"
                    % (index, column, row)
                )

            cells.append((column, row))

        return cells

    def make_cmd(self):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = 3
        msg.contact = 15
        msg.duration = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        msg.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.ctrl_point = [0.0, 0.0, 0.0]
        msg.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.step_height = [0.06, 0.06]
        return msg

    def publish_cmd(self, vx, vy, yaw):
        self.cmd.mode = 11
        self.cmd.gait_id = 3
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [float(vx), float(vy), float(yaw)]
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def publish_recovery_stand_cmd(self):
        self.cmd.mode = 12
        self.cmd.gait_id = 0
        self.cmd.contact = 0
        self.cmd.duration = 5000
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def damper_stop(self):
        self.cmd.mode = 7
        self.cmd.gait_id = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.duration = 0
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    @staticmethod
    def yaw_to_quaternion(yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return Quaternion(x=0.0, y=0.0, z=qz, w=qw)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def world_to_body_cmd(dx, dy, yaw, max_speed):
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return 0.0, 0.0

        world_vx = max_speed * dx / dist
        world_vy = max_speed * dy / dist
        body_vx = math.cos(yaw) * world_vx + math.sin(yaw) * world_vy
        body_vy = -math.sin(yaw) * world_vx + math.cos(yaw) * world_vy
        return body_vx, body_vy

    @staticmethod
    def format_target(cell):
        if cell is None:
            return "None"
        return "(%d,%d)" % (cell[0], cell[1])

    @staticmethod
    def format_ball(ball):
        if ball is None:
            return "None"
        return "(%.3f, %.3f)" % (ball[0], ball[1])

    @staticmethod
    def format_hit_point(hit_point):
        if hit_point is None:
            return "None"
        return "(%.3f, %.3f, %.3f)" % (hit_point[0], hit_point[1], hit_point[2])

    @staticmethod
    def format_waypoints(waypoints):
        if not waypoints:
            return "[]"
        return "[%s]" % ", ".join(
            "(%.3f, %.3f, %.3f)" % (point[0], point[1], point[2])
            for point in waypoints
        )

    def throttled_info(self, key, message):
        now = time.time()
        last = self.last_log_times.get(key)
        if last is None or now - last >= self.log_period:
            self.last_log_times[key] = now
            self.get_logger().info(message)

    def switch_state(self, state):
        old_state = self.state
        self.state = state
        self.state_start_time = time.time()
        self.get_logger().info("状态切换：%s -> %s" % (old_state, state))

        if state == "DONE":
            if self.done_reason == "finished":
                self.get_logger().info(
                    "任务结束：原因=finished | 已撞击=%d/%d | 已到达终点"
                    % (self.hit_count, len(self.target_sequence))
                )
            else:
                self.get_logger().info(
                    "任务结束：原因=%s | 已撞击=%d/%d | 当前目标=%s"
                    % (
                        self.done_reason,
                        self.hit_count,
                        len(self.target_sequence),
                        self.format_target(self.current_target_cell),
                    )
                )

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

    def start_reset_robot_pose(self):
        if not self.reset_client.wait_for_service(timeout_sec=0.1):
            self.throttled_info(
                "reset_wait_service",
                "等待服务：/gazebo/set_entity_state 不可用，继续等待。",
            )
            return False

        state = EntityState()
        state.name = self.model_name
        state.reference_frame = "world"
        state.pose = Pose(
            position=Point(x=self.start_x, y=self.start_y, z=self.start_z),
            orientation=self.yaw_to_quaternion(self.start_yaw),
        )
        state.twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )

        request = SetEntityState.Request()
        request.state = state
        self.reset_future = self.reset_client.call_async(request)
        self.reset_requested = True
        self.get_logger().info(
            "已发送位姿重置请求：目标位姿=(%.4f, %.4f, %.4f, %.3f)"
            % (self.start_x, self.start_y, self.start_z, self.start_yaw)
        )
        return False

    def poll_reset_robot_pose(self):
        if not self.reset_requested:
            return self.start_reset_robot_pose()

        if self.reset_future is None or not self.reset_future.done():
            self.throttled_info("reset_wait_result", "等待机器人位姿重置完成。")
            return False

        try:
            response = self.reset_future.result()
        except Exception as exc:
            self.get_logger().warning("重置机器人位姿失败：%s" % exc)
            return True

        if response is None:
            self.get_logger().warning("重置机器人位姿失败：空响应")
            return True

        if getattr(response, "success", False):
            self.get_logger().info(
                "机器人位姿已重置到 (%.4f, %.4f, %.4f, %.3f)"
                % (self.start_x, self.start_y, self.start_z, self.start_yaw)
            )
        else:
            self.get_logger().warning(
                "重置机器人位姿失败：%s"
                % getattr(response, "status_message", "未知")
            )

        return True

    def goto_xy_yaw(self, target_x, target_y, target_yaw, state_name):
        if not self.received_pose or self.latest_pose is None:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.throttled_info(
                "%s_wait_pose" % state_name,
                "等待位姿：状态=%s" % state_name,
            )
            return False

        x, y, z, yaw = self.latest_pose
        dx = target_x - x
        dy = target_y - y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > self.goto_pos_deadband:
            vx, vy = self.world_to_body_cmd(dx, dy, yaw, self.goto_speed)
            self.publish_cmd(vx, vy, 0.0)
            self.throttled_info(
                "%s_move" % state_name,
                "移动中：状态=%s | 当前位姿=(%.3f, %.3f, %.3f) | 目标点=(%.3f, %.3f) | 距离=%.3f m | 机体系速度=(%.3f, %.3f)"
                % (state_name, x, y, z, target_x, target_y, dist, vx, vy)
            )
            return False

        yaw_error = self.normalize_angle(yaw - target_yaw)
        if abs(yaw_error) > self.goto_yaw_deadband:
            yaw_cmd = self.goto_turn_speed if yaw_error < 0.0 else -self.goto_turn_speed
            self.publish_cmd(0.0, 0.0, yaw_cmd)
            self.throttled_info(
                "%s_turn" % state_name,
                "转向中：状态=%s | 当前yaw=%.3f | 目标yaw=%.3f | 误差=%.3f | 角速度=%.3f"
                % (state_name, yaw, target_yaw, yaw_error, yaw_cmd)
            )
            return False

        self.publish_cmd(0.0, 0.0, 0.0)
        self.get_logger().info(
            "到达导航目标：状态=%s | 当前位姿=(%.3f, %.3f, %.3f) | 目标位姿=(%.3f, %.3f, %.3f)"
            % (state_name, x, y, z, target_x, target_y, target_yaw)
        )
        return True

    def clear_current_target(self):
        self.current_target_cell = None
        self.current_target_ball = None
        self.current_hit_point = None
        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.hit_reverse = False

    def build_safe_waypoints(
        self, column, row, ball_x, ball_y, hit_x, hit_y, hit_yaw, hit_reverse
    ):
        waypoints = []
        corridor_x = 2.60

        if column == self.special_column:
            if row == 3:
                return [
                    (2.750, 0.985, 1.000),
                    (corridor_x, 2.60, 3.14),
                    (hit_x, hit_y, hit_yaw),
                ]
        elif column == 3:
            if row == 2:
                waypoints.append((2.60, 2.60, 3.14))
        elif column == 2:
            if row == 1:
                waypoints.append((1.40, 1.76, 3.14))
        elif column == 1:
            if row == 4:
                waypoints.append((0.20, 1.76, 3.14))
                waypoints.append((0.20, 3.44, 3.14))

        del ball_x
        del ball_y
        del hit_reverse

        waypoints.append((hit_x, hit_y, hit_yaw))
        return waypoints

    def prepare_finish(self):
        self.finish_waypoints = [
            (0.20, 3.44, 3.14),
            (0.20, 4.10, self.finish_yaw),
            (self.finish_x, self.finish_y, self.finish_yaw),
        ]
        self.finish_waypoint_index = 0
        self.get_logger().info(
            "准备前往终点：路径点=%s"
            % self.format_waypoints(self.finish_waypoints)
        )
        return True

    def prepare_next_target(self):
        if self.current_target_index >= len(self.target_sequence):
            self.done_reason = "all_targets_done"
            self.switch_state("DONE")
            return False

        cell = self.target_sequence[self.current_target_index]
        ball = self.ball_grid.get(cell)
        if ball is None:
            self.done_reason = "missing_ball_grid_%d_%d" % (cell[0], cell[1])
            self.get_logger().warning(
                "准备撞击失败：目标=%s | 原因=BALL_GRID 缺少该坐标。"
                % self.format_target(cell)
            )
            self.switch_state("DONE")
            return False

        column, row = cell
        ball_x, ball_y = ball
        if column == self.special_column:
            hit_x = ball_x + self.special_hit_offset_x
            hit_y = ball_y + self.special_hit_offset_y
            hit_yaw = self.special_hit_yaw
            hit_reverse = True
        else:
            hit_x = ball_x + self.normal_hit_offset_x
            hit_y = ball_y + self.normal_hit_offset_y
            hit_yaw = self.normal_hit_yaw
            hit_reverse = False

        self.current_target_cell = cell
        self.current_target_ball = ball
        self.current_hit_point = (hit_x, hit_y, hit_yaw)
        self.current_waypoints = self.build_safe_waypoints(
            column,
            row,
            ball_x,
            ball_y,
            hit_x,
            hit_y,
            hit_yaw,
            hit_reverse,
        )
        self.current_waypoint_index = 0
        self.hit_reverse = hit_reverse

        self.get_logger().info(
            "准备撞击：目标=(%d,%d) | 球心=(%.3f,%.3f) | 撞击点=(%.3f,%.3f,%.3f) | 路径点=%s | 倒车撞击=%s"
            % (
                column,
                row,
                ball_x,
                ball_y,
                hit_x,
                hit_y,
                hit_yaw,
                self.format_waypoints(self.current_waypoints),
                str(hit_reverse),
            )
        )
        return True

    def publish_impact_cmd(self):
        if self.hit_reverse:
            vx = -self.reverse_impact_speed
        else:
            vx = self.impact_speed

        self.publish_cmd(vx, 0.0, 0.0)
        return vx

    def current_impact_time(self):
        if self.hit_reverse:
            return self.reverse_impact_time
        return self.impact_time

    def control_loop(self):
        if self.final_stopped:
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        now = time.time()
        elapsed = now - self.state_start_time
        total_elapsed = now - self.total_start_time

        if total_elapsed > self.max_total_time and self.state != "DONE":
            self.publish_cmd(0.0, 0.0, 0.0)
            self.done_reason = "max_total_time_exceeded"
            self.switch_state("DONE")
            return

        if self.state == "START":
            if self.reset_start_pose:
                self.switch_state("RESET_POSE")
            else:
                self.switch_state("RECOVERY_STAND")
            return

        if self.state == "RESET_POSE":
            if self.poll_reset_robot_pose():
                self.switch_state("RECOVERY_STAND")
            return

        if self.state == "RECOVERY_STAND":
            if elapsed < self.recovery_stand_time:
                self.publish_recovery_stand_cmd()
                self.throttled_info(
                    "RECOVERY_STAND_progress",
                    "恢复站立中：已用时间=%.2f/%.2f s"
                    % (elapsed, self.recovery_stand_time),
                )
                return

            self.cmd = self.make_cmd()
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().info("恢复站立完成。")
            self.switch_state("PREPARE_NEXT_TARGET")
            return

        if self.state == "PREPARE_NEXT_TARGET":
            if self.prepare_next_target():
                self.switch_state("GO_HIT_POINT")
            return

        if self.state == "GO_HIT_POINT":
            if self.current_hit_point is None:
                self.done_reason = "missing_hit_point"
                self.switch_state("DONE")
                return

            if not self.current_waypoints:
                self.done_reason = "missing_waypoints"
                self.switch_state("DONE")
                return

            if self.current_waypoint_index >= len(self.current_waypoints):
                self.switch_state("HIT_TARGET")
                return

            waypoint = self.current_waypoints[self.current_waypoint_index]
            waypoint_count = len(self.current_waypoints)

            self.throttled_info(
                "GO_HIT_POINT_progress",
                "前往路径点：目标=%s | 路径点=%d/%d | 当前路径点=(%.3f, %.3f, %.3f) | 最终撞击点=%s"
                % (
                    self.format_target(self.current_target_cell),
                    self.current_waypoint_index + 1,
                    waypoint_count,
                    waypoint[0],
                    waypoint[1],
                    waypoint[2],
                    self.format_hit_point(self.current_hit_point),
                ),
            )

            if self.goto_xy_yaw(
                waypoint[0],
                waypoint[1],
                waypoint[2],
                "GO_HIT_POINT",
            ):
                self.get_logger().info(
                    "到达路径点 %d/%d"
                    % (self.current_waypoint_index + 1, waypoint_count)
                )
                self.current_waypoint_index += 1
                if self.current_waypoint_index >= waypoint_count:
                    self.switch_state("HIT_TARGET")
            return

        if self.state == "HIT_TARGET":
            impact_time = self.current_impact_time()
            if elapsed < impact_time:
                vx = self.publish_impact_cmd()
                self.throttled_info(
                    "HIT_TARGET_progress",
                    "撞击中：目标=%s | vx=%.3f | 已用时间=%.2f/%.2f s | 倒车撞击=%s"
                    % (
                        self.format_target(self.current_target_cell),
                        vx,
                        elapsed,
                        impact_time,
                        str(self.hit_reverse),
                    )
                )
                return

            self.publish_cmd(0.0, 0.0, 0.0)
            self.hit_count += 1
            self.get_logger().info(
                "撞击完成：目标=%s | 已完成=%d/%d"
                % (
                    self.format_target(self.current_target_cell),
                    self.hit_count,
                    len(self.target_sequence),
                )
            )
            self.switch_state("POST_HIT_PAUSE")
            return

        if self.state == "POST_HIT_PAUSE":
            self.publish_cmd(0.0, 0.0, 0.0)
            if elapsed >= self.post_hit_pause_time:
                self.clear_current_target()
                if self.hit_count >= len(self.target_sequence) and self.target_sequence:
                    if self.prepare_finish():
                        self.switch_state("GO_FINISH")
                    else:
                        self.done_reason = "prepare_finish_failed"
                        self.switch_state("DONE")
                    return

                self.current_target_index += 1
                self.switch_state("PREPARE_NEXT_TARGET")
            return

        if self.state == "GO_FINISH":
            if not self.finish_waypoints:
                self.done_reason = "missing_finish_waypoints"
                self.switch_state("DONE")
                return

            if self.finish_waypoint_index >= len(self.finish_waypoints):
                self.done_reason = "finished"
                self.switch_state("DONE")
                return

            waypoint = self.finish_waypoints[self.finish_waypoint_index]
            waypoint_count = len(self.finish_waypoints)

            self.throttled_info(
                "GO_FINISH_progress",
                "前往终点路径点：路径点=%d/%d | 当前路径点=(%.3f, %.3f, %.3f) | 终点=(%.3f, %.3f, %.3f)"
                % (
                    self.finish_waypoint_index + 1,
                    waypoint_count,
                    waypoint[0],
                    waypoint[1],
                    waypoint[2],
                    self.finish_x,
                    self.finish_y,
                    self.finish_yaw,
                ),
            )

            if self.goto_xy_yaw(
                waypoint[0],
                waypoint[1],
                waypoint[2],
                "GO_FINISH",
            ):
                self.get_logger().info(
                    "到达终点路径点 %d/%d"
                    % (self.finish_waypoint_index + 1, waypoint_count)
                )
                self.finish_waypoint_index += 1
                if self.finish_waypoint_index >= waypoint_count:
                    self.done_reason = "finished"
                    self.switch_state("DONE")
            return

        if self.state == "DONE":
            self.publish_cmd(0.0, 0.0, 0.0)
            self.final_stopped = True
            return

        self.done_reason = "unknown_state"
        self.switch_state("DONE")


def main():
    rclpy.init()
    node = HitOrangeBallsBase()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到 KeyboardInterrupt，执行阻尼停止。")
        node.damper_stop()
    finally:
        node.publish_cmd(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()