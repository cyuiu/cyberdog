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
from cyberdog_msg.msg import YamlParam

from robot_control_cmd_lcmt import robot_control_cmd_lcmt

# ── 蹲姿参数 ──
CROUCH_HEIGHT = [0.0, 0.0, 0.15]
NORMAL_HEIGHT = [0.0, 0.0, 0.25]
NORMAL_HEIGHT_MOTION = [0.0, 0.0, 0.225]
NORMAL_HEIGHT_STAIR = [0.0, 0.0, 0.225]

CROUCH_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("step_height_max",              YamlParam.DOUBLE,       0.04),
]

RESTORE_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_MOTION),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_STAIR),
    ("step_height_max",              YamlParam.DOUBLE,       0.06),
]


class CombinedSlopeWalk(Node):
    def __init__(self):
        super().__init__("combined_slope_walk")

        # ==================== 第一阶段：上台阶专用参数 ====================
        self.declare_parameter("model_name", "robot")
        self.declare_parameter("max_time", 900.0)          # 增加总超时
        self.declare_parameter("control_period", 0.1)

        # 初始传送位置（台阶下方）
        self.declare_parameter("spawn_x", 3.1112)
        self.declare_parameter("spawn_y", 7.1091)
        self.declare_parameter("spawn_z", 0.0542)
        self.declare_parameter("spawn_qx", -0.000106)
        self.declare_parameter("spawn_qy", 0.000142)
        self.declare_parameter("spawn_qz", 0.6923)
        self.declare_parameter("spawn_qw", 0.7216)

        # 台阶上方起点位置（用于计算目标 yaw）
        self.declare_parameter("stage1_end_x", 3.121)
        self.declare_parameter("stage1_end_y", 7.923)
        self.declare_parameter("stage1_end_z", 0.250)

        # 上台阶步态参数（调大步伐与步高）
        self.declare_parameter("stage1_forward_speed", 0.22)   # 原0.15，加大步速
        self.declare_parameter("stage1_step_height", 0.14)     # 原0.10，加大步高
        self.declare_parameter("stage1_body_height", 0.35)
        self.declare_parameter("stage1_pitch_angle", -0.22)
        self.declare_parameter("stage1_gait_id", 26)
        self.declare_parameter("step_obstacle_height", 0.05)

        # 第一阶段最终终点（快速行走到达的位置）
        self.declare_parameter("stage2_end_x", 3.15293)
        self.declare_parameter("stage2_end_y", 12.33792)
        self.declare_parameter("stage2_end_z", 0.23161)

        # 快速行走步态参数
        self.declare_parameter("stage2_forward_speed", 0.30)
        self.declare_parameter("stage2_step_height", 0.22)
        self.declare_parameter("stage2_body_height", 0.32)
        self.declare_parameter("stage2_pitch_angle", -0.18)
        self.declare_parameter("stage2_gait_id", 27)

        # 停止检测误差范围
        self.declare_parameter("stop_x_tolerance", 0.20)
        self.declare_parameter("stop_y_tolerance", 0.05)

        # 方向对准参数
        self.declare_parameter("yaw_deadband", 0.05)
        self.declare_parameter("turn_speed", 0.1)
        self.declare_parameter("max_align_time", 10.0)

        # 左转参数（保留但第一阶段不用）
        self.declare_parameter("turn_angle", 90.0)
        self.declare_parameter("turn_vel", 0.4)

        # Y坐标触发线后移
        self.declare_parameter("stage1_trigger_y", 8.100)

        # ==================== 第二阶段：斜坡侧向行走参数 ====================
        # 第二段第一终点
        self.declare_parameter("dest1_x", -0.17466)
        self.declare_parameter("dest1_y", 12.42385)
        self.declare_parameter("dest1_z", 0.26596)

        # 第二段第二终点
        self.declare_parameter("dest2_x", -0.34364)
        self.declare_parameter("dest2_y", 15.16933)
        self.declare_parameter("dest2_z", 0.24047)

        # 第二段第三终点
        self.declare_parameter("dest3_x", 3.36096)
        self.declare_parameter("dest3_y", 15.36093)
        self.declare_parameter("dest3_z", 0.21635)

        # 侧向行走参数
        self.declare_parameter("slope_forward_speed", 0.12)
        self.declare_parameter("slope_step_height", 0.12)
        self.declare_parameter("slope_body_height", 0.22)
        self.declare_parameter("slope_pitch_angle", 0.0)
        self.declare_parameter("slope_roll_angle", 0.0)
        self.declare_parameter("slope_gait_id", 26)

        # 轴线锁定速度
        self.declare_parameter("lateral_correction_speed", 0.06)

        # 斜坡旋转参数（两次顺时针 90°，以前肢为圆心）
        self.declare_parameter("pivot_angle", -90.0)
        self.declare_parameter("pivot_vel", 0.4)
        self.declare_parameter("wheel_base", 0.45)

        # 斜坡行走各阶段目标 yaw
        self.yaw_walk1 = math.pi / 2.0          # 面朝 +Y
        self.yaw_walk2 = 0.0                    # 面朝 +X
        self.yaw_walk3 = -math.pi / 2.0         # 面朝 -Y

        # ==================== 第三阶段：额外直线行走参数 ====================
        self.declare_parameter("extra_dest_x", 3.11908)
        self.declare_parameter("extra_dest_y", 13.9438)
        self.declare_parameter("extra_forward_speed", 0.12)   # 慢速稳定
        self.declare_parameter("extra_step_height", 0.12)
        self.declare_parameter("extra_body_height", 0.30)     # 较高身体，适应平地
        self.declare_parameter("extra_pitch_angle", -0.10)
        self.declare_parameter("extra_roll_angle", 0.0)
        self.declare_parameter("extra_gait_id", 26)

        # 最终转向参数
        self.declare_parameter("final_yaw", -3.1406)

        # 蹲姿前进目标
        self.declare_parameter("crouch_target_x", 2.2120)
        self.declare_parameter("football_x", 2.334)
        self.declare_parameter("football_y", 13.344)
        self.declare_parameter("football_yaw", 3.13)

        # ==================== 获取所有参数 ====================
        self.model_name = self.get_parameter("model_name").value
        self.max_time = float(self.get_parameter("max_time").value)
        self.control_period = float(self.get_parameter("control_period").value)

        # 第一阶段参数
        self.spawn_x = float(self.get_parameter("spawn_x").value)
        self.spawn_y = float(self.get_parameter("spawn_y").value)
        self.spawn_z = float(self.get_parameter("spawn_z").value)
        self.spawn_qx = float(self.get_parameter("spawn_qx").value)
        self.spawn_qy = float(self.get_parameter("spawn_qy").value)
        self.spawn_qz = float(self.get_parameter("spawn_qz").value)
        self.spawn_qw = float(self.get_parameter("spawn_qw").value)

        self.stage1_end_x = float(self.get_parameter("stage1_end_x").value)
        self.stage1_end_y = float(self.get_parameter("stage1_end_y").value)
        self.stage1_end_z = float(self.get_parameter("stage1_end_z").value)

        self.stage1_forward_speed = float(self.get_parameter("stage1_forward_speed").value)
        self.stage1_step_height = float(self.get_parameter("stage1_step_height").value)
        self.stage1_body_height = float(self.get_parameter("stage1_body_height").value)
        self.stage1_pitch_angle = float(self.get_parameter("stage1_pitch_angle").value)
        self.stage1_gait_id = int(self.get_parameter("stage1_gait_id").value)
        self.step_obstacle_height = float(self.get_parameter("step_obstacle_height").value)

        self.stage2_end_x = float(self.get_parameter("stage2_end_x").value)
        self.stage2_end_y = float(self.get_parameter("stage2_end_y").value)
        self.stage2_end_z = float(self.get_parameter("stage2_end_z").value)

        self.stage2_forward_speed = float(self.get_parameter("stage2_forward_speed").value)
        self.stage2_step_height = float(self.get_parameter("stage2_step_height").value)
        self.stage2_body_height = float(self.get_parameter("stage2_body_height").value)
        self.stage2_pitch_angle = float(self.get_parameter("stage2_pitch_angle").value)
        self.stage2_gait_id = int(self.get_parameter("stage2_gait_id").value)

        self.stop_x_tolerance = float(self.get_parameter("stop_x_tolerance").value)
        self.stop_y_tolerance = float(self.get_parameter("stop_y_tolerance").value)
        self.yaw_deadband = float(self.get_parameter("yaw_deadband").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.max_align_time = float(self.get_parameter("max_align_time").value)
        self.turn_angle = float(self.get_parameter("turn_angle").value)
        self.turn_vel = float(self.get_parameter("turn_vel").value)
        self.stage1_trigger_y = float(self.get_parameter("stage1_trigger_y").value)

        # 第二阶段参数
        self.dest1_x = float(self.get_parameter("dest1_x").value)
        self.dest1_y = float(self.get_parameter("dest1_y").value)
        self.dest1_z = float(self.get_parameter("dest1_z").value)

        self.dest2_x = float(self.get_parameter("dest2_x").value)
        self.dest2_y = float(self.get_parameter("dest2_y").value)
        self.dest2_z = float(self.get_parameter("dest2_z").value)

        self.dest3_x = float(self.get_parameter("dest3_x").value)
        self.dest3_y = float(self.get_parameter("dest3_y").value)
        self.dest3_z = float(self.get_parameter("dest3_z").value)

        self.slope_forward_speed = float(self.get_parameter("slope_forward_speed").value)
        self.slope_step_height = float(self.get_parameter("slope_step_height").value)
        self.slope_body_height = float(self.get_parameter("slope_body_height").value)
        self.slope_pitch_angle = float(self.get_parameter("slope_pitch_angle").value)
        self.slope_roll_angle = float(self.get_parameter("slope_roll_angle").value)
        self.slope_gait_id = int(self.get_parameter("slope_gait_id").value)

        self.lateral_correction_speed = float(self.get_parameter("lateral_correction_speed").value)

        self.pivot_angle = math.radians(float(self.get_parameter("pivot_angle").value))
        self.pivot_vel = float(self.get_parameter("pivot_vel").value)
        self.wheel_base = float(self.get_parameter("wheel_base").value)

        # 第三阶段参数
        self.extra_dest_x = float(self.get_parameter("extra_dest_x").value)
        self.extra_dest_y = float(self.get_parameter("extra_dest_y").value)
        self.extra_forward_speed = float(self.get_parameter("extra_forward_speed").value)
        self.extra_step_height = float(self.get_parameter("extra_step_height").value)
        self.extra_body_height = float(self.get_parameter("extra_body_height").value)
        self.extra_pitch_angle = float(self.get_parameter("extra_pitch_angle").value)
        self.extra_roll_angle = float(self.get_parameter("extra_roll_angle").value)
        self.extra_gait_id = int(self.get_parameter("extra_gait_id").value)

        self.final_yaw = float(self.get_parameter("final_yaw").value)

        self.crouch_target_x = float(self.get_parameter("crouch_target_x").value)
        self.football_x = float(self.get_parameter("football_x").value)
        self.football_y = float(self.get_parameter("football_y").value)
        self.football_yaw = float(self.get_parameter("football_yaw").value)

        # ==================== 初始化通信和状态 ====================
        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        # 初始命令使用第一阶段参数
        self.cmd = self.make_cmd(self.stage1_gait_id, self.stage1_pitch_angle,
                                 self.stage1_body_height, self.stage1_step_height)

        self.latest_pose = None
        self.received_pose = False
        self.state = "RECOVERY"
        self.start_time = None
        self.final_stopped = False
        self.last_report_time = 0
        self._restored = False

        # 状态变量
        self.start_yaw = None
        self.align_start_time = None
        self.pivot_start_yaw = None

        # ROS2 参数发布器（蹲姿用）
        self.param_pub = self.create_publisher(YamlParam, "yaml_parameter", 10)

        self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self.on_model_states,
            qos_profile_sensor_data,
        )
        self.create_timer(self.control_period, self.control_loop)

        # 计算从 spawn 到 stage1_end 的目标 yaw 角
        dx = self.stage1_end_x - self.spawn_x
        dy = self.stage1_end_y - self.spawn_y
        self.spawn_to_stage1_yaw = math.atan2(dy, dx)

        self.get_logger().info("="*50)
        self.get_logger().info("=== COMBINED WALK + SLOPE + EXTRA WALK ===")
        self.get_logger().info("Phase1: steps to (%.3f, %.3f)" % (self.stage2_end_x, self.stage2_end_y))
        self.get_logger().info("Phase2: slope sideways to (%.3f, %.3f)" % (self.dest3_x, self.dest3_y))
        self.get_logger().info("Phase3: extra walk to (%.3f, %.3f)" % (self.extra_dest_x, self.extra_dest_y))
        self.get_logger().info("="*50)

    def make_cmd(self, gait_id, pitch_angle, body_height, step_height):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = gait_id
        msg.contact = 15
        msg.duration = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.rpy_des = [0.0, pitch_angle, 0.0]
        msg.pos_des = [0.0, 0.0, body_height]
        msg.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.ctrl_point = [0.0, 0.0, 0.0]
        msg.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.step_height = [step_height, step_height]
        return msg

    def publish_cmd(self, vx, vy, yaw_rate, gait_id=None, step_height=None,
                    body_height=None, pitch=None, roll=None):
        self.cmd.mode = 11
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [float(vx), float(vy), float(yaw_rate)]

        if gait_id is not None:
            self.cmd.gait_id = int(gait_id)
        if pitch is not None:
            if roll is not None:
                self.cmd.rpy_des = [float(roll), float(pitch), 0.0]
            else:
                self.cmd.rpy_des = [0.0, float(pitch), 0.0]
        if roll is not None and pitch is None:
            self.cmd.rpy_des = [float(roll), self.cmd.rpy_des[1], 0.0]
        if body_height is not None:
            self.cmd.pos_des = [0.0, 0.0, float(body_height)]
        if step_height is not None:
            self.cmd.step_height = [float(step_height), float(step_height)]

        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def damper_stop(self):
        self.cmd.mode = 7
        self.cmd.gait_id = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.duration = 0
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def stand_stay(self, gait_id=26):
        self.cmd.mode = 11
        self.cmd.gait_id = gait_id
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.rpy_des = [0.0, 0.0, 0.0]
        self.cmd.pos_des = [0.0, 0.0, self.slope_body_height]
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

        self.cmd = self.make_cmd(self.stage1_gait_id, self.stage1_pitch_angle,
                                 self.stage1_body_height, self.stage1_step_height)

    def set_entity_state(self):
        self.get_logger().info("Spawning robot to (%.3f, %.3f, %.3f)..." %
                               (self.spawn_x, self.spawn_y, self.spawn_z))
        cmd = [
            "ros2", "service", "call", "/gazebo/set_entity_state",
            "gazebo_msgs/srv/SetEntityState",
            "{state: {name: 'robot', pose: {position: {x: %.3f, y: %.3f, z: %.3f}, "
            "orientation: {x: %.4f, y: %.4f, z: %.4f, w: %.4f}}, "
            "twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}, "
            "reference_frame: 'world'}}"
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

    def send_yaml_params(self, param_list):
        for name, kind, value in param_list:
            m = YamlParam()
            m.name = name
            m.kind = kind
            m.is_user = 1
            if kind == YamlParam.VEC_X_DOUBLE:
                arr = [0.0] * 12
                for i, v in enumerate(value):
                    arr[i] = float(v)
                m.vecxd_value = arr
            elif kind == YamlParam.DOUBLE:
                m.double_value = float(value)
            self.param_pub.publish(m)
            time.sleep(0.05)

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

    def calculate_target_yaw(self, current_x, current_y, target_x, target_y):
        dx = target_x - current_x
        dy = target_y - current_y
        return math.atan2(dy, dx)

    def check_reach_stage1_end(self, y):
        return y > self.stage1_trigger_y

    def check_reach_dest(self, x, y, dest_x, dest_y):
        x_ok = abs(x - dest_x) <= self.stop_x_tolerance
        y_ok = abs(y - dest_y) <= self.stop_y_tolerance
        return x_ok and y_ok

    def control_loop(self):
        if self.final_stopped:
            self.stand_stay(gait_id=self.extra_gait_id)
            return

        if self.start_time is not None and time.time() - self.start_time > self.max_time:
            self.get_logger().warn("Task timeout! Stopping robot...")
            self.final_stopped = True
            return

        # ==================== 状态机 ====================
        if self.state == "RECOVERY":
            self.recovery_stand()
            self.state = "ALIGN_STAGE1_YAW"
            self.get_logger().info("State: RECOVERY → ALIGN_STAGE1_YAW")
            return

        if not self.received_pose or self.latest_pose is None:
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        x, y, z, yaw = self.latest_pose
        now = time.time()

        if now - self.last_report_time >= 0.5:
            self.get_logger().info("POS | x=%.3f | y=%.3f | z=%.3f | yaw=%.3f | State: %s" %
                                   (x, y, z, yaw, self.state))
            self.last_report_time = now

        # ==================== 第一阶段：上台阶 ====================
        if self.state == "ALIGN_STAGE1_YAW":
            yaw_error = self.normalize_angle(yaw - self.spawn_to_stage1_yaw)
            if abs(yaw_error) > self.yaw_deadband:
                yaw_cmd = self.turn_speed if yaw_error < 0 else -self.turn_speed
                self.publish_cmd(0.0, 0.0, yaw_cmd)
            else:
                self.state = "WALK_STAGE1"
                self.start_time = time.time()
                self.get_logger().info("State: ALIGN_STAGE1_YAW → WALK_STAGE1")
            return

        if self.state == "WALK_STAGE1":
            if self.check_reach_stage1_end(y):
                self.get_logger().info("Reached top of steps, aligning to final direction...")
                self.state = "ALIGN_STAGE2_YAW"
                self.align_start_time = now
                return

            yaw_error = self.normalize_angle(yaw - self.spawn_to_stage1_yaw)
            yaw_correction = -0.6 * yaw_error if abs(yaw_error) > 0.03 else 0.0
            yaw_correction = max(-0.12, min(0.12, yaw_correction))

            step_h = max(self.stage1_step_height, self.step_obstacle_height + 0.02)
            self.publish_cmd(self.stage1_forward_speed, 0.0, yaw_correction,
                             step_height=step_h)
            return

        if self.state == "ALIGN_STAGE2_YAW":
            if now - self.align_start_time > self.max_align_time:
                self.get_logger().warn("Alignment timeout, proceeding...")
                self.state = "WALK_STAGE2"
                return

            target_yaw = self.calculate_target_yaw(x, y, self.stage2_end_x, self.stage2_end_y)
            yaw_error = self.normalize_angle(yaw - target_yaw)
            if abs(yaw_error) > self.yaw_deadband:
                yaw_cmd = self.turn_speed if yaw_error < 0 else -self.turn_speed
                self.publish_cmd(0.0, 0.0, yaw_cmd)
            else:
                self.state = "WALK_STAGE2"
                self.get_logger().info("Aligned, start fast walk to final point.")
            return

        if self.state == "WALK_STAGE2":
            if self.check_reach_dest(x, y, self.stage2_end_x, self.stage2_end_y):
                self.get_logger().info("Reached stage2 end. Switching to slope phase...")
                # 转入斜坡准备：对准 +Y
                self.state = "ALIGN_TO_PLUS_Y"
                self.align_start_time = now   # 复用此变量做超时
                return

            target_yaw = self.calculate_target_yaw(x, y, self.stage2_end_x, self.stage2_end_y)
            yaw_error = self.normalize_angle(yaw - target_yaw)
            yaw_correction = -0.6 * yaw_error if abs(yaw_error) > 0.03 else 0.0
            yaw_correction = max(-0.12, min(0.12, yaw_correction))

            self.publish_cmd(self.stage2_forward_speed * 1.2, 0.0, yaw_correction,
                             gait_id=self.stage2_gait_id,
                             step_height=self.stage2_step_height,
                             body_height=self.stage2_body_height,
                             pitch=self.stage2_pitch_angle)
            return

        # ==================== 斜坡阶段：对准 +Y ====================
        if self.state == "ALIGN_TO_PLUS_Y":
            target_yaw = math.pi / 2.0
            yaw_error = self.normalize_angle(yaw - target_yaw)
            if abs(yaw_error) > self.yaw_deadband:
                yaw_cmd = self.turn_speed if yaw_error < 0 else -self.turn_speed
                self.publish_cmd(0.0, 0.0, yaw_cmd)
            else:
                self.get_logger().info("Aligned to +Y, starting slope sideways walk 1.")
                self.state = "WALK_SIDEWAYS1"
                self.start_time = time.time()   # 重新开始计时
            return

        # ==================== 斜坡第一段侧向行走 ====================
        if self.state == "WALK_SIDEWAYS1":
            if self.check_reach_dest(x, y, self.dest1_x, self.dest1_y):
                self.state = "PIVOT1"
                self.pivot_start_yaw = None
                self.get_logger().info("State: WALK_SIDEWAYS1 → PIVOT1")
                return

            yaw_error = self.normalize_angle(yaw - self.yaw_walk1)
            yaw_correction = -0.8 * yaw_error if abs(yaw_error) > 0.03 else 0.0
            yaw_correction = max(-0.15, min(0.15, yaw_correction))

            vy = abs(self.slope_forward_speed)
            dy = self.dest1_y - y
            vx = 0.0
            if abs(dy) > 0.02:
                vx = self.lateral_correction_speed if dy > 0 else -self.lateral_correction_speed

            self.publish_cmd(vx, vy, yaw_correction,
                             gait_id=self.slope_gait_id,
                             step_height=self.slope_step_height,
                             body_height=self.slope_body_height,
                             pitch=self.slope_pitch_angle,
                             roll=self.slope_roll_angle)
            return

        # ==================== 第一个旋转（+Y → +X） ====================
        if self.state == "PIVOT1":
            if self.pivot_start_yaw is None:
                self.pivot_start_yaw = yaw
                self.get_logger().info("Start PIVOT1 clockwise")

            yaw_diff = self.normalize_angle(yaw - self.pivot_start_yaw)
            remaining = self.pivot_angle - yaw_diff

            if abs(remaining) < 0.03:
                self.state = "WALK_SIDEWAYS2"
                self.pivot_start_yaw = None
                self.get_logger().info("PIVOT1 done → WALK_SIDEWAYS2")
                return

            yaw_rate = self.pivot_vel if remaining > 0 else -self.pivot_vel
            vy = -yaw_rate * self.wheel_base / 2.0
            self.publish_cmd(0.0, vy, yaw_rate,
                             gait_id=self.slope_gait_id,
                             step_height=self.slope_step_height,
                             body_height=self.slope_body_height,
                             pitch=self.slope_pitch_angle,
                             roll=self.slope_roll_angle)
            return

        # ==================== 斜坡第二段侧向行走 ====================
        if self.state == "WALK_SIDEWAYS2":
            if self.check_reach_dest(x, y, self.dest2_x, self.dest2_y):
                self.state = "PIVOT2"
                self.pivot_start_yaw = None
                self.get_logger().info("State: WALK_SIDEWAYS2 → PIVOT2")
                return

            yaw_error = self.normalize_angle(yaw - self.yaw_walk2)
            yaw_correction = -0.8 * yaw_error if abs(yaw_error) > 0.03 else 0.0
            yaw_correction = max(-0.15, min(0.15, yaw_correction))

            vy = abs(self.slope_forward_speed)
            dx = self.dest2_x - x
            vx = 0.0
            if abs(dx) > 0.02:
                vx = self.lateral_correction_speed if dx > 0 else -self.lateral_correction_speed

            self.publish_cmd(vx, vy, yaw_correction,
                             gait_id=self.slope_gait_id,
                             step_height=self.slope_step_height,
                             body_height=self.slope_body_height,
                             pitch=self.slope_pitch_angle,
                             roll=self.slope_roll_angle)
            return

        # ==================== 第二个旋转（+X → -Y） ====================
        if self.state == "PIVOT2":
            if self.pivot_start_yaw is None:
                self.pivot_start_yaw = yaw
                self.get_logger().info("Start PIVOT2 clockwise")

            yaw_diff = self.normalize_angle(yaw - self.pivot_start_yaw)
            remaining = self.pivot_angle - yaw_diff

            if abs(remaining) < 0.03:
                self.state = "WALK_SIDEWAYS3"
                self.pivot_start_yaw = None
                self.get_logger().info("PIVOT2 done → WALK_SIDEWAYS3")
                return

            yaw_rate = self.pivot_vel if remaining > 0 else -self.pivot_vel
            vy = -yaw_rate * self.wheel_base / 2.0
            self.publish_cmd(0.0, vy, yaw_rate,
                             gait_id=self.slope_gait_id,
                             step_height=self.slope_step_height,
                             body_height=self.slope_body_height,
                             pitch=self.slope_pitch_angle,
                             roll=self.slope_roll_angle)
            return

        # ==================== 斜坡第三段侧向行走 ====================
        if self.state == "WALK_SIDEWAYS3":
            if self.check_reach_dest(x, y, self.dest3_x, self.dest3_y):
                self.get_logger().info("Reached slope dest. Starting extra walk...")
                self.state = "ALIGN_EXTRA"
                self.align_start_time = now
                return

            yaw_error = self.normalize_angle(yaw - self.yaw_walk3)
            yaw_correction = -0.8 * yaw_error if abs(yaw_error) > 0.03 else 0.0
            yaw_correction = max(-0.15, min(0.15, yaw_correction))

            vy = abs(self.slope_forward_speed)
            dy = self.dest3_y - y
            vx = 0.0
            if abs(dy) > 0.02:
                vx = -self.lateral_correction_speed if dy > 0 else self.lateral_correction_speed

            self.publish_cmd(vx, vy, yaw_correction,
                             gait_id=self.slope_gait_id,
                             step_height=self.slope_step_height,
                             body_height=self.slope_body_height,
                             pitch=self.slope_pitch_angle,
                             roll=self.slope_roll_angle)
            return

        # ==================== 额外行走：对准新目标 ====================
        if self.state == "ALIGN_EXTRA":
            if now - self.align_start_time > self.max_align_time:
                self.get_logger().warn("Extra alignment timeout, walking anyway.")
                self.state = "WALK_EXTRA"
                self.start_time = now
                return

            target_yaw = self.calculate_target_yaw(x, y, self.extra_dest_x, self.extra_dest_y)
            yaw_error = self.normalize_angle(yaw - target_yaw)
            if abs(yaw_error) > self.yaw_deadband:
                yaw_cmd = self.turn_speed if yaw_error < 0 else -self.turn_speed
                self.publish_cmd(0.0, 0.0, yaw_cmd)
            else:
                self.state = "WALK_EXTRA"
                self.start_time = now
                self.get_logger().info("Aligned to extra target, start walking.")
            return

        # ==================== 额外行走：直线前进 ====================
        if self.state == "WALK_EXTRA":
            if self.check_reach_dest(x, y, self.extra_dest_x, self.extra_dest_y):
                self.get_logger().info("Reached extra dest. Aligning to final yaw...")
                self.state = "ALIGN_FINAL_YAW"
                self.align_start_time = now
                return

            target_yaw = self.calculate_target_yaw(x, y, self.extra_dest_x, self.extra_dest_y)
            yaw_error = self.normalize_angle(yaw - target_yaw)
            yaw_correction = -0.6 * yaw_error if abs(yaw_error) > 0.03 else 0.0
            yaw_correction = max(-0.12, min(0.12, yaw_correction))

            self.publish_cmd(self.extra_forward_speed, 0.0, yaw_correction,
                             gait_id=self.extra_gait_id,
                             step_height=self.extra_step_height,
                             body_height=self.extra_body_height,
                             pitch=self.extra_pitch_angle,
                             roll=self.extra_roll_angle)
            return

        # ==================== 最终转向（比例控制，防振荡）====================
        if self.state == "ALIGN_FINAL_YAW":
            if now - self.align_start_time > 30.0:
                self.get_logger().warn("Final alignment timeout, starting crouch walk.")
                self.state = "CROUCH_SETUP"
                return

            yaw_error = self.normalize_angle(yaw - self.final_yaw)
            if abs(yaw_error) > 0.05:
                # 比例控制：离目标越远转越快，接近时减速，防止振荡
                yaw_cmd = max(-0.4, min(0.4, yaw_error * 0.8))
                self.publish_cmd(0.0, 0.0, -yaw_cmd)
            else:
                self.get_logger().info("Aligned to final yaw (%.3f). Starting crouch walk..." % yaw)
                self.state = "CROUCH_SETUP"
            return

        # ==================== 蹲姿准备（完全重置命令，和 run.py 一致）====================
        if self.state == "CROUCH_SETUP":
            self.get_logger().info("Resetting cmd + sending crouch params...")
            # 完全重置 cmd，和 run.py 的 make_cmd() 一样
            self.cmd = robot_control_cmd_lcmt()
            self.cmd.mode = 11
            self.cmd.gait_id = 3
            self.cmd.contact = 15
            self.cmd.duration = 0
            self.cmd.vel_des = [0.0, 0.0, 0.0]
            self.cmd.rpy_des = [0.0, 0.25, 0.0]
            self.cmd.pos_des = [0.0, 0.0, 0.0]
            self.cmd.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.cmd.ctrl_point = [0.0, 0.0, 0.0]
            self.cmd.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.cmd.step_height = [0.05, 0.05]
            # 发送蹲姿参数
            self.send_yaml_params(CROUCH_PARAMS)
            self.state = "CROUCH_WAIT"
            self.align_start_time = now
            return

        # ==================== 等待身体降下（和 run.py 一样：零速3秒）====================
        if self.state == "CROUCH_WAIT":
            self.cmd.vel_des = [0.0, 0.0, 0.0]
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            if now - self.align_start_time > 3.0:
                self.get_logger().info("Body lowered. Walking forward to x=%.3f..." % self.crouch_target_x)
                self.state = "CROUCH_WALK"
                self.start_time = now
            return

        # ==================== 蹲姿前进（和 run.py 完全一样的发送方式）====================
        if self.state == "CROUCH_WALK":
            if x >= self.crouch_target_x:
                self.get_logger().info("Reached x=%.3f. Standing up..." % x)
                self.send_yaml_params(RESTORE_PARAMS)
                self.cmd.vel_des = [0.0, 0.0, 0.0]
                self.cmd.life_count = (self.cmd.life_count + 1) % 128
                self.lc.publish("robot_control_cmd", self.cmd.encode())
                self.state = "CROUCH_DONE"
                self.align_start_time = now
                return

            if now - self.start_time > 60.0:
                self.get_logger().warn("CROUCH_WALK timeout at x=%.3f!" % x)
                self.send_yaml_params(RESTORE_PARAMS)
                self.state = "CROUCH_DONE"
                self.align_start_time = now
                return

            # 和 run.py 完全一样：vx=0.08, pitch=0.25
            self.cmd.vel_des = [0.08, 0.0, 0.0]
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())

            if now - self.last_report_time >= 1.0:
                self.get_logger().info("CROUCH_WALK | x=%.3f y=%.3f" % (x, y))
                self.last_report_time = now
            return

        # ==================== 蹲姿结束，等待身体恢复 ====================
        if self.state == "CROUCH_DONE":
            self.cmd.vel_des = [0.0, 0.0, 0.0]
            self.cmd.rpy_des = [0.0, 0.0, 0.0]
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            if now - self.align_start_time > 3.0:
                self.get_logger().info("Height restored. All tasks complete!")
                self.state = "FINISHED"
                self.final_stopped = True
            return

        if self.state == "FINISHED":
            if not self._restored:
                self.send_yaml_params(RESTORE_PARAMS)
                self._restored = True
            self.stand_stay(gait_id=self.extra_gait_id)
            return


def main():
    rclpy.init()
    node = CombinedSlopeWalk()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — restoring height...")
        node.send_yaml_params(RESTORE_PARAMS)
        node.damper_stop()
    finally:
        node.publish_cmd(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()