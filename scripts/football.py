#!/usr/bin/env python3
import math
import sys
import time

LCM_PYTHON_PATH = "/home/lcm/build/python"
CONTROL_PATH = "/home/loco_example/sequential_motion"
LCM_URL = "udpm://239.255.76.67:7671?ttl=255"

if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)
if CONTROL_PATH not in sys.path:
    sys.path.insert(0, CONTROL_PATH)

import lcm
import rclpy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose, Quaternion, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from robot_control_cmd_lcmt import robot_control_cmd_lcmt


class FootballTest3(Node):
    def __init__(self):
        super().__init__("football_test3")

        self.declare_parameter("robot_name", "robot")
        self.declare_parameter("ball_name", "football3")

        self.declare_parameter("ball_default_x", 0.4)
        self.declare_parameter("ball_default_y", 14.7)
        self.declare_parameter("ball_default_z", 0.1)

        self.declare_parameter("kick_pose_x", 0.8613135427491895)
        self.declare_parameter("kick_pose_y", 14.782376214480617)
        self.declare_parameter("kick_pose_z", 0.25)
        self.declare_parameter("kick_pose_qx", 0.0)
        self.declare_parameter("kick_pose_qy", 0.0)
        self.declare_parameter("kick_pose_qz", 0.9998685028783119)
        self.declare_parameter("kick_pose_qw", -0.016215372605750324)
        self.declare_parameter("stage_start_x", 2.3344570329174847)
        self.declare_parameter("stage_start_y", 13.34445691007769)
        self.declare_parameter("stage_start_z", 0.25)
        self.declare_parameter("stage_start_yaw", 3.13)
        self.declare_parameter("first_route_mid_x", 1.45)
        self.declare_parameter("first_route_mid_y", 13.90)
        self.declare_parameter("first_push_yaw", -3.109)

        self.declare_parameter("target_exit_x", 3.188726689378719)
        self.declare_parameter("target_exit_y", 12.938645248236686)

        self.declare_parameter("push_vx", 0.08)
        self.declare_parameter("push_vy", 0.04)
        self.declare_parameter("push_yaw", 0.0)
        self.declare_parameter("push_step_height", 0.02)
        self.declare_parameter("push_max_time", 30.0)
        self.declare_parameter("ball_move_threshold", 0.008)

        self.declare_parameter("backoff_vx", -0.04)
        self.declare_parameter("backoff_time", 0.7)

        self.declare_parameter("stable_check_interval", 4.0)
        self.declare_parameter("stable_threshold", 0.02)
        self.declare_parameter("stable_samples", 2)
        self.declare_parameter("settle_timeout", 45.0)
        self.declare_parameter("move_start_delta_threshold", 0.06)
        self.declare_parameter("move_start_check_interval", 4.0)

        self.declare_parameter("stand_height_threshold", 0.20)
        self.declare_parameter("stand_good_samples", 10)
        self.declare_parameter("recovery_stand_timeout", 10.0)
        self.declare_parameter("stand_hold_time", 1.5)

        self.declare_parameter("ball_reset_tolerance", 0.05)
        self.declare_parameter("robot_reset_tolerance", 0.08)
        self.declare_parameter("ball_reset_confirm_timeout", 2.0)
        self.declare_parameter("robot_reset_confirm_timeout", 2.5)

        self.declare_parameter("approach_distance", 0.45)
        self.declare_parameter("safe_x_min", 0.05)
        self.declare_parameter("safe_x_max", 3.10)
        self.declare_parameter("safe_y_min", 12.80)
        self.declare_parameter("safe_y_max", 14.85)
        self.declare_parameter("after_first_touch_wait", 5.0)
        self.declare_parameter("turn_to_neg_y_yaw", -1.654)
        self.declare_parameter("b_pose_x", 0.29146899556577405)
        self.declare_parameter("b_pose_y", 14.57312701083567)
        self.declare_parameter("b_pose_z", 0.25)
        self.declare_parameter("b_pose_yaw", -1.654)
        self.declare_parameter("right_shift_speed", 0.05)
        self.declare_parameter("c_pose_x", 0.23330677453129753)
        self.declare_parameter("c_pose_y", 13.135809135192238)
        self.declare_parameter("c_pose_z", 0.25)
        self.declare_parameter("c_pose_yaw", -1.636)
        self.declare_parameter("manual_side_walk_timeout", 160.0)
        self.declare_parameter("manual_side_push_vx", 0.0)
        self.declare_parameter("manual_side_push_vy", 0.12)
        self.declare_parameter("manual_side_push_yaw", 0.0)
        self.declare_parameter("manual_side_push_step_height", 0.02)
        self.declare_parameter("manual_side_ball_move_threshold", 0.03)
        self.declare_parameter("manual_side_push_min_time", 3.0)
        self.declare_parameter("manual_side_push_max_time", 6.0)
        self.declare_parameter("manual_side_push_yaw_hold_gain", 0.4)
        self.declare_parameter("manual_side_push_yaw_cmd_limit", 0.04)
        self.declare_parameter("finish_x", 3.1546944792817944)
        self.declare_parameter("finish_y", 12.94576515677228)
        self.declare_parameter("finish_z", 0.25229174503664764)
        self.declare_parameter("finish_yaw", 0.0113)
        self.declare_parameter("total_timeout", 420.0)
        self.declare_parameter("walk_speed", 0.07)
        self.declare_parameter("walk_turn_speed", 0.10)
        self.declare_parameter("walk_yaw_deadband", 0.12)
        self.declare_parameter("walk_pos_tolerance", 0.08)
        self.declare_parameter("walk_replan_interval", 1.0)
        self.declare_parameter("walk_replan_distance", 0.08)

        self.declare_parameter("control_period", 0.10)
        self.declare_parameter("status_log_period", 2.0)
        self.declare_parameter("model_states_wait_timeout", 5.0)
        self.declare_parameter("model_states_fresh_timeout", 1.0)
        self.declare_parameter("set_state_timeout", 3.0)
        self.declare_parameter("get_state_timeout", 2.0)

        self.robot_name = str(self.get_parameter("robot_name").value)
        self.ball_name = str(self.get_parameter("ball_name").value)

        self.ball_default = (
            float(self.get_parameter("ball_default_x").value),
            float(self.get_parameter("ball_default_y").value),
            float(self.get_parameter("ball_default_z").value),
        )
        self.kick_pose = (
            float(self.get_parameter("kick_pose_x").value),
            float(self.get_parameter("kick_pose_y").value),
            float(self.get_parameter("kick_pose_z").value),
        )
        self.kick_quat = (
            float(self.get_parameter("kick_pose_qx").value),
            float(self.get_parameter("kick_pose_qy").value),
            float(self.get_parameter("kick_pose_qz").value),
            float(self.get_parameter("kick_pose_qw").value),
        )
        self.stage_start_pose = (
            float(self.get_parameter("stage_start_x").value),
            float(self.get_parameter("stage_start_y").value),
            float(self.get_parameter("stage_start_z").value),
        )
        self.stage_start_yaw = float(self.get_parameter("stage_start_yaw").value)
        self.first_route_mid_pose = (
            float(self.get_parameter("first_route_mid_x").value),
            float(self.get_parameter("first_route_mid_y").value),
            float(self.get_parameter("kick_pose_z").value),
        )
        self.first_push_yaw = float(self.get_parameter("first_push_yaw").value)
        self.target_exit = (
            float(self.get_parameter("target_exit_x").value),
            float(self.get_parameter("target_exit_y").value),
        )

        self.push_vx = float(self.get_parameter("push_vx").value)
        self.push_vy = float(self.get_parameter("push_vy").value)
        self.push_yaw = float(self.get_parameter("push_yaw").value)
        self.push_step_height = float(self.get_parameter("push_step_height").value)
        self.push_max_time = float(self.get_parameter("push_max_time").value)
        self.ball_move_threshold = float(
            self.get_parameter("ball_move_threshold").value
        )
        self.backoff_vx = float(self.get_parameter("backoff_vx").value)
        self.backoff_time = float(self.get_parameter("backoff_time").value)

        self.stable_check_interval = float(
            self.get_parameter("stable_check_interval").value
        )
        self.stable_threshold = float(self.get_parameter("stable_threshold").value)
        self.stable_samples = int(self.get_parameter("stable_samples").value)
        self.settle_timeout = float(self.get_parameter("settle_timeout").value)
        self.move_start_delta_threshold = float(
            self.get_parameter("move_start_delta_threshold").value
        )
        self.move_start_check_interval = float(
            self.get_parameter("move_start_check_interval").value
        )

        self.stand_height_threshold = float(
            self.get_parameter("stand_height_threshold").value
        )
        self.stand_good_samples = int(
            self.get_parameter("stand_good_samples").value
        )
        self.recovery_stand_timeout = float(
            self.get_parameter("recovery_stand_timeout").value
        )
        self.stand_hold_time = float(self.get_parameter("stand_hold_time").value)

        self.ball_reset_tolerance = float(
            self.get_parameter("ball_reset_tolerance").value
        )
        self.robot_reset_tolerance = float(
            self.get_parameter("robot_reset_tolerance").value
        )
        self.ball_reset_confirm_timeout = float(
            self.get_parameter("ball_reset_confirm_timeout").value
        )
        self.robot_reset_confirm_timeout = float(
            self.get_parameter("robot_reset_confirm_timeout").value
        )

        self.approach_distance = float(
            self.get_parameter("approach_distance").value
        )
        self.safe_x_min = float(self.get_parameter("safe_x_min").value)
        self.safe_x_max = float(self.get_parameter("safe_x_max").value)
        self.safe_y_min = float(self.get_parameter("safe_y_min").value)
        self.safe_y_max = float(self.get_parameter("safe_y_max").value)
        self.after_first_touch_wait = float(
            self.get_parameter("after_first_touch_wait").value
        )
        self.turn_to_neg_y_yaw = float(
            self.get_parameter("turn_to_neg_y_yaw").value
        )
        self.b_pose_x = float(self.get_parameter("b_pose_x").value)
        self.b_pose_y = float(self.get_parameter("b_pose_y").value)
        self.b_pose_z = float(self.get_parameter("b_pose_z").value)
        self.b_pose_yaw = float(self.get_parameter("b_pose_yaw").value)
        self.right_shift_speed = float(self.get_parameter("right_shift_speed").value)
        self.c_pose = (
            float(self.get_parameter("c_pose_x").value),
            float(self.get_parameter("c_pose_y").value),
            float(self.get_parameter("c_pose_z").value),
        )
        self.c_pose_yaw = float(self.get_parameter("c_pose_yaw").value)
        self.manual_side_walk_timeout = float(
            self.get_parameter("manual_side_walk_timeout").value
        )
        self.manual_side_push_vx = float(
            self.get_parameter("manual_side_push_vx").value
        )
        self.manual_side_push_vy = float(
            self.get_parameter("manual_side_push_vy").value
        )
        self.manual_side_push_yaw = float(
            self.get_parameter("manual_side_push_yaw").value
        )
        self.manual_side_push_step_height = float(
            self.get_parameter("manual_side_push_step_height").value
        )
        self.manual_side_ball_move_threshold = float(
            self.get_parameter("manual_side_ball_move_threshold").value
        )
        self.manual_side_push_min_time = float(
            self.get_parameter("manual_side_push_min_time").value
        )
        self.manual_side_push_max_time = float(
            self.get_parameter("manual_side_push_max_time").value
        )
        self.manual_side_push_yaw_hold_gain = float(
            self.get_parameter("manual_side_push_yaw_hold_gain").value
        )
        self.manual_side_push_yaw_cmd_limit = float(
            self.get_parameter("manual_side_push_yaw_cmd_limit").value
        )
        self.finish_pose = (
            float(self.get_parameter("finish_x").value),
            float(self.get_parameter("finish_y").value),
            float(self.get_parameter("finish_z").value),
        )
        self.finish_yaw = float(self.get_parameter("finish_yaw").value)
        self.total_timeout = float(self.get_parameter("total_timeout").value)
        self.walk_speed = float(self.get_parameter("walk_speed").value)
        self.walk_turn_speed = float(self.get_parameter("walk_turn_speed").value)
        self.walk_yaw_deadband = float(
            self.get_parameter("walk_yaw_deadband").value
        )
        self.walk_pos_tolerance = float(
            self.get_parameter("walk_pos_tolerance").value
        )
        self.walk_replan_interval = float(
            self.get_parameter("walk_replan_interval").value
        )
        self.walk_replan_distance = float(
            self.get_parameter("walk_replan_distance").value
        )
        self.control_period = float(self.get_parameter("control_period").value)
        self.status_log_period = float(
            self.get_parameter("status_log_period").value
        )
        self.model_states_wait_timeout = float(
            self.get_parameter("model_states_wait_timeout").value
        )
        self.model_states_fresh_timeout = float(
            self.get_parameter("model_states_fresh_timeout").value
        )
        self.set_state_timeout = float(self.get_parameter("set_state_timeout").value)
        self.get_state_timeout = float(self.get_parameter("get_state_timeout").value)

        self.lc = lcm.LCM(LCM_URL)
        self.cmd = self.make_cmd()

        self.robot_pose = None
        self.robot_twist = None
        self.ball_pose = None
        self.ball_twist = None
        self.robot_in_model_states = False
        self.ball_in_model_states = False
        self.last_model_states_time = 0.0
        self.received_model_states = False
        self.model_states_logged = False
        self.last_missing_log_time = 0.0

        self.state = "INIT"
        self.script_start_time = time.time()
        self.state_start_time = time.time()
        self.last_status_log_time = 0.0

        self.pending_set_future = None
        self.pending_set_name = ""
        self.pending_set_start_time = 0.0

        self.pending_get_future = None
        self.pending_get_name = ""
        self.pending_get_start_time = 0.0

        self.stand_good_counter = 0
        self.settle_last_pose = None
        self.settle_last_check_time = 0.0
        self.settle_stable_counter = 0

        self.ball_before = None
        self.ball_before_source = ""
        self.first_move_pos = None
        self.first_move_time = None
        self.current_ball_pose = None
        self.first_plan_ball_pose = None
        self.first_kick_success = False
        self.distance_to_exit = None
        self.next_kick_pose = None
        self.next_kick_yaw = None
        self.next_kick_safe = None
        self.manual_side_exec_reason = "未执行人工横推"
        self.manual_side_ball_before = None
        self.manual_side_first_move_pos = None
        self.recheck_ball_pose = None
        self.recheck_error = None
        self.manual_side_push_executed = False
        self.manual_side_final_ball_pose = None
        self.manual_side_push_hold_yaw = None
        self.manual_side_push_actual_duration = None
        self.manual_side_last_push_log_time = 0.0
        self.finish_reached = False
        self.finish_distance = None
        self.final_damper_start_time = 0.0
        self.first_after_ball_pose = None
        self.walk_target_pose = None
        self.walk_target_yaw = None
        self.walk_target_kind = ""
        self.walk_last_replan_time = 0.0
        self.walk_reference_ball_pose = None
        self.manual_stage_name = ""
        self.manual_side_wait_start_time = 0.0
        self.manual_side_walk_start_time = 0.0
        self.after_first_touch_wait_done = False
        self.manual_side_reached_pose = False
        self.manual_side_arrival_total_time = None
        self.final_summary_printed = False
        self.settle_result = "未开始"
        self.stop_reason = ""

        self.set_state_client = self.create_client(
            SetEntityState,
            "/gazebo/set_entity_state",
        )
        self.get_state_client = self.create_client(
            GetEntityState,
            "/gazebo/get_entity_state",
        )

        self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self.on_model_states,
            qos_profile_sensor_data,
        )
        self.create_timer(self.control_period, self.control_loop)

        self.log_event(
            "football_test3 启动 | 白球默认位置=%s | 第一次推球点=%s | 目标出口=(%.6f, %.6f)"
            % (
                self.format_pose(self.ball_default),
                self.format_pose(self.kick_pose),
                self.target_exit[0],
                self.target_exit[1],
            )
        )
        self.log_event(
            "第一脚默认参数 | push_vx=%.2f, push_vy=%.2f | yaw=%.1f | step_height=%.2f | move_threshold=%.3f"
            % (
                self.push_vx,
                self.push_vy,
                self.push_yaw,
                self.push_step_height,
                self.ball_move_threshold,
            )
        )
        self.log_event(
            "起点与终点参数 | stage_start=%s | stage_start_yaw=%.3f | first_route_mid=(%.3f, %.3f) | first_push_yaw=%.3f | finish=(%.6f, %.6f, %.6f) | finish_yaw=%.3f"
            % (
                self.format_pose(self.stage_start_pose),
                self.stage_start_yaw,
                self.first_route_mid_pose[0],
                self.first_route_mid_pose[1],
                self.first_push_yaw,
                self.finish_pose[0],
                self.finish_pose[1],
                self.finish_pose[2],
                self.finish_yaw,
            )
        )
        self.log_event(
            "第二脚路径参数 | turn_to_neg_y_yaw=%.3f | b_pose=(%.6f, %.6f, %.6f) | b_pose_yaw=%.3f | right_shift_speed=%.3f | c_pose=%s | c_pose_yaw=%.3f | manual_side_push_vy=%.3f | push_min=%.1f | push_max=%.1f | ball_threshold=%.3f | walk_timeout=%.1f | total_timeout=%.1f"
            % (
                self.turn_to_neg_y_yaw,
                self.b_pose_x,
                self.b_pose_y,
                self.b_pose_z,
                self.b_pose_yaw,
                self.right_shift_speed,
                self.format_pose(self.c_pose),
                self.c_pose_yaw,
                self.manual_side_push_vy,
                self.manual_side_push_min_time,
                self.manual_side_push_max_time,
                self.manual_side_ball_move_threshold,
                self.manual_side_walk_timeout,
                self.total_timeout,
            )
        )
        self.switch_state("WAIT_MODEL_STATES")

    def make_cmd(self):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = 3
        msg.contact = 15
        msg.life_count = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        msg.acc_des = [0.0] * 6
        msg.ctrl_point = [0.0, 0.0, 0.0]
        msg.foot_pose = [0.0] * 6
        msg.step_height = [0.03, 0.03]
        msg.value = 0
        msg.duration = 0
        return msg

    def publish_cmd(self):
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def publish_walk_cmd(self, vx, vy, yaw_rate, step_height):
        self.cmd.mode = 11
        self.cmd.gait_id = 3
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [float(vx), float(vy), float(yaw_rate)]
        self.cmd.step_height = [float(step_height), float(step_height)]
        self.publish_cmd()

    def publish_zero_hold(self):
        self.publish_walk_cmd(0.0, 0.0, 0.0, 0.03)

    def publish_push_cmd(self):
        self.publish_walk_cmd(
            self.push_vx,
            self.push_vy,
            self.push_yaw,
            self.push_step_height,
        )

    def publish_backoff_cmd(self):
        self.publish_walk_cmd(self.backoff_vx, 0.0, 0.0, 0.03)

    def publish_manual_side_push_cmd(self):
        self.publish_walk_cmd(
            self.manual_side_push_vx,
            self.manual_side_push_vy,
            self.manual_side_push_yaw,
            self.manual_side_push_step_height,
        )

    def publish_manual_side_push_hold_cmd(self, yaw_cmd):
        self.publish_walk_cmd(
            0.0,
            self.manual_side_push_vy,
            yaw_cmd,
            self.manual_side_push_step_height,
        )

    def publish_manual_side_backoff_cmd(self):
        lateral = -0.04 if self.manual_side_push_vy > 0.0 else 0.04
        self.publish_walk_cmd(0.0, lateral, 0.0, 0.03)

    def log_second_kick_motion(
        self,
        label,
        robot_pos,
        target_pose,
        x_error,
        y_error,
        yaw_error,
        vx,
        vy,
        yaw_cmd,
    ):
        self.log_status(
            "%s | robot=(%.3f,%.3f) | target=(%.3f,%.3f) | x_error=%.3f | y_error=%.3f | yaw_error=%.3f | vx=%.3f | vy=%.3f | yaw_cmd=%.3f"
            % (
                label,
                robot_pos[0],
                robot_pos[1],
                target_pose[0],
                target_pose[1],
                x_error,
                y_error,
                yaw_error,
                vx,
                vy,
                yaw_cmd,
            )
        )

    def log_finish_motion(self, robot_pos, dist, yaw_error, vx, yaw_cmd):
        self.log_status(
            "前往终点 | robot=(%.3f,%.3f) | finish=(%.3f, %.3f) | dist=%.3f | yaw_error=%.3f | vx=%.3f | yaw_cmd=%.3f"
            % (
                robot_pos[0],
                robot_pos[1],
                self.finish_pose[0],
                self.finish_pose[1],
                dist,
                yaw_error,
                vx,
                yaw_cmd,
            )
        )

    def log_first_push_motion(self, robot_pos, target_pose, dist, yaw_error, vx, yaw_cmd):
        self.log_status(
            "前往第一踢球点 | robot=(%.3f,%.3f) | target=(%.3f,%.3f) | dist=%.3f | yaw_error=%.3f | vx=%.3f | yaw_cmd=%.3f"
            % (
                robot_pos[0],
                robot_pos[1],
                target_pose[0],
                target_pose[1],
                dist,
                yaw_error,
                vx,
                yaw_cmd,
            )
        )

    def reset_manual_side_tracking(self):
        self.manual_side_exec_reason = "未执行人工横推"
        self.manual_side_ball_before = None
        self.manual_side_first_move_pos = None
        self.recheck_ball_pose = None
        self.recheck_error = None
        self.manual_side_push_executed = False
        self.manual_side_final_ball_pose = None
        self.manual_side_push_hold_yaw = None
        self.manual_side_push_actual_duration = None
        self.manual_side_last_push_log_time = 0.0
        self.finish_reached = False
        self.finish_distance = None
        self.final_damper_start_time = 0.0
        self.walk_target_pose = None
        self.walk_target_yaw = None
        self.walk_target_kind = ""
        self.walk_last_replan_time = 0.0
        self.walk_reference_ball_pose = None
        self.manual_stage_name = ""
        self.manual_side_wait_start_time = 0.0
        self.manual_side_walk_start_time = 0.0
        self.after_first_touch_wait_done = False
        self.manual_side_reached_pose = False
        self.manual_side_arrival_total_time = None

    def publish_recovery_stand(self):
        self.cmd.mode = 12
        self.cmd.gait_id = 0
        self.cmd.contact = 0
        self.cmd.duration = 5000
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.step_height = [0.0, 0.0]
        self.publish_cmd()

    def publish_damper_stop(self):
        self.cmd.mode = 7
        self.cmd.gait_id = 0
        self.cmd.contact = 0
        self.cmd.duration = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.step_height = [0.0, 0.0]
        self.publish_cmd()

    def on_model_states(self, msg):
        self.received_model_states = True
        self.last_model_states_time = time.time()

        if not self.model_states_logged:
            self.log_event("收到 /gazebo/model_states | 模型列表=%s" % list(msg.name))
            self.model_states_logged = True

        robot_index = None
        ball_index = None
        for index, name in enumerate(msg.name):
            if name == self.robot_name:
                robot_index = index
            elif name == self.ball_name:
                ball_index = index

        self.robot_in_model_states = robot_index is not None
        self.ball_in_model_states = ball_index is not None

        if robot_index is not None:
            self.robot_pose = msg.pose[robot_index]
            self.robot_twist = msg.twist[robot_index] if robot_index < len(msg.twist) else None

        if ball_index is not None:
            self.ball_pose = msg.pose[ball_index]
            self.ball_twist = msg.twist[ball_index] if ball_index < len(msg.twist) else None
            self.current_ball_pose = self.get_ball_position()

        if robot_index is None or ball_index is None:
            now = time.time()
            if now - self.last_missing_log_time >= self.status_log_period:
                if robot_index is None and ball_index is None:
                    self.get_logger().warn("model_states 中暂未找到 robot 和 football3")
                elif robot_index is None:
                    self.get_logger().warn("model_states 中暂未找到 robot")
                else:
                    self.get_logger().warn("model_states 中暂未找到 football3")
                self.last_missing_log_time = now

    def log_event(self, text):
        self.get_logger().info(text)
        self.last_status_log_time = time.time()

    def elapsed_total(self):
        return time.time() - self.script_start_time

    def elapsed_state(self):
        return time.time() - self.state_start_time

    def with_timing(self, text):
        return "总时间=%.2fs | 状态时间=%.2fs | %s" % (
            self.elapsed_total(),
            self.elapsed_state(),
            text,
        )

    def log_status(self, text):
        now = time.time()
        if now - self.last_status_log_time >= self.status_log_period:
            self.get_logger().info(self.with_timing(text))
            self.last_status_log_time = now

    def switch_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        if new_state in (
            "CONFIRM_BALL_RESET",
            "CONFIRM_DOG_STAGE_START",
        ):
            self.clear_pending_get()
        if new_state == "RECOVERY_STAND":
            self.stand_good_counter = 0
        if new_state in (
            "WAIT_BALL_SETTLE_OR_TIMEOUT",
            "WAIT_AFTER_FIRST_TOUCH_FOR_MANUAL_SIDE",
        ):
            self.settle_last_pose = None
            self.settle_last_check_time = 0.0
            self.settle_stable_counter = 0
        if new_state == "FINAL_DAMPER_STOP":
            self.final_damper_start_time = 0.0
        self.log_event(
            "状态切换 -> %s | 总时间=%.2fs"
            % (new_state, self.elapsed_total())
        )

    def clear_pending_set(self):
        self.pending_set_future = None
        self.pending_set_name = ""
        self.pending_set_start_time = 0.0

    def clear_pending_get(self):
        self.pending_get_future = None
        self.pending_get_name = ""
        self.pending_get_start_time = 0.0

    @staticmethod
    def format_pose(pose):
        if pose is None:
            return "不可用"
        return "(%.6f, %.6f, %.6f)" % (pose[0], pose[1], pose[2])

    @staticmethod
    def distance_xy(p0, p1):
        return math.hypot(p0[0] - p1[0], p0[1] - p1[1])

    @staticmethod
    def distance_xyz(p0, p1):
        return math.sqrt(
            (p0[0] - p1[0]) * (p0[0] - p1[0])
            + (p0[1] - p1[1]) * (p0[1] - p1[1])
            + (p0[2] - p1[2]) * (p0[2] - p1[2])
        )

    @staticmethod
    def normalize_2d(dx, dy):
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return None
        return (dx / norm, dy / norm)

    @staticmethod
    def quaternion_from_yaw(yaw):
        half = yaw * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    @staticmethod
    def yaw_from_quaternion(quat):
        if quat is None:
            return None
        x = float(quat.x)
        y = float(quat.y)
        z = float(quat.z)
        w = float(quat.w)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def clamp_safe_pose(self, x, y):
        return (
            self.clamp(x, self.safe_x_min, self.safe_x_max),
            self.clamp(y, self.safe_y_min, self.safe_y_max),
        )

    def get_robot_position(self):
        if self.robot_pose is None:
            return None
        return (
            float(self.robot_pose.position.x),
            float(self.robot_pose.position.y),
            float(self.robot_pose.position.z),
        )

    def get_robot_yaw(self):
        if self.robot_pose is None:
            return None
        return self.yaw_from_quaternion(self.robot_pose.orientation)

    def get_ball_position(self):
        if self.ball_pose is None:
            return None
        return (
            float(self.ball_pose.position.x),
            float(self.ball_pose.position.y),
            float(self.ball_pose.position.z),
        )

    def get_cached_pose(self, model_name):
        if not self.received_model_states:
            return None, "尚未收到 model_states"
        if time.time() - self.last_model_states_time > self.model_states_fresh_timeout:
            return None, "model_states 缓存过旧"
        if model_name == self.robot_name:
            if not self.robot_in_model_states or self.robot_pose is None:
                return None, "model_states 中没有 robot"
            return self.get_robot_position(), None
        if model_name == self.ball_name:
            if not self.ball_in_model_states or self.ball_pose is None:
                return None, "model_states 中没有 football3"
            return self.get_ball_position(), None
        return None, "未知模型名: %s" % model_name

    def make_set_state_request(self, name, pose_xyz, quat_xyzw):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.reference_frame = "world"
        req.state.pose = Pose()
        req.state.pose.position.x = float(pose_xyz[0])
        req.state.pose.position.y = float(pose_xyz[1])
        req.state.pose.position.z = float(pose_xyz[2])
        req.state.pose.orientation = Quaternion(
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
            w=float(quat_xyzw[3]),
        )
        req.state.twist = Twist()
        return req

    def make_get_state_request(self, name):
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = "world"
        return req

    def begin_set_state(self, request_name, req):
        self.pending_set_future = self.set_state_client.call_async(req)
        self.pending_set_name = request_name
        self.pending_set_start_time = time.time()
        self.log_event("已发送 set_entity_state | %s" % request_name)

    def poll_set_state(self, next_state):
        if self.pending_set_future is None:
            return False

        if self.pending_set_future.done():
            try:
                resp = self.pending_set_future.result()
            except Exception as exc:
                self.get_logger().warn(
                    "%s 异常: %s，继续进入坐标确认阶段" % (self.pending_set_name, exc)
                )
                resp = None

            if resp is not None and resp.success:
                self.log_event("%s 返回成功，进入坐标确认" % self.pending_set_name)
            else:
                message = ""
                if resp is not None:
                    message = str(resp.status_message)
                if message:
                    self.get_logger().warn("%s 返回失败: %s" % (self.pending_set_name, message))
                else:
                    self.get_logger().warn("%s 未确认成功，进入坐标确认" % self.pending_set_name)

            self.clear_pending_set()
            self.switch_state(next_state)
            return True

        if time.time() - self.pending_set_start_time >= self.set_state_timeout:
            self.get_logger().warn("%s 等待超时，进入坐标确认" % self.pending_set_name)
            self.clear_pending_set()
            self.switch_state(next_state)
            return True

        self.log_status("状态=%s | 等待 %s 返回" % (self.state, self.pending_set_name))
        return False

    def request_get_entity_pose(self, model_name, request_name):
        if not self.get_state_client.service_is_ready():
            return "service_not_ready", None

        if self.pending_get_future is None:
            req = self.make_get_state_request(model_name)
            self.pending_get_future = self.get_state_client.call_async(req)
            self.pending_get_name = request_name
            self.pending_get_start_time = time.time()
            self.log_event("已发送 get_entity_state | %s" % request_name)
            return "pending", None

        if not self.pending_get_future.done():
            if time.time() - self.pending_get_start_time >= self.get_state_timeout:
                self.clear_pending_get()
                return "error", "%s 超时" % request_name
            self.log_status("状态=%s | 等待 %s 返回" % (self.state, self.pending_get_name))
            return "pending", None

        try:
            resp = self.pending_get_future.result()
        except Exception as exc:
            self.clear_pending_get()
            return "error", "%s 异常: %s" % (request_name, exc)

        self.clear_pending_get()
        if resp is None or not resp.success:
            message = "空响应" if resp is None else str(resp.status_message)
            return "error", "%s 失败: %s" % (request_name, message)

        pose = (
            float(resp.state.pose.position.x),
            float(resp.state.pose.position.y),
            float(resp.state.pose.position.z),
        )
        if model_name == self.robot_name:
            self.current_ball_pose = self.current_ball_pose
        elif model_name == self.ball_name:
            self.current_ball_pose = pose
        return "done", pose

    def confirm_with_cache_or_get(
        self,
        model_name,
        target_pose,
        tolerance,
        metric_name,
        confirm_timeout,
        next_state,
        success_text,
        fail_text,
    ):
        pose, reason = self.get_cached_pose(model_name)
        distance = None

        if pose is not None:
            if metric_name == "xy":
                distance = self.distance_xy(pose, target_pose)
            else:
                distance = self.distance_xyz(pose, target_pose)
            if distance < tolerance:
                self.log_event(
                    "%s | 来源=model_states | 当前=%s | 误差=%.6f"
                    % (success_text, self.format_pose(pose), distance)
                )
                self.switch_state(next_state)
                return

            elapsed = time.time() - self.state_start_time
            if elapsed < confirm_timeout:
                self.log_status(
                    "状态=%s | 等待 %s 到位 | 当前=%s | 目标=%s | 误差=%.6f"
                    % (
                        self.state,
                        model_name,
                        self.format_pose(pose),
                        self.format_pose(target_pose),
                        distance,
                    )
                )
                return
        else:
            self.log_status(
                "状态=%s | model_states 不能确认 %s | 原因=%s，切换 get_entity_state 备用确认"
                % (self.state, model_name, reason)
            )

        status, result = self.request_get_entity_pose(
            model_name,
            "确认 %s 实际坐标" % model_name,
        )
        if status == "pending":
            return
        if status == "service_not_ready":
            self.get_logger().error("%s：/gazebo/get_entity_state 服务不可用" % fail_text)
            self.enter_final_stop("%s：get_entity_state 不可用" % model_name)
            return
        if status == "error":
            self.get_logger().error("%s：%s" % (fail_text, result))
            self.enter_final_stop("%s：坐标确认失败" % model_name)
            return

        pose = result
        if metric_name == "xy":
            distance = self.distance_xy(pose, target_pose)
        else:
            distance = self.distance_xyz(pose, target_pose)

        if distance < tolerance:
            self.log_event(
                "%s | 来源=get_entity_state | 当前=%s | 误差=%.6f"
                % (success_text, self.format_pose(pose), distance)
            )
            self.switch_state(next_state)
            return

        self.get_logger().error(
            "%s | 当前=%s | 目标=%s | 误差=%.6f"
            % (fail_text, self.format_pose(pose), self.format_pose(target_pose), distance)
        )
        self.enter_final_stop("%s 未到目标位" % model_name)

    def plan_next_kick(self):
        current_ball = self.get_ball_position()
        if current_ball is None:
            current_ball = self.current_ball_pose
        self.current_ball_pose = current_ball
        self.first_plan_ball_pose = current_ball

        if current_ball is None:
            self.distance_to_exit = None
            self.next_kick_pose = None
            self.next_kick_yaw = None
            self.next_kick_safe = None
            self.get_logger().warn("PLAN_NEXT_KICK：当前 football3 坐标不可用")
            return

        push_dx = self.target_exit[0] - current_ball[0]
        push_dy = self.target_exit[1] - current_ball[1]
        push_dir = self.normalize_2d(push_dx, push_dy)
        self.distance_to_exit = math.hypot(push_dx, push_dy)

        self.log_event("当前白球坐标 | %s" % self.format_pose(current_ball))
        self.log_event("到出口距离 | %.6f" % self.distance_to_exit)

        if push_dir is None:
            self.next_kick_pose = None
            self.next_kick_yaw = None
            self.next_kick_safe = False
            self.get_logger().warn("PLAN_NEXT_KICK：白球已与出口目标重合，无法计算下一脚")
            return

        kick_x = current_ball[0] - push_dir[0] * self.approach_distance
        kick_y = current_ball[1] - push_dir[1] * self.approach_distance
        kick_z = self.kick_pose[2]
        self.next_kick_pose = (kick_x, kick_y, kick_z)
        self.next_kick_yaw = math.atan2(push_dy, push_dx)
        self.next_kick_safe = (
            self.safe_x_min <= kick_x <= self.safe_x_max
            and self.safe_y_min <= kick_y <= self.safe_y_max
        )

        self.log_event(
            "建议下一脚狗站位 kick_pose | %s" % self.format_pose(self.next_kick_pose)
        )
        self.log_event("建议 yaw | %.6f" % self.next_kick_yaw)
        self.log_event(
            "是否在安全范围内 | %s | x=[%.2f, %.2f] y=[%.2f, %.2f]"
            % (
                "是" if self.next_kick_safe else "否",
                self.safe_x_min,
                self.safe_x_max,
                self.safe_y_min,
                self.safe_y_max,
            )
        )
        if not self.next_kick_safe:
            self.get_logger().warn("建议下一脚站位超出安全范围，暂不自动修正")

    def drive_walk_to_target(
        self,
        pos_tolerance=None,
        yaw_tolerance=None,
        walk_speed=None,
        turn_speed=None,
    ):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None
        if self.walk_target_pose is None or self.walk_target_yaw is None:
            self.publish_zero_hold()
            return False, None
        if pos_tolerance is None:
            pos_tolerance = self.walk_pos_tolerance
        if yaw_tolerance is None:
            yaw_tolerance = self.walk_yaw_deadband
        if walk_speed is None:
            walk_speed = self.walk_speed
        if turn_speed is None:
            turn_speed = self.walk_turn_speed

        dx = self.walk_target_pose[0] - robot_pos[0]
        dy = self.walk_target_pose[1] - robot_pos[1]
        dist = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx) if dist > 1e-6 else self.walk_target_yaw
        target_yaw_for_motion = target_heading if dist > pos_tolerance else self.walk_target_yaw
        yaw_error = self.normalize_angle(target_yaw_for_motion - robot_yaw)
        final_yaw_error = self.normalize_angle(self.walk_target_yaw - robot_yaw)

        if dist < pos_tolerance and abs(final_yaw_error) < yaw_tolerance:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "dist": dist,
                "yaw_error": final_yaw_error,
            }

        if abs(yaw_error) > yaw_tolerance:
            yaw_rate = turn_speed if yaw_error > 0.0 else -turn_speed
            self.publish_walk_cmd(0.0, 0.0, yaw_rate, 0.03)
        else:
            yaw_rate = max(
                -turn_speed,
                min(turn_speed, yaw_error * 0.8),
            )
            self.publish_walk_cmd(walk_speed, 0.0, yaw_rate, 0.03)

        return False, {
            "robot": robot_pos,
            "dist": dist,
            "yaw_error": final_yaw_error,
        }

    def drive_turn_to_neg_y_after_first_push(self):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None

        yaw_error = self.normalize_angle(self.b_pose_yaw - robot_yaw)
        if abs(yaw_error) < 0.14:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "target": (robot_pos[0], robot_pos[1], robot_pos[2]),
                "x_error": 0.0,
                "y_error": 0.0,
                "yaw_error": yaw_error,
                "vx": 0.0,
                "vy": 0.0,
                "yaw_cmd": 0.0,
            }

        yaw_cmd = self.clamp(0.5 * yaw_error, -0.10, 0.10)
        self.publish_walk_cmd(0.0, 0.0, yaw_cmd, 0.03)

        return False, {
            "robot": robot_pos,
            "target": (robot_pos[0], robot_pos[1], robot_pos[2]),
            "x_error": 0.0,
            "y_error": 0.0,
            "yaw_error": yaw_error,
            "vx": 0.0,
            "vy": 0.0,
            "yaw_cmd": yaw_cmd,
        }

    def drive_side_shift_right_to_b(self):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None

        yaw_error = self.normalize_angle(self.b_pose_yaw - robot_yaw)
        x_error = self.b_pose_x - robot_pos[0]
        y_error = self.b_pose_y - robot_pos[1]
        b_target = (self.b_pose_x, self.b_pose_y, self.b_pose_z)
        dist = math.hypot(x_error, y_error)

        if dist < 0.10 and abs(yaw_error) < 0.18:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "target": b_target,
                "x_error": x_error,
                "y_error": y_error,
                "yaw_error": yaw_error,
                "vx": 0.0,
                "vy": 0.0,
                "yaw_cmd": 0.0,
            }

        vx = self.clamp(-0.6 * y_error, -0.08, 0.08)
        vy = self.clamp(0.8 * x_error, -0.05, 0.05)
        yaw_cmd = self.clamp(0.4 * yaw_error, -0.05, 0.05)
        self.publish_walk_cmd(vx, vy, yaw_cmd, 0.03)
        return False, {
            "robot": robot_pos,
            "target": b_target,
            "x_error": x_error,
            "y_error": y_error,
            "yaw_error": yaw_error,
            "vx": vx,
            "vy": vy,
            "yaw_cmd": yaw_cmd,
        }

    def drive_walk_neg_y_to_c(self):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None

        x_error = self.c_pose[0] - robot_pos[0]
        y_error = self.c_pose[1] - robot_pos[1]
        dist = math.hypot(x_error, y_error)
        yaw_error = self.normalize_angle(self.c_pose_yaw - robot_yaw)

        if dist < 0.12 and abs(yaw_error) < 0.20:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "target": self.c_pose,
                "x_error": x_error,
                "y_error": y_error,
                "dist": dist,
                "yaw_error": yaw_error,
                "vx": 0.0,
                "vy": 0.0,
                "yaw_cmd": 0.0,
            }

        vx = self.clamp(-0.7 * y_error, 0.06, 0.18)
        if abs(y_error) < 0.25:
            vx = self.clamp(-0.7 * y_error, 0.04, 0.09)
        vy = self.clamp(0.5 * x_error, -0.04, 0.04)
        yaw_cmd = self.clamp(0.45 * yaw_error, -0.06, 0.06)

        self.publish_walk_cmd(vx, vy, yaw_cmd, 0.03)
        return False, {
            "robot": robot_pos,
            "target": self.c_pose,
            "x_error": x_error,
            "y_error": y_error,
            "dist": dist,
            "yaw_error": yaw_error,
            "vx": vx,
            "vy": vy,
            "yaw_cmd": yaw_cmd,
        }

    def drive_turn_to_finish_yaw(self):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None

        yaw_error = self.normalize_angle(self.finish_yaw - robot_yaw)
        if abs(yaw_error) < 0.15:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "dist": self.distance_xy(robot_pos, self.finish_pose),
                "yaw_error": yaw_error,
                "vx": 0.0,
                "yaw_cmd": 0.0,
            }

        yaw_cmd = self.clamp(0.5 * yaw_error, -0.12, 0.12)
        self.publish_walk_cmd(0.0, 0.0, yaw_cmd, 0.03)
        return False, {
            "robot": robot_pos,
            "dist": self.distance_xy(robot_pos, self.finish_pose),
            "yaw_error": yaw_error,
            "vx": 0.0,
            "yaw_cmd": yaw_cmd,
        }

    def drive_walk_to_finish(self):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None

        dx = self.finish_pose[0] - robot_pos[0]
        dy = self.finish_pose[1] - robot_pos[1]
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(target_yaw - robot_yaw)

        if dist < 0.12:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "dist": dist,
                "yaw_error": yaw_error,
                "vx": 0.0,
                "yaw_cmd": 0.0,
            }

        if abs(yaw_error) > 0.45:
            vx = 0.0
            yaw_cmd = self.clamp(0.5 * yaw_error, -0.12, 0.12)
        else:
            vx = 0.16
            if dist < 0.5:
                vx = 0.08
            yaw_cmd = self.clamp(0.5 * yaw_error, -0.10, 0.10)

        self.publish_walk_cmd(vx, 0.0, yaw_cmd, 0.03)
        return False, {
            "robot": robot_pos,
            "dist": dist,
            "yaw_error": yaw_error,
            "vx": vx,
            "yaw_cmd": yaw_cmd,
        }

    def drive_walk_to_first_target(self, target_pose, pos_tolerance):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None

        dx = target_pose[0] - robot_pos[0]
        dy = target_pose[1] - robot_pos[1]
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(target_yaw - robot_yaw)

        if dist < pos_tolerance:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "dist": dist,
                "yaw_error": yaw_error,
                "vx": 0.0,
                "yaw_cmd": 0.0,
            }

        if abs(yaw_error) > 0.45:
            vx = 0.0
            yaw_cmd = self.clamp(0.5 * yaw_error, -0.12, 0.12)
        else:
            vx = 0.16
            if dist < 0.35:
                vx = 0.08
            yaw_cmd = self.clamp(0.5 * yaw_error, -0.10, 0.10)

        self.publish_walk_cmd(vx, 0.0, yaw_cmd, 0.03)
        return False, {
            "robot": robot_pos,
            "dist": dist,
            "yaw_error": yaw_error,
            "vx": vx,
            "yaw_cmd": yaw_cmd,
        }

    def drive_align_first_push_yaw(self):
        robot_pos = self.get_robot_position()
        robot_yaw = self.get_robot_yaw()
        if robot_pos is None or robot_yaw is None:
            self.publish_zero_hold()
            return False, None

        yaw_error = self.normalize_angle(self.first_push_yaw - robot_yaw)
        if abs(yaw_error) < 0.15:
            self.publish_zero_hold()
            return True, {
                "robot": robot_pos,
                "yaw_error": yaw_error,
                "vx": 0.0,
                "yaw_cmd": 0.0,
            }

        yaw_cmd = self.clamp(0.5 * yaw_error, -0.10, 0.10)
        self.publish_walk_cmd(0.0, 0.0, yaw_cmd, 0.03)
        return False, {
            "robot": robot_pos,
            "yaw_error": yaw_error,
            "vx": 0.0,
            "yaw_cmd": yaw_cmd,
        }

    def print_final_summary(self):
        if self.final_summary_printed:
            return

        final_robot_pos = self.get_robot_position()

        self.log_event(self.with_timing("FINAL_SUMMARY 汇总开始"))
        self.log_event(
            self.with_timing(
                "第一脚推球前球位置 | %s | 来源=%s"
                % (self.format_pose(self.ball_before), self.ball_before_source)
            )
        )
        self.log_event(
            self.with_timing("第一脚后球位置 | %s" % self.format_pose(self.first_after_ball_pose))
        )
        self.log_event(
            self.with_timing(
                "第一脚后等待是否完成 | %s" % ("是" if self.after_first_touch_wait_done else "否")
            )
        )
        self.log_event(
            self.with_timing(
                "是否到达第二踢球点 | %s" % ("是" if self.manual_side_reached_pose else "否")
            )
        )
        self.log_event(
            self.with_timing(
                "到达第二踢球点的总时间 | %s"
                % (
                    "%.2fs" % self.manual_side_arrival_total_time
                    if self.manual_side_arrival_total_time is not None
                    else "不可用"
                )
            )
        )
        self.log_event(
            self.with_timing(
                "人工第二踢球点 c_pose | %s" % self.format_pose(self.c_pose)
            )
        )
        self.log_event(
            self.with_timing(
                "是否执行人工横推 | %s" % ("是" if self.manual_side_push_executed else "否")
            )
        )
        self.log_event(
            self.with_timing(
                "人工横推前球位置 | %s" % self.format_pose(self.manual_side_ball_before)
            )
        )
        self.log_event(
            self.with_timing(
                "人工横推首次移动位置 | %s" % self.format_pose(self.manual_side_first_move_pos)
            )
        )
        self.log_event(
            self.with_timing(
                "人工横推后当前球位置 | %s" % self.format_pose(self.manual_side_final_ball_pose)
            )
        )
        self.log_event(
            self.with_timing(
                "人工横推实际执行时间 | %s"
                % (
                    "%.2fs" % self.manual_side_push_actual_duration
                    if self.manual_side_push_actual_duration is not None
                    else "不可用"
                )
            )
        )
        self.log_event(
            self.with_timing(
                "第二脚后球位置 | %s" % self.format_pose(self.manual_side_final_ball_pose)
            )
        )
        self.log_event(
            self.with_timing(
                "是否到达终点 | %s" % ("是" if self.finish_reached else "否")
            )
        )
        self.log_event(
            self.with_timing(
                "最终 robot 坐标 | %s" % self.format_pose(final_robot_pos)
            )
        )
        self.log_event(
            self.with_timing(
                "终点距离 | %s"
                % (
                    "%.6f" % self.finish_distance
                    if self.finish_distance is not None
                    else "不可用"
                )
            )
        )
        self.log_event(
            self.with_timing(
                "当前 football3 到 target_exit 距离 | %s"
                % (
                    "%.6f" % self.distance_to_exit
                    if self.distance_to_exit is not None
                    else "不可用"
                )
            )
        )
        if not self.manual_side_reached_pose:
            self.log_event(
                self.with_timing("未执行人工横推，因为未到达第二踢球点")
            )
        elif not self.manual_side_push_executed:
            self.log_event(
                self.with_timing("未执行人工横推原因 | %s" % self.manual_side_exec_reason)
            )
        self.log_event(
            self.with_timing("结束原因 | %s" % (self.stop_reason if self.stop_reason else "未记录"))
        )
        self.final_summary_printed = True

    def enter_final_stop(self, reason):
        self.stop_reason = reason
        if self.state != "FINAL_STOP":
            self.switch_state("FINAL_STOP")
        self.log_event("状态=FINAL_STOP | 原因=%s" % reason)

    def control_loop(self):
        if self.state == "FINAL_STOP":
            self.publish_damper_stop()
            return

        if self.elapsed_total() >= self.total_timeout and self.state not in (
            "FINAL_SUMMARY",
            "FINAL_STOP",
            "FINAL_DAMPER_STOP",
        ):
            self.stop_reason = "总流程超时"
            self.switch_state("FINAL_SUMMARY")
            return

        if self.state == "INIT":
            self.switch_state("WAIT_MODEL_STATES")
            return

        if self.state == "WAIT_MODEL_STATES":
            self.publish_zero_hold()
            if self.robot_pose is not None and self.ball_pose is not None:
                self.switch_state("RESET_BALL")
                return

            if time.time() - self.state_start_time >= self.model_states_wait_timeout:
                self.get_logger().error("5 秒内未同时读到 robot 和 football3，停止流程")
                self.enter_final_stop("WAIT_MODEL_STATES 超时")
                return

            self.log_status("状态=WAIT_MODEL_STATES | 等待 robot 和 football3 坐标")
            return

        if self.state == "RESET_BALL":
            self.publish_zero_hold()
            if not self.set_state_client.service_is_ready():
                self.log_status("状态=RESET_BALL | 等待 /gazebo/set_entity_state 服务")
                return
            if self.pending_set_future is None:
                req = self.make_set_state_request(
                    self.ball_name,
                    self.ball_default,
                    (0.0, 0.0, 0.0, 1.0),
                )
                self.begin_set_state("重置 football3 到默认位置", req)
                return
            self.poll_set_state("CONFIRM_BALL_RESET")
            return

        if self.state == "CONFIRM_BALL_RESET":
            self.publish_zero_hold()
            self.confirm_with_cache_or_get(
                self.ball_name,
                self.ball_default,
                self.ball_reset_tolerance,
                "xyz",
                self.ball_reset_confirm_timeout,
                "RESET_DOG_TO_STAGE_START",
                "football3 已回到默认位置",
                "football3 坐标确认失败",
            )
            return

        if self.state == "RESET_DOG_TO_STAGE_START":
            self.publish_zero_hold()
            if not self.set_state_client.service_is_ready():
                self.log_status("状态=RESET_DOG_TO_STAGE_START | 等待 /gazebo/set_entity_state 服务")
                return
            if self.pending_set_future is None:
                req = self.make_set_state_request(
                    self.robot_name,
                    self.stage_start_pose,
                    self.quaternion_from_yaw(self.stage_start_yaw),
                )
                self.begin_set_state("重置 robot 到第六关起点", req)
                return
            self.poll_set_state("CONFIRM_DOG_STAGE_START")
            return

        if self.state == "CONFIRM_DOG_STAGE_START":
            self.publish_zero_hold()
            self.confirm_with_cache_or_get(
                self.robot_name,
                self.stage_start_pose,
                self.robot_reset_tolerance,
                "xy",
                self.robot_reset_confirm_timeout,
                "RECOVERY_STAND",
                "robot 已到第六关起点",
                "robot 坐标确认失败",
            )
            return

        if self.state == "RECOVERY_STAND":
            self.publish_recovery_stand()
            robot_pos = self.get_robot_position()
            if robot_pos is None:
                self.log_status("状态=RECOVERY_STAND | 等待 robot 高度")
            else:
                if robot_pos[2] >= self.stand_height_threshold:
                    self.stand_good_counter += 1
                else:
                    self.stand_good_counter = 0
                if self.stand_good_counter >= self.stand_good_samples:
                    self.log_event(
                        "robot 已恢复站立 | z=%.6f | 连续样本=%d"
                        % (robot_pos[2], self.stand_good_counter)
                    )
                    self.switch_state("WALK_TO_FIRST_ROUTE_MID")
                    return
                self.log_status(
                    "状态=RECOVERY_STAND | robot_z=%.6f | 连续合格=%d/%d"
                    % (
                        robot_pos[2],
                        self.stand_good_counter,
                        self.stand_good_samples,
                    )
                )

            if time.time() - self.state_start_time >= self.recovery_stand_timeout:
                self.get_logger().error("RECOVERY_STAND 超时，狗未成功站起")
                self.enter_final_stop("RECOVERY_STAND 超时")
            return

        if self.state == "WALK_TO_FIRST_ROUTE_MID":
            reached, info = self.drive_walk_to_first_target(
                self.first_route_mid_pose,
                0.18,
            )
            if info is not None:
                self.log_first_push_motion(
                    info["robot"],
                    self.first_route_mid_pose,
                    info["dist"],
                    info["yaw_error"],
                    info["vx"],
                    info["yaw_cmd"],
                )
            if reached:
                self.switch_state("WALK_TO_FIRST_PUSH_POSE")
            return

        if self.state == "WALK_TO_FIRST_PUSH_POSE":
            reached, info = self.drive_walk_to_first_target(
                self.kick_pose,
                0.12,
            )
            if info is not None:
                self.log_first_push_motion(
                    info["robot"],
                    self.kick_pose,
                    info["dist"],
                    info["yaw_error"],
                    info["vx"],
                    info["yaw_cmd"],
                )
            if reached:
                self.log_event(self.with_timing("到达第一踢球点，准备第一脚"))
                self.switch_state("ALIGN_FIRST_PUSH_YAW")
            return

        if self.state == "ALIGN_FIRST_PUSH_YAW":
            aligned, info = self.drive_align_first_push_yaw()
            if info is not None:
                self.log_first_push_motion(
                    info["robot"],
                    self.kick_pose,
                    0.0,
                    info["yaw_error"],
                    info["vx"],
                    info["yaw_cmd"],
                )
            if aligned:
                self.switch_state("STAND_HOLD")
            return

        if self.state == "STAND_HOLD":
            self.publish_zero_hold()
            elapsed = time.time() - self.state_start_time
            if elapsed >= self.stand_hold_time:
                self.switch_state("RECORD_BEFORE")
                return
            self.log_status(
                "状态=STAND_HOLD | 保持站立中 | %.2f / %.2f 秒"
                % (elapsed, self.stand_hold_time)
            )
            return

        if self.state == "RECORD_BEFORE":
            self.publish_zero_hold()
            pose, reason = self.get_cached_pose(self.ball_name)
            if pose is None:
                status, result = self.request_get_entity_pose(
                    self.ball_name,
                    "读取推球前 football3 坐标",
                )
                if status == "pending":
                    return
                if status == "service_not_ready":
                    self.get_logger().error(
                        "RECORD_BEFORE 失败：/gazebo/get_entity_state 服务不可用 | 原因=%s"
                        % reason
                    )
                    self.enter_final_stop("推球前未读到 football3")
                    return
                if status == "error":
                    self.get_logger().error(
                        "RECORD_BEFORE 失败：未读到 football3 | 原因=%s | 备用查询=%s"
                        % (reason, result)
                    )
                    self.enter_final_stop("推球前未读到 football3")
                    return
                pose = result
                self.ball_before_source = "get_entity_state"
            else:
                self.ball_before_source = "model_states"
            self.ball_before = pose
            self.current_ball_pose = pose
            self.log_event("记录推球前球位置 | %s" % self.format_pose(self.ball_before))
            self.switch_state("PUSH")
            return

        if self.state == "PUSH":
            self.publish_push_cmd()
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball
                moved_distance = self.distance_xy(current_ball, self.ball_before)
                if moved_distance > self.ball_move_threshold:
                    self.first_kick_success = True
                    self.first_move_pos = current_ball
                    self.first_move_time = time.time()
                    self.publish_zero_hold()
                    self.log_event(
                        "检测到白球首次移动 | football3=%s | moved_dist=%.6f"
                        % (self.format_pose(current_ball), moved_distance)
                    )
                    self.switch_state("BACK_OFF_AFTER_TOUCH")
                    return
                self.log_status(
                    "推球 | football3=%s | delta=%.6f"
                    % (self.format_pose(current_ball), moved_distance)
                )
            else:
                self.log_status("推球 | football3=不可用 | delta=不可用")

            if time.time() - self.state_start_time >= self.push_max_time:
                self.first_kick_success = False
                self.get_logger().warn("第一脚未推动白球")
                self.stop_reason = "第一脚未推动白球"
                self.switch_state("FINAL_SUMMARY")
            return

        if self.state == "BACK_OFF_AFTER_TOUCH":
            elapsed = time.time() - self.state_start_time
            if elapsed < self.backoff_time:
                self.publish_backoff_cmd()
                self.log_status(
                    "等待下一脚时机 | football3=%s | delta=后退脱离中"
                    % self.format_pose(self.current_ball_pose)
                )
                return
            self.publish_zero_hold()
            self.switch_state("WAIT_AFTER_FIRST_TOUCH_FOR_MANUAL_SIDE")
            return

        if self.state == "WAIT_AFTER_FIRST_TOUCH_FOR_MANUAL_SIDE":
            self.publish_zero_hold()
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball

            self.log_status(
                "第一脚后等待 %.1f 秒准备前往第二踢球点 | football3=%s"
                % (
                    self.after_first_touch_wait,
                    self.format_pose(self.current_ball_pose),
                )
            )
            if time.time() - self.state_start_time >= self.after_first_touch_wait:
                self.reset_manual_side_tracking()
                self.after_first_touch_wait_done = True
                if current_ball is not None:
                    self.first_after_ball_pose = tuple(current_ball)
                self.manual_side_walk_start_time = time.time()
                self.manual_side_exec_reason = "第一脚后等待完成，开始原地转体"
                self.log_event(
                    self.with_timing(
                        "第二踢球点行走超时 | manual_side_walk_timeout=%.1f"
                        % self.manual_side_walk_timeout
                    )
                )
                self.switch_state("TURN_TO_NEG_Y_AFTER_FIRST_PUSH")
            return

        if self.state == "TURN_TO_NEG_Y_AFTER_FIRST_PUSH":
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball
            reached, walk_info = self.drive_turn_to_neg_y_after_first_push()
            if walk_info is not None:
                self.log_second_kick_motion(
                    "第一脚后原地转体",
                    walk_info["robot"],
                    walk_info["target"],
                    walk_info["x_error"],
                    walk_info["y_error"],
                    walk_info["yaw_error"],
                    walk_info["vx"],
                    walk_info["vy"],
                    walk_info["yaw_cmd"],
                )
            if reached:
                self.switch_state("SIDE_SHIFT_RIGHT_TO_B")
                return
            if time.time() - self.manual_side_walk_start_time >= self.manual_side_walk_timeout:
                self.get_logger().warn(
                    "总时间=%.2fs | 未到达第二踢球点，禁止横推"
                    % self.elapsed_total()
                )
                self.manual_side_reached_pose = False
                self.manual_side_exec_reason = "未到达第二踢球点，禁止横推"
                self.stop_reason = self.manual_side_exec_reason
                self.switch_state("FINAL_SUMMARY")
            return

        if self.state == "SIDE_SHIFT_RIGHT_TO_B":
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball
            reached, walk_info = self.drive_side_shift_right_to_b()
            if walk_info is not None:
                self.log_second_kick_motion(
                    "前往 B 点",
                    walk_info["robot"],
                    walk_info["target"],
                    walk_info["x_error"],
                    walk_info["y_error"],
                    walk_info["yaw_error"],
                    walk_info["vx"],
                    walk_info["vy"],
                    walk_info["yaw_cmd"],
                )
            if reached:
                self.switch_state("WALK_NEG_Y_TO_C")
                return
            if time.time() - self.manual_side_walk_start_time >= self.manual_side_walk_timeout:
                self.get_logger().warn(
                    "总时间=%.2fs | 未到达第二踢球点，禁止横推"
                    % self.elapsed_total()
                )
                self.manual_side_reached_pose = False
                self.manual_side_exec_reason = "未到达第二踢球点，禁止横推"
                self.stop_reason = self.manual_side_exec_reason
                self.switch_state("FINAL_SUMMARY")
            return

        if self.state == "WALK_NEG_Y_TO_C":
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball
            reached, walk_info = self.drive_walk_neg_y_to_c()
            if walk_info is not None:
                self.log_second_kick_motion(
                    "前往 C 点",
                    walk_info["robot"],
                    walk_info["target"],
                    walk_info["x_error"],
                    walk_info["y_error"],
                    walk_info["yaw_error"],
                    walk_info["vx"],
                    walk_info["vy"],
                    walk_info["yaw_cmd"],
                )
            if reached:
                self.manual_side_reached_pose = True
                self.manual_side_arrival_total_time = self.elapsed_total()
                self.log_event(self.with_timing("到达 C 点，准备向左横推"))
                self.manual_side_exec_reason = "已到达第二踢球点，允许人工横推"
                self.switch_state("RECHECK_MANUAL_SIDE_BALL")
                return
            if time.time() - self.manual_side_walk_start_time >= self.manual_side_walk_timeout:
                self.get_logger().warn(
                    "总时间=%.2fs | 未到达第二踢球点，禁止横推"
                    % self.elapsed_total()
                )
                self.manual_side_reached_pose = False
                self.manual_side_exec_reason = "未到达第二踢球点，禁止横推"
                self.stop_reason = self.manual_side_exec_reason
                self.switch_state("FINAL_SUMMARY")
            return

        if self.state == "RECHECK_MANUAL_SIDE_BALL":
            self.publish_zero_hold()
            pose, reason = self.get_cached_pose(self.ball_name)
            if pose is None:
                status, result = self.request_get_entity_pose(
                    self.ball_name,
                    "人工横推前读取 football3 坐标",
                )
                if status == "pending":
                    return
                if status == "service_not_ready":
                    self.manual_side_exec_reason = "人工横推前读取球坐标失败：get_entity_state 不可用"
                    self.stop_reason = self.manual_side_exec_reason
                    self.switch_state("FINAL_SUMMARY")
                    return
                if status == "error":
                    self.manual_side_exec_reason = "人工横推前读取球坐标失败：读不到 football3"
                    self.stop_reason = self.manual_side_exec_reason
                    self.switch_state("FINAL_SUMMARY")
                    return
                pose = result
            robot_yaw = self.get_robot_yaw()
            if robot_yaw is None:
                self.manual_side_exec_reason = "人工横推前读取 robot yaw 失败"
                self.stop_reason = self.manual_side_exec_reason
                self.switch_state("FINAL_SUMMARY")
                return
            self.recheck_ball_pose = pose
            self.current_ball_pose = pose
            self.manual_side_ball_before = pose
            self.manual_side_push_hold_yaw = robot_yaw
            self.manual_side_push_actual_duration = None
            self.manual_side_last_push_log_time = 0.0
            self.log_event(
                self.with_timing("人工横推前球位置 | football3=%s" % self.format_pose(pose))
            )
            self.manual_side_exec_reason = "开始执行人工横推"
            self.switch_state("MANUAL_SIDE_PUSH")
            return

        if self.state == "MANUAL_SIDE_PUSH":
            robot_yaw = self.get_robot_yaw()
            if robot_yaw is None or self.manual_side_push_hold_yaw is None:
                self.publish_zero_hold()
                self.manual_side_exec_reason = "人工横推中读取 robot yaw 失败"
                self.stop_reason = self.manual_side_exec_reason
                self.switch_state("FINAL_SUMMARY")
                return

            elapsed = time.time() - self.state_start_time
            yaw_error = self.normalize_angle(self.manual_side_push_hold_yaw - robot_yaw)
            yaw_cmd = self.clamp(
                self.manual_side_push_yaw_hold_gain * yaw_error,
                -self.manual_side_push_yaw_cmd_limit,
                self.manual_side_push_yaw_cmd_limit,
            )
            self.publish_manual_side_push_hold_cmd(yaw_cmd)

            moved_distance = 0.0
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball
                if self.manual_side_ball_before is not None:
                    moved_distance = self.distance_xy(
                        current_ball, self.manual_side_ball_before
                    )

            now = time.time()
            if (
                self.manual_side_last_push_log_time == 0.0
                or now - self.manual_side_last_push_log_time >= 1.0
            ):
                ball_text = (
                    self.format_pose(self.current_ball_pose)
                    if self.current_ball_pose is not None
                    else "不可用"
                )
                self.log_event(
                    self.with_timing(
                        "人工横推中 | elapsed=%.2f | vy=%.3f | yaw_error=%.3f | ball_moved=%.3f | football3=%s"
                        % (
                            elapsed,
                            self.manual_side_push_vy,
                            yaw_error,
                            moved_distance,
                            ball_text,
                        )
                    )
                )
                self.manual_side_last_push_log_time = now

            if elapsed < self.manual_side_push_min_time:
                return

            if moved_distance >= self.manual_side_ball_move_threshold:
                self.manual_side_push_executed = True
                self.manual_side_first_move_pos = current_ball
                self.manual_side_push_actual_duration = elapsed
                self.publish_zero_hold()
                self.log_event(
                    "检测到人工横推首次明显移动 | football3=%s | moved_dist=%.6f | elapsed=%.2f"
                    % (self.format_pose(current_ball), moved_distance, elapsed)
                )
                self.switch_state("BACK_OFF_AFTER_MANUAL_SIDE_PUSH")
                return

            if elapsed >= self.manual_side_push_max_time:
                self.manual_side_push_executed = (
                    moved_distance >= self.manual_side_ball_move_threshold
                )
                if self.manual_side_push_executed and self.manual_side_first_move_pos is None:
                    self.manual_side_first_move_pos = current_ball
                self.manual_side_push_actual_duration = elapsed
                self.publish_zero_hold()
                self.log_event(
                    "人工横推达到最大执行时间 | elapsed=%.2f | moved_dist=%.6f"
                    % (elapsed, moved_distance)
                )
                self.switch_state("BACK_OFF_AFTER_MANUAL_SIDE_PUSH")
                return
            return

        if self.state == "BACK_OFF_AFTER_MANUAL_SIDE_PUSH":
            elapsed = time.time() - self.state_start_time
            if elapsed < self.backoff_time:
                self.publish_manual_side_backoff_cmd()
                self.log_status(
                    "等待下一脚时机 | football3=%s | delta=后退脱离中"
                    % self.format_pose(self.current_ball_pose)
                )
                return
            self.publish_zero_hold()
            if self.current_ball_pose is not None:
                self.manual_side_final_ball_pose = tuple(self.current_ball_pose)
            self.switch_state("TURN_TO_FINISH_YAW")
            return

        if self.state == "TURN_TO_FINISH_YAW":
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball
            reached, info = self.drive_turn_to_finish_yaw()
            if info is not None:
                self.finish_distance = info["dist"]
                self.log_finish_motion(
                    info["robot"],
                    info["dist"],
                    info["yaw_error"],
                    info["vx"],
                    info["yaw_cmd"],
                )
            if reached:
                self.switch_state("WALK_TO_FINISH")
                return
            return

        if self.state == "WALK_TO_FINISH":
            current_ball = self.get_ball_position()
            if current_ball is not None:
                self.current_ball_pose = current_ball
            reached, info = self.drive_walk_to_finish()
            if info is not None:
                self.finish_distance = info["dist"]
                self.log_finish_motion(
                    info["robot"],
                    info["dist"],
                    info["yaw_error"],
                    info["vx"],
                    info["yaw_cmd"],
                )
            if reached:
                self.finish_reached = True
                self.finish_distance = info["dist"] if info is not None else 0.0
                self.switch_state("FINAL_DAMPER_STOP")
                return
            return

        if self.state == "FINAL_DAMPER_STOP":
            self.publish_damper_stop()
            if self.final_damper_start_time == 0.0:
                self.final_damper_start_time = time.time()
            if time.time() - self.final_damper_start_time >= 1.0:
                self.stop_reason = self.stop_reason or "到达终点并执行阻尼停止"
                self.switch_state("FINAL_STOP")
            return

        if self.state == "PLAN_DYNAMIC_SECOND_KICK":
            self.manual_side_exec_reason = "旧的上方第二脚逻辑已停用"
            self.stop_reason = self.manual_side_exec_reason
            self.switch_state("FINAL_SUMMARY")
            return

        if self.state in (
            "WAIT_BALL_SETTLE_OR_TIMEOUT",
            "WALK_TO_MANUAL_CORRIDOR",
            "WALK_TO_MANUAL_SIDE_POSE",
            "WALK_TO_DYNAMIC_SECOND_KICK",
            "RECHECK_SECOND_BALL",
            "SECOND_PUSH",
            "BACK_OFF_AFTER_SECOND_TOUCH",
            "PLAN_EXIT_PUSH",
            "WALK_TO_EXIT_PUSH_POSE",
            "RECHECK_EXIT_BALL",
            "EXIT_PUSH",
            "BACK_OFF_AFTER_EXIT_TOUCH",
            "TRACK_BALL_FOR_SIDE_PUSH",
            "PLAN_SIDE_PUSH_POSE",
            "WALK_TO_SIDE_PUSH_POSE",
            "RECHECK_SIDE_PUSH_BALL",
            "SIDE_PUSH",
            "BACK_OFF_AFTER_SIDE_PUSH",
            "WAIT_AFTER_SIDE_PUSH",
        ):
            self.manual_side_exec_reason = "旧的动态横推/第三脚逻辑已停用"
            self.stop_reason = self.manual_side_exec_reason
            self.switch_state("FINAL_SUMMARY")
            return

        if self.state == "FINAL_SUMMARY":
            self.publish_zero_hold()
            if self.current_ball_pose is None:
                self.current_ball_pose = self.get_ball_position()
            if self.manual_side_final_ball_pose is None and self.manual_side_push_executed:
                self.manual_side_final_ball_pose = self.current_ball_pose
            robot_pos = self.get_robot_position()
            if robot_pos is not None:
                self.finish_distance = self.distance_xy(robot_pos, self.finish_pose)
            final_ball = self.current_ball_pose
            if final_ball is not None:
                self.distance_to_exit = self.distance_xy(
                    final_ball,
                    (self.target_exit[0], self.target_exit[1], final_ball[2]),
                )
            self.print_final_summary()
            if self.finish_reached:
                self.switch_state("FINAL_STOP")
            else:
                self.enter_final_stop(self.stop_reason or "流程结束")
            return


def main(args=None):
    rclpy.init(args=args)
    node = FootballTest3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.log_event("收到 KeyboardInterrupt，执行 damper stop")
        try:
            for _ in range(5):
                node.publish_damper_stop()
                time.sleep(0.05)
        except Exception:
            pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()