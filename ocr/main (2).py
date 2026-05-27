"""
用于真狗测试
"""
import lcm
import sys
import os
import time
import math
import copy
import toml
from threading import Thread, Lock
import numpy as np
from datetime import datetime
import traceback

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pyzbar.pyzbar import decode
from std_msgs.msg import String
import struct
import numpy as np

# 尝试导入语音播报模块
try:
    from protocol.msg import AudioPlayExtend, BmsStatus
    SPEECH_AVAILABLE = True
    print("语音播报模块导入成功")
except ImportError as e:
    print(f"语音播报模块导入失败: {e}")
    SPEECH_AVAILABLE = False

from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from robot_control_response_lcmt import robot_control_response_lcmt



# 导入customized_gait模块
import subprocess

# 导入识别模块
try:
    from ocr_recognition import OCRRecognizer, recognize_image
    OCR_AVAILABLE = True
    print("文字识别模块导入成功")
except ImportError as e:
    print(f"文字识别模块导入失败: {e}")
    OCR_AVAILABLE = False

# 黄灯检测相关常量
PIXEL_PER_METER_YELLOW = 385  # 1米约180像素
STOP_DISTANCE = 0.5  # 50cm
distance1 = 0.2

# 限高杆检测相关常量（来自height_limit.py）
PIXEL_PER_METER_RED = 180  # 1米约180像素（用于红色限高杆检测）
LIMIT_STOP_DISTANCE = 1.0  # 1米处开始执行自定义步态

# s_bby功能相关常量
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])

# 导入s_bby功能需要的模块
from file_send_lcmt import file_send_lcmt

# s_bby功能：全局robot_cmd字典
robot_cmd = {
    'mode':0, 'gait_id':0, 'contact':0, 'life_count':0,
    'vel_des':[0.0, 0.0, 0.0],
    'rpy_des':[0.0, 0.0, 0.0],
    'pos_des':[0.0, 0.0, 0.0],
    'acc_des':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'ctrl_point':[0.0, 0.0, 0.0],
    'foot_pose':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'step_height':[0.0, 0.0],
    'value':0,  'duration':0
    }

def recognize_photo_labels(image_filename):
    """
    识别照片中的标签
    
    Args:
        image_filename: 图像文件名
        
    Returns:
        str: 识别到的标签字符串，如果识别失败返回空字符串
    """
    if not OCR_AVAILABLE:
        print("模块不可用，跳过识别")
        print("默认返回B-1标签")
        return "B-1"
    
    try:
        print("起立...")
        robot.msg.mode = 12  
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.msg.duration = 3000
        robot.ctrl.Send_cmd(robot.msg)
        
        print("开始识别...")
        # 识别刚拍摄的图片，使用完整路径
        image_path = f"saved_images/{image_filename}"
        if not os.path.exists(image_path):
            # 如果saved_images目录下没有，尝试当前目录
            image_path = image_filename
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                print("跳过识别")
                print("默认返回B-1标签")
                return "B-1"
        
        print(f"使用图像文件: {image_path}")
        result = recognize_image(image_path)
        
        # 检查结果是否有效
        if result is None:
            print("识别失败: 返回结果为空")
            print("默认返回B-1标签")
            return "B-1"
        elif not isinstance(result, dict):
            print(f"识别失败: 返回结果格式错误，类型为 {type(result)}")
            print("默认返回B-1标签")
            return "B-1"
        elif 'success' not in result:
            print(f"识别失败: 返回结果缺少success字段，结果为: {result}")
            print("未识别到任何标签")
            print("默认返回B-1标签")
            return "B-1"
        elif result.get('success', False):
            detected_labels = result.get('detected_labels', [])
            print(f"识别成功！检测到标签: {detected_labels}")
            print(f"识别方法: {result.get('recognition_method', '未知')}")
            
            # 显示详细信息
            matched_labels = result.get('matched_labels', [])
            for match in matched_labels:
                method_type = "(二维码)" if match.get('type') == 'qrcode' else "(文字)"
                print(f"  {match.get('label', '未知')} {method_type}: '{match.get('original_text', '')}' (置信度: {match.get('confidence', 0):.2f})")
            
            # 返回第一个识别到的标签，如果没有则返回空字符串
            return detected_labels[0] if detected_labels else "B-1"
        else:
            print(f"识别失败: {result.get('error', '未知错误')}")
            print("未识别到任何标签")
            print("默认返回B-1标签")
            return "B-1"
            
    except Exception as e:
        print(f"识别过程出错: {e}")
        import traceback
        traceback.print_exc()
        print("识别异常，默认返回B-1标签")
        return "B-1"



# PID控制器类
# 用于实现比例-积分-微分控制算法，带有积分限制和输出限制
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, integral_limit=None):
        """
        初始化PID控制器变体1
        
        参数:
            kp (float): 比例系数 - 控制系统对当前误差的响应强度
            ki (float): 积分系数 - 控制系统对累积误差的响应强度
            kd (float): 微分系数 - 控制系统对误差变化率的响应强度
            setpoint (float): 目标值/设定点 - 系统期望达到的状态值
            integral_limit (float, optional): 积分项限制值 - 防止积分饱和
        """
        self.kp = kp                  # 比例系数
        self.ki = ki                  # 积分系数
        self.kd = kd                  # 微分系数
        self.setpoint = setpoint      # 目标值/设定点
        self.error_sum = 0.0          # 误差累积和（用于积分项）
        self.error_prev = 0.0         # 上一次的误差值（用于微分项）
        self.integral_limit = integral_limit  # 积分限制，防止积分饱和
        self.last_time = time.time()  # 上次更新的时间戳
        self.output_limit = None      # 输出限制，防止控制量过大

    def update(self, value, dt=None):
        """
        更新PID控制器并计算输出控制量
        
        参数:
            value (float): 当前测量值/反馈值
            dt (float, optional): 时间间隔，如果为None则自动计算
            
        返回:
            float: 计算得到的控制输出值
        """
        # 如果未提供时间间隔，则自动计算
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time  # 计算时间差
            self.last_time = current_time      # 更新上次时间戳
            if dt <= 0:                        # 防止除以零错误
                dt = 1e-16                     # 设置一个极小的正数
                
        # 计算当前误差（目标值 - 当前值）
        error = self.setpoint - value
        
        # 更新积分项（误差的累积）
        self.error_sum += error * dt
        
        # 应用积分限制，防止积分饱和
        if self.integral_limit is not None:
            self.error_sum = max(min(self.error_sum, self.integral_limit), -self.integral_limit)
            
        # 计算微分项（误差的变化率）
        error_diff = (error - self.error_prev) / dt if dt > 0 else 0
        self.error_prev = error  # 更新上一次误差
        
        # 计算PID输出：比例项 + 积分项 + 微分项
        output = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        
        # 应用输出限制，防止控制量过大
        if self.output_limit is not None:
            output = max(min(output, self.output_limit), -self.output_limit)
            
        return output  # 返回计算得到的控制输出值

class Robot_Ctrl(object):
    def __init__(self):
        self.rec_thread = Thread(target=self.rec_responce)
        self.send_thread = Thread(target=self.send_publish)
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = Lock()
        self.delay_cnt = 0
        self.mode_ok = 0
        self.gait_ok = 0
        self.runing = 1

    def run(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        self.send_thread.start()
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)
        if(self.rec_msg.order_process_bar >= 95):
            self.mode_ok = self.rec_msg.mode
        else:
            self.mode_ok = 0

    def rec_responce(self):
        while self.runing:
            self.lc_r.handle()
            time.sleep( 0.002 )

    def Wait_finish(self, mode, gait_id,dura=800):
        count = 0
        while self.runing and count < dura: 
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            else:
                time.sleep(0.003)
                count += 1

    def Wait_finish_stone(self, mode, gait_id):
    #石板路步态用
        count = 0
        while self.runing and count < 6000: 
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            else:
                time.sleep(0.005)
                count += 1

    def Wait_finish_low_height(self, mode, gait_id):
    #限高杆步态用
        count = 0
        while self.runing and count < 1700: 
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            else:
                time.sleep(0.005)
                count += 1

    def send_publish(self):
        while self.runing:
            self.send_lock.acquire()
            if self.delay_cnt > 5: # Heartbeat signal 20HZ, It is used to maintain the heartbeat when life count is not updated
                self.lc_s.publish("robot_control_cmd",self.cmd_msg.encode())
                self.delay_cnt = 0
            self.delay_cnt += 1
            self.send_lock.release()
            time.sleep( 0.003 )

    def Send_cmd(self, msg):
        self.send_lock.acquire()
        self.delay_cnt = 50
        # 确保life_count在有符号字节范围内(-128到127)
        msg.life_count = msg.life_count % 128
        if msg.life_count > 127:
            msg.life_count = msg.life_count - 128
        # 确保value和duration是整数类型
        msg.value = int(msg.value) if msg.value is not None else 0
        msg.duration = int(msg.duration) if msg.duration is not None else 0
        self.cmd_msg = msg
        self.send_lock.release()

    def quit(self):
        self.runing = 0
        self.rec_thread.join()
        self.send_thread.join()
    
    def safe_increment_life_count(self, msg):
        """安全地递增life_count，确保在有符号字节范围内"""
        msg.life_count = (msg.life_count + 1) % 128
        return msg

class LowPassFilter:
    """简易低通滤波器"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

# s_bby功能：曲线处理器类
class CurveProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.lock = Lock()
        self.current_level = 0  # 当前偏移程度
        
        # 性能优化：预分配缓存变量
        self._hsv_cache = None
        self._mask_cache = None
        self._last_image_shape = None
        
        # 预计算常量
        self._roi_height = 100  # ROI区域高度
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._font_thickness = 2
        
        # 颜色常量（避免重复创建元组）
        self._green_color = (0, 255, 0)
        self._red_color = (0, 0, 255)
        self._blue_color = (255, 0, 0)
        self._white_color = (255, 255, 255)
        
    def image_callback(self, msg):
        try:
            # 与s弯检测处理保持一致，直接转换为bgr8格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.image = cv_image
                # 重置缓存（图像更新时）
                self._hsv_cache = None
                self._mask_cache = None
        except Exception as e:
            print(f'Error processing image: {str(e)}')
    
    def process_image(self, input_image=None):
        # 如果提供了输入图像，使用它；否则使用内部图像
        if input_image is not None:
            if input_image is None:
                return None, None, None
            # 性能优化：避免不必要的copy，直接使用输入图像
            image = input_image
            use_input_image = True
        else:
            with self.lock:
                if self.image is None:
                    return None, None, None
                # 性能优化：只在需要绘制时才复制图像
                image = self.image
                use_input_image = False
        
        h, w = image.shape[:2]
        mid_x = w >> 1  # 位运算替代除法
        
        # 性能优化：检查图像尺寸是否变化，重用HSV和mask缓存
        current_shape = (h, w)
        if (not use_input_image and 
            self._last_image_shape == current_shape and 
            self._hsv_cache is not None and 
            self._mask_cache is not None):
            # 重用缓存的HSV和mask
            mask = self._mask_cache
        else:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 创建黄色掩膜
            mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
            
            # 缓存结果（仅对内部图像）
            if not use_input_image:
                self._hsv_cache = hsv
                self._mask_cache = mask
                self._last_image_shape = current_shape
        
        # 性能优化：直接切片获取底部区域，避免额外变量
        roi_start = h - self._roi_height
        bottom_region = mask[roi_start:, :]
        
        # 查找左半部分最靠近中线的黄点
        left_point = None
        # 性能优化：使用range的step参数，减少循环次数
        for x in range(mid_x-1, -1, -1):
            col = bottom_region[:, x]
            # 性能优化：使用np.any的axis参数，更高效
            if np.any(col):
                y = np.argmax(col)
                left_point = (x, y + roi_start)  # 直接使用roi_start
                break
        
        # 如果没有找到黄点，使用左下角
        if left_point is None:
            left_point = (0, h-1)
        
        # 查找右半部分最靠近中线的黄点
        right_point = None
        for x in range(mid_x, w):
            col = bottom_region[:, x]
            if np.any(col):
                y = np.argmax(col)
                right_point = (x, y + roi_start)  # 直接使用roi_start
                break
        
        # 如果没有找到黄点，使用右下角
        if right_point is None:
            right_point = (w-1, h-1)
        
        # 计算点到中线的距离
        left_dist = mid_x - left_point[0]
        right_dist = right_point[0] - mid_x
        
        # 计算偏移程度（-6到6）
        diff = left_dist - right_dist
        max_diff = w >> 2  # 位运算替代除法
        
        if diff == 0:
            level = 0
        else:
            # 将差值映射到-6到6范围
            level = round(6 * diff / max_diff)
            level = max(min(level, 6), -6)  # 限制在-6到6之间
        
        # 更新当前偏移程度
        self.current_level = level
        
        # 性能优化：只在需要时才创建图像副本用于绘制
        if use_input_image:
            display_image = input_image.copy()
        else:
            display_image = image.copy()
        
        # 在图像上标记点和信息
        cv2.circle(display_image, left_point, 8, self._green_color, -1)  # 左点绿色
        cv2.circle(display_image, right_point, 8, self._red_color, -1)  # 右点红色
        cv2.line(display_image, (mid_x, 0), (mid_x, h), self._blue_color, 2)  # 中线蓝色
        
        # 性能优化：预格式化字符串，减少f-string开销
        left_text = f"Left: {left_dist}"
        right_text = f"Right: {right_dist}"
        level_text = f"Level: {level}"
        
        # 添加文本信息
        cv2.putText(display_image, left_text, (10, 30), 
                    self._font, self._font_scale, self._green_color, self._font_thickness)
        cv2.putText(display_image, right_text, (w-200, 30), 
                    self._font, self._font_scale, self._red_color, self._font_thickness)
        cv2.putText(display_image, level_text, (mid_x-100, 30), 
                    self._font, self._font_scale, self._white_color, self._font_thickness)
        
        return display_image, level, (left_dist, right_dist)

#控制类
class ImageSubscriber(Node):
    def __init__(self, robot_controller, topic_name="/image", stop_distance=STOP_DISTANCE):
        super().__init__("camera_subscriber")
        self.print_count = 1  # 只打印一次初始化信息
        self.cv_bridge = CvBridge()
        self.robot_controller = robot_controller
        self.stop_requested = False  # 用作打印控制，不实际停止
        self.last_detect_time = 0.0
        self.cooldown = 1.0  # 秒，避免频繁重复打印
        self.stop_distance = stop_distance
        self.current_image = None  # 存储当前图像

        from rclpy.qos import QoSProfile
        qos = QoSProfile(depth=10)  # 默认深度10，兼容性较好

        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_cb,
            qos
        )
        self.get_logger().info(f"订阅 {topic_name} 话题成功，等待图像...")

    def image_cb(self, msg: Image) -> None:
        if self.print_count:
            self.get_logger().info(f"收到来自 {self.subscription.topic_name} 话题的图像: {msg.width}x{msg.height} 编码: {msg.encoding}")
            self.print_count = 0

        try:
            # 根据图像编码格式进行适当转换
            if msg.encoding == "rgb8":
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
                # 将RGB转换为BGR以便OpenCV处理
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgr8":
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            else:
                # 尝试转换为bgr8
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}，编码格式: {msg.encoding}")


class RobotController:
    def __init__(self):
        """初始化控制器、消息对象"""
        # 初始化ROS2
        rclpy.init()
        

        
        # 创建图像订阅器
        self.image_subscriber = ImageSubscriber(self, '/image')
        
        # 创建RGB图像订阅器
        self.image_rgb_subscriber = ImageSubscriber(self, '/image_rgb')
        
        # 初始化s_bby功能：曲线处理器
        self.curve_processor = CurveProcessor()
        
        # 连接RGB图像订阅器到曲线处理器
        self.image_rgb_subscriber.subscription = self.image_rgb_subscriber.create_subscription(
            Image,
            '/image_rgb',
            self.curve_processor.image_callback,
            10)
        
        # 初始化s_bby功能的LCM通信
        self.lcm_usergait = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        

        
        # 初始化语音播报功能
        if SPEECH_AVAILABLE:
            # 需要一个ROS节点来创建发布器，暂时使用image_subscriber
            self.speech_publisher = self.image_subscriber.create_publisher(
                AudioPlayExtend, '/mi_desktop_48_b0_2d_7b_02_dc/speech_play_extend', 10)
            print("语音播报功能已初始化")
        else:
            self.speech_publisher = None
            print("语音播报功能不可用")
        
        # 初始化语音识别订阅器
        if SPEECH_AVAILABLE:
            self.asr_subscription = self.image_subscriber.create_subscription(
                String, '/mi_desktop_48_b0_2d_7b_02_dc/asr_text',
                self.asr_callback, 10)
            print("语音识别功能已初始化")
        else:
            self.asr_subscription = None
            print("语音识别功能不可用")
        
        # 初始化电池状态监控
        if SPEECH_AVAILABLE:
            self.bms_subscription = self.image_subscriber.create_subscription(
                BmsStatus, '/mi_desktop_48_b0_2d_7b_02_dc/bms_status',
                self.bms_callback, 10)
            print("电池状态监控已初始化")
        else:
            self.bms_subscription = None
            print("电池状态监控不可用")
        
        # 语音检测相关状态变量
        self.countdown_active = False
        self.current_count = 0
        self.countdown_timer = None
        self.is_charging = False
        self.was_charging = False
        
        # 二维码检测相关变量
        self.qr_content = None
        self.qr_detection_stopped = False
        
        # 黄灯检测相关变量
        self.yellow_light_detected = False
        self.last_yellow_detect_time = 0.0
        self.yellow_cooldown = 1.0  # 秒，避免频繁重复打印
        self.stop_distance = STOP_DISTANCE
        
        # 限高杆检测相关变量
        self.limit_barrier_detected = False
        self.last_limit_detect_time = 0.0
        self.limit_cooldown = 1.0  # 秒，避免频繁重复打印
        self.limit_stop_distance = LIMIT_STOP_DISTANCE
        

        
        # 创建ROS2执行线程
        self.ros_thread = Thread(target=self.ros_spin)
        self.ros_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
        self.ros_thread.start()
        
        # 初始化机器人控制
        self.ctrl = Robot_Ctrl()
        self.msg = robot_control_cmd_lcmt()
        self.ctrl.run()  # 启动通信线程
        
        # 初始化拍照功能
        self.save_dir = "saved_images"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 等待系统稳定
        time.sleep(1.0)

    def __del__(self):
        """析构时清理资源"""
        self.ctrl.quit()
        rclpy.shutdown()
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止机器人控制
            if hasattr(self, 'ctrl'):
                self.ctrl.quit()
            
            # 销毁ROS2节点
            if hasattr(self, 'image_subscriber'):
                self.image_subscriber.destroy_node()
            if hasattr(self, 'image_rgb_subscriber'):
                self.image_rgb_subscriber.destroy_node()
            
            # 关闭ROS2
            if rclpy.ok():
                rclpy.shutdown()
                
            # 关闭OpenCV窗口
            cv2.destroyAllWindows()
            
            print("资源清理完成")
        except Exception as e:
            print(f"资源清理异常: {e}")
        
    def ros_spin(self):
        """ROS2消息处理循环"""
        while rclpy.ok():
            rclpy.spin_once(self.image_subscriber, timeout_sec=0.05)
            rclpy.spin_once(self.image_rgb_subscriber, timeout_sec=0.05)
            

        

    
    def safe_increment_life_count(self):
        """安全地递增self.msg.life_count，确保在有符号字节范围内"""
        self.msg.life_count = (self.msg.life_count + 1) % 128
        return self.msg.life_count
    
    def play_speech(self, text):
        """语音播报功能"""
        if SPEECH_AVAILABLE and self.speech_publisher is not None:
            try:
                msg = AudioPlayExtend()
                msg.module_name = "voice_interaction"
                msg.is_online = True
                msg.text = text
                self.speech_publisher.publish(msg)
                print(f'语音播报: {text}')
            except Exception as e:
                print(f"语音播报失败: {e}")
        else:
            print(f"语音播报不可用，文本内容: {text}")
    
    def asr_callback(self, msg):
        """语音识别回调函数"""
        text = msg.data.strip()
        print(f"收到语音指令: {text}")
        
        if text == "完成":
            print("[终端输出] 检测到完成指令")
            self.play_speech("完成确认")
            # 设置语音完成检测标志
            self.voice_completion_detected = True
            
        # 其他语音指令不进行播报，只在终端显示
    
    def bms_callback(self, msg):
        """电池状态回调函数"""
        self.was_charging = self.is_charging
        self.is_charging = msg.power_wired_charging
        
        if self.was_charging and not self.is_charging:
            self.handle_disconnection()
            
        elif not self.was_charging and self.is_charging:
            self.play_speech("充电中")
    
    def start_countdown(self):
        """黄灯倒计时处理"""
        if self.countdown_timer:
            self.countdown_timer.cancel()
        self.countdown_active = True
        self.current_count = 5
        self.countdown_timer = self.image_subscriber.create_timer(1.0, self.countdown_callback)
    
    def handle_disconnection(self):
        """充电线拔出处理流程"""
        self.play_speech("充电线拔出")
        
        # 严格遵循官方例程的动作顺序
        self.msg.mode = 12  # Recovery stand
        self.msg.gait_id = 0
        self.msg.life_count = (self.msg.life_count + 1) % 128
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(12, 0)
        
        self.msg.mode = 11  # Locomotion
        self.msg.gait_id = 27  # TROT_SLOW
        self.msg.vel_des = [0.2, 0, 0]
        self.msg.duration = 4000
        self.msg.life_count = (self.msg.life_count + 1) % 128
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)
        
        self.msg.mode = 7  # PureDamper
        self.msg.life_count = (self.msg.life_count + 1) % 128
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(7, 0)
    
    def countdown_callback(self):
        """倒计时处理"""
        if self.current_count > 0:
            self.play_speech(str(self.current_count))
            self.current_count -= 1
        else:
            self.countdown_timer.cancel()
            self.countdown_active = False
    

    
    def wait_for_voice_completion(self, duration=30.0):
        """等待语音"完成"指令"""
        print(f"等待语音'完成'指令，最长等待时间: {duration}秒")
        print("请说'完成'来继续...")
        
        # 添加检测标志
        self.voice_completion_detected = False
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                # 检查是否检测到"完成"语音指令
                if hasattr(self, 'voice_completion_detected') and self.voice_completion_detected:
                    print("检测到'完成'指令，继续执行")
                    return True
                time.sleep(0.1)  # 100ms间隔
        except KeyboardInterrupt:
            print("\n语音检测被中断")
        
        print("语音检测超时")
        return False
    
    # ===== s_bby功能相关方法 =====
    def start_image_processor(self):
        """启动图像处理器"""
        # 使用现有的RGB图像订阅器的图像数据
        print("曲线检测器已启动")
        # 等待图像数据稳定
        time.sleep(2.0)
    
    def wait_for_action(self, cmd_msg, mode, duration):
        """等待动作执行完成的同时处理图像"""
        wait_time = duration / 1000.0
        steps = int(wait_time * 5)  # 每0.2秒发送一次心跳
        levels = []  # 存储检测到的偏移程度
        valid_levels = []  # 存储该次循环中的有效数据
        
        for _ in range(steps):
            # 处理当前图像
            processed_img, level, dists = self.curve_processor.process_image()
            
            if processed_img is not None:
                # 显示处理后的图像
                #cv2.imshow('Curve Detection', processed_img)
                #cv2.waitKey(1)
                
                # 打印偏移程度
                if level is not None:
                    # 检查是否为无效数据（左距离为0，右距离为1）或（左距离为1，右距离为0）或（320，319）
                    if (dists[0] == 0 and dists[1] == 1) or (dists[0] == 1 and dists[1] == 0) or (dists[0] == 320 and dists[1] == 319) or (dists[0] == 319 and dists[1] == 320):
                        # 如果是无效数据，使用该次循环最近的有效数据
                        if valid_levels:
                            level = valid_levels[-1]
                            print(f"检测到无效数据({dists[0]},{dists[1]})，使用该次循环最近有效数据: {level}")
                        else:
                            print(f"检测到无效数据({dists[0]},{dists[1]})，但暂无有效数据可用，跳过此次检测")
                            # 即使无效数据也要添加到levels中以保持循环计数
                            levels.append(0)  # 添加默认值0
                            # 发送心跳命令
                            self.ctrl.Send_cmd(cmd_msg)
                            time.sleep(0.1)
                            continue
                    else:
                        # 如果是有效数据，添加到有效数据列表
                        valid_levels.append(level)
                    
                    print(f"偏移程度: {level} (左距离: {dists[0]}, 右距离: {dists[1]})")
                    levels.append(level)
            
            # 发送心跳命令
            self.ctrl.Send_cmd(cmd_msg)
            time.sleep(0.1)
        
        # 返回检测到的偏移程度列表
        return levels
    
    def recover_stand(self):
        """让机器狗站起来"""
        print("执行恢复站立...")
        self.msg.mode = 12  # Recovery stand
        self.msg.gait_id = 0
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        time.sleep(2)
    
    def forward_walk(self, duration=5000):
        """使用官方步态向前走并处理图像"""
        print("执行官方步态向前走并检测曲线...")
        self.msg.mode = 11  # 运动模式
        self.msg.gait_id = 27  # 官方步态ID
        self.msg.duration = duration
        self.msg.vel_des = [0.2, 0, 0]  # x方向速度0.2
        self.msg.rpy_des = [0, 0, 0]
        self.msg.pos_des = [0, 0, 0.27]
        self.msg.step_height = [0.06, 0.06]
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        levels = self.wait_for_action(self.msg, self.msg.mode, duration)
        return levels
    
    def stand_pose(self, duration=1000):
        """保持站立姿态"""
        print("保持站立姿态...")
        self.msg.mode = 11
        self.msg.gait_id = 1
        self.msg.duration = duration
        self.msg.vel_des = [0, 0, 0]
        self.msg.rpy_des = [0, 0, 0]
        self.msg.pos_des = [0, 0, 0.25]
        self.msg.step_height = [0.06, 0.06]
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        levels = self.wait_for_action(self.msg, self.msg.mode, duration)
        return levels
    
    def get_gait_params_index(self, usergait_list_index):
        """获取步态参数索引映射"""
        gait_mapping = {
            0: 0,   # 自定义低机身高度前进
            1: 1,   # 自定义前倾直行
            2: 2,   # 自定义前倾左转，转向速度0.05
            3: 3,   # 自定义前倾左转，转向速度0.1
            4: 4,   # 自定义前倾左转，转向速度0.15
            5: 5,   # 自定义前倾左转，转向速度0.2
            6: 6,   # 自定义前倾左转，转向速度0.25
            7: 7,   # 自定义前倾左转，转向速度0.3
            8: 8,   # 自定义前倾右转，转向速度0.05
            9: 9,   # 自定义前倾右转，转向速度0.1
            10: 10, # 自定义前倾右转，转向速度0.15
            11: 11, # 自定义前倾右转，转向速度0.2
            12: 12, # 自定义前倾右转，转向速度0.25
            13: 13, # 自定义前倾右转，转向速度0.3
        }
        return gait_mapping.get(usergait_list_index, None)
    
    def execute_single_gait(self, gait_index, duration=2000):
        """执行单个步态并返回检测到的偏移程度列表"""
        print(f"执行步态索引 {gait_index}")
        gait_params_index = self.get_gait_params_index(gait_index)
        if gait_params_index is not None:
            self.initialize_gait_files(gait_params_index)
            try:
                with open("Gait_Params_full.toml", 'r') as f:
                    msg = file_send_lcmt()
                    msg.data = f.read()
                    self.lcm_usergait.publish("user_gait_file", msg.encode())
                    time.sleep(0.1)
            except FileNotFoundError:
                print("Gait_Params_full.toml文件未找到，跳过LCM发布")
        
        levels = self.execute_gait_sequence(gait_index, 1, duration)
        return levels
    
    def execute_gait_sequence(self, start_idx, count, duration=2000):
        """执行步态序列并返回检测到的偏移程度列表"""
        try:
            with open("Usergait_List.toml", 'r') as user_gait_list:
                steps = toml.load(user_gait_list)
        except FileNotFoundError:
            print("Usergait_List.toml文件未找到")
            return []
        
        all_levels = []  # 存储所有检测到的偏移程度
        
        for step in steps['step'][start_idx:start_idx + count]:
            self.msg.mode = step['mode']
            self.msg.value = step['value']
            self.msg.contact = step['contact']
            self.msg.gait_id = step['gait_id']
            self.msg.duration = duration  # 使用传入的持续时间
            self.msg.life_count += 1
            
            for i in range(3):
                self.msg.vel_des[i] = step['vel_des'][i]
                self.msg.rpy_des[i] = step['rpy_des'][i]
                self.msg.pos_des[i] = step['pos_des'][i]
                self.msg.acc_des[i] = step['acc_des'][i]
                self.msg.acc_des[i+3] = step['acc_des'][i+3]
                self.msg.foot_pose[i] = step['foot_pose'][i]
                self.msg.ctrl_point[i] = step['ctrl_point'][i]
            
            for i in range(2):
                self.msg.step_height[i] = step['step_height'][i]
            
            self.ctrl.Send_cmd(self.msg)
            if duration > 0:
                levels = self.wait_for_action(self.msg, step['mode'], duration)
                all_levels.extend(levels)
        
        return all_levels
    
    def initialize_gait_files(self, gait_index=None):
        """初始化步态文件"""
        try:
            steps = toml.load("Gait_Params.toml")
        except FileNotFoundError:
            print("Gait_Params.toml文件未找到")
            return
        
        robot_cmd = {
            'mode':0, 'gait_id':0, 'contact':0, 'life_count':0,
            'vel_des':[0.0, 0.0, 0.0],
            'rpy_des':[0.0, 0.0, 0.0],
            'pos_des':[0.0, 0.0, 0.0],
            'acc_des':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'ctrl_point':[0.0, 0.0, 0.0],
            'foot_pose':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'step_height':[0.0, 0.0],
            'value':0,  'duration':0
        }
        
        full_steps = {'step':[]}
        
        if gait_index is not None:
            if gait_index < len(steps['step']):
                i = steps['step'][gait_index]
                cmd = copy.deepcopy(robot_cmd)
                cmd['duration'] = i['duration']
                if i['type'] == 'usergait':
                    cmd['mode'] = 11
                    cmd['gait_id'] = i['gait_id']
                    cmd['vel_des'] = i['body_vel_des']
                    cmd['rpy_des'] = i['body_pos_des'][0:3]
                    cmd['pos_des'] = i['body_pos_des'][3:6]
                    cmd['foot_pose'][0:2] = i['landing_pos_des'][0:2]
                    cmd['foot_pose'][2:4] = i['landing_pos_des'][3:5]
                    cmd['foot_pose'][4:6] = i['landing_pos_des'][6:8]
                    cmd['ctrl_point'][0:2] = i['landing_pos_des'][9:11]
                    cmd['step_height'][0] = math.ceil(i['step_height'][0] * 1e3) + math.ceil(i['step_height'][1] * 1e3) * 1e3
                    cmd['step_height'][1] = math.ceil(i['step_height'][2] * 1e3) + math.ceil(i['step_height'][3] * 1e3) * 1e3
                    cmd['acc_des'] = i['weight']
                    cmd['value'] = i['use_mpc_traj']
                    cmd['contact'] = math.floor(i['landing_gain'] * 1e1)
                    cmd['ctrl_point'][2] = i['mu']
                    full_steps['step'].append(cmd)
        
        with open("Gait_Params_full.toml", 'w') as f:
            f.write("# Gait Params\n")
            f.writelines(toml.dumps(full_steps))
    
    def get_next_gait_index(self, level):
        """根据偏移程度获取下一个步态索引"""
        gait_mapping = {
            0: 1,   # 直行
            1: 2,   # 左转0.05
            2: 3,   # 左转0.1
            3: 4,   # 左转0.15
            4: 5,   # 左转0.2
            5: 6,   # 左转0.25
            6: 7,   # 左转0.3
            -1: 8,  # 右转0.05
            -2: 9,  # 右转0.1
            -3: 10, # 右转0.15
            -4: 11, # 右转0.2
            -5: 12, # 右转0.25
            -6: 13, # 右转0.3
        }
        return gait_mapping.get(level, 1)  # 默认为直行
    
    def s_bby_main_function(self):
        """s_bby主要功能：基于相机数据的自适应步态控制"""
        # 启动图像处理器
        self.start_image_processor()
        
        try:
            # 初始化步态文件
            self.initialize_gait_files()
                    
            # 发送步态定义文件
            with open("Gait_Def.toml", 'r') as f:
                usergait_msg = file_send_lcmt()
                usergait_msg.data = f.read()
                self.lcm_usergait.publish("user_gait_file", usergait_msg.encode())
                time.sleep(0.5)
                
            with open("Gait_Params.toml", 'r') as f:
                usergait_msg = file_send_lcmt()
                usergait_msg.data = f.read()
                self.lcm_usergait.publish("user_gait_file", usergait_msg.encode())
                time.sleep(0.1)
            
            # # 1. 起立 (索引14)
            # print("执行起立动作...")
            # self.execute_single_gait(14, 2000)
            
            # 初始步态：直行
            current_gait_index = 1
            count = 0
            # 主循环：根据偏移程度切换步态
            while count <= 83 :
                # 执行当前步态2秒，并获取检测到的偏移程度
                levels = self.execute_single_gait(current_gait_index, 1500)
                
                if levels:
                    # 取最后一个偏移程度作为决策依据
                    last_level = levels[-1]
                    print(f"当前偏移程度: {last_level}, 执行步态索引: {current_gait_index}")
                    count = count + 1
                    print(f"执行次数: {count}/84    ")
                    
                    # 根据偏移程度获取下一个步态索引
                    next_gait_index = self.get_next_gait_index(last_level)
                    
                    # 更新当前步态索引
                    current_gait_index = next_gait_index
                else:
                    # 如果没有检测到偏移程度，保持当前步态
                    print("未检测到偏移程度，保持当前步态")
                    count = count + 1
                    print(f"执行次数: {count}/65    ")
            
            # # 5. 趴下 (索引16)
            # print("执行趴下动作...")
            # self.execute_single_gait(16, 2000)
            
        except KeyboardInterrupt:
            # 在中断时执行趴下动作
            print("收到中断信号，执行趴下动作...")
            self.execute_single_gait(16, 2000)
            self.msg.mode = 7  # PureDamper
            self.msg.gait_id = 0
            self.msg.duration = 0
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
        # finally:
        #     cv2.destroyAllWindows()
    
    
    def origin_a_qr(self):
        """从出库到扫描二维码"""
        

        # 2. 第一段直线前进 - 持续行走并检测黄色边界
        print("开始前进，持续检测黄色边界直到距离大于2.0米...")
        
        # 开始持续行走模式
        self.msg.mode = 11  # 行走模式
        self.msg.gait_id = 27  # 步态类型
        self.msg.vel_des = [0.2, 0, 0]  # 期望速度[x, y, yaw] (m/s)
        self.msg.step_height = [0.1, 0.1]  # 步高(m)
        self.msg.duration = 10000  # 设置较长的持续时间，实际由检测结果控制
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        
        # 持续检测黄色边界
        start_time = time.time()
        detection_count = 0
        max_walk_time = 30.0  # 最大行走时间（安全限制）
        
        print("开始边界检测循环...")
        while time.time() - start_time < max_walk_time:
            # 执行黄色边界检测
            detected, distance, position = self.detect_yellow_boundary_rgb(enable_debug_visual=False)
            detection_count += 1
            
            if detected or distance > 0:  # 检测到黄色边界或发现黄色区域
                current_time = time.time() - start_time
                if distance > 2.45:
                     print(f"[检测{detection_count}次] [{current_time:.1f}s] 检测到黄色边界距离{distance:.2f}m > 2.45m，停止前进")
                     break
                else:
                     print(f"[检测{detection_count}次] [{current_time:.1f}s] 检测到黄色边界距离{distance:.2f}m ≤ 2.45m，继续前进")
            else:
                # 每50次检测输出一次状态
                if detection_count % 50 == 0:
                    current_time = time.time() - start_time
                    print(f"[检测{detection_count}次] [{current_time:.1f}s] 未检测到黄色边界，继续前进")
            
            # 控制检测频率（20Hz）
            time.sleep(0.05)
        
        # 停止行走
        print("停止前进...")
        self.msg.mode = 12  # 站立模式
        self.msg.gait_id = 0
        self.msg.duration = 0
        self.msg.life_count = (self.msg.life_count + 1) % 128
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(12, 0)
        
        total_time = time.time() - start_time
        print(f"第一段前进完成，总用时: {total_time:.1f}秒，检测次数: {detection_count}次")

        print("转弯90度...")
        dura = 2600  
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0, 0, -0.8]  
        self.msg.duration = dura
        self.msg.step_height = [0.02, 0.02]  # 转向时降低步高
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27,500)

        # 4. 第二段直线前进
        print("开始前进...")
        # dura = 2500
        dura = 5000
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0.25, 0, 0]  # 纯前进
        self.msg.step_height = [0.1, 0.1]
        self.msg.duration = dura
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        time.sleep(4)  # 等待移动完成


    def a_qr_s(self,detected_labels):
        """二维码到配货到s弯"""



        if detected_labels == "A-2":  # A-2
            self.play_speech("A区库位2")


            # print("开始前进...")
            # dura = 2700
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            detected, walk_time = self.walk_straight_until_yellow_line(
                max_duration=25000,  # 最大15秒
                velocity=0.1,       # 较慢的速度便于观察
                enable_debug=False,   # 启用调试输出
                detection_position = "quarter"
            )

            # print("开始前进...")
            # dura = 700
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27,500)  

            print("开始左移...")
            dura = 5300
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [0, 0.25, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            # print("开始后退...")
            # dura = 2300
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [-0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)


            # detected, walk_time = robot.walk_straight_until_yellow_line(
            #     max_duration=8000,  # 最大15秒
            #     velocity=-0.1,       # 较慢的速度便于观察
            #     enable_debug=False,   # 启用调试输出
            #     detection_position = "quarter"
            # )

            print("开始后退...")
            dura = 2800
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [-0.25, 0, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            print("到达装货区")


            print("开始配货")
            dura = 5000
            self.msg.mode = 7
            self.msg.gait_id = 1
            self.msg.duration = dura
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(7,1)

        
            # 等待语音"完成"指令
            self.wait_for_voice_completion(duration=30.0)
            time.sleep(3)

            self.msg.mode = 12  # Recovery stand
            self.msg.gait_id = 0
            self.msg.life_count += 1  # Command will take effect when life_count update
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(12, 0)

            # print("开始前进")
            # dura = 3000
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)


            detected, walk_time = self.walk_straight_until_yellow_line(
                max_duration=40000,  # 最大15秒
                velocity=0.1,       # 较慢的速度便于观察
                enable_debug=False,   # 启用调试输出
                detection_position = "quarter"
            )

            # print("开始前进...")
            # dura = 1200
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27,500)

            print("开始右移...")
            dura = 4800
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [0, -0.25, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

        elif detected_labels == "A-1":  # A-1
            self.play_speech("A区库位1")

            # print("开始前进...")
            # dura = 4000
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            # print("开始前进...")
            # dura = 3500
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            detected, walk_time = self.walk_straight_until_yellow_line(
                max_duration=25000,  # 最大15秒
                velocity=0.1,       # 较慢的速度便于观察
                enable_debug=False,   # 启用调试输出
                detection_position = "quarter"
            )

            # print("开始前进...")
            # dura = 700
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27,500)  

            print("开始右移...")
            dura = 4300
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [0, -0.25, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            # print("开始后退...")
            # dura = 3000
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [-0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            # detected, walk_time = robot.walk_straight_until_yellow_line(
            #     max_duration=8000,  # 最大15秒
            #     velocity=-0.1,       # 较慢的速度便于观察
            #     enable_debug=False,   # 启用调试输出
            #     detection_position = "quarter"
            # )


            print("开始后退...")
            dura = 2800
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [-0.25, 0, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            print("到达装货区")

            print("开始配货")
            dura = 5000
            self.msg.mode = 7
            self.msg.gait_id = 1
            self.msg.duration = dura
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(7,1)



            # 等待语音"完成"指令
            self.wait_for_voice_completion(duration=30.0)
            time.sleep(3)

            self.msg.mode = 12  # Recovery stand
            self.msg.gait_id = 0
            self.msg.life_count += 1  # Command will take effect when life_count update
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(12, 0)

            # print("开始前进")
            # dura = 3000
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            detected, walk_time = robot.walk_straight_until_yellow_line(
                max_duration=12000,  # 最大15秒
                velocity=0.1,       # 较慢的速度便于观察
                enable_debug=False,   # 启用调试输出
                detection_position = "quarter"
            )

            # print("开始前进...")
            # dura = 1000
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27,500)

            print("开始左移...")
            dura = 4300
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [0, 0.25, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)
        else:
            print("未识别到有效值")

        detected, walk_time = robot.walk_straight_until_yellow_line_2(
            max_duration=8000,  # 最大15秒
            velocity=-0.1,       # 较慢的速度便于观察
            enable_debug=False,   # 启用调试输出
            detection_position = "quarter"
        )


        print("开始后退...")
        dura = 5800
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [-0.25, 0, 0]
        self.msg.step_height = [0.1, 0.1]
        self.msg.duration = dura  # Until close to turn
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        print("左转弯90度...")
        dura = 2600  
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0, 0, 0.8]  
        self.msg.duration = dura
        self.msg.step_height = [0.02, 0.02]  # 转向时降低步高
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        print("正在进入S弯")
        # S弯
        dura = 1000
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0.31, 0.02, 0]
        self.msg.duration = dura
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        self.msg.mode = 12  # Recovery stand
        self.msg.gait_id = 0
        self.msg.life_count += 1  # Command will take effect when life_count update
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(12, 0)  
    
    def arrow_s(self,arrow_result):
        if arrow_result:
            print(f"箭头检测测试成功，检测到方向: {arrow_result}")
            if arrow_result == "left":
                # print("开始前进...")
                # dura = 1000
                # self.msg.mode = 11
                # self.msg.gait_id = 27
                # self.msg.vel_des = [0.25, 0, 0]
                # self.msg.step_height = [0.1, 0.1]
                # self.msg.duration = dura  
                # self.msg.life_count += 1
                # self.ctrl.Send_cmd(self.msg)
                # self.ctrl.Wait_finish(11, 27)
                
                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=5000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False,   # 启用调试输出
                    detection_position = "last"
                )  

                self.msg.mode = 12  # 站立模式
                self.msg.gait_id = 0
                self.msg.duration = 0
                self.msg.life_count = (self.msg.life_count + 1) % 128
                self.ctrl.Send_cmd(self.msg)

                print("开始前进...")
                dura = 1700
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.1, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,500)

                print("开始左移...")
                dura = 4600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0, 0.25, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)
            elif arrow_result == "right":
                # print("开始前进...")
                # dura = 1000
                # self.msg.mode = 11
                # self.msg.gait_id = 27
                # self.msg.vel_des = [0.25, 0, 0]
                # self.msg.step_height = [0.1, 0.1]
                # self.msg.duration = dura  
                # self.msg.life_count += 1
                # self.ctrl.Send_cmd(self.msg)
                # self.ctrl.Wait_finish(11, 27)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=5000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False,   # 启用调试输出
                    detection_position = "last"
                )  

                self.msg.mode = 12  # 站立模式
                self.msg.gait_id = 0
                self.msg.duration = 0
                self.msg.life_count = (self.msg.life_count + 1) % 128
                self.ctrl.Send_cmd(self.msg)

                print("开始前进...")
                dura = 1700
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.1, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,500)

                print("开始右移...")
                dura = 4700
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0, -0.25, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)
            else:
                print("无效结果")
        else:
            print("箭头检测测试超时，未检测到箭头")
    
    def hill_return(self):
        """坡道 - 使用固定参数进行上下坡"""
        print("开始上下坡控制...")
        
        # 上坡
        self.simple_hill_up(walk_time=12.0)
        
        # print("左移...")
        # self.msg.mode = 11 
        # self.msg.gait_id = 27
        # self.msg.vel_des = [0.0, 0.2, 0.0]
        # self.msg.duration = 700
        # self.msg.step_height = [0.06, 0.06]
        # self.msg.life_count += 1
        # self.ctrl.Send_cmd(self.msg)
        # self.ctrl.Wait_finish(11, 27,500)

        # 下坡
        self.simple_hill_down(walk_time=9.0)

        detected, walk_time = self.walk_straight_until_yellow_line(
            max_duration=40000,  
            velocity=0.08,       # 较慢的速度便于观察
            enable_debug=False   # 启用调试输出
        )  

        print("前进...")
        dura = 700
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0.3, 0, 0]
        self.msg.duration = dura
        self.msg.step_height = [0.1, 0.1]
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

    def hill(self):
        """坡道 - 使用固定参数进行上下坡"""
        print("开始上下坡控制...")
        
        # 上坡
        self.simple_hill_up(walk_time=12.0)
        
        # 下坡
        self.simple_hill_down(walk_time=12.0)



    # 上坡路PID循迹功能
    def simple_hill_up(self, walk_time=20.0):
        """上坡简单控制功能，使用固定参数"""
        print("进入上坡简单控制")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < walk_time:
                # 准备控制命令
                msg = robot_control_cmd_lcmt()
                msg.mode = 11  # 行走模式
                msg.gait_id = 26  
                msg.duration = 100  # 持续时间
                
                # 设置固定速度
                msg.vel_des = [0.2, 0, 0]  # 前进速度0.3m/s
                
                # 设置固定姿态
                msg.rpy_des = [0, 0.3, 0]  # pitch设为0.3
                
                # 身体高度
                msg.pos_des = [0, 0, 0.23]  # 保持一定高度
                
                # 步高
                msg.step_height = [0.06, 0.06]
                
                # 确保 life_count 在有效范围内
                msg.life_count = (self.msg.life_count + 1) % 127
                self.ctrl.Send_cmd(msg)
                self.msg.life_count = msg.life_count
                
                time.sleep(0.1)  # 100ms间隔
                
        except Exception as e:
            print(f"上坡简单控制出错: {str(e)}")
        finally:
            print("上坡简单控制结束")

    def simple_hill_down(self, walk_time=20.0):
        """下坡简单控制功能，使用固定参数"""
        print("进入下坡简单控制")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < walk_time:
                # 准备控制命令
                msg = robot_control_cmd_lcmt()
                msg.mode = 11  # 行走模式
                msg.gait_id = 26  
                msg.duration = 100  # 持续时间
                
                # 设置固定速度
                msg.vel_des = [0.2, 0, 0]  # 前进速度0.3m/s
                
                # 设置固定姿态
                msg.rpy_des = [0, -0.25, 0]  # pitch设为-0.25
                
                # 身体高度
                msg.pos_des = [0, 0, 0.23]  # 保持一定高度
                
                # 步高
                msg.step_height = [0.05, 0.05]
                
                # 确保 life_count 在有效范围内
                msg.life_count = (self.msg.life_count + 1) % 127
                self.ctrl.Send_cmd(msg)
                self.msg.life_count = msg.life_count
                
                time.sleep(0.1)  # 100ms间隔
                
        except Exception as e:
            print(f"下坡简单控制出错: {str(e)}")
        finally:
            print("下坡简单控制结束")

    def stone_road(self):
        """石板路 - 使用PID控制进行石板路循迹"""

        print("开始石板路PID控制...")
        
        # 石板路PID循迹

        self.execute_custom_gait_stone_road_pid()

    
    def qr_detection_logic(self):
        """二维码检测核心逻辑"""
        if self.image_subscriber.current_image is None:
            return None
            
        # 图像预处理，增强对比度
        alpha = 2.0
        beta = 200
        adjusted_image = cv2.convertScaleAbs(self.image_subscriber.current_image, alpha=alpha, beta=beta)
        gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
        
        # 识别二维码
        decoded = decode(gray)
        if decoded:
            qr_info = decoded[0]
            content = qr_info.data.decode('utf-8')
            print(f"检测到二维码内容: {content}")
            self.qr_content = content
            self.qr_detection_stopped = True
            
            # 绘制二维码边框和内容（可选）
            rect = qr_info.rect
            cv2.rectangle(self.image_subscriber.current_image,
                          (rect.left, rect.top),
                          (rect.left + rect.width, rect.top + rect.height),
                          (0, 255, 0), 2)
            cv2.putText(self.image_subscriber.current_image, content,
                        (rect.left, rect.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return content
        else:
            return None
    
    def detect_qr_code(self, timeout=28.0):
        """检测二维码的主函数
        
        Args:
            timeout: 检测超时时间（秒）
            
        Returns:
            str: 检测到的二维码内容，如果超时则返回None
        """
        print("开始二维码检测...")
        self.qr_content = None
        self.qr_detection_stopped = False
        
        start_time = time.time()
        
        while time.time() - start_time < timeout and not self.qr_detection_stopped:
            content = self.qr_detection_logic()
            if content:
                print(f"二维码检测成功: {content}")
                return content
            time.sleep(0.1)  # 控制检测频率
        
        if time.time() - start_time >= timeout:
            print("二维码检测超时")
        
        return None
    
    def detect_arrow(self, image, min_area=500):
        """箭头检测核心算法（基于arrows.py的完整实现）
        
        Args:
            image: 输入图像
            min_area: 最小轮廓面积阈值
            
        Returns:
            tuple: (箭头方向, 掩膜图像, 可视化图像)
        """
        result_vis = None
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 绿色阈值（可按现场光照调整）
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # 形态学清理
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # 找轮廓，兼容 OpenCV 不同版本返回值
            contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            if not contours:
                return None, mask, None

            # 选取面积最大的轮廓（更稳定）
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            arrow_direction = None
            vis = image.copy()

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    # 后面的轮廓面积更小，直接跳出
                    break

                # 计算凸包（索引形式）并求凸缺陷
                hull = cv2.convexHull(cnt, returnPoints=False)
                if hull is None or len(hull) < 3:
                    continue

                defects = cv2.convexityDefects(cnt, hull)
                if defects is None:
                    continue

                # 找到深度最大的缺陷点（通常对应箭尾窝点）
                max_depth = 0
                tip_point = None
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > max_depth:
                        max_depth = d
                        tip_point = tuple(cnt[f][0])

                # 轮廓中心
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                if tip_point is None:
                    continue

                # 计算箭头方向角度（注意 y 轴向下）
                dx = tip_point[0] - cx
                dy = cy - tip_point[1]  # 令上为正
                angle = np.degrees(np.arctan2(dy, dx))

                # 方向判断（只识别左右方向）
                if -90 <= angle <= 90:
                    arrow_direction = "右"
                else:
                    arrow_direction = "左"

                # 可视化: 轮廓、中心与顶点
                cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(vis, tip_point, 6, (0, 0, 255), -1)
                cv2.putText(vis, f"{arrow_direction} ({angle:.1f})", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 发现有效箭头则返回（只取最大轮廓）
                result_vis = vis
                return arrow_direction, mask, result_vis, angle

        except Exception as e:
            print(f"[detect_arrow] 异常: {e}\n{traceback.format_exc()}")

        return None, mask if 'mask' in locals() else None, result_vis, None
    
    def arrow_detection_logic(self):
        """箭头检测逻辑
        
        Returns:
            str: 检测到的箭头方向或None
        """
        if self.image_subscriber.current_image is None:
            return None
            
        arrow_direction, mask, vis, angle = self.detect_arrow(self.image_subscriber.current_image)
        
        # 显示可视化窗口
        # if vis is not None:
        #     cv2.imshow("Arrow Detection", vis)
        #     cv2.waitKey(1)
        
        # if mask is not None:
        #     cv2.imshow("Arrow Mask", mask)
        #     cv2.waitKey(1)
        
        if arrow_direction:
            if angle is not None:
                print(f"检测到箭头方向: {arrow_direction}, 角度: {angle:.1f}°")
            else:
                print(f"检测到箭头方向: {arrow_direction}")
            return arrow_direction
        else:
            return None
    
    def detect_arrow_direction(self, timeout=28.0):
        """检测箭头方向的主函数（基于arrows.py的完整实现）
        
        Args:
            timeout: 检测超时时间（秒）
            
        Returns:
            str: 检测到的箭头方向（"left"或"right"），如果超时则返回None
        """
        print("开始箭头检测...")
        
        start_time = time.time()
        last_detection_time = 0
        cooldown_duration = 3.0  # 3秒冷却时间，与arrows.py保持一致
        
        while time.time() - start_time < timeout:
            current_time = time.time()
            
            # 控制检测频率和冷却时间
            if current_time - last_detection_time < cooldown_duration:
                time.sleep(0.1)
                continue
                
            direction = self.arrow_detection_logic()
            if direction:
                # 语音播报
                if direction == "左":
                    result = "left"
                    speech_text = "左侧路线"
                elif direction == "右":
                    result = "right"
                    speech_text = "右侧路线"
                elif direction == "上":
                    result = "up"
                    speech_text = "上方路线"
                elif direction == "下":
                    result = "down"
                    speech_text = "下方路线"
                else:
                    result = direction
                    speech_text = f"{direction}路线"
                
                # 播报语音
                print(f"箭头检测成功: {result}，播报: {speech_text}")
                self.play_speech(speech_text)
                
                # 更新检测时间
                last_detection_time = current_time
                
                # 只返回左右方向，其他方向继续检测
                if result in ["left", "right"]:
                    return result
                
            time.sleep(0.1)  # 控制检测频率
        
        print("箭头检测超时")
        return None

    def test_arrow_detection(self, timeout=60.0):
        """测试箭头检测功能，检测到箭头后继续显示窗口
        
        Args:
            timeout: 测试超时时间（秒）
        """
        print("开始箭头检测测试...")
        print("检测到箭头后将继续显示可视化窗口")
        print("按Ctrl+C或等待超时结束测试")
        
        start_time = time.time()
        last_detection_time = 0
        cooldown_duration = 1.0  # 1秒冷却时间，用于测试
        
        try:
            while time.time() - start_time < timeout:
                current_time = time.time()
                
                # 控制检测频率和冷却时间
                if current_time - last_detection_time < cooldown_duration:
                    time.sleep(0.1)
                    continue
                    
                direction = self.arrow_detection_logic()
                if direction:
                    # 语音播报
                    if direction == "左":
                        result = "left"
                        speech_text = "左侧路线"
                    elif direction == "右":
                        result = "right"
                        speech_text = "右侧路线"
                    elif direction == "上":
                        result = "up"
                        speech_text = "上方路线"
                    elif direction == "下":
                        result = "down"
                        speech_text = "下方路线"
                    else:
                        result = direction
                        speech_text = f"{direction}路线"
                    
                    # 播报语音
                    print(f"箭头检测成功: {result}，播报: {speech_text}")
                    self.play_speech(speech_text)
                    
                    # 更新检测时间
                    last_detection_time = current_time
                    
                    # 注意：这里不返回，继续检测和显示
                    
                time.sleep(0.1)  # 控制检测频率
                
        except KeyboardInterrupt:
            print("\n箭头检测测试被用户中断")
        
        print("箭头检测测试结束")
        # 关闭可视化窗口
        cv2.destroyAllWindows()

    # 新增工具函数（放在类内）
    def apply_deadzone(self, value, deadzone):
        return 0 if abs(value) < deadzone else value

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def take_photo(self, filename=None, use_rgb=False):
        """拍照功能，保存当前图像但不显示可视化窗口
        
        Args:
            filename: 指定文件名，如果为None则使用时间戳
            use_rgb: 是否使用RGB图像，默认使用普通图像
            
        Returns:
            str: 保存的文件路径，如果失败返回None
        """
        try:
            # 选择图像源
            current_image = None
            if use_rgb and hasattr(self, 'image_rgb_subscriber') and self.image_rgb_subscriber.current_image is not None:
                current_image = self.image_rgb_subscriber.current_image
                source = "RGB"
            elif hasattr(self, 'image_subscriber') and self.image_subscriber.current_image is not None:
                current_image = self.image_subscriber.current_image
                source = "普通"
            else:
                print("拍照失败：没有可用的图像数据")
                return None
            
            # 生成文件名
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"photo_{timestamp}.jpg"
            
            # 确保文件名有正确的扩展名
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'
            
            # 完整文件路径
            filepath = os.path.join(self.save_dir, filename)
            
            # 保存图像
            success = cv2.imwrite(filepath, current_image)
            
            if success:
                print(f"拍照成功：{source}图像已保存到 {filepath}")
                return filepath
            else:
                print(f"拍照失败：无法保存图像到 {filepath}")
                return None
                
        except Exception as e:
            print(f"拍照异常: {e}")
            return None
    
    
    def pid_straight_walk_with_boundary_detection(self, walk_time=30.0, target_velocity=0.3, enable_debug_visual=False):
        """基于边界偏移检测的PID直走控制函数（使用类似s弯的方法）
        
        Args:
            walk_time: 直走时间（秒）
            target_velocity: 目标前进速度（m/s）
            enable_debug_visual: 是否启用调试可视化
        """
        print(f"开始基于边界偏移检测的PID直走控制，目标速度: {target_velocity}m/s，持续时间: {walk_time}s")
        
        # ===== 1. 初始化PID控制器 =====
        # 姿态控制PID
        roll_pid = PIDController(
            kp=1.2, ki=0.05, kd=0.3, 
            setpoint=0.0, integral_limit=0.15
        )
        pitch_pid = PIDController(
            kp=1.2, ki=0.05, kd=0.3,
            setpoint=0.0, integral_limit=0.15
        )
        # 基于边界偏移检测的直走PID控制器（使用类似s弯的参数）
        slope_pid = PIDController(
            kp=0.008, ki=0.001, kd=0.003,  # 使用类似s弯的PID参数
            setpoint=0.0, integral_limit=0.1
        )
        
        # 滤波器初始化
        self.roll_filter = LowPassFilter(alpha=0.3)
        self.pitch_filter = LowPassFilter(alpha=0.3)
        self.slope_filter = LowPassFilter(alpha=0.7)  # 边界偏移滤波器
        
        # ===== 2. 控制参数 =====
        turn_sensitivity = 0.3   # 转向灵敏度（类似s弯参数）
        max_tilt_angle = math.radians(25)  # 安全倾斜阈值
        slope_deadzone = 5.0    # 边界偏移死区（像素）
        max_turn_rate = 0.3      # 最大转向速率（类似s弯参数）
        
        # ===== 3. 主控制循环 ===== 
        start_time = time.time()
        last_control_time = time.time()
        last_debug_time = time.time()
        safety_counter = 0
        
        try:
            while time.time() - start_time < walk_time:
                # 频率控制（100Hz）
                current_time = time.time()
                dt = current_time - last_control_time
                if dt < 0.01:
                    time.sleep(0.01 - dt)
                    continue
                last_control_time = current_time
                
                # ---- 传感器数据获取 ----
                # IMU数据已移除，使用默认值
                raw_roll, raw_pitch = 0.0, 0.0
                roll = self.roll_filter.update(raw_roll)
                pitch = self.pitch_filter.update(raw_pitch)
                
                # ---- 黄线边界偏移检测 ----
                boundary_error, left_distance, right_distance = self.detect_boundary_error_for_straight_walk()
                
                # 可选的调试可视化
                if enable_debug_visual:
                    self.debug_boundary_visual_for_straight_walk(enable_debug=False)
                
                # 对边界误差进行滤波和死区处理
                filtered_boundary_error = self.slope_filter.update(boundary_error)
                boundary_error_with_deadzone = self.apply_deadzone(filtered_boundary_error, slope_deadzone)
                
                # ---- 安全检测 ----
                if abs(roll) > max_tilt_angle or abs(pitch) > max_tilt_angle:
                    safety_counter += 1
                    if safety_counter > 5:  # 持续50ms超限
                        print(f"危险倾斜！Roll:{math.degrees(roll):.1f}° Pitch:{math.degrees(pitch):.1f}°")
                        break
                else:
                    safety_counter = 0
                
                # ---- PID计算 ----
                # 姿态控制
                deadzone = math.radians(2)  # 姿态死区
                roll_control = roll_pid.update(
                    self.apply_deadzone(roll, deadzone), dt
                )
                pitch_control = pitch_pid.update(
                    self.apply_deadzone(pitch, deadzone), dt
                )
                
                # 基于边界偏移的转向控制（类似s弯方法）
                # boundary_error > 0: 右侧距离更大，偏向左侧，需要右转（正转向速度）
                # boundary_error < 0: 左侧距离更大，偏向右侧，需要左转（负转向速度）
                target_yaw_rate = slope_pid.update(boundary_error_with_deadzone, dt)  # 正偏移量对应右转
                target_yaw_rate = np.clip(target_yaw_rate, -max_turn_rate, max_turn_rate)
                
                # ---- 运动控制指令 ----
                msg = robot_control_cmd_lcmt()
                msg.mode = 11  # Locomotion mode
                msg.gait_id = 27  # Trot gait
                
                # 速度控制（根据转向幅度动态调整前进速度）
                forward_ratio = 1.0 - min(abs(boundary_error_with_deadzone)/30.0, 0.25)  # 调整为像素单位
                msg.vel_des = [
                    target_velocity,  # 动态前进速度
                    0,  # 侧向速度
                    target_yaw_rate * turn_sensitivity  # 转向速度
                ]
                
                # 姿态控制
                msg.rpy_des = [roll_control, pitch_control, 0]
                msg.pos_des = [0, 0, 0.22]  # 机身高度
                msg.step_height = [0.06, 0.06]  # 步高
                
                # 发送指令
                msg.life_count = (self.msg.life_count + 1) % 127
                self.ctrl.Send_cmd(msg)
                self.msg.life_count = msg.life_count
                
                # ---- 调试输出 ----
                if (current_time - last_debug_time) > 0.09:
                    direction_str = "直走" if abs(boundary_error_with_deadzone) < slope_deadzone else ("右转" if boundary_error_with_deadzone > 0 else "左转")
                    print(f"[直走PID] 速度X:{msg.vel_des[0]:.2f}m/s "
                          f"转向速度:{msg.vel_des[2]:.2f}rad/s "
                          f"左侧距离:{left_distance:.1f}像素 "
                          f"右侧距离:{right_distance:.1f}像素 "
                          f"偏移量:{boundary_error:.2f}像素 "
                          f"滤波后:{boundary_error_with_deadzone:.2f}像素 "
                          f"状态:{direction_str}")
                    last_debug_time = current_time
                        
        except Exception as e:
            print(f"直走PID控制异常: {str(e)}")
            raise
        finally:
            # 清理调试窗口
            if enable_debug_visual:
                cv2.destroyAllWindows()
            print("基于边界偏移检测的PID直走控制结束")
    
    def detect_boundary_error_for_straight_walk(self):
        """检测边界偏移误差用于直走控制（从中心线往两边找第一个黄点）
        
        Returns:
            tuple: (偏移误差, 左侧距离, 右侧距离)
        """
        if self.curve_processor.image is None:
            return 0.0, 0.0, 0.0
            
        try:
            with self.curve_processor.lock:
                cv_image = self.curve_processor.image.copy()
            h, w = cv_image.shape[:2]
            image_center_x = w // 2
            
            # 使用与s弯相同的HSV范围
            yellow_hsv_low = np.array([20, 100, 100])
            yellow_hsv_high = np.array([30, 255, 255])
            
            # 取图像从下往上三分之一行进行检测
            middle_y = int(h * 2/3)  # 从下往上三分之一行
            middle_row = cv_image[middle_y:middle_y+1, :]  # 取该行
            hsv_middle = cv2.cvtColor(middle_row, cv2.COLOR_BGR2HSV)
            mask_middle = cv2.inRange(hsv_middle, yellow_hsv_low, yellow_hsv_high)
            
            mask_line = mask_middle[0]  # 获取一维数组
            
            # 从中心线向左侧搜索第一个黄点
            left_point_x = None
            for x in range(image_center_x - 1, -1, -1):
                if mask_line[x] > 0:
                    left_point_x = x
                    break
            
            # 从中心线向右侧搜索第一个黄点
            right_point_x = None
            for x in range(image_center_x + 1, w):
                if mask_line[x] > 0:
                    right_point_x = x
                    break
            
            # 计算左右两侧到中心线的距离
            left_distance = image_center_x - left_point_x if left_point_x is not None else 0.0
            right_distance = right_point_x - image_center_x if right_point_x is not None else 0.0
            
            # 如果没有识别到其中一侧边界（0像素），赋值为220
            if left_distance == 0.0:
                left_distance = 220.0
            if right_distance == 0.0:
                right_distance = 220.0
            
            # 计算偏移量：右侧距离 - 左侧距离
            # 正值表示右侧黄点更远（偏向左侧，需要右转）
            # 负值表示左侧黄点更远（偏向右侧，需要左转）
            boundary_error = right_distance - left_distance
            
            return boundary_error, left_distance, right_distance
            
        except Exception as e:
            print(f"边界偏移检测异常: {e}")
            return 0.0, 0.0, 0.0
    

    
    def detect_yellow_boundary_rgb(self, enable_debug_visual=False, side="right"):
        """基于/image_rgb话题的黄色边界检测功能（来自begin.py）
        
        Args:
            enable_debug_visual: 是否启用调试可视化
            side: 检测区域 ('right' 表示右五分之二, 'left' 表示左五分之二)
            
        Returns:
            tuple: (是否检测到边界, 估计距离, 检测位置)
        """
        if self.image_rgb_subscriber.current_image is None:
            return False, 0.0, (0, 0)
            
        try:
            cv_image = self.image_rgb_subscriber.current_image.copy()
            h, w = cv_image.shape[:2]

            # 区域选择：右五分之二或左五分之二
            if side == "right":
                start_x = int(w * 3 / 5)  # 右侧2/5
                region = cv_image[:, start_x:]
            elif side == "left":
                end_x = int(w * 3 / 5)    # 左侧2/5
                start_x = 0
                region = cv_image[:, :end_x]
            else:
                raise ValueError("参数 side 必须是 'left' 或 'right'")

            # HSV转换和模糊处理
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

            # 黄色检测范围（与begin.py保持一致）
            lower_yellow = np.array([15, 80, 80])
            upper_yellow = np.array([35, 255, 255])
            mask = cv2.inRange(blurred, lower_yellow, upper_yellow)

            # 形态学处理
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            min_point_global = None
            min_distance = float('inf')
            min_box = None

            for contour in contours:
                if cv2.contourArea(contour) <= 200:
                    continue

                pts = contour.reshape(-1, 2)
                lowest_y = int(np.max(pts[:, 1]))
                bottom_pts = pts[pts[:, 1] == lowest_y]
                lowest_x_rel = int(np.mean(bottom_pts[:, 0]))

                lowest_x_global = lowest_x_rel + start_x
                lowest_y_global = lowest_y
                distance = lowest_y_global / 180  # 使用begin.py中的像素到米转换

                if min_point_global is None or lowest_y_global > min_point_global[1]:
                    min_point_global = (lowest_x_global, lowest_y_global)
                    min_distance = distance
                    rect = cv2.minAreaRect(contour)
                    box = np.intp(cv2.boxPoints(rect))
                    box[:, 0] += start_x
                    min_box = box

            # 调试可视化和结果处理
            display_img = cv_image.copy()
            
            # 添加时间戳
            timestamp = time.time()
            cv2.putText(display_img, f"Time: {timestamp:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 添加检测状态信息
            if min_point_global is not None:
                px, py = min_point_global

                if min_box is not None:
                    cv2.drawContours(display_img, [min_box], 0, (0, 255, 0), 2)
                cv2.circle(display_img, (int(px), int(py)), 6, (0, 0, 255), -1)
                cv2.putText(display_img, f"{min_distance:.2f}m", (int(px)+6, int(py)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 显示检测状态
                detected = min_distance <= 0.5
                status_text = f"检测到: 距离{min_distance:.2f}m" if detected else f"发现但距离过远: {min_distance:.2f}m"
                status_color = (0, 255, 0) if detected else (0, 255, 255)
                cv2.putText(display_img, status_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            else:
                cv2.putText(display_img, "未检测到黄色边界", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                detected = False

            # 显示处理后的mask
            if enable_debug_visual:
                roi_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combined = cv2.hconcat([display_img, cv2.resize(roi_bgr, (w, h))])
                cv2.imshow("RGB黄色边界检测", combined)
                cv2.waitKey(1)

            if min_point_global is not None:
                detected = min_distance <= 0.5  # 检测阈值
                return detected, min_distance, min_point_global
            else:
                return False, 0.0, (0, 0)

        except Exception as e:
            print(f"RGB黄色边界检测异常: {e}")
            return False, 0.0, (0, 0)

    def test_rgb_boundary_detection(self, duration=None):
        """测试RGB黄色边界检测功能
        
        Args:
            duration: 测试持续时间（秒），None表示持续运行直到用户中断
        """
        if duration is None:
            print("开始RGB黄色边界检测测试，持续运行（按Ctrl+C停止）")
        else:
            print(f"开始RGB黄色边界检测测试，持续时间: {duration}s")
        
        # 创建调试窗口
        cv2.namedWindow("RGB黄色边界检测", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB黄色边界检测", 960, 480)
        
        start_time = time.time()
        frame_count = 0
        last_output_time = time.time()
        
        try:
            while True:
                # 检查是否超过指定时间
                if duration is not None and time.time() - start_time >= duration:
                    break
                    
                # 执行检测
                detected, distance, position = self.detect_yellow_boundary_rgb(enable_debug_visual=True)
                frame_count += 1
                current_time = time.time()
                
                # 持续输出检测结果
                if detected:
                     print(f"[帧{frame_count}] [{current_time:.2f}s] ✓ 检测到黄色边界，距离: {distance:.2f}m, 位置: {position}")
                elif distance > 0:  # 发现了黄色区域但距离过远
                     print(f"[帧{frame_count}] [{current_time:.2f}s] ○ 发现黄色区域但距离过远: {distance:.2f}m, 位置: {position}")
                else:
                     # 每秒输出一次未检测到的信息
                    if current_time - last_output_time >= 1.0:
                         print(f"[帧{frame_count}] [{current_time:.2f}s] × 未检测到黄色边界")
                         last_output_time = current_time
                
                # 控制检测频率（10Hz）
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n用户中断测试")
        finally:
            cv2.destroyAllWindows()
            print("RGB黄色边界检测测试完成")
    
    def debug_boundary_visual_for_straight_walk(self, enable_debug=False):
        """直走控制的边界偏移检测可视化调试功能（展示检测到的点和偏移）
        
        Args:
            enable_debug: 是否启用调试显示
        """
        if not enable_debug or self.curve_processor.image is None:
            return
            
        try:
            with self.curve_processor.lock:
                cv_image = self.curve_processor.image.copy()
            h, w = cv_image.shape[:2]
            image_center_x = w // 2
            
            # 使用与s弯相同的HSV范围
            yellow_hsv_low = np.array([20, 100, 100])
            yellow_hsv_high = np.array([30, 255, 255])
            
            # 取图像从下往上三分之一行进行检测（与新的检测方法一致）
            middle_y = int(h * 2/3)  # 从下往上三分之一行
            middle_row = cv_image[middle_y:middle_y+1, :]  # 取该行
            hsv_middle = cv2.cvtColor(middle_row, cv2.COLOR_BGR2HSV)
            mask_middle = cv2.inRange(hsv_middle, yellow_hsv_low, yellow_hsv_high)
            
            mask_line = mask_middle[0]  # 获取一维数组
            
            # 绘制图像中心线
            cv2.line(cv_image, (image_center_x, 0), (image_center_x, h), (0, 255, 0), 2)
            
            # 绘制检测行
            cv2.line(cv_image, (0, middle_y), (w, middle_y), (255, 100, 0), 2)
            cv2.putText(cv_image, "Detection Line", (10, middle_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            
            # 从中心线向左侧搜索第一个黄点
            left_point_x = None
            for x in range(image_center_x - 1, -1, -1):
                if mask_line[x] > 0:
                    left_point_x = x
                    break
            
            # 从中心线向右侧搜索第一个黄点
            right_point_x = None
            for x in range(image_center_x + 1, w):
                if mask_line[x] > 0:
                    right_point_x = x
                    break
            
            # 计算左右两侧到中心线的距离
            left_distance = image_center_x - left_point_x if left_point_x is not None else 0.0
            right_distance = right_point_x - image_center_x if right_point_x is not None else 0.0
            
            # 计算偏移量：右侧距离 - 左侧距离
            boundary_error = right_distance - left_distance
            
            # 绘制检测到的黄点
            if left_point_x is not None:
                # 绘制左侧黄点
                left_point = (left_point_x, middle_y)
                cv2.circle(cv_image, left_point, 8, (0, 255, 0), -1)  # 绿色实心圆
                cv2.circle(cv_image, left_point, 12, (255, 255, 255), 2)  # 白色边框
                
                # 绘制左侧距离线
                cv2.line(cv_image, (image_center_x, middle_y), left_point, (0, 255, 0), 2)
                
                # 标注左侧距离
                mid_x = (image_center_x + left_point_x) // 2
                cv2.putText(cv_image, f"L:{left_distance:.0f}", (mid_x - 20, middle_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if right_point_x is not None:
                # 绘制右侧黄点
                right_point = (right_point_x, middle_y)
                cv2.circle(cv_image, right_point, 8, (255, 0, 0), -1)  # 蓝色实心圆
                cv2.circle(cv_image, right_point, 12, (255, 255, 255), 2)  # 白色边框
                
                # 绘制右侧距离线
                cv2.line(cv_image, (image_center_x, middle_y), right_point, (255, 0, 0), 2)
                
                # 标注右侧距离
                mid_x = (image_center_x + right_point_x) // 2
                cv2.putText(cv_image, f"R:{right_distance:.0f}", (mid_x - 20, middle_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 显示检测信息
            cv2.putText(cv_image, f"Left Distance: {left_distance:.1f} pixels", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(cv_image, f"Right Distance: {right_distance:.1f} pixels", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(cv_image, f"Offset (R-L): {boundary_error:.2f} pixels", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 方向判断
            if abs(boundary_error) < 5.0:  # 死区阈值
                direction = "直行"
                color = (0, 255, 0)  # 绿色
            elif boundary_error > 0:
                direction = "右转"  # 右侧距离更大，偏向左侧，需要右转
                color = (0, 165, 255)  # 橙色
            else:
                direction = "左转"  # 左侧距离更大，偏向右侧，需要左转
                color = (255, 0, 255)  # 紫色
            
            cv2.putText(cv_image, f"Direction: {direction}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 如果没有检测到任何黄点
            if left_point_x is None and right_point_x is None:
                cv2.putText(cv_image, "No Yellow Points Detected", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示调试窗口
            cv2.imshow("Boundary Offset Detection - Debug", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
             print(f"边界偏移可视化调试异常: {e}")
    
    
    def detect_yellow_line(self, enable_debug_window=False, stop_on_detection=False, detection_position="quarter"):
        """检测/image_rgb图像指定位置中间四分之一区域的黄色
        
        Args:
            enable_debug_window: 是否启用调试窗口显示检测过程
            stop_on_detection: 检测到黄线后是否立即停止调试窗口显示
            detection_position: 检测位置，"last"表示最后一行，"quarter"表示从下往上四分之一行，"middle"表示中间一行，"top"表示最上一行
        
        Returns:
            bool: 如果检测到黄色返回True，否则返回False
        """
        if not hasattr(self, 'image_rgb_subscriber') or self.image_rgb_subscriber.current_image is None:
            return False
            
        try:
            cv_image = self.image_rgb_subscriber.current_image.copy()
            h, w = cv_image.shape[:2]
            
            # 根据detection_position参数选择检测位置
            if detection_position == "last":
                # 获取最后一行
                target_row_y = h - 1
                target_row = cv_image[target_row_y:target_row_y+1, :]
            elif detection_position == "middle":
                # 获取中间一行
                target_row_y = h // 2
                target_row = cv_image[target_row_y:target_row_y+1, :]
            elif detection_position == "top":
                # 获取最上一行
                target_row_y = 0
                target_row = cv_image[target_row_y:target_row_y+1, :]
            else:  # detection_position == "quarter"
                # 获取从下往上四分之一的那一行
                target_row_y = int(h * 3 / 4)  # 从下往上四分之一位置
                target_row = cv_image[target_row_y:target_row_y+1, :]
            
            # 计算中间四分之一的范围
            start_x = 7 * w // 16
            end_x = 9 * w // 16
            
            # 提取中间四分之一区域
            center_region = target_row[:, start_x:end_x]
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            
            # 黄色检测范围
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            
            # 创建黄色掩码
            mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            
            # 检查是否有黄色像素
            yellow_pixels = cv2.countNonZero(mask)
            
            # 检测结果
            detection_result = yellow_pixels > 0
            
            # 调试窗口显示
            if enable_debug_window:
                try:
                    # 创建调试图像
                    debug_image = cv_image.copy()
                    
                    # 在原图上标记检测区域
                    cv2.rectangle(debug_image, (start_x, target_row_y), (end_x, target_row_y+1), (0, 255, 0), 2)
                    if detection_position == "last":
                        position_text = "Last Row"
                    elif detection_position == "middle":
                        position_text = "Middle Row"
                    elif detection_position == "top":
                        position_text = "Top Row"
                    else:
                        position_text = "Quarter Row"
                    cv2.putText(debug_image, f"Detection Area ({position_text} Center 1/4)", 
                            (start_x, target_row_y-10 if target_row_y > 10 else target_row_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 计算放大区域的目标宽度，确保一致性
                    target_width = max(300, center_region.shape[1] * 10)  # 至少放大10倍
                    enlarged_region = cv2.resize(center_region, (target_width, 50), interpolation=cv2.INTER_NEAREST)
                    enlarged_hsv = cv2.resize(hsv, (target_width, 50), interpolation=cv2.INTER_NEAREST)
                    enlarged_mask = cv2.resize(mask, (target_width, 50), interpolation=cv2.INTER_NEAREST)
                    
                    # 将掩码转换为3通道以便显示
                    enlarged_mask_colored = cv2.cvtColor(enlarged_mask, cv2.COLOR_GRAY2BGR)
                    
                    # 创建信息文本
                    result_text = f"Yellow Pixels: {yellow_pixels} | Result: {'DETECTED' if detection_result else 'NOT DETECTED'}"
                    info_text = f"Detection Region: [{start_x}:{end_x}] (Width: {end_x-start_x}px)"
                    
                    # 创建组合显示图像
                    display_height = debug_image.shape[0] + 200  # 为放大区域和文本预留空间
                    display_width = max(debug_image.shape[1], target_width * 3 + 40)  # 适应三个放大区域的宽度
                    combined_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                    
                    # 放置原图
                    combined_image[:debug_image.shape[0], :debug_image.shape[1]] = debug_image
                    
                    # 放置放大的检测区域
                    y_offset = debug_image.shape[0] + 10
                    x_offset1 = 10
                    x_offset2 = x_offset1 + target_width + 10
                    x_offset3 = x_offset2 + target_width + 10
                    
                    combined_image[y_offset:y_offset+50, x_offset1:x_offset1+target_width] = enlarged_region
                    combined_image[y_offset:y_offset+50, x_offset2:x_offset2+target_width] = enlarged_hsv
                    combined_image[y_offset:y_offset+50, x_offset3:x_offset3+target_width] = enlarged_mask_colored
                    
                    # 添加标签
                    cv2.putText(combined_image, "Original Region", (x_offset1, y_offset-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(combined_image, "HSV Region", (x_offset2, y_offset-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(combined_image, "Yellow Mask", (x_offset3, y_offset-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 添加结果信息
                    cv2.putText(combined_image, result_text, (10, y_offset+80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if detection_result else (0, 0, 255), 2)
                    cv2.putText(combined_image, info_text, (10, y_offset+110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(combined_image, "HSV Range: [20,100,100] - [30,255,255]", (10, y_offset+130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 如果检测到黄线且设置了停止标志，添加停止提示
                    if detection_result and stop_on_detection:
                        cv2.putText(combined_image, "YELLOW LINE DETECTED - STOPPING DEBUG", (10, y_offset+160), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 显示调试窗口
                    cv2.imshow("Yellow Line Detection Debug", combined_image)
                    
                    # 如果检测到黄线且设置了停止标志，等待用户查看后关闭窗口
                    if detection_result and stop_on_detection:
                        cv2.waitKey(2000)  # 显示2秒让用户看到结果
                        cv2.destroyWindow("Yellow Line Detection Debug")
                        return detection_result
                    else:
                        cv2.waitKey(1)  # 非阻塞显示，允许图像更新
                    
                except Exception as debug_e:
                    print(f"调试窗口显示错误: {debug_e}")
            
            # 返回检测结果
            return detection_result
        except Exception as debug_e:
            print(f"错误: {debug_e}")

    def walk_straight_until_yellow_line_2(
        self, max_duration=2000, velocity=0.15, enable_debug=False,
        detection_position="quarter", timeout_return=True):
        """持续直走直到检测到黄线

        Args:
            max_duration: 最大行走时间（毫秒），防止无限行走
            velocity: 前进速度（m/s）
            enable_debug: 是否启用调试输出
            detection_position: 检测位置，"last"表示最后一行，"quarter"表示从下往上四分之一行
            timeout_return: 超时时是否直接返回，True表示直接返回，False表示继续等待

        Returns:
            tuple: (是否检测到黄线, 实际行走时间 毫秒)
        """
        print(f"开始直走，速度: {velocity}m/s，最大时长: {max_duration}ms")
        print("持续检测黄线，检测到后立即停止")

        start_time = time.time()
        detection_count = 0

        try:
            # 开始持续行走
            self.msg.mode = 11  # 行走模式
            self.msg.gait_id = 27  # 步态类型
            self.msg.vel_des = [velocity, 0, 0]  # 期望速度[x, y, yaw]
            self.msg.step_height = [0.1, 0.1]  # 步高
            self.msg.duration = max_duration  # 毫秒
            self.msg.life_count = (self.msg.life_count + 1) % 128
            self.ctrl.Send_cmd(self.msg)

            # 持续检测循环
            while (time.time() - start_time) * 1000 < max_duration:
                # 执行黄线检测
                yellow_detected = self.detect_yellow_line(
                    enable_debug_window=enable_debug,
                    stop_on_detection=True,
                    detection_position=detection_position
                )
                detection_count += 1

                current_time_ms = int((time.time() - start_time) * 1000)

                if yellow_detected:
                    print(f"[{current_time_ms}ms] 检测到黄线！停止前进")
                    print(f"总检测次数: {detection_count}")

                    # 立即停止
                    self.msg.mode = 12  # 站立模式
                    self.msg.gait_id = 0
                    self.msg.duration = 0
                    self.msg.life_count = (self.msg.life_count + 1) % 128
                    self.ctrl.Send_cmd(self.msg)
                    time.sleep(1)
                    return True, current_time_ms

                if enable_debug and detection_count % 1 == 0:
                    print(f"[{current_time_ms}ms] 检测{detection_count}次，未发现黄线，继续前进")

                # 控制检测频率（50Hz）
                time.sleep(0.02)

            # 超时处理
            print(f"达到最大行走时间 {max_duration}ms，直接结束并返回")
            print(f"总检测次数: {detection_count}")

            self.msg.mode = 12  # 站立模式
            self.msg.gait_id = 0
            self.msg.duration = 0
            self.msg.life_count = (self.msg.life_count + 1) % 128
            self.ctrl.Send_cmd(self.msg)

            return False, int((time.time() - start_time) * 1000)

        except Exception as e:
            print(f"直走检测过程出错: {e}")
            try:
                self.msg.mode = 12
                self.msg.gait_id = 0
                self.msg.duration = 0
                self.msg.life_count = (self.msg.life_count + 1) % 128
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0, 500)
            except:
                pass
            return False, int((time.time() - start_time) * 1000)



    def detect_yellow_line_bottom_center(self, enable_debug_window=False, stop_on_detection=False, detection_position="quarter"):
        """
        检测/image_rgb图像下四分之一中间区域的黄线，并估算距离

        Returns:
            float | None: 返回估算的距离(米)，未检测到黄线返回 None
        """
        if not hasattr(self, 'image_rgb_subscriber') or self.image_rgb_subscriber.current_image is None:
            return None

        try:
            cv_image = self.image_rgb_subscriber.current_image.copy()
            h, w = cv_image.shape[:2]

            # ROI：取图像底部 1/4 且横向中间 1/3 区域
            x1, x2 = w // 3, 2 * w // 3
            y1, y2 = 3 * h // 4, h   # 下四分之一
            roi = cv_image[y1:y2, x1:x2]

            if roi.size == 0:
                print("ROI区域无效，请检查图像尺寸")
                return None

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([15, 80, 80])
            upper_yellow = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # 形态学去噪
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            points = cv2.findNonZero(mask)

            if points is not None:
                y_min = np.min(points[:, 0, 1])
                # y_min 在 ROI 内，要加上 y1 转换回全图坐标
                distance = (h - (y1 + y_min)) / PIXEL_PER_METER_YELLOW

                if enable_debug_window:
                    x_pt = points[np.argmin(points[:, 0, 1]), 0, 0]
                    cv2.circle(roi, (x_pt, y_min), 5, (0, 0, 255), -1)
                    cv2.putText(roi, f"{distance:.2f}m", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    display = cv_image.copy()
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    display[y1:y2, x1:x2] = roi
                    cv2.imshow("Yellow Line Detection Debug", display)

                    if stop_on_detection and distance < distance1:  # distance1 需在外部定义
                        cv2.waitKey(2000)
                        cv2.destroyWindow("Yellow Line Detection Debug")
                    else:
                        cv2.waitKey(1)

                return distance
            else:
                return None

        except Exception as e:
            print(f"黄线检测错误: {e}")
            return None



    def walk_straight_until_yellow_line(
        self, max_duration=200000, velocity=0.15, enable_debug=False,
        detection_position="quarter", timeout_return=True):
        """持续直走直到黄线小于阈值时停止"""
        print(f"开始直走，速度: {velocity}m/s，最大时长: {max_duration}ms")
        print(f"持续检测黄线，距离 < {distance1}m 时停止前进")

        start_time = time.time()
        detection_count = 0

        try:
            # 开始持续行走
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [velocity, 0, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = max_duration
            self.msg.life_count = (self.msg.life_count + 1) % 128
            self.ctrl.Send_cmd(self.msg)

            while (time.time() - start_time) * 1000 < max_duration:
                distance = self.detect_yellow_line_bottom_center(
                    enable_debug_window=enable_debug,
                    stop_on_detection=True,
                    detection_position=detection_position
                )
                detection_count += 1
                current_time_ms = int((time.time() - start_time) * 1000)

                if distance is not None:
                    print(f"[{current_time_ms}ms] 检测到黄线，距离 {distance:.2f}m")
                    if distance < distance1:
                        print(f"距离小于阈值 {distance1}m，停止前进")
                        self.msg.mode = 12
                        self.msg.gait_id = 0
                        self.msg.duration = 0
                        self.msg.life_count = (self.msg.life_count + 1) % 128
                        self.ctrl.Send_cmd(self.msg)
                        time.sleep(1)
                        return True, current_time_ms

                if enable_debug and detection_count % 10 == 0:
                    print(f"[{current_time_ms}ms] 检测{detection_count}次，未满足停止条件")

                time.sleep(0.02)  # 控制频率 50Hz

            # 超时
            print(f"达到最大行走时间 {max_duration}ms，直接结束并返回")
            print(f"总检测次数: {detection_count}")
            self.msg.mode = 12
            self.msg.gait_id = 0
            self.msg.duration = 0
            self.msg.life_count = (self.msg.life_count + 1) % 128
            self.ctrl.Send_cmd(self.msg)
            time.sleep(1)
            return False, int((time.time() - start_time) * 1000)

        except Exception as e:
            print(f"直走检测过程出错: {e}")
            try:
                self.msg.mode = 12
                self.msg.gait_id = 0
                self.msg.duration = 0
                self.msg.life_count = (self.msg.life_count + 1) % 128
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0, 500)
            except:
                pass
            return False, int((time.time() - start_time) * 1000)


    def detect_yellow_light(self, enable_debug_visual=False):
        """
        检测黄灯功能
        
        Args:
            enable_debug_visual: 是否启用调试可视化
        Returns:
            tuple: (是否检测到黄灯, 估计距离)
        """
        if self.image_subscriber.current_image is None:
            return False, 0.0

        try:
            cv_image = self.image_subscriber.current_image
            h, w = cv_image.shape[:2]

            # 只处理上四分之三
            upper_three_quarters = cv_image[: (h * 3) // 4, :].copy()

            hsv = cv2.cvtColor(upper_three_quarters, cv2.COLOR_BGR2HSV)

            # 黄色HSV范围
            # lower_yellow = np.array([20, 100, 100])
            lower_yellow = np.array([20, 150, 150])
            upper_yellow = np.array([50, 255, 235])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # 形态学处理去噪
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

            now = time.time()
            detected = False
            distance = 0.0

            if valid_contours:
                max_contour = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)
                if area > 200:
                    x, y, w_box, h_box = cv2.boundingRect(max_contour)
                    center_y = y + h_box // 2
                    distance = center_y / PIXEL_PER_METER_YELLOW  # 简单线性估距，单位米

                    # 调试可视化
                    if enable_debug_visual:
                        cv2.rectangle(upper_three_quarters, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        cv2.drawContours(upper_three_quarters, [max_contour], -1, (0, 0, 255), 2)
                        cv2.putText(upper_three_quarters, f"{distance:.2f}m", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    if distance <= self.stop_distance:
                        if now - self.last_yellow_detect_time > self.yellow_cooldown:
                            print(f"检测到黄灯，估计距离 {distance:.2f} 米（<= {self.stop_distance}m），需要停止")
                            self.last_yellow_detect_time = now
                            self.yellow_light_detected = True
                            detected = True
                    else:
                        if now - self.last_yellow_detect_time > self.yellow_cooldown:
                            print(f"检测到黄灯，但距离 {distance:.2f} 米，大于阈值 {self.stop_distance}m，继续观察")
                            self.last_yellow_detect_time = now
                            self.yellow_light_detected = False
            else:
                if self.yellow_light_detected:
                    print("黄灯不再可见，重置检测状态")
                    self.yellow_light_detected = False

            if enable_debug_visual:
                cv2.imshow("黄灯检测", upper_three_quarters)
                cv2.waitKey(1)

            return detected, distance

        except Exception as e:
            print(f"黄灯检测异常: {e}")
            return False, 0.0

    
    def yellow_light_detection_loop(self, detection_time=30.0, enable_debug_visual=False):
        """黄灯检测循环
        
        Args:
            detection_time: 检测时间（秒）
            enable_debug_visual: 是否启用调试可视化
            
        Returns:
            bool: 是否检测到黄灯并需要停止
        """
        print(f"开始黄灯检测循环，检测时间: {detection_time}s")
        
        start_time = time.time()
        last_check_time = time.time()
        
        try:
            while time.time() - start_time < detection_time:
                current_time = time.time()
                dt = current_time - last_check_time
                
                # 控制检测频率（20Hz）
                if dt < 0.05:
                    time.sleep(0.05 - dt)
                    continue
                last_check_time = current_time
                
                # 执行黄灯检测
                detected, distance = self.detect_yellow_light(True)
                
                if detected:
                    print(f"黄灯检测完成，距离: {distance:.2f}m")
                    return True
                    
        except Exception as e:
            print(f"黄灯检测循环异常: {str(e)}")
        finally:
            # 清理调试窗口
            if enable_debug_visual:
                cv2.destroyAllWindows()
            print("黄灯检测循环结束")
            
        return False
    
    def continuous_yellow_light_monitoring(self, enable_debug_visual=False):
        """持续黄灯监控（用于在运动过程中监控）
        
        Args:
            enable_debug_visual: 是否启用调试可视化
            
        Returns:
            bool: 当前是否检测到需要停止的黄灯
        """
        detected, distance = self.detect_yellow_light(enable_debug_visual)
        return detected


    def detect_limit_barrier(self, enable_debug_visual=False):
        """检测限高杆功能（使用物理尺寸测距）

        Returns:
            tuple: (是否检测到限高杆, 估计距离)
        """
        if self.image_subscriber.current_image is None:
            return False, 0.0

        try:
            cv_image = self.image_subscriber.current_image
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # ==== Step1: 红色掩膜 ====
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([150, 150, 100])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # ==== Step2: 去噪 ====
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # ==== Step3: 找轮廓 ====
            contours_data = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_data) == 3:
                _, contours, _ = contours_data
            else:
                contours, _ = contours_data

            now = time.time()
            detected, distance = False, 0.0

            if contours:
                # ==== Step4: 最大轮廓作为候选横杆 ====
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(max_contour)

                # 横杆筛选条件
                if w_box >= 150 or h_box <= 50:  # 宽度足够或高度很小
                    # ---- 测距方法 ----
                    H_real = 0.10  # 横杆厚度（m）
                    F_pix = 400.0  # 相机焦距经验值
                    distance = F_pix * H_real / max(h_box, 1)  # 防止除零

                    # 调试可视化
                    if enable_debug_visual:
                        # 轮廓框
                        cv2.rectangle(cv_image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        cv2.drawContours(cv_image, [max_contour], -1, (0, 0, 255), 2)
                        # 标记最上面一行
                        cv2.line(cv_image, (x, y), (x + w_box, y), (255, 0, 0), 2)
                        cv2.putText(cv_image, f"{distance:.2f}m", (x, max(0, y - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # 检测逻辑
                    if distance <= self.limit_stop_distance:
                        if now - self.last_limit_detect_time > self.limit_cooldown:
                            print(f"检测到红色限高杆，距离 {distance:.2f} 米（<= {self.limit_stop_distance}m），开始执行自定义步态")
                            self.last_limit_detect_time = now
                            self.limit_barrier_detected = True
                            detected = True
                    else:
                        if now - self.last_limit_detect_time > self.limit_cooldown:
                            print(f"检测到红色限高杆，距离 {distance:.2f} 米，大于阈值，继续前进")
                            self.last_limit_detect_time = now
                            self.limit_barrier_detected = False
            else:
                if self.limit_barrier_detected:
                    print("红色限高杆不再可见，重置检测状态")
                    self.limit_barrier_detected = False

            if enable_debug_visual:
                cv2.imshow("红色限高杆检测", cv_image)
                cv2.waitKey(1)

            return detected, distance

        except Exception as e:
            print(f"红色限高杆检测异常: {e}")
            return False, 0.0


    def hill_b_qr(self):
        """坡道下后到卸货码,持续检测黄灯，在24秒时拍照识别"""
        print("开始坡道下后到卸货码的移动，同时持续检测黄灯")
        
        # 初始化变量
        detected_labels = None
        
        # 设置黄灯检测距离阈值为70cm
        original_stop_distance = self.stop_distance
        self.stop_distance = 0.7   # 70cm停止距离
        yellow_stop_executed = False  # 标记是否已执行过黄灯停止
        photo_taken = False  # 标记是否已拍照
        
        try:
            # 开始移动并持续检测黄灯
            start_time = time.time()
            total_walk_time = 21.0 
            
            while time.time() - start_time < total_walk_time:
                current_time = time.time() - start_time
                
                # 在20秒时拍照并进行识别
                if current_time >= 19.0 and not photo_taken:
                    print("到达20秒，开始拍照和识别...")
                    
                    # 起立
                    self.msg.mode = 12  # Recovery stand 模式
                    self.msg.gait_id = 0
                    self.msg.life_count += 1
                    self.ctrl.Send_cmd(self.msg)
                    self.ctrl.Wait_finish(12, 0)
                    
                    # 单张拍照
                    self.take_photo("test_single.jpg")
                    time.sleep(2)
                    # 使用RGB图像拍照
                    # self.take_photo("test_rgb.jpg", use_rgb=True)
                    self.take_photo("test_single.jpg")
                    
                    # 调用识别函数
                    detected_labels = recognize_photo_labels("test_single.jpg")
                    if detected_labels:
                        print(f"识别到的标签: {detected_labels}")
                    else:
                        print("未识别到任何标签")
                    
                    photo_taken = True
                
                # 持续检测黄灯（如果还没有执行过黄灯停止）
                if not yellow_stop_executed:
                    yellow_detected, distance = self.detect_yellow_light(enable_debug_visual=False)
                    
                    if yellow_detected and distance <= 0.7:
                        print(f"检测到黄灯，距离约{distance:.2f}米，开始停止5秒")
                        
                        # 停止机器狗
                        self.msg.mode = 12  # 停止模式
                        self.msg.gait_id = 0
                        self.msg.duration = 0
                        self.msg.life_count += 1
                        self.ctrl.Send_cmd(self.msg)
                        
                        # 等待5秒并播报倒数
                        for countdown in [5, 4, 3, 2, 1]:
                            self.play_speech(str(countdown))
                            print(f"倒数: {countdown}")
                            time.sleep(1.0)
                        
                        print("停止5秒完成，继续直走")
                        yellow_stop_executed = True
                        continue
                
                # 执行移动控制（短时间步进）
                self.pid_straight_walk_with_boundary_detection(walk_time=0.5, target_velocity=0.2, enable_debug_visual=False)
                
                # 短暂延时（减少延时以增加检测频率）
                time.sleep(0.01)
                
        except Exception as e:
            print(f"hill_b_qr执行异常: {e}")
        finally:
            # 恢复原始停止距离
            self.stop_distance = original_stop_distance
            print(f"hill_b_qr任务完成，总用时: {time.time() - start_time:.2f}秒")
            if detected_labels:
                print(f"成功识别到标签: {detected_labels}")
            else:
                print("未识别到任何标签")

        self.walk_straight_until_yellow_line_2(5700,0.1,False,"middle")        
        # 返回检测结果
        return detected_labels

    def stone_b_qr(self):
        """石板路下后到卸货码,持续检测限高杆，在24秒时拍照识别"""
        print("开始stone_b_qr任务，检测限高杆")
        
        # 初始化变量
        detected_labels = ""
        # 保存原始停止距离
        original_limit_stop_distance = self.limit_stop_distance
        
        # 设置检测距离阈值
        self.limit_stop_distance = 0.65# 限高杆检测距离
        
        try:
            start_time = time.time()
            total_walk_time = 35.0 # 总直走时间
            limit_gait_executed = False  # 标记是否已执行过限高杆自定义步态
            photo_taken = False  # 标记是否已拍照
            
            while time.time() - start_time < total_walk_time:
                current_time = time.time() - start_time
                
                # 在24秒时拍照并进行识别
                if current_time >= 28.0 and not photo_taken:
                    print("到达28秒，开始拍照和识别...")
                    
                    # 起立
                    self.msg.mode = 12  # Recovery stand 模式
                    self.msg.gait_id = 0
                    self.msg.life_count += 1
                    self.ctrl.Send_cmd(self.msg)
                    self.ctrl.Wait_finish(12, 0)
                    
                    # 单张拍照
                    self.take_photo("test_single.jpg")
                    time.sleep(2)
                    # 使用RGB图像拍照
                    # self.take_photo("test_rgb.jpg", use_rgb=True)
                    self.take_photo("test_single.jpg")
                    
                    # 调用识别函数
                    detected_labels = recognize_photo_labels("test_single.jpg")
                    if detected_labels:
                        print(f"识别到的标签: {detected_labels}")
                    else:
                        print("未识别到任何标签")
                    
                    photo_taken = True
                
                # 持续检测限高杆（如果还没有执行过自定义步态）
                if not limit_gait_executed:
                    limit_detected, limit_distance = self.detect_limit_barrier(enable_debug_visual=False)
                    
                    if limit_detected and limit_distance <= 0.65:
                        print(f"检测到限高杆，距离约{limit_distance:.2f}米，开始执行自定义步态")
                        
                        # 执行自定义步态（低高度步态）
                        print("执行低高度自定义步态...")
                        self.execute_custom_gait_low_height()
                        
                        print("自定义步态执行完毕，继续直走")
                        limit_gait_executed = True
                        continue
                
                # 执行正常直走控制（短时间步进）
                self.pid_straight_walk_with_boundary_detection(walk_time=0.5, target_velocity=0.15, enable_debug_visual=False)
                
                # 短暂延时（进一步提高限高杆检测频率）
                time.sleep(0.005)
                
        except Exception as e:
            print(f"stone_b_qr执行异常: {e}")
        finally:
            # 恢复原始停止距离
            self.limit_stop_distance = original_limit_stop_distance
            print(f"stone_b_qr任务完成，总用时: {time.time() - start_time:.2f}秒")
            if detected_labels:
                print(f"成功识别到标签: {detected_labels}")
            else:
                print("未识别到任何标签")
            if limit_gait_executed:
                print("已执行限高杆自定义步态")
        
        print("stone_b_qr任务结束")
        self.walk_straight_until_yellow_line_2(7000,0.1,False,"middle")
        
        # 返回检测结果
        return detected_labels
    
    def stoneside_b(self,detected_labels):
        """石板路侧b区卸装货(一侧路口到另一侧路口)"""

        print("开始前进...")
        dura = 2500
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0.25, 0, 0]
        self.msg.step_height = [0.1, 0.1]
        self.msg.duration = dura  
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        if detected_labels:
            if detected_labels == "B-2":
                print("B区2库位卸货，B区1库位装货")
                self.play_speech("B区库位2")
                #卸货
                print("右移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, -0.2, 0.0]
                self.msg.duration = 8500
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,2500)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=20000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )     

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                # 改为交互
                print("开始卸货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)


                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)


                print("开始后退")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                #配货
                print("左移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.2, 0.0]
                self.msg.duration = 8200
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,2500)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=20000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )     

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                # 改为交互
                print("开始配货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)


                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                print("开始后退")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                print("右移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, -0.2, 0.0]
                self.msg.duration = 8200
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,2500)

                # self.msg.mode = 12  # Recovery stand
                # self.msg.gait_id = 0
                # self.msg.life_count += 1  # Command will take effect when life_count update
                # self.ctrl.Send_cmd(self.msg)
                # self.ctrl.Wait_finish(12, 0)

                print("转身...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.0, 0.8]
                self.msg.duration = 5100
                self.msg.step_height = [0.02, 0.02]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 10,1000)

            elif detected_labels == "B-1":
                print("B区1库位卸货，B区2库位配货")
                self.play_speech("B区库位1")
                #卸货
                # print("开始前进...")
                # dura = 3250
                # self.msg.mode = 11
                # self.msg.gait_id = 27
                # self.msg.vel_des = [0.25, 0, 0]
                # self.msg.step_height = [0.1, 0.1]
                # self.msg.duration = dura  
                # self.msg.life_count += 1
                # self.ctrl.Send_cmd(self.msg)
                # self.ctrl.Wait_finish(11, 27)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=40000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )                

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                # 改为交互
                print("开始卸货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                print("开始后退")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                print("右移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, -0.2, 0.0]
                self.msg.duration = 8500
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(robot.msg)
                self.ctrl.Wait_finish(11, 27,2500)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=40000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )     

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                print("开始配货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                print("开始后退")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,2000)

                print("转身...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.0, 0.8]
                self.msg.duration = 5100
                self.msg.step_height = [0.02, 0.02]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 10,1000)
            else:
                print("未识别到有效标签")
        else:
            print("未检测到任何标签")
    
    def hillside_b(self,detected_labels):
        """坡道侧b区卸装货(一侧路口到另一侧路口)"""

        print("开始前进...")
        dura = 2500
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0.25, 0, 0]
        self.msg.step_height = [0.1, 0.1]
        self.msg.duration = dura  
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        if detected_labels:
            if detected_labels == "B-2":
                print("B区2库位卸货，B区1库位装货")
                self.play_speech("B区库位2")
                #卸货
                # print("开始前进...")
                # dura = 3350
                # self.msg.mode = 11
                # self.msg.gait_id = 27
                # self.msg.vel_des = [0.25, 0, 0]
                # self.msg.step_height = [0.1, 0.1]
                # self.msg.duration = dura  
                # self.msg.life_count += 1
                # self.ctrl.Send_cmd(self.msg)
                # self.ctrl.Wait_finish(11, 27)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=40000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )  

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                # 改为交互
                print("开始卸货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                print("开始后退")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                #配货
                print("左移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.2, 0.0]
                self.msg.duration = 8500
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(robot.msg)
                self.ctrl.Wait_finish(11, 27,2500)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=40000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )  

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                # 改为交互
                print("开始配货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                print("开始后退")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)


                print("转身...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.0, 0.8]
                self.msg.duration = 5100
                self.msg.step_height = [0.02, 0.02]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 10,1000)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)
                time.sleep(1)

            elif detected_labels == "B-1":
                print("B区1库位卸货，B区2库位配货")
                self.play_speech("B区库位1")
                #卸货
                print("左移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.2, 0.0]
                self.msg.duration = 8600
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,2500)
                # time.sleep(1)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                # print("开始前进...")
                # dura = 3350
                # self.msg.mode = 11
                # self.msg.gait_id = 27
                # self.msg.vel_des = [0.25, 0, 0]
                # self.msg.step_height = [0.1, 0.1]
                # self.msg.duration = dura  
                # self.msg.life_count += 1
                # self.ctrl.Send_cmd(self.msg)
                # self.ctrl.Wait_finish(11, 27)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=40000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )  

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                # 改为交互
                print("开始卸货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)


                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                print("开始后退...")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                print("右移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, -0.2, 0.0]
                self.msg.duration = 8400
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(robot.msg)
                self.ctrl.Wait_finish(11, 27,2500)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                #配货
                # print("开始前进...")
                # dura = 3350
                # self.msg.mode = 11
                # self.msg.gait_id = 27
                # self.msg.vel_des = [0.25, 0, 0]
                # self.msg.step_height = [0.1, 0.1]
                # self.msg.duration = dura  
                # self.msg.life_count += 1
                # self.ctrl.Send_cmd(self.msg)
                # self.ctrl.Wait_finish(11, 27)

                detected, walk_time = self.walk_straight_until_yellow_line(
                    max_duration=40000,  
                    velocity=0.10,       # 较慢的速度便于观察
                    enable_debug=False   # 启用调试输出
                )  

                print("开始前进...")
                dura = 600
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                # 改为交互
                print("开始配货")
                dura = 5000
                self.msg.mode = 7
                self.msg.gait_id = 1
                self.msg.duration = dura
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(7,1)

                # 等待语音"完成"指令
                self.wait_for_voice_completion(duration=30.0)
                time.sleep(3)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)

                print("开始后退")
                dura = 4000
                self.msg.mode = 11
                self.msg.gait_id = 27
                self.msg.vel_des = [-0.25, 0, 0]
                self.msg.step_height = [0.1, 0.1]
                self.msg.duration = dura  # Until close to turn
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27)

                print("左移...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.2, 0.0]
                self.msg.duration = 8400
                self.msg.step_height = [0.06, 0.06]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 27,2500)



                print("转身...")
                self.msg.mode = 11 
                self.msg.gait_id = 27
                self.msg.vel_des = [0.0, 0.0, 0.8]
                self.msg.duration = 5100
                self.msg.step_height = [0.02, 0.02]
                self.msg.life_count += 1
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(11, 10,1000)

                self.msg.mode = 12  # Recovery stand
                self.msg.gait_id = 0
                self.msg.life_count += 1  # Command will take effect when life_count update
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)
                time.sleep(1)

            else:
                print("未识别到有效标签")

        else:
            print("未检测到任何标签")

    def b_qr_hill(self):
        """b二维码到上坡前，持续检测黄灯"""
        print("开始b二维码到上坡前的直走，同时检测黄灯")
        
        # 设置黄灯检测距离阈值为70cm（需要根据实际像素比例调整）
        original_stop_distance = self.stop_distance
        self.stop_distance = 0.7  # 70cm停止距离
        
        try:
            # 开始直走并持续检测黄灯
            start_time = time.time()
            total_walk_time = 44.5  # 总直走时间25秒
            yellow_stop_executed = False  # 标记是否已执行过黄灯停止
            
            while time.time() - start_time < total_walk_time:
                # 如果还没有执行过黄灯停止，继续检测黄灯
                if not yellow_stop_executed:
                    yellow_detected, distance = self.detect_yellow_light(enable_debug_visual=False)
                    
                    if yellow_detected and distance <= 0.7:
                        print(f"检测到黄灯，距离约{distance:.2f}米，开始停止5秒")
                        
                        # 停止机器狗
                        self.msg.mode = 12  # 停止模式
                        self.msg.gait_id = 0
                        self.msg.duration = 0
                        self.msg.life_count += 1
                        self.ctrl.Send_cmd(self.msg)
                        
                        # 等待5秒并播报倒数
                        for countdown in [5, 4, 3, 2, 1]:
                            self.play_speech(str(countdown))
                            print(f"倒数: {countdown}")
                            time.sleep(1.0)
                        
                        print("停止5秒完成，继续直走")
                        yellow_stop_executed = True
                        continue
                
                # 执行基于偏移量的直走控制（短时间）
                self.pid_straight_walk_with_boundary_detection(walk_time=0.5, target_velocity=0.1, enable_debug_visual=False)
                
                # 短暂延时（增大限高杆检测频率）
                # time.sleep(0.005)
                
        except Exception as e:
            print(f"b_qr_hill执行异常: {e}")
        finally:
            # 恢复原始停止距离
            self.stop_distance = original_stop_distance
            print(f"b二维码到上坡前的任务完成，总用时: {time.time() - start_time:.2f}秒")
    


    def b_qr_stone(self):
        """b二维码到石板路前，持续检测红色限高杆"""
        print("开始b_qr_stone任务，从b二维码到石板路前，持续检测红色限高杆")
        
        # 保存原始停止距离
        original_limit_stop_distance = self.limit_stop_distance
        
        # 设置检测距离阈值（使用height_limit.py的设置）
        self.limit_stop_distance = 0.65  # 红色限高杆检测距离
        
        try:
            start_time = time.time()
            total_walk_time = 34 # 总直走时间
            limit_gait_executed = False  # 标记是否已执行过限高杆自定义步态
            
            while time.time() - start_time < total_walk_time:
                current_time = time.time()
                
                # 持续检测红色限高杆（如果还没有执行过自定义步态）
                if not limit_gait_executed:
                    limit_detected, limit_distance = self.detect_limit_barrier(enable_debug_visual=False)
                    
                    if limit_detected and limit_distance <= 0.65:
                        print(f"检测到红色限高杆，距离约{limit_distance:.2f}米，开始执行自定义步态")
                        
                        # 执行自定义步态（低高度步态）
                        print("执行低高度自定义步态...")

                        self.execute_custom_gait_low_height()
                        
                        print("自定义步态执行完毕，继续直走")
                        limit_gait_executed = True
                        continue
                
                # 执行基于偏移量的直走控制（短时间步进）
                self.pid_straight_walk_with_boundary_detection(walk_time=0.5, target_velocity=0.15, enable_debug_visual=False)
                
                # 短暂延时
                time.sleep(0.01)
                
        except Exception as e:
            print(f"b_qr_stone执行异常: {e}")
        finally:
            # 恢复原始停止距离
            self.limit_stop_distance = original_limit_stop_distance
            print(f"b_qr_stone任务完成，总用时: {time.time() - start_time:.2f}秒")
            if limit_gait_executed:
                print("已执行红色限高杆自定义步态")
            else:
                print("未检测到红色限高杆")
        
        print("b_qr_stone任务结束")

    def execute_custom_gait_low_height(self):
        """自定义限高杆步态"""
        print("开始执行自定义限高杆步态...")
        try:
            from file_send_lcmt import file_send_lcmt
            
            # 定义robot_cmd模板
            robot_cmd = {
                'mode':0, 'gait_id':0, 'contact':0, 'life_count':0,
                'vel_des':[0.0, 0.0, 0.0],
                'rpy_des':[0.0, 0.0, 0.0],
                'pos_des':[0.0, 0.0, 0.0],
                'acc_des':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'ctrl_point':[0.0, 0.0, 0.0],
                'foot_pose':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'step_height':[0.0, 0.0],
                'value':0,  'duration':0
            }
            
            usergait_msg = file_send_lcmt()
            cmd_msg = robot_control_cmd_lcmt()
            
            # 加载步态参数
            steps = toml.load("Gait_Params_low_height.toml")
            full_steps = {'step':[robot_cmd]}
            k = 0
            
            for i in steps['step']:
                cmd = copy.deepcopy(robot_cmd)
                cmd['duration'] = i['duration']
                if i['type'] == 'usergait':                
                    cmd['mode'] = 11 # LOCOMOTION
                    cmd['gait_id'] = 110 # USERGAIT
                    cmd['vel_des'] = i['body_vel_des']
                    cmd['rpy_des'] = i['body_pos_des'][0:3]
                    cmd['pos_des'] = i['body_pos_des'][3:6]
                    cmd['foot_pose'][0:2] = i['landing_pos_des'][0:2]
                    cmd['foot_pose'][2:4] = i['landing_pos_des'][3:5]
                    cmd['foot_pose'][4:6] = i['landing_pos_des'][6:8]
                    cmd['ctrl_point'][0:2] = i['landing_pos_des'][9:11]
                    cmd['step_height'][0] = math.ceil(i['step_height'][0] * 1e3) + math.ceil(i['step_height'][1] * 1e3) * 1e3
                    cmd['step_height'][1] = math.ceil(i['step_height'][2] * 1e3) + math.ceil(i['step_height'][3] * 1e3) * 1e3
                    cmd['acc_des'] = i['weight']
                    cmd['value'] = i['use_mpc_traj']
                    cmd['contact'] = math.floor(i['landing_gain'] * 1e1)
                    cmd['ctrl_point'][2] =  i['mu']
                if k == 0:
                    full_steps['step'] = [cmd]
                else:
                    full_steps['step'].append(cmd)
                k = k + 1
                
            # 生成完整步态参数文件
            f = open("Gait_Params_low_height_full.toml", 'w')
            f.write("# Gait Params\n")
            f.writelines(toml.dumps(full_steps))
            f.close()

            # 发送步态定义文件
            file_obj_gait_def = open("Gait_Def_low_height.toml",'r')
            file_obj_gait_params = open("Gait_Params_low_height_full.toml",'r')
            usergait_msg.data = file_obj_gait_def.read()
            self.ctrl.lc_s.publish("user_gait_file", usergait_msg.encode())
            time.sleep(0.5)
            usergait_msg.data = file_obj_gait_params.read()
            self.ctrl.lc_s.publish("user_gait_file", usergait_msg.encode())
            time.sleep(0.1)
            file_obj_gait_def.close()
            file_obj_gait_params.close()

            # 执行步态列表
            user_gait_list = open("Usergait_List_low_height.toml",'r')
            steps = toml.load(user_gait_list)
            for step in steps['step']:
                print(f"执行步态: mode={step['mode']}, gait_id={step['gait_id']}")
                cmd_msg.mode = step['mode']
                cmd_msg.value = step['value']
                cmd_msg.contact = step['contact']
                cmd_msg.gait_id = step['gait_id']
                cmd_msg.duration = step['duration']  
                cmd_msg.life_count += 1
                for i in range(3):
                    cmd_msg.vel_des[i] = step['vel_des'][i]
                    cmd_msg.rpy_des[i] = step['rpy_des'][i]
                    cmd_msg.pos_des[i] = step['pos_des'][i]
                    cmd_msg.acc_des[i] = step['acc_des'][i]
                    cmd_msg.acc_des[i+3] = step['acc_des'][i+3]
                    cmd_msg.foot_pose[i] = step['foot_pose'][i]
                    cmd_msg.ctrl_point[i] = step['ctrl_point'][i]
                for i in range(2):
                    cmd_msg.step_height[i] = step['step_height'][i]
                
                # 发送命令并等待完成
                self.ctrl.Send_cmd(cmd_msg)
                self.ctrl.Wait_finish_low_height(step['mode'], step['gait_id'])
                print(f"步态执行完成: mode={step['mode']}, gait_id={step['gait_id']}")
                
            # 维持心跳2秒
            for i in range(10): 
                self.ctrl.lc_s.publish("robot_control_cmd", cmd_msg.encode())
                time.sleep(0.2)
                
            print("自定义前向步态执行完成")
                    
        except Exception as e:
            print(f"执行自定义前向步态时出错: {e}")
            raise

    def execute_custom_gait_stone_road_pid_step(self, duration=0.5):
        """执行一步石板路PID控制（参考s弯实现）
        
        Args:
            duration: 执行时长（秒）
        """
        try:
            # 初始化PID控制器（仅用于横移控制）
            slope_pid = PIDController(
                kp=0.008, ki=0.001, kd=0.003,
                setpoint=0.0, integral_limit=0.1
            )
            
            # 滤波器
            if not hasattr(self, 'slope_filter'):
                self.slope_filter = LowPassFilter(alpha=0.7)
            
            # 控制参数
            turn_sensitivity = 0.05
            slope_deadzone = 5.0
            max_turn_rate = 0.2
            
            # 检测边界偏移
            boundary_error, left_distance, right_distance = self.detect_boundary_error_for_straight_walk()
            
            # 滤波和死区处理
            filtered_boundary_error = self.slope_filter.update(boundary_error)
            boundary_error_with_deadzone = self.apply_deadzone(filtered_boundary_error, slope_deadzone)
            
            # PID计算转向速度
            target_yaw_velocity = slope_pid.update(boundary_error_with_deadzone, duration)
            yaw_velocity = float(np.clip(target_yaw_velocity * turn_sensitivity, -max_turn_rate, max_turn_rate))
            
            # 每次循环重新生成和发送更新转向速度的步态参数文件（参考限高杆实现）
            from file_send_lcmt import file_send_lcmt
            import toml
            
            try:
                # 定义robot_cmd模板
                robot_cmd = {
                    'mode':0, 'gait_id':0, 'contact':0, 'life_count':0,
                    'vel_des':[0.0, 0.0, 0.0],
                    'rpy_des':[0.0, 0.0, 0.0],
                    'pos_des':[0.0, 0.0, 0.0],
                    'acc_des':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'ctrl_point':[0.0, 0.0, 0.0],
                    'foot_pose':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'step_height':[0.0, 0.0],
                    'value':0,  'duration':0
                }
                
                # 加载原始步态参数
                steps = toml.load("Gait_Params_stone_road.toml")
                full_steps = {'step':[robot_cmd]}
                k = 0
                
                for i in steps['step']:
                    cmd = copy.deepcopy(robot_cmd)
                    cmd['duration'] = i['duration']
                    if i['type'] == 'usergait':                
                        cmd['mode'] = 11 # LOCOMOTION
                        cmd['gait_id'] = 110 # USERGAIT
                        # 更新转向速度到body_vel_des
                        cmd['vel_des'] = [i['body_vel_des'][0], i['body_vel_des'][1], yaw_velocity]
                        cmd['rpy_des'] = i['body_pos_des'][0:3]
                        cmd['pos_des'] = i['body_pos_des'][3:6]
                        cmd['foot_pose'][0:2] = i['landing_pos_des'][0:2]
                        cmd['foot_pose'][2:4] = i['landing_pos_des'][3:5]
                        cmd['foot_pose'][4:6] = i['landing_pos_des'][6:8]
                        cmd['ctrl_point'][0:2] = i['landing_pos_des'][9:11]
                        cmd['step_height'][0] = math.ceil(i['step_height'][0] * 1e3) + math.ceil(i['step_height'][1] * 1e3) * 1e3
                        cmd['step_height'][1] = math.ceil(i['step_height'][2] * 1e3) + math.ceil(i['step_height'][3] * 1e3) * 1e3
                        cmd['acc_des'] = i['weight']
                        cmd['value'] = i['use_mpc_traj']
                        cmd['contact'] = math.floor(i['landing_gain'] * 1e1)
                        cmd['ctrl_point'][2] =  i['mu']
                    if k == 0:
                        full_steps['step'] = [cmd]
                    else:
                        full_steps['step'].append(cmd)
                    k = k + 1
                    
                # 生成完整步态参数文件
                f = open("Gait_Params_stone_road_full.toml", 'w')
                f.write("# Gait Params\n")
                f.writelines(toml.dumps(full_steps))
                f.close()
                
                # 发送更新后的步态参数文件
                usergait_msg = file_send_lcmt()
                with open("Gait_Params_stone_road_full.toml", 'r') as f:
                    usergait_msg.data = f.read()
                    self.ctrl.lc_s.publish("user_gait_file", usergait_msg.encode())
                    time.sleep(0.1)  # 等待参数更新
                
            except Exception as e:
                print(f"生成和发送步态参数文件失败: {e}")
            
            # 使用自定义石板路步态
            msg = robot_control_cmd_lcmt()
            msg.mode = 62  # User defined gait mode
            msg.gait_id = 110  # User gait ID
            msg.vel_des = [0.15, 0, yaw_velocity]  # 前进速度0.15，转向速度由PID控制
            msg.rpy_des = [0, 0, 0]
            msg.pos_des = [0, 0, 0]
            msg.step_height = [0, 0]
            msg.duration = int(duration * 1000)
            msg.life_count = (self.msg.life_count + 1) % 127
            
            # 发送指令
            self.ctrl.Send_cmd(msg)
            self.msg.life_count = msg.life_count
            
            # 等待执行完成
            time.sleep(duration)
            
            # 调试输出
            direction_str = "直走" if abs(boundary_error_with_deadzone) < slope_deadzone else ("右转" if boundary_error_with_deadzone > 0 else "左转")
            print(f"[石板路PID] 转向速度:{yaw_velocity:.3f}rad/s "
                  f"边界偏移:{boundary_error:.2f}像素 "
                  f"左距离:{left_distance:.1f} 右距离:{right_distance:.1f} "
                  f"状态:{direction_str}")
            
        except Exception as e:
            print(f"石板路PID单步执行异常: {e}")
    
    def execute_custom_gait_stone_road_pid_loop(self, loop_count=50):
        """循环执行石板路PID控制（参考s弯实现）
        
        Args:
            loop_count: 循环次数，默认50次
        """
        print(f"开始石板路PID循环控制，共{loop_count}次...")
        
        try:
            # 发送自定义步态文件（只需发送一次）
            from file_send_lcmt import file_send_lcmt
            usergait_msg = file_send_lcmt()
            
            # 发送步态定义文件
            with open("Gait_Def_stone_road.toml", 'r') as f:
                usergait_msg.data = f.read()
                self.ctrl.lc_s.publish("user_gait_file", usergait_msg.encode())
                time.sleep(0.5)
            
            # 发送步态参数文件
            with open("Gait_Params_stone_road.toml", 'r') as f:
                usergait_msg.data = f.read()
                self.ctrl.lc_s.publish("user_gait_file", usergait_msg.encode())
                time.sleep(0.1)
            
            # 循环执行PID控制
            for i in range(loop_count):
                print(f"执行第{i+1}/{loop_count}次石板路PID控制...")
                self.execute_custom_gait_stone_road_pid_step(duration=0.5)
                time.sleep(0.1)  # 短暂延时
            
            print("石板路PID循环控制完成")
            
        except Exception as e:
            print(f"石板路PID循环执行异常: {e}")
            raise
        
    def execute_custom_gait_stone_road_pid(self):
        """自定义石板路步态（调用循环函数）"""
        print("开始执行自定义石板路步态...")
        # 调用循环执行函数
        self.execute_custom_gait_stone_road_pid_loop(loop_count=41)
        print("自定义石板路步态执行完成")
    
    def execute_custom_gait_stone_road_pid_return(self, max_loop_count=45, enable_debug=False):
        """石板路返程PID控制函数，持续检测黄线直到检测到为止
        
        Args:
            max_loop_count: 最大循环次数，默认45次
            enable_debug: 是否启用调试输出
            
        Returns:
            tuple: (是否检测到黄线, 实际执行时间)
        """
        print(f"开始石板路返程PID控制，最大循环次数: {max_loop_count}次")
        print("持续检测黄线，检测到后立即停止石板路自定义步态")
        
        start_time = time.time()
        detection_count = 0
        step_count = 0
        loop_count = 0
        
        try:
            # 发送自定义步态文件（只需发送一次）
            from file_send_lcmt import file_send_lcmt
            usergait_msg = file_send_lcmt()
            
            # 发送步态定义文件
            with open("Gait_Def_stone_road.toml", 'r') as f:
                usergait_msg.data = f.read()
                self.ctrl.lc_s.publish("user_gait_file", usergait_msg.encode())
                time.sleep(0.5)
            
            # 发送步态参数文件
            with open("Gait_Params_stone_road.toml", 'r') as f:
                usergait_msg.data = f.read()
                self.ctrl.lc_s.publish("user_gait_file", usergait_msg.encode())
                time.sleep(0.1)
            
            # 持续执行石板路PID控制直到检测到黄线
            while loop_count < max_loop_count:
                loop_count += 1
                step_count += 1
                current_time = time.time() - start_time
                
                if enable_debug:
                    print(f"[{current_time:.1f}s] 第{loop_count}/{max_loop_count}次循环，执行第{step_count}次石板路PID控制...")
                
                # 执行一步石板路PID控制
                self.execute_custom_gait_stone_road_pid_step(duration=0.5)
                
                # 在每次PID控制步骤后检测黄线
                yellow_detected = self.detect_yellow_line(enable_debug_window=False, stop_on_detection=False)
                detection_count += 1
                
                if yellow_detected:
                    current_time = time.time() - start_time
                    print(f"[{current_time:.1f}s] 检测到黄线！停止石板路返程")
                    print(f"总循环次数: {loop_count}/{max_loop_count}, 总执行步数: {step_count}, 总检测次数: {detection_count}")

                    self.execute_custom_gait_stone_road_pid_step(duration=2.0)
                    
                    # # 立即停止自定义步态，切换到站立模式
                    # self.msg.mode = 12  # 站立模式
                    # self.msg.gait_id = 0
                    # self.msg.duration = 0
                    # self.msg.life_count = (self.msg.life_count + 1) % 128
                    # self.ctrl.Send_cmd(self.msg)
                    # self.ctrl.Wait_finish(12, 0)
                    
                    return True, current_time
                
                # 调试输出
                if enable_debug and detection_count % 10 == 0:  # 每10次检测输出一次
                    print(f"[{current_time:.1f}s] 第{loop_count}/{max_loop_count}次循环，执行{step_count}步，检测{detection_count}次，未发现黄线，继续石板路返程")
                
                # 短暂延时
                time.sleep(0.1)
            
            # 达到最大循环次数停止
            print(f"达到最大循环次数 {max_loop_count}次，停止石板路返程")
            print(f"总循环次数: {loop_count}/{max_loop_count}, 总执行步数: {step_count}, 总检测次数: {detection_count}")

            
            # 停止自定义步态
            self.msg.mode = 12  # 站立模式
            self.msg.gait_id = 0
            self.msg.duration = 0
            self.msg.life_count = (self.msg.life_count + 1) % 128
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(12, 0)
            
            return False, max_duration
            
        except Exception as e:
            print(f"石板路返程PID控制过程出错: {e}")
            # 确保停止运动
            try:
                self.msg.mode = 12
                self.msg.gait_id = 0
                self.msg.duration = 0
                self.msg.life_count = (self.msg.life_count + 1) % 128
                self.ctrl.Send_cmd(self.msg)
                self.ctrl.Wait_finish(12, 0)
            except:
                pass
            return False, time.time() - start_time
        


    def s_finish(self,detected_labels):
        """出s弯到结束"""

        print("左转...")
        dura = 300  
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0, 0, 0.8]  
        self.msg.duration = dura
        self.msg.step_height = [0.02, 0.02]  # 转向时降低步高
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27,300)

        # 开始持续行走模式
        self.msg.mode = 11  # 行走模式
        self.msg.gait_id = 27  # 步态类型
        self.msg.vel_des = [0.1, 0, 0]  # 期望速度[x, y, yaw] (m/s)
        self.msg.step_height = [0.1, 0.1]  # 步高(m)
        self.msg.duration = 10000  # 设置较长的持续时间，实际由检测结果控制
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)

        # 持续检测黄色边界
        start_time = time.time()
        detection_count = 0
        max_walk_time = 30.0  # 最大行走时间（安全限制）

        print("开始边界检测循环...")
        while time.time() - start_time < max_walk_time:
            # 执行黄色边界检测
            detected, distance, position = self.detect_yellow_boundary_rgb(enable_debug_visual=False, side="left")
            detection_count += 1
            
            if detected or distance > 0:  # 检测到黄色边界或发现黄色区域
                current_time = time.time() - start_time
                if distance > 2.65:
                     print(f"[检测{detection_count}次] [{current_time:.1f}s] 检测到黄色边界距离{distance:.2f}m > 2.7m，停止前进")
                     break
                else:
                     print(f"[检测{detection_count}次] [{current_time:.1f}s] 检测到黄色边界距离{distance:.2f}m ≤ 2.7m，继续前进")
            else:
                # 每50次检测输出一次状态
                if detection_count % 10 == 0:
                    current_time = time.time() - start_time
                    print(f"[检测{detection_count}次] [{current_time:.1f}s] 未检测到黄色边界，继续前进")
            
            # 控制检测频率（20Hz）
            time.sleep(0.05)
        
        # 停止行走
        print("停止前进...")
        self.msg.mode = 12  # 站立模式
        self.msg.gait_id = 0
        self.msg.duration = 0
        self.msg.life_count = (self.msg.life_count + 1) % 128
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(12, 0)
        
        total_time = time.time() - start_time
        print(f"前进完成，总用时: {total_time:.1f}秒，检测次数: {detection_count}次")


        print("左转弯90度...")
        dura = 2050  
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0, 0, 0.8]  
        self.msg.duration = dura
        self.msg.step_height = [0.02, 0.02]  # 转向时降低步高
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27,500)

        self.msg.mode = 12  # 站立模式
        self.msg.gait_id = 0
        self.msg.duration = 0
        self.msg.life_count = (self.msg.life_count + 1) % 128
        self.ctrl.Send_cmd(self.msg)
        time.sleep(2)


        detected, walk_time = self.walk_straight_until_yellow_line(
            max_duration=80000,  # 最大15秒
            velocity=0.1,       # 较慢的速度便于观察
            enable_debug=False,   # 启用调试输出
            detection_position = "quarter"
        )

        # print("开始前进...")
        # dura = 700
        # self.msg.mode = 11
        # self.msg.gait_id = 27
        # self.msg.vel_des = [0.25, 0, 0]
        # self.msg.step_height = [0.1, 0.1]
        # self.msg.duration = dura  
        # self.msg.life_count += 1
        # self.ctrl.Send_cmd(self.msg)
        # self.ctrl.Wait_finish(11, 27,500)  

        if detected_labels == "A-2":
            print("A区2库位卸货")

            print("开始右移...")
            dura = 4800
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [0, -0.25, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            #卸货
            # print("开始后退...")
            # dura = 3000
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [-0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            # detected, walk_time = self.walk_straight_until_yellow_line(
            #     max_duration=8000,  # 最大15秒
            #     velocity=-0.1,       # 较慢的速度便于观察
            #     enable_debug=False,   # 启用调试输出
            #     detection_position = "quarter"
            # )

            print("开始后退...")
            dura = 2800
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [-0.25, 0, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            print("开始卸货")
            dura = 5000
            self.msg.mode = 7
            self.msg.gait_id = 1
            self.msg.duration = dura
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(7, 1)

            # 等待语音"完成"指令
            self.wait_for_voice_completion(duration=30.0)
            time.sleep(3)

            self.msg.mode = 12  # Recovery stand
            self.msg.gait_id = 0
            self.msg.life_count += 1  # Command will take effect when life_count update
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(12, 0)

            # print("开始前进")
            # dura = 3100
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            detected, walk_time = self.walk_straight_until_yellow_line(
                max_duration=40000,  # 最大15秒
                velocity=0.1,       # 较慢的速度便于观察
                enable_debug=False,   # 启用调试输出
                detection_position = "quarter"
            )

            # print("开始前进...")
            # dura = 1200
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27,500)

            print("左移...")
            dura = 4700
            self.msg.mode = 11 
            self.msg.gait_id = 27
            self.msg.vel_des = [0.0, 0.2, 0.0]
            self.msg.duration = 3900
            self.msg.step_height = [0.1, 0.1]
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

        else:
            print("A区1库位卸货")

            print("开始左移...")
            dura = 4700
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [0, 0.25, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            #卸货
            # print("开始后退...")
            # dura = 3000
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [-0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            # detected, walk_time = self.walk_straight_until_yellow_line(
            #     max_duration=8000,  # 最大15秒
            #     velocity=-0.1,       # 较慢的速度便于观察
            #     enable_debug=False,   # 启用调试输出
            #     detection_position = "quarter"
            # )

            print("开始后退...")
            dura = 2800
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [-0.25, 0, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

            # 改为交互
            print("开始卸货")
            dura = 5000
            self.msg.mode = 7
            self.msg.gait_id = 1
            self.msg.duration = dura
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(7, 1)

            # 等待语音"完成"指令
            self.wait_for_voice_completion(duration=30.0)
            time.sleep(3)

            self.msg.mode = 12  # Recovery stand
            self.msg.gait_id = 0
            self.msg.life_count += 1  # Command will take effect when life_count update
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(12, 0)

            # print("开始前进")
            # dura = 3100
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27)

            detected, walk_time = self.walk_straight_until_yellow_line(
                max_duration=40000,  # 最大15秒
                velocity=0.1,       # 较慢的速度便于观察
                enable_debug=False,   # 启用调试输出
                detection_position = "quarter"
            )

            # print("开始前进...")
            # dura = 1200
            # self.msg.mode = 11
            # self.msg.gait_id = 27
            # self.msg.vel_des = [0.25, 0, 0]
            # self.msg.step_height = [0.1, 0.1]
            # self.msg.duration = dura  # Until close to turn
            # self.msg.life_count += 1
            # self.ctrl.Send_cmd(self.msg)
            # self.ctrl.Wait_finish(11, 27,500)

            print("开始右移...")
            dura = 5000
            self.msg.mode = 11
            self.msg.gait_id = 27
            self.msg.vel_des = [0, -0.25, 0]
            self.msg.step_height = [0.1, 0.1]
            self.msg.duration = dura  # Until close to turn
            self.msg.life_count += 1
            self.ctrl.Send_cmd(self.msg)
            self.ctrl.Wait_finish(11, 27)

        detected, walk_time = robot.walk_straight_until_yellow_line(
            max_duration=8000,  # 最大15秒
            velocity=-0.08,       # 较慢的速度便于观察
            enable_debug=False,   # 启用调试输出
            detection_position = "quarter"
        )

        print("后退...")
        dura = 5800
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [-0.3, 0, 0]
        self.msg.duration = dura
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        print("左转弯90度...")
        dura = 2500  
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [0, 0, 0.8]  
        self.msg.duration = dura
        self.msg.step_height = [0.02, 0.02]  # 转向时降低步高
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        print("后退...")
        dura = 2900
        self.msg.mode = 11
        self.msg.gait_id = 27
        self.msg.vel_des = [-0.3, 0, 0]
        self.msg.duration = dura
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(11, 27)

        print("结束")
        dura = 5000
        self.msg.mode = 7
        self.msg.gait_id = 1
        self.msg.duration = dura
        self.msg.life_count += 1
        self.ctrl.Send_cmd(self.msg)
        self.ctrl.Wait_finish(7, 1)



#主函数
def main():
    try:

        #测试用
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")



        robot.origin_a_qr()

        detected, walk_time = robot.walk_straight_until_yellow_line(
            max_duration=25000,  # 最大15秒
            velocity=0.1,       # 较慢的速度便于观察
            enable_debug=False,   # 启用调试输出
            detection_position = "quarter"
        )

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)




        # # 单张拍照
        robot.take_photo("test_single.jpg")
        time.sleep(2)
        # 使用RGB图像拍照
        # robot.take_photo("test_rgb.jpg", use_rgb=True)
        # robot.take_photo("test_single.jpg")
        
        # 调用OCR识别函数
        a_detected_labels = ""
        a_detected_labels = recognize_photo_labels("test_single.jpg")
        if a_detected_labels:
            print(f"识别到的标签: {a_detected_labels}")
        else:
            print("未识别到任何标签")

        robot.a_qr_s(a_detected_labels)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        print("开始左转...")
        dura = 700
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0 ,0 , 0.25]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27,500)

        robot.msg.mode = 12  # Recovery stand 模式\
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        arrow_result = "right"
        # 箭头检测功能
        arrow_result = robot.detect_arrow_direction(timeout=15.0)
        robot.arrow_s(arrow_result)
        if arrow_result == "left":
            robot.execute_custom_gait_stone_road_pid()
            b_detected_labels = ""
            b_detected_labels = robot.stone_b_qr()
            robot.stoneside_b(b_detected_labels)
            robot.b_qr_hill()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.hill_return()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)


            #进入s弯
            print("开始右移...")
            dura = 4500
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0, -0.25, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  # Until close to turn
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            print("开始前进...")
            dura = 1000
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0.25, 0, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_bby_main_function()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_finish(a_detected_labels)
            
        elif arrow_result == "right":
            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.hill()
            b_detected_labels = ""
            b_detected_labels = robot.hill_b_qr()
            robot.hillside_b(b_detected_labels)
            robot.b_qr_stone()
            robot.execute_custom_gait_stone_road_pid_return(45)
            #进入s弯
            print("开始左移...")
            dura = 4300
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0, 0.25, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  # Until close to turn
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            print("开始前进...")
            dura = 1000
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0.25, 0, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)
            
            robot.s_bby_main_function()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_finish(a_detected_labels)

        else:
            print("无效检测")

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)


def s_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")


        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        print("开始左转...")
        dura = 700
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0 ,0 , 0.25]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27,500)

        robot.msg.mode = 12  # Recovery stand 模式\
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        arrow_result = "right"
        # 箭头检测功能
        arrow_result = robot.detect_arrow_direction(timeout=15.0)
        robot.arrow_s(arrow_result)
        if arrow_result == "left":
            robot.execute_custom_gait_stone_road_pid()
            b_detected_labels = ""
            b_detected_labels = robot.stone_b_qr()
            robot.stoneside_b(b_detected_labels)
            robot.b_qr_hill()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.hill_return()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)


            #进入s弯
            print("开始右移...")
            dura = 4500
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0, -0.25, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  # Until close to turn
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            print("开始前进...")
            dura = 1000
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0.25, 0, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_bby_main_function()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_finish(a_detected_labels)
            
        elif arrow_result == "right":
            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.hill()
            b_detected_labels = ""
            b_detected_labels = robot.hill_b_qr()
            robot.hillside_b(b_detected_labels)
            robot.b_qr_stone()
            robot.execute_custom_gait_stone_road_pid_return(45)
            #进入s弯
            print("开始左移...")
            dura = 4300
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0, 0.25, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  # Until close to turn
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            print("开始前进...")
            dura = 1000
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0.25, 0, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)
            
            robot.s_bby_main_function()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_finish(a_detected_labels)

        else:
            print("无效检测")

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)



def arrow_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")



        robot.msg.mode = 12  # Recovery stand 模式\
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        arrow_result = "right"
        # 箭头检测功能
        arrow_result = robot.detect_arrow_direction(timeout=15.0)
        robot.arrow_s(arrow_result)
        if arrow_result == "left":
            robot.execute_custom_gait_stone_road_pid()
            b_detected_labels = ""
            b_detected_labels = robot.stone_b_qr()
            robot.stoneside_b(b_detected_labels)
            robot.b_qr_hill()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.hill_return()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)


            #进入s弯
            print("开始右移...")
            dura = 4500
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0, -0.25, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  # Until close to turn
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            print("开始前进...")
            dura = 1000
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0.25, 0, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_bby_main_function()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_finish(a_detected_labels)
            
        elif arrow_result == "right":
            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.hill()
            b_detected_labels = ""
            b_detected_labels = robot.hill_b_qr()
            robot.hillside_b(b_detected_labels)
            robot.b_qr_stone()
            robot.execute_custom_gait_stone_road_pid_return(45)
            #进入s弯
            print("开始左移...")
            dura = 4300
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0, 0.25, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  # Until close to turn
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            print("开始前进...")
            dura = 1000
            robot.msg.mode = 11
            robot.msg.gait_id = 27
            robot.msg.vel_des = [0.25, 0, 0]
            robot.msg.step_height = [0.1, 0.1]
            robot.msg.duration = dura  
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(11, 27)

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)
            
            robot.s_bby_main_function()

            robot.msg.mode = 12  # Recovery stand 模式
            robot.msg.gait_id = 0
            robot.msg.life_count += 1
            robot.ctrl.Send_cmd(robot.msg)
            robot.ctrl.Wait_finish(12, 0)
            time.sleep(2)

            robot.s_finish(a_detected_labels)

        else:
            print("无效检测")

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




def hill_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改

        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.hill()
        b_detected_labels = ""
        b_detected_labels = robot.hill_b_qr()
        robot.hillside_b(b_detected_labels)
        robot.b_qr_stone()
        robot.execute_custom_gait_stone_road_pid_return(45)
        #进入s弯
        print("开始左移...")
        dura = 4300
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, 0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)
        
        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)


        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)



def hill_b_qr_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改

        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")

        b_detected_labels = robot.hill_b_qr()
        robot.hillside_b(b_detected_labels)
        robot.b_qr_stone()
        robot.execute_custom_gait_stone_road_pid_return(45)
        #进入s弯
        print("开始左移...")
        dura = 4300
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, 0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)
        
        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)


        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




def hillside_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        b_detected_labels = "B-1"
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")


        # b_detected_labels = robot.hill_b_qr()
        robot.hillside_b(b_detected_labels)
        robot.b_qr_stone()
        robot.execute_custom_gait_stone_road_pid_return(45)
        #进入s弯
        print("开始左移...")
        dura = 4300
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, 0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)
        
        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)


        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)



def b_qr_stone_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")


        # b_detected_labels = robot.hill_b_qr()
        robot.b_qr_stone()
        robot.execute_custom_gait_stone_road_pid_return(45)
        #进入s弯
        print("开始左移...")
        dura = 4300
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, 0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)
        
        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)


        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




def stone_return_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")


        robot.execute_custom_gait_stone_road_pid_return(45)
        #进入s弯
        print("开始左移...")
        dura = 4000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, 0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)
        
        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)


        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)



def stone_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")

        robot.execute_custom_gait_stone_road_pid()
        b_detected_labels = ""
        b_detected_labels = robot.stone_b_qr()
        robot.stoneside_b(b_detected_labels)
        robot.b_qr_hill()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.hill_return()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)


        #进入s弯
        print("开始右移...")
        dura = 4500
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, -0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)
        

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)



def stone_b_qr_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")

        b_detected_labels = robot.stone_b_qr()
        robot.stoneside_b(b_detected_labels)
        robot.b_qr_hill()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.hill_return()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)


        #进入s弯
        print("开始右移...")
        dura = 4500
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, -0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)
        

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




def stoneside_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        b_detected_labels = "B-1"

        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")



        robot.stoneside_b(b_detected_labels)
        robot.b_qr_hill()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.hill_return()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)


        #进入s弯
        print("开始右移...")
        dura = 4500
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, -0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)
     

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)



def b_qr_hill_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改

        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")



        robot.b_qr_hill()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.hill_return()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)


        #进入s弯
        print("开始右移...")
        dura = 4500
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, -0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)
     

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)



def hill_return_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改

        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")




        robot.hill_return()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)


        #进入s弯
        print("开始右移...")
        dura = 4500
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0, -0.25, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  # Until close to turn
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        print("开始前进...")
        dura = 1000
        robot.msg.mode = 11
        robot.msg.gait_id = 27
        robot.msg.vel_des = [0.25, 0, 0]
        robot.msg.step_height = [0.1, 0.1]
        robot.msg.duration = dura  
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(11, 27)

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)
     

        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




def s_return_start():
    try:
        a_detected_labels = "A-1"#此处根据识别更改
        b_detected_labels = "B-2"
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")



        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)
        
        robot.s_bby_main_function()

        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)


        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




def s_finish_start():
    try:
        a_detected_labels = "A-2"#此处根据识别更改
        print("起立...")
        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count = (robot.msg.life_count + 1) % 128
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)

        # 等待图像数据到达
        print("等待图像数据...")
        max_wait_time = 10.0  # 最大等待时间10秒
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查是否两个图像话题都有数据
            if (hasattr(robot.image_subscriber, 'current_image') and 
                robot.image_subscriber.current_image is not None) and \
               (hasattr(robot.image_rgb_subscriber, 'current_image') and 
                robot.image_rgb_subscriber.current_image is not None):
                print("两个图像数据都已接收，开始执行任务...")
                break
            time.sleep(0.1)
        else:
            print("等待图像数据超时，继续执行...")





        robot.msg.mode = 12  # Recovery stand 模式
        robot.msg.gait_id = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        robot.ctrl.Wait_finish(12, 0)
        time.sleep(2)

        robot.s_finish(a_detected_labels)


        # 持续检测电源状态30秒，检测到充电中后程序结束
        print("开始检测电源状态，持续30秒...")
        start_time = time.time()
        check_duration = 30.0  # 检测30秒
        
        while time.time() - start_time < check_duration:
            # 检查是否正在充电
            if hasattr(robot, 'is_charging') and robot.is_charging:
                print("检测到充电中，程序即将结束")
                robot.play_speech("检测到充电，程序结束")
                time.sleep(2)  # 等待语音播放完成
                break
            
            # 每秒检测一次
            time.sleep(1.0)
            elapsed_time = time.time() - start_time
            remaining_time = check_duration - elapsed_time
            if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
                print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        if time.time() - start_time >= check_duration:
            print("电源状态检测超时，程序正常结束")


        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.cleanup()  # 清理所有资源
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.cleanup()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




# 测试函数
def test():
    try:
        # 起立
        # print("起立...")
        # robot.msg.mode = 12  # Recovery stand 模式
        # robot.msg.gait_id = 0
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(12, 0)

        # b_detected_labels = robot.stone_b_qr()
        # robot.stoneside_b(b_detected_labels)
        # robot.hill_b_qr()
        # robot.b_qr_hill()
        # robot.hill_return()
        # robot.take_photo("test_single.jpg")

        detected_labels = recognize_photo_labels("test_single.jpg")

        # robot.execute_custom_gait_stone_road_pid()
        # b_detected_labels = "B-2"
        # robot.b_qr_stone()
        # b_detected_labels = robot.stone_b_qr()
        # robot.stoneside_b(b_detected_labels)
        # robot.b_qr_hill()
        # robot.hill_return()


        # #进入s弯
        # print("开始右移...")
        # dura = 4700
        # robot.msg.mode = 11
        # robot.msg.gait_id = 27
        # robot.msg.vel_des = [0, -0.25, 0]
        # robot.msg.step_height = [0.1, 0.1]
        # robot.msg.duration = dura  # Until close to turn
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(11, 27)

        # print("开始前进...")
        # dura = 1000
        # robot.msg.mode = 11
        # robot.msg.gait_id = 27
        # robot.msg.vel_des = [0.25, 0, 0]
        # robot.msg.step_height = [0.1, 0.1]
        # robot.msg.duration = dura  
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(11, 27)

        # robot.msg.mode = 12  # Recovery stand 模式
        # robot.msg.gait_id = 0
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(12, 0)

        # robot.s_bby_main_function()

        # robot.msg.mode = 12  # Recovery stand 模式
        # robot.msg.gait_id = 0
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(12, 0)

        # robot.s_finish(a_detected_labels)
            

        # robot.hill()
        # b_detected_labels = ""
        # b_detected_labels = robot.hill_b_qr()
        # robot.hillside_b(b_detected_labels)
        # robot.b_qr_stone()
        # robot.execute_custom_gait_stone_road_pid_return(45)
        # #进入s弯
        # print("开始左移...")
        # dura = 4300
        # robot.msg.mode = 11
        # robot.msg.gait_id = 27
        # robot.msg.vel_des = [0, 0.25, 0]
        # robot.msg.step_height = [0.1, 0.1]
        # robot.msg.duration = dura  # Until close to turn
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(11, 27)

        # print("开始前进...")
        # dura = 1000
        # robot.msg.mode = 11
        # robot.msg.gait_id = 27
        # robot.msg.vel_des = [0.25, 0, 0]
        # robot.msg.step_height = [0.1, 0.1]
        # robot.msg.duration = dura  
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(11, 27)

        # robot.msg.mode = 12  # Recovery stand 模式
        # robot.msg.gait_id = 0
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(12, 0)
        
        # robot.s_bby_main_function()

        # robot.msg.mode = 12  # Recovery stand 模式
        # robot.msg.gait_id = 0
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(12, 0)

        # robot.s_finish(a_detected_labels)







        # # 停止
        # print("停止...")
        # robot.msg.mode = 7 
        # robot.msg.gait_id = 1
        # robot.msg.duration = 0
        # robot.msg.life_count += 1
        # robot.ctrl.Send_cmd(robot.msg)
        # robot.ctrl.Wait_finish(7,1)

        # 持续检测电源状态30秒，检测到充电中后程序结束
        # print("开始检测电源状态，持续30秒...")
        # start_time = time.time()
        # check_duration = 30.0  # 检测30秒
        
        # while time.time() - start_time < check_duration:
        #     # 检查是否正在充电
        #     if hasattr(robot, 'is_charging') and robot.is_charging:
        #         print("检测到充电中，程序即将结束")
        #         # robot.play_speech("检测到充电，程序结束")
        #         time.sleep(2)  # 等待语音播放完成
        #         break
            
        #     # 每秒检测一次
        #     time.sleep(1.0)
        #     elapsed_time = time.time() - start_time
        #     remaining_time = check_duration - elapsed_time
        #     if int(elapsed_time) % 5 == 0:  # 每5秒打印一次状态
        #         print(f"电源状态检测中... 剩余时间: {remaining_time:.1f}秒")
        
        # if time.time() - start_time >= check_duration:
        #     print("电源状态检测超时，程序正常结束")

        
        # 程序执行完成，清理资源并退出
        print("程序执行完成，正在退出...")
        robot.ctrl.quit()  # 停止控制线程
        time.sleep(1)  # 等待线程完全停止
        print("程序已退出")
        sys.exit(0)  # 正常退出程序
        
    except KeyboardInterrupt:
        print("检测到中断信号，正在安全退出...")
        # 发送停止命令
        robot.msg.mode = 7  # PureDamper
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.ctrl.quit()
        time.sleep(1)
        print("程序已安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 发送停止命令
        robot.msg.mode = 7
        robot.msg.gait_id = 0
        robot.msg.duration = 0
        robot.msg.life_count += 1
        robot.ctrl.Send_cmd(robot.msg)
        # 清理资源
        robot.ctrl.quit()
        time.sleep(1)
        print("程序已退出")
        sys.exit(1)




if __name__ == '__main__':
    robot = RobotController()
    # test()
    main()
    # s_start()
    # arrow_start()
    # hill_start()
    # hillside_start()
    # b_qr_stone_start()
    # stone_return_start()

    # stone_start()
    # stone_b_qr_start()
    # stoneside_start()
    # b_qr_hill_start()
    # hill_return_start()

    # s_return_start()
    # s_finish_start()