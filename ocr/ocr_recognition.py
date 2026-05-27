#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合OCR识别模块
腾讯云二维码识别 + 阿里云文字识别
专门用于识别A-1、B-1、A-2、B-2标识
支持二维码优先识别功能

使用示例:
from ocr_recognition import OCRRecognizer

# 创建识别器实例
ocr = OCRRecognizer()

# 识别图像
result = ocr.recognize_image('path/to/image.jpg')
if result['success']:
    # print(f"识别到标签: {result['detected_labels']}")
else:
    # print(f"识别失败: {result.get('error', '未知错误')}")

# 配置二维码识别
ocr.set_qrcode_config(enable_qrcode=True, qrcode_priority=True)
"""

import os
import cv2
import numpy as np
import base64
import re
from typing import List, Dict
from PIL import Image
import io
import json

# 阿里云SDK导入
from alibabacloud_ocr_api20210707.client import Client as OcrClient
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_ocr_api20210707 import models as ocr_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

# 腾讯云SDK导入
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models as tencent_models

# --------  配置 --------
TENCENT_SECRET_ID = ""
TENCENT_SECRET_KEY = ""
TENCENT_REGION = "ap-beijing"  # 腾讯云区域

# -------- 阿里云API配置 --------
ALIYUN_ACCESS_KEY_ID = ""
ALIYUN_ACCESS_KEY_SECRET = ""
ALIYUN_REGION = "cn-hangzhou"  # 使用杭州区域
ALIYUN_ENDPOINT = f"ocr-api.{ALIYUN_REGION}.aliyuncs.com"

# -------- 识别配置 --------
VALID_LABELS = ['A-1', 'A-2', 'B-1', 'B-2']
IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.bmp']
CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值

# 二维码识别配置
ENABLE_QRCODE = True  # 是否启用二维码识别
QRCODE_PRIORITY = True  # 二维码优先识别


class OCRRecognizer:
    """
    混合OCR识别器
    腾讯云二维码识别 + 阿里云文字识别
    专门用于识别A-1、B-1、A-2、B-2标识
    """
    
    def __init__(self, enable_qrcode: bool = ENABLE_QRCODE, qrcode_priority: bool = QRCODE_PRIORITY):
        """
        初始化混合OCR客户端
        
        Args:
            enable_qrcode: 是否启用二维码识别
            qrcode_priority: 是否优先进行二维码识别
        """
        self.enable_qrcode = enable_qrcode
        self.qrcode_priority = qrcode_priority
        
        # 初始化阿里云OCR客户端
        try:
            config = open_api_models.Config(
                access_key_id=ALIYUN_ACCESS_KEY_ID,
                access_key_secret=ALIYUN_ACCESS_KEY_SECRET
            )
            config.endpoint = ALIYUN_ENDPOINT
            self.aliyun_client = OcrClient(config)
            # print("阿里云OCR初始化成功")
        except Exception as e:
            # print(f"阿里云OCR初始化失败: {e}")
            raise e
        
        # 初始化 客户端
        try:
            cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "ocr.tencentcloudapi.com"
            
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            
            self.tencent_client = ocr_client.OcrClient(cred, TENCENT_REGION, clientProfile)
            # print(" 初始化成功")
        except Exception as e:
            # print(f" 初始化失败: {e}")
            raise e
    
    def preprocess_image(self, image_path: str) -> bytes:
        """
        预处理图像，转换为base64编码
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像的base64编码字节数据
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return image_data
        except Exception as e:
            # print(f"图像预处理失败: {e}")
            return None
    
    def image_to_base64(self, image_path: str) -> str:
        """
        将图像转换为base64字符串
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的字符串
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            # print(f"图像base64转换失败: {e}")
            return None
    
    def recognize_text(self, image_path: str) -> List[Dict]:
        """
        使用阿里云OCR识别图像中的文字
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            识别结果列表
        """
        try:
            # print(f"正在处理图像: {os.path.basename(image_path)}")
            
            # 读取图像二进制数据
            image_data = self.preprocess_image(image_path)
            
            # print("正在调用阿里云OCR API...")
            
            # 创建识别请求
            request = ocr_models.RecognizeAllTextRequest()
            request.body = image_data
            request.type = "General"  # 设置识别类型为通用文字识别
            
            # 创建运行时选项
            runtime = util_models.RuntimeOptions()
            
            # 调用API
            response = self.aliyun_client.recognize_all_text_with_options(request, runtime)
            
            # print(f"阿里云OCR识别完成")
            
            # 解析结果
            texts = []
            if response.body and response.body.data:
                content = response.body.data.content
                if content:
                    # print(f"检测到 {len(content)} 个文字区域")
                    
                    for item in content:
                        # 处理字符串格式的结果
                        if isinstance(item, str):
                            text_info = {
                                'text': item,
                                'confidence': 1.0,
                                'location': {
                                    'left': 0,
                                    'top': 0,
                                    'width': 100,
                                    'height': 30
                                }
                            }
                        else:
                            # 处理对象格式的结果
                            text_info = {
                                'text': getattr(item, 'text', '') if hasattr(item, 'text') else str(item),
                                'confidence': getattr(item, 'confidence', 1.0) if hasattr(item, 'confidence') else 1.0,
                                'location': {
                                    'left': int(item.text_rectangles[0]) if hasattr(item, 'text_rectangles') and item.text_rectangles and len(item.text_rectangles) >= 1 else 0,
                                    'top': int(item.text_rectangles[1]) if hasattr(item, 'text_rectangles') and item.text_rectangles and len(item.text_rectangles) >= 2 else 0,
                                    'width': int(item.text_rectangles[2] - item.text_rectangles[0]) if hasattr(item, 'text_rectangles') and item.text_rectangles and len(item.text_rectangles) >= 3 else 100,
                                    'height': int(item.text_rectangles[3] - item.text_rectangles[1]) if hasattr(item, 'text_rectangles') and item.text_rectangles and len(item.text_rectangles) >= 4 else 30
                                }
                            }
                        
                        texts.append(text_info)
                        # print(f"  检测到文字: '{text_info['text']}' (置信度: {text_info['confidence']:.2f})")
            
            # print(f"总共检测到 {len(texts)} 个文字区域")
            return texts
            
        except Exception as error:
            # print(f"阿里云OCR识别失败: {error}")
            return []
    
    def recognize_qrcode_with_tencent_api(self, image_path: str) -> List[Dict]:
        """
        使用  API的QrcodeOCR接口识别二维码
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            二维码识别结果列表
        """
        try:
            # print(f"正在使用  API进行二维码识别: {os.path.basename(image_path)}")
            
            # 将图像转换为base64
            image_base64 = self.image_to_base64(image_path)
            if not image_base64:
                return []
            
            # print("正在调用腾讯云QrcodeOCR API...")
            
            # 创建请求对象
            req = tencent_models.QrcodeOCRRequest()
            
            # 设置请求参数
            params = {
                "ImageBase64": image_base64
            }
            req.from_json_string(json.dumps(params))
            
            # 调用API
            resp = self.tencent_client.QrcodeOCR(req)
            
            # print(f" 二维码识别完成")
            
            # 解析二维码结果
            qrcodes = []
            if resp.CodeResults:
                # print(f"检测到 {len(resp.CodeResults)} 个二维码/条形码")
                
                for code_result in resp.CodeResults:
                    # 获取二维码内容
                    qr_text = getattr(code_result, 'Url', '') or getattr(code_result, 'Text', '')
                    type_name = getattr(code_result, 'TypeName', 'UNKNOWN')
                    
                    if qr_text and qr_text.strip():
                        # print(f" 检测到二维码内容: {qr_text}")
                        
                        # 检查二维码内容是否包含目标标签
                        found_label = False
                        for label in VALID_LABELS:
                            if label in qr_text:
                                found_label = True
                                # 获取位置信息
                                left, top, width, height = 0, 0, 100, 100
                                if hasattr(code_result, 'Position') and code_result.Position:
                                    position = code_result.Position
                                    if hasattr(position, 'LeftTop') and hasattr(position, 'RightBottom'):
                                        left_top = position.LeftTop
                                        right_bottom = position.RightBottom
                                        if hasattr(left_top, 'X') and hasattr(left_top, 'Y'):
                                            left = int(left_top.X)
                                            top = int(left_top.Y)
                                        if hasattr(right_bottom, 'X') and hasattr(right_bottom, 'Y'):
                                            width = int(right_bottom.X - left)
                                            height = int(right_bottom.Y - top)
                                
                                qr_info = {
                                    'text': qr_text,
                                    'label': label,
                                    'confidence': 1.0,  #  识别置信度
                                    'type': 'qrcode',
                                    'method': f'  API ({type_name})',
                                    'location': {
                                        'left': left,
                                        'top': top,
                                        'width': width,
                                        'height': height
                                    }
                                }
                                qrcodes.append(qr_info)
                                # print(f"检测到二维码标签: '{label}' 内容: '{qr_text}' 类型: {type_name}")
                                break
                        
                        if not found_label:
                            print(f"二维码内容不包含目标标签 (A-1, A-2, B-1, B-2): '{qr_text}'")
                    else:
                        print("二维码数据为空或无效")
            else:
                print(" 未检测到二维码")
            
            print(f" 总共检测到 {len(qrcodes)} 个有效二维码标签")
            return qrcodes
            
        except TencentCloudSDKException as error:
            # print(f" 二维码识别失败: {error}")
            return []
        except Exception as error:
            # print(f" 二维码识别失败: {error}")
            return []
    
    def recognize_qrcode(self, image_path: str) -> List[Dict]:
        """
        使用  API进行二维码识别
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            二维码识别结果列表
        """
        try:
            # print(f"正在进行 二维码识别: {os.path.basename(image_path)}")
            
            # print("使用 (QrcodeOCR) 进行识别...")
            qrcodes = self.recognize_qrcode_with_tencent_api(image_path)
            
            if qrcodes:
                # print(" 二维码识别成功！")
                return qrcodes
            else:
                # print(" 二维码识别未找到有效标签")
                return []
            
        except Exception as error:
            # print(f"二维码识别失败: {error}")
            return []
    
    def set_qrcode_config(self, enable_qrcode: bool = True, qrcode_priority: bool = True):
        """
        设置二维码识别配置
        
        Args:
            enable_qrcode: 是否启用二维码识别
            qrcode_priority: 是否优先进行二维码识别
        """
        self.enable_qrcode = enable_qrcode
        self.qrcode_priority = qrcode_priority
        # print(f"二维码识别配置已更新: 启用={enable_qrcode}, 优先={qrcode_priority}")
    
    def normalize_text(self, text: str) -> str:
        """
        标准化文本，用于提高匹配准确率
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        if not text:
            return ''
        
        # 移除空格和特殊字符
        text = re.sub(r'[\s\-_]', '', text)
        
        # 转换为大写
        text = text.upper()
        
        # 处理常见的OCR错误
        text = text.replace('O', '0')  # 字母O替换为数字0
        text = text.replace('I', '1')  # 字母I替换为数字1
        text = text.replace('L', '1')  # 字母L替换为数字1
        
        # 处理中文字符的OCR错误
        text = text.replace('口', '0')  # 中文"口"替换为数字0
        text = text.replace('一', '1')  # 中文"一"替换为数字1
        
        return text
    
    def combine_consecutive_chars(self, ocr_results: List[Dict]) -> List[str]:
        """
        组合连续的字符，用于处理被分割的标签
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            组合后的字符串列表
        """
        if not ocr_results:
            return []
        
        # 按位置排序（从左到右，从上到下）
        sorted_results = sorted(ocr_results, key=lambda x: (x['location']['top'], x['location']['left']))
        
        # 提取所有文字
        all_texts = [item['text'] for item in sorted_results]
        
        # 组合连续字符
        combinations = []
        
        # 单个字符
        for text in all_texts:
            combinations.append(text)
        
        # 两个连续字符的组合
        for i in range(len(all_texts) - 1):
            combinations.append(all_texts[i] + all_texts[i + 1])
        
        # 三个连续字符的组合
        for i in range(len(all_texts) - 2):
            combinations.append(all_texts[i] + all_texts[i + 1] + all_texts[i + 2])
        
        # 四个连续字符的组合
        for i in range(len(all_texts) - 3):
            combinations.append(all_texts[i] + all_texts[i + 1] + all_texts[i + 2] + all_texts[i + 3])
        
        # 完整文本序列
        full_text = ''.join(all_texts)
        combinations.append(full_text)
        
        return combinations
    
    def match_target_labels(self, ocr_results: List[Dict]) -> List[Dict]:
        """
        匹配目标标签
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            匹配到的标签列表
        """
        matched_labels = []
        
        # print("\n开始匹配目标标签...")
        
        # 直接匹配单个文字区域
        for result in ocr_results:
            original_text = result['text']
            normalized_text = self.normalize_text(original_text)
            
            # print(f"  原文: '{original_text}' -> 标准化: '{normalized_text}'")
            
            for label in VALID_LABELS:
                normalized_label = self.normalize_text(label)
                if normalized_label == normalized_text:
                    matched_labels.append({
                        'label': label,
                        'original_text': original_text,
                        'confidence': result['confidence'],
                        'location': result['location'],
                        'type': 'text'
                    })
                    # print(f"    匹配成功: {original_text} -> {label}")
                    break
        
        # 如果没有直接匹配，尝试连续字符组合匹配
        if not matched_labels:
            # print("\n尝试连续字符组合匹配...")
            combinations = self.combine_consecutive_chars(ocr_results)
            
            for combination in combinations:
                normalized_combination = self.normalize_text(combination)
                # print(f"完整文本序列: '{combination}' -> 标准化: '{normalized_combination}'")
                
                for label in VALID_LABELS:
                    normalized_label = self.normalize_text(label)
                    if normalized_label in normalized_combination:
                        # 计算平均位置和置信度
                        avg_confidence = sum(r['confidence'] for r in ocr_results) / len(ocr_results)
                        avg_location = {
                            'left': int(sum(r['location']['left'] for r in ocr_results) / len(ocr_results)),
                            'top': int(sum(r['location']['top'] for r in ocr_results) / len(ocr_results)),
                            'width': int(sum(r['location']['width'] for r in ocr_results) / len(ocr_results)),
                            'height': int(sum(r['location']['height'] for r in ocr_results) / len(ocr_results))
                        }
                        
                        matched_labels.append({
                            'label': label,
                            'original_text': combination,
                            'confidence': avg_confidence,
                            'location': avg_location,
                            'type': 'text'
                        })
                        # print(f"    组合匹配成功: {combination} -> {label}")
                        break
                
                if matched_labels:
                    break
        
        return matched_labels
    
    def recognize_image(self, image_path: str, debug: bool = False) -> Dict:
        """
        识别单张图像中的目标标签
        
        Args:
            image_path: 图像文件路径
            debug: 是否启用调试模式
            
        Returns:
            识别结果字典
        """
        # print(f"\n{'='*60}")
        # print(f"开始识别图像: {image_path}")
        # print(f"{'='*60}")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            # 尝试添加扩展名
            found = False
            for ext in IMAGE_EXTS:
                test_path = image_path + ext
                if os.path.exists(test_path):
                    image_path = test_path
                    found = True
                    break
            
            if not found:
                return {'error': f'图像文件不存在: {image_path}'}
        
        try:
            matched_labels = []
            recognition_method = "文字识别"
            
            # 优先进行二维码识别
            if self.enable_qrcode and self.qrcode_priority:
                # print("\n优先进行二维码识别...")
                qr_results = self.recognize_qrcode(image_path)
                
                if qr_results:
                    # 将二维码结果转换为匹配标签格式
                    for qr_info in qr_results:
                        matched_labels.append({
                            'label': qr_info['label'],
                            'original_text': qr_info['text'],
                            'confidence': qr_info['confidence'],
                            'location': qr_info['location'],
                            'type': 'qrcode'
                        })
                    recognition_method = "二维码识别"
                    print(f"二维码识别成功，找到 {len(matched_labels)} 个标签，跳过文字识别")
                else:
                    print("二维码识别未找到有效标签，继续进行文字识别...")
            
            # 如果二维码识别未找到结果，或者未启用二维码识别，则进行文字识别
            if not matched_labels:
                # print("\n开始文字识别...")
                # 执行OCR识别
                ocr_results = self.recognize_text(image_path)
                
                if not ocr_results:
                    return {'error': '未检测到任何文字'}
                
                # 匹配目标标签
                matched_labels = self.match_target_labels(ocr_results)
                recognition_method = "文字识别"
            
            # print(f"\n匹配结果: 找到 {len(matched_labels)} 个目标标签")
            
            # 构建结果
            result = {
                'image_path': image_path,
                'ocr_service': '混合OCR (腾讯云二维码 + 阿里云文字)',
                'recognition_method': recognition_method,
                'total_text_regions': len(ocr_results) if 'ocr_results' in locals() else 0,
                'matched_labels': matched_labels,
                'detected_labels': [match['label'] for match in matched_labels],
                'success': len(matched_labels) > 0
            }
            
            # 打印最终结果
            # print(f"\n{'='*60}")
            # print(f"识别方法: {recognition_method}")
            if matched_labels:
                # print("识别成功！检测到以下标签:")
                for match in matched_labels:
                    method_type = "(二维码)" if match.get('type') == 'qrcode' else "(文字)"
                    # print(f"{match['label']} {method_type}: '{match['original_text']}' (置信度: {match['confidence']:.2f})")
            else:
                print("未识别到目标标签 (A-1, A-2, B-1, B-2)，默认返回B-1")
                # 识别不到时默认返回B-1
                matched_labels.append({
                    'label': 'B-1',
                    'original_text': 'B-1',
                    'confidence': 0.5,  # 默认置信度
                    'location': None,
                    'type': 'default'
                })
                result['matched_labels'] = matched_labels
                result['detected_labels'] = ['B-1']
                result['success'] = True
            # print(f"{'='*60}")
            
            return result
            
        except Exception as e:
            error_msg = f"识别过程出错: {e}"
            # print(f"{error_msg}")
            return {'error': error_msg}


def find_image_file(image_path: str) -> str:
    """
    查找图像文件，支持自动添加扩展名
    
    Args:
        image_path: 图像路径（可能不包含扩展名）
        
    Returns:
        完整的图像文件路径，如果找不到则返回None
    """
    # 如果文件已存在，直接返回
    if os.path.exists(image_path):
        return image_path
    
    # 尝试添加不同的扩展名
    for ext in IMAGE_EXTS:
        test_path = image_path + ext
        if os.path.exists(test_path):
            return test_path
    
    return None


# 便捷函数
def recognize_image(image_path: str, enable_qrcode: bool = True, qrcode_priority: bool = True) -> Dict:
    """
    便捷的图像识别函数
    
    Args:
        image_path: 图像文件路径
        enable_qrcode: 是否启用二维码识别
        qrcode_priority: 是否优先进行二维码识别
        
    Returns:
        识别结果字典
    """
    ocr = OCRRecognizer(enable_qrcode=enable_qrcode, qrcode_priority=qrcode_priority)
    return ocr.recognize_image(image_path)


def recognize_text_only(image_path: str) -> Dict:
    """
    仅使用文字识别的便捷函数
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        识别结果字典
    """
    ocr = OCRRecognizer(enable_qrcode=False, qrcode_priority=False)
    return ocr.recognize_image(image_path)


def recognize_qrcode_only(image_path: str) -> Dict:
    """
    仅使用二维码识别的便捷函数
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        识别结果字典
    """
    ocr = OCRRecognizer(enable_qrcode=True, qrcode_priority=True)
    # 临时禁用文字识别回退
    original_method = ocr.recognize_image
    
    def qrcode_only_recognize(image_path: str, debug: bool = False) -> Dict:
        # print(f"\n{'='*60}")
        # print(f"开始二维码识别: {image_path}")
        # print(f"{'='*60}")
        
        if not os.path.exists(image_path):
            for ext in IMAGE_EXTS:
                test_path = image_path + ext
                if os.path.exists(test_path):
                    image_path = test_path
                    break
            else:
                return {'error': f'图像文件不存在: {image_path}'}
        
        try:
            qr_results = ocr.recognize_qrcode(image_path)
            matched_labels = []
            
            if qr_results:
                for qr_info in qr_results:
                    matched_labels.append({
                        'label': qr_info['label'],
                        'original_text': qr_info['text'],
                        'confidence': qr_info['confidence'],
                        'location': qr_info['location'],
                        'type': 'qrcode'
                    })
            
            result = {
                'image_path': image_path,
                'ocr_service': '腾讯云二维码识别',
                'recognition_method': '二维码识别',
                'total_text_regions': 0,
                'matched_labels': matched_labels,
                'detected_labels': [match['label'] for match in matched_labels],
                'success': len(matched_labels) > 0
            }
            
            # print(f"\n{'='*60}")
            if matched_labels:
                # print("二维码识别成功！检测到以下标签:")
                for match in matched_labels:
                    print(f"{match['label']} (二维码): '{match['original_text']}' (置信度: {match['confidence']:.2f})")
            else:
                print("未识别到目标标签 (A-1, A-2, B-1, B-2)，默认返回B-1")
                # 识别不到时默认返回B-1
                matched_labels.append({
                    'label': 'B-1',
                    'original_text': 'B-1',
                    'confidence': 0.5,  # 默认置信度
                    'location': None,
                    'type': 'default'
                })
                result['matched_labels'] = matched_labels
                result['detected_labels'] = ['B-1']
                result['success'] = True
            # print(f"{'='*60}")
            
            return result
            
        except Exception as e:
            error_msg = f"二维码识别过程出错: {e}"
            # print(f"{error_msg}")
            return {'error': error_msg}
    
    return qrcode_only_recognize(image_path)