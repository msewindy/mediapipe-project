#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数
包含绘图等通用功能
"""

import cv2


def draw_landmarks_manually(image, landmarks, connections, color=(0, 255, 0), thickness=2):
    """
    手动绘制关键点和连接线
    
    Args:
        image: 要绘制的图像
        landmarks: 关键点列表，可以是元组格式 (x, y, z) 或字典格式 {'x': x, 'y': y, 'z': z}
        connections: 连接线列表，每个元素是 (start_idx, end_idx)
        color: 绘制颜色，默认绿色
        thickness: 线条粗细，默认2
    """
    if not landmarks:
        return
    
    h, w = image.shape[:2]
    
    # 绘制关键点
    for landmark in landmarks:
        if isinstance(landmark, dict):
            # 字典格式: {'x': x, 'y': y, 'z': z}
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
        else:
            # 元组格式: (x, y, z)
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
        
        cv2.circle(image, (x, y), 3, color, -1)
    
    # 绘制连接线
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            if isinstance(landmarks[start_idx], dict):
                start_x = int(landmarks[start_idx]['x'] * w)
                start_y = int(landmarks[start_idx]['y'] * h)
            else:
                start_x = int(landmarks[start_idx][0] * w)
                start_y = int(landmarks[start_idx][1] * h)
            
            if isinstance(landmarks[end_idx], dict):
                end_x = int(landmarks[end_idx]['x'] * w)
                end_y = int(landmarks[end_idx]['y'] * h)
            else:
                end_x = int(landmarks[end_idx][0] * w)
                end_y = int(landmarks[end_idx][1] * h)
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)


def create_test_video():
    """
    创建一个简单的测试视频（如果不存在的话）
    """
    import os
    
    if os.path.exists("video.mp4"):
        print("测试视频已存在")
        return
    
    print("创建测试视频...")
    # 这里可以添加创建测试视频的代码
    # 暂时只是占位符 