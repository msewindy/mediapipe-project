#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程处理器：分离检测和显示线程
包含多线程模式的实现
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import queue
from utils import draw_landmarks_manually


def process_video_multithread(video_path, optimize_performance=False):
    """
    多线程版本：分离检测和显示线程
    """
    print(f"开始多线程模式处理视频: {video_path}")
    print(f"性能优化模式: {'开启' if optimize_performance else '关闭'}")
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {total_frames}")
    
    # 创建窗口并设置大小
    window_name = 'MediaPipe Holistic - MultiThread'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    # 线程间通信队列
    frame_queue = queue.Queue(maxsize=10)  # 原始帧队列
    result_queue = queue.Queue(maxsize=10)  # 检测结果队列
    display_queue = queue.Queue(maxsize=5)  # 显示帧队列
    
    # 控制标志
    stop_event = threading.Event()
    
    # 统计变量
    stats = {
        'frame_count': 0,
        'detection_count': 0,
        'display_count': 0,
        'start_time': time.time(),
        'fps_start_time': time.time(),
        'fps_frame_count': 0,
        'detection_fps': 0,
        'display_fps': 0
    }
    
    # 线程锁
    stats_lock = threading.Lock()
    
    def detection_worker():
        """检测线程：处理MediaPipe检测"""
        print("检测线程启动")
        
        mp_holistic = mp.solutions.holistic
        model_complexity = 0 if optimize_performance else 1
        
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            
            while not stop_event.is_set():
                try:
                    # 从队列获取帧，设置超时避免阻塞
                    frame_data = frame_queue.get(timeout=0.1)
                    frame, frame_id = frame_data
                    
                    # 转换为RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # MediaPipe检测
                    results = holistic.process(rgb_frame)
                    
                    # 将MediaPipe结果转换为可序列化的格式
                    serializable_results = {
                        'pose_landmarks': None,
                        'face_landmarks': None,
                        'left_hand_landmarks': None,
                        'right_hand_landmarks': None
                    }
                    
                    # 转换姿态关键点
                    if results.pose_landmarks:
                        pose_landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            pose_landmarks.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z,
                                'visibility': landmark.visibility
                            })
                        serializable_results['pose_landmarks'] = pose_landmarks
                    
                    # 转换面部关键点
                    if results.face_landmarks:
                        face_landmarks = []
                        for landmark in results.face_landmarks.landmark:
                            face_landmarks.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        serializable_results['face_landmarks'] = face_landmarks
                    
                    # 转换左手关键点
                    if results.left_hand_landmarks:
                        left_hand_landmarks = []
                        for landmark in results.left_hand_landmarks.landmark:
                            left_hand_landmarks.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        serializable_results['left_hand_landmarks'] = left_hand_landmarks
                    
                    # 转换右手关键点
                    if results.right_hand_landmarks:
                        right_hand_landmarks = []
                        for landmark in results.right_hand_landmarks.landmark:
                            right_hand_landmarks.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        serializable_results['right_hand_landmarks'] = right_hand_landmarks
                    
                    # 将结果放入结果队列
                    result_queue.put((frame, serializable_results, frame_id), timeout=0.1)
                    
                    # 更新统计
                    with stats_lock:
                        stats['detection_count'] += 1
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"检测线程错误: {e}")
                    break
        
        print("检测线程结束")
    
    def display_worker():
        """显示线程：处理绘制和显示"""
        print("显示线程启动")
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        
        # 初始化绘图工具
        drawing_utils = mp_drawing
        drawing_styles = mp_drawing_styles
        holistic = mp_holistic
        
        while not stop_event.is_set():
            try:
                # 从结果队列获取数据
                frame, results, frame_id = result_queue.get(timeout=0.1)
                
                # 创建显示帧
                display_frame = frame.copy()
                display_frame.flags.writeable = True
                
                # 绘制检测结果
                if results['pose_landmarks']:
                    drawing_utils.draw_landmarks(
                        display_frame,
                        results['pose_landmarks'],
                        holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style())
                
                if results['face_landmarks']:
                    drawing_utils.draw_landmarks(
                        display_frame,
                        results['face_landmarks'],
                        holistic.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
                    drawing_utils.draw_landmarks(
                        display_frame,
                        results['face_landmarks'],
                        holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
                
                if results['left_hand_landmarks']:
                    drawing_utils.draw_landmarks(
                        display_frame,
                        results['left_hand_landmarks'],
                        holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=drawing_styles.get_default_hand_connections_style())
                
                if results['right_hand_landmarks']:
                    drawing_utils.draw_landmarks(
                        display_frame,
                        results['right_hand_landmarks'],
                        holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=drawing_styles.get_default_hand_connections_style())
                
                # 添加信息显示
                cv2.putText(display_frame, f'Frame: {frame_id}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                status_text = []
                if results['pose_landmarks']:
                    status_text.append("Pose")
                if results['face_landmarks']:
                    status_text.append("Face")
                if results['left_hand_landmarks']:
                    status_text.append("L-Hand")
                if results['right_hand_landmarks']:
                    status_text.append("R-Hand")
                
                cv2.putText(display_frame, f'Detected: {", ".join(status_text) if status_text else "None"}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                with stats_lock:
                    cv2.putText(display_frame, f'Detection FPS: {stats["detection_fps"]:.1f}', (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f'Display FPS: {stats["display_fps"]:.1f}', (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f'MultiThread Mode', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 将显示帧放入显示队列
                display_queue.put(display_frame, timeout=0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"显示线程错误: {e}")
                break
        
        print("显示线程结束")
    
    # 启动工作线程
    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    display_thread = threading.Thread(target=display_worker, daemon=True)
    
    detection_thread.start()
    display_thread.start()
    
    print("主线程：开始读取视频帧")
    
    # 主线程：读取视频帧
    frame_count = 0
    last_display_time = time.time()
    
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束")
            break
        
        frame_count += 1
        
        # 将帧放入检测队列
        try:
            frame_queue.put((frame, frame_count), timeout=0.1)
        except queue.Full:
            # 队列满了，跳过这一帧
            continue
        
        # 显示处理
        try:
            display_frame = display_queue.get_nowait()
            
            # 控制显示帧率
            current_time = time.time()
            if current_time - last_display_time >= 1.0 / fps:
                cv2.imshow(window_name, display_frame)
                last_display_time = current_time
                
                with stats_lock:
                    stats['display_count'] += 1
                    
        except queue.Empty:
            # 没有可显示的帧，显示原始帧
            cv2.imshow(window_name, frame)
        
        # 更新统计
        with stats_lock:
            stats['frame_count'] += 1
            stats['fps_frame_count'] += 1
            
            # 每秒更新FPS统计
            current_time = time.time()
            if current_time - stats['fps_start_time'] >= 1.0:
                stats['detection_fps'] = stats['detection_count'] / (current_time - stats['fps_start_time'])
                stats['display_fps'] = stats['display_count'] / (current_time - stats['fps_start_time'])
                stats['fps_frame_count'] = 0
                stats['fps_start_time'] = current_time
                stats['detection_count'] = 0
                stats['display_count'] = 0
                
                print(f"\n=== 多线程模式统计 (帧 {frame_count}) ===")
                print(f"检测FPS: {stats['detection_fps']:.2f}")
                print(f"显示FPS: {stats['display_fps']:.2f}")
                print(f"目标FPS: {fps:.2f}")
                print("=" * 40)
        
        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            stop_event.set()
            break
        elif key == 32:  # 空格键
            cv2.waitKey(0)  # 暂停直到按任意键
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 等待线程结束
    stop_event.set()
    detection_thread.join(timeout=2)
    display_thread.join(timeout=2)
    
    print("多线程模式结束") 