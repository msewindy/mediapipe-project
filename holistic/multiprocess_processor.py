#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程处理器：使用独立进程进行检测
包含多进程模式的实现
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import queue
import multiprocessing as mp_proc
from multiprocessing import Process, Queue, Event
from utils import draw_landmarks_manually


def detection_process_worker(frame_queue, result_queue, stop_event, optimize_performance=False):
    """
    检测进程工作函数 - 以25 FPS固定帧率工作
    """
    print(f"检测进程启动 (PID: {os.getpid()})")
    
    # 初始化MediaPipe
    mp_holistic = mp.solutions.holistic
    model_complexity = 0 if optimize_performance else 1
    print(f"检测进程 - 模型复杂度: {model_complexity} (0=轻量级, 1=标准, 2=重型)")
    print(f"检测进程 - 性能优化模式: {'开启' if optimize_performance else '关闭'}")
    
    init_start = time.time()
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    init_time = time.time() - init_start
    print(f"检测进程 - MediaPipe初始化耗时: {init_time*1000:.2f}ms")
    
    # 统计变量
    processed_frames = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    detection_fps = 0
    
    # 详细耗时统计变量
    total_convert_time = 0
    total_mediapipe_time = 0
    total_serialize_time = 0
    total_queue_get_time = 0
    total_queue_put_time = 0
    total_frame_copy_time = 0
    total_loop_time = 0
    total_wait_time = 0
    total_frame_size = 0
    
    # 固定帧率控制
    target_fps = 25.0
    frame_interval = 1.0 / target_fps
    last_process_time = time.time()
    
    print(f"检测进程：以 {target_fps} FPS 处理帧数据...")
    
    # 预热模型（处理几帧让模型稳定）
    print("检测进程：开始模型预热...")
    warmup_frames = 0
    while warmup_frames < 10 and not stop_event.is_set():
        try:
            frame, frame_id = frame_queue.get(timeout=0.1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            warmup_frames += 1
            print(f"检测进程：预热帧 {warmup_frames}/10")
        except queue.Empty:
            break
    print("检测进程：模型预热完成")
    
    while not stop_event.is_set():
        # 控制处理帧率到25 FPS
        current_time = time.time()
        if current_time - last_process_time < frame_interval:
            wait_start = time.time()
            time.sleep(0.001)  # 短暂休眠
            total_wait_time += time.time() - wait_start
            continue
        
        last_process_time = current_time
        loop_start = time.time()
        
        try:
            # 从队列获取帧，设置超时避免无限等待
            queue_get_start = time.time()
            frame, frame_id = frame_queue.get(timeout=0.1)
            queue_get_time = time.time() - queue_get_start
            total_queue_get_time += queue_get_time
            
            # 记录帧大小
            frame_size = frame.nbytes
            total_frame_size += frame_size
            
            processed_frames += 1
            fps_frame_count += 1
            
            # 颜色转换
            convert_start = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convert_time = time.time() - convert_start
            total_convert_time += convert_time
            
            # MediaPipe处理
            mediapipe_start = time.time()
            results = holistic.process(frame_rgb)
            mediapipe_time = time.time() - mediapipe_start
            total_mediapipe_time += mediapipe_time
            
            # 序列化结果
            serialize_start = time.time()
            serializable_results = {
                'pose_landmarks': [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark] if results.pose_landmarks else None,
                'face_landmarks': [(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark] if results.face_landmarks else None,
                'left_hand_landmarks': [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else None,
                'right_hand_landmarks': [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else None
            }
            serialize_time = time.time() - serialize_start
            total_serialize_time += serialize_time
            
            # 帧拷贝（模拟队列传输的数据拷贝）
            frame_copy_start = time.time()
            frame_copy = frame.copy()  # 模拟队列传输时的数据拷贝
            frame_copy_time = time.time() - frame_copy_start
            total_frame_copy_time += frame_copy_time
            
            # 将结果放入队列
            queue_put_start = time.time()
            try:
                result_queue.put_nowait((frame_copy, serializable_results, frame_id))
            except queue.Full:
                # 结果队列满了，丢弃这个结果
                pass
            queue_put_time = time.time() - queue_put_start
            total_queue_put_time += queue_put_time
            
            loop_time = time.time() - loop_start
            total_loop_time += loop_time
            
            # 每秒更新FPS统计和详细耗时分析
            if current_time - fps_start_time >= 1.0:
                detection_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
                
                avg_convert = total_convert_time / processed_frames if processed_frames > 0 else 0
                avg_mediapipe = total_mediapipe_time / processed_frames if processed_frames > 0 else 0
                avg_serialize = total_serialize_time / processed_frames if processed_frames > 0 else 0
                avg_queue_get = total_queue_get_time / processed_frames if processed_frames > 0 else 0
                avg_queue_put = total_queue_put_time / processed_frames if processed_frames > 0 else 0
                avg_frame_copy = total_frame_copy_time / processed_frames if processed_frames > 0 else 0
                avg_wait = total_wait_time / processed_frames if processed_frames > 0 else 0
                avg_loop = total_loop_time / processed_frames if processed_frames > 0 else 0
                avg_frame_size = total_frame_size / processed_frames if processed_frames > 0 else 0
                
                print(f"检测进程 - 处理帧数: {processed_frames}, FPS: {detection_fps:.2f}")
                print(f"  详细耗时分析 (ms):")
                print(f"    队列获取等待: {avg_queue_get*1000:.2f}")
                print(f"    颜色转换: {avg_convert*1000:.2f}")
                print(f"    MediaPipe处理: {avg_mediapipe*1000:.2f}")
                print(f"    序列化关键点: {avg_serialize*1000:.2f}")
                print(f"    帧数据拷贝: {avg_frame_copy*1000:.2f}")
                print(f"    队列写入: {avg_queue_put*1000:.2f}")
                print(f"    固定帧率等待: {avg_wait*1000:.2f}")
                print(f"    总循环时间: {avg_loop*1000:.2f}")
                print(f"    理论FPS: {1/avg_loop:.1f}")
                print(f"    平均帧大小: {avg_frame_size/1024:.1f}KB")
                print(f"    纯MediaPipe时间: {avg_mediapipe*1000:.2f}ms (对比无显示模式: 37ms)")
                print(f"    额外开销: {(avg_loop-avg_mediapipe)*1000:.2f}ms")
                print(f"    MediaPipe性能下降: {((avg_mediapipe-0.037)/0.037*100):.1f}%")
                
        except queue.Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            print(f"检测进程错误: {e}")
            continue
    
    # 清理资源
    holistic.close()
    print(f"检测进程结束 (PID: {os.getpid()})")


def process_video_multiprocess(video_path, optimize_performance=False):
    """
    多进程版本：使用独立进程进行检测，真正利用多核CPU
    播放和检测都以25 FPS固定帧率工作
    """
    print(f"开始多进程模式处理视频: {video_path}")
    print(f"性能优化模式: {'开启' if optimize_performance else '关闭'}")
    print(f"CPU核心数: {mp_proc.cpu_count()}")
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 25.0  # 固定目标帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, 原始FPS: {original_fps}, 目标FPS: {target_fps}, 总帧数: {total_frames}")
    
    # 创建窗口并设置大小
    window_name = 'MediaPipe Holistic - MultiProcess'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    # 进程间通信队列
    frame_queue = Queue(maxsize=50)  # 适中的队列大小
    result_queue = Queue(maxsize=50)  # 适中的队列大小
    
    # 控制标志
    stop_event = Event()
    
    # 统计变量
    frame_count = 0
    display_count = 0
    detection_count = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    detection_fps = 0
    display_fps = 0
    read_fps = 0
    
    # 启动检测进程
    detection_process = Process(
        target=detection_process_worker,
        args=(frame_queue, result_queue, stop_event, optimize_performance),
        daemon=True
    )
    
    detection_process.start()
    print(f"检测进程已启动 (PID: {detection_process.pid})")
    
    # 定义关键点连接
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),
        (16, 18), (16, 20), (16, 22), (18, 20), (20, 22),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
        (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    print("主进程：开始以25 FPS读取视频帧")
    
    # 主进程：以25 FPS读取视频帧和显示
    last_frame_time = time.time()
    frame_interval = 1.0 / target_fps  # 每帧间隔时间
    
    while cap.isOpened() and not stop_event.is_set():
        # 控制读取帧率到25 FPS
        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            time.sleep(0.001)  # 短暂休眠
            continue
        
        last_frame_time = current_time
        
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束")
            break
        
        frame_count += 1
        fps_frame_count += 1
        
        # 将帧放入检测队列
        try:
            frame_queue.put_nowait((frame, frame_count))
        except queue.Full:
            # 队列满了，跳过这一帧
            continue
        
        # 显示处理 - 尝试获取检测结果
        display_frame = None
        try:
            # 从结果队列获取检测结果
            frame_result, serializable_results, frame_id = result_queue.get_nowait()
            detection_count += 1
            
            # 创建显示帧
            display_frame = frame_result.copy()
            display_frame.flags.writeable = True
            
            # 手动绘制检测结果
            if serializable_results['pose_landmarks']:
                draw_landmarks_manually(display_frame, serializable_results['pose_landmarks'], 
                                      pose_connections, color=(0, 255, 0), thickness=2)
            
            if serializable_results['left_hand_landmarks']:
                draw_landmarks_manually(display_frame, serializable_results['left_hand_landmarks'], 
                                      hand_connections, color=(255, 0, 0), thickness=2)
            
            if serializable_results['right_hand_landmarks']:
                draw_landmarks_manually(display_frame, serializable_results['right_hand_landmarks'], 
                                      hand_connections, color=(0, 0, 255), thickness=2)
            
            # 添加信息显示
            cv2.putText(display_frame, f'Frame: {frame_id}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            status_text = []
            if serializable_results['pose_landmarks']:
                status_text.append("Pose")
            if serializable_results['face_landmarks']:
                status_text.append("Face")
            if serializable_results['left_hand_landmarks']:
                status_text.append("L-Hand")
            if serializable_results['right_hand_landmarks']:
                status_text.append("R-Hand")
            
            cv2.putText(display_frame, f'Detected: {", ".join(status_text) if status_text else "None"}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f'Detection FPS: {detection_fps:.1f}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f'Display FPS: {display_fps:.1f}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f'Target FPS: {target_fps:.1f}', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f'MultiProcess Mode', (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        except queue.Empty:
            # 没有可显示的帧，使用原始帧
            display_frame = frame.copy()
            cv2.putText(display_frame, f'Frame: {frame_count} (No Detection)', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f'Waiting for detection...', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, f'Target FPS: {target_fps:.1f}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f'MultiProcess Mode', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示帧
        cv2.imshow(window_name, display_frame)
        display_count += 1
        
        # 每秒更新FPS统计
        if current_time - fps_start_time >= 1.0:
            read_fps = fps_frame_count / (current_time - fps_start_time)
            detection_fps = detection_count / (current_time - fps_start_time)
            display_fps = display_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
            display_count = 0
            detection_count = 0
            
            print(f"\n=== 多进程模式统计 (帧 {frame_count}) ===")
            print(f"读取FPS: {read_fps:.2f}")
            print(f"检测FPS: {detection_fps:.2f}")
            print(f"显示FPS: {display_fps:.2f}")
            print(f"目标FPS: {target_fps:.2f}")
            print(f"检测进程状态: {'运行中' if detection_process.is_alive() else '已停止'}")
            print(f"检测队列大小: {frame_queue.qsize()}")
            print(f"结果队列大小: {result_queue.qsize()}")
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
    
    # 停止检测进程
    stop_event.set()
    detection_process.join(timeout=3)
    
    print("多进程模式结束") 