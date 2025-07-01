#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化多进程处理器：主进程负责读取和检测，子进程负责显示
最大化检测性能，最小化检测进程的额外开销
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import queue
import multiprocessing as mp_proc
from multiprocessing import Process, Queue, Event
from collections import deque


def draw_landmarks_manually(image, landmarks, connections, color=(0, 255, 0), thickness=2):
    """
    手动绘制关键点和连接线
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


def display_process_worker(display_queue, stop_event):
    """
    显示进程工作函数 - 专门负责绘制和显示
    """
    print(f"显示进程启动 (PID: {os.getpid()})")
    
    # 统计变量
    display_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    display_fps = 0
    
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
    
    # 创建窗口
    window_name = 'MediaPipe Holistic - Optimized MultiProcess'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("显示进程：开始处理显示队列...")
    
    while not stop_event.is_set():
        try:
            # 从队列获取显示数据
            frame, serializable_results, frame_id, detection_fps = display_queue.get(timeout=0.1)
            display_count += 1
            fps_frame_count += 1
            
            # 创建显示帧
            display_frame = frame.copy()
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
            
            cv2.putText(display_frame, f'Optimized MultiProcess', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示帧
            cv2.imshow(window_name, display_frame)
            
            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                print("显示进程：收到ESC键，停止显示")
                stop_event.set()
                break
            elif key == 32:  # 空格键
                cv2.waitKey(0)  # 暂停直到按任意键
            
            # 每秒更新FPS统计
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                display_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
                
                print(f"显示进程 - 显示帧数: {display_count}, FPS: {display_fps:.2f}")
                
        except queue.Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            print(f"显示进程错误: {e}")
            continue
    
    cv2.destroyAllWindows()
    print(f"显示进程结束 (PID: {os.getpid()})")


def process_video_optimized_multiprocess(video_path, optimize_performance=False):
    """
    优化多进程版本：主进程负责读取和检测，子进程负责显示
    最大化检测性能，最小化检测进程的额外开销
    """
    print(f"开始优化多进程模式处理视频: {video_path}")
    print(f"性能优化模式: {'开启' if optimize_performance else '关闭'}")
    print(f"CPU核心数: {mp_proc.cpu_count()}")
    print("架构: 主进程(读取+检测) + 子进程(显示)")
    
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
    window_name = 'MediaPipe Holistic - Optimized MultiProcess'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    # 进程间通信队列（只用于显示）
    display_queue = Queue(maxsize=30)  # 适中的队列大小
    
    # 控制标志
    stop_event = Event()
    
    # 统计变量
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    detection_fps = 0
    
    # 性能统计变量
    total_read_time = 0
    total_convert_time = 0
    total_mediapipe_time = 0
    total_serialize_time = 0
    total_queue_put_time = 0
    total_loop_time = 0
    last_frame_time = time.time()
    
    # 启动显示进程
    display_process = Process(
        target=display_process_worker,
        args=(display_queue, stop_event),
        daemon=True
    )
    
    display_process.start()
    print(f"显示进程已启动 (PID: {display_process.pid})")
    
    # 主进程：初始化MediaPipe
    mp_holistic = mp.solutions.holistic
    model_complexity = 0 if optimize_performance else 1
    print(f"主进程 - 模型复杂度: {model_complexity} (0=轻量级, 1=标准, 2=重型)")
    
    init_start = time.time()
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        init_time = time.time() - init_start
        print(f"主进程 - MediaPipe初始化耗时: {init_time*1000:.2f}ms")
        
        print("主进程：开始以25 FPS读取视频帧并进行检测")
        
        # 主进程：以25 FPS读取视频帧和检测
        frame_interval = 1.0 / target_fps  # 每帧间隔时间
        
        while cap.isOpened() and not stop_event.is_set():
            # 控制读取帧率到25 FPS
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # 短暂休眠
                continue
            
            last_frame_time = current_time
            loop_start = time.time()
            
            # 读取视频帧
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            
            if not ret:
                print("视频播放结束")
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # 颜色转换
            convert_start = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convert_time = time.time() - convert_start
            
            # MediaPipe检测（核心操作）
            mediapipe_start = time.time()
            results = holistic.process(frame_rgb)
            mediapipe_time = time.time() - mediapipe_start
            
            detection_count += 1
            
            # 序列化结果
            serialize_start = time.time()
            serializable_results = {
                'pose_landmarks': [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark] if results.pose_landmarks else None,
                'face_landmarks': [(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark] if results.face_landmarks else None,
                'left_hand_landmarks': [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else None,
                'right_hand_landmarks': [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else None
            }
            serialize_time = time.time() - serialize_start
            
            # 异步发送到显示队列（不阻塞检测）
            queue_put_start = time.time()
            try:
                display_queue.put_nowait((frame, serializable_results, frame_count, detection_fps))
                
                # 每10帧打印一次写入队列的信息
                if frame_count % 10 == 0:
                    print(f"主线程 - 成功写入帧 {frame_count} 到队列")
                    print(f"  队列大小: {display_queue.qsize()}/{display_queue._maxsize}")
                    print(f"  检测结果: Pose={serializable_results['pose_landmarks'] is not None}, "
                          f"Face={serializable_results['face_landmarks'] is not None}, "
                          f"L-Hand={serializable_results['left_hand_landmarks'] is not None}, "
                          f"R-Hand={serializable_results['right_hand_landmarks'] is not None}")
                    print(f"  帧形状: {frame.shape}, 检测FPS: {detection_fps:.1f}")
                
            except queue.Full:
                # 显示队列满了，丢弃这个结果（不影响检测性能）
                print(f"主线程 - 队列已满，丢弃帧 {frame_count}")
                pass
            queue_put_time = time.time() - queue_put_start
            
            loop_time = time.time() - loop_start
            
            # 累计性能统计
            total_read_time += read_time
            total_convert_time += convert_time
            total_mediapipe_time += mediapipe_time
            total_serialize_time += serialize_time
            total_queue_put_time += queue_put_time
            total_loop_time += loop_time
            
            # 每秒更新FPS统计和详细耗时分析
            if current_time - fps_start_time >= 1.0:
                detection_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
                
                avg_read = total_read_time / frame_count if frame_count > 0 else 0
                avg_convert = total_convert_time / frame_count if frame_count > 0 else 0
                avg_mediapipe = total_mediapipe_time / frame_count if frame_count > 0 else 0
                avg_serialize = total_serialize_time / frame_count if frame_count > 0 else 0
                avg_queue_put = total_queue_put_time / frame_count if frame_count > 0 else 0
                avg_loop = total_loop_time / frame_count if frame_count > 0 else 0
                
                print(f"\n=== 优化多进程模式统计 (帧 {frame_count}) ===")
                print(f"检测FPS: {detection_fps:.2f}")
                print(f"目标FPS: {target_fps:.2f}")
                print(f"显示进程状态: {'运行中' if display_process.is_alive() else '已停止'}")
                print(f"显示队列大小: {display_queue.qsize()}")
                print(f"  详细耗时分析 (ms):")
                print(f"    帧读取: {avg_read*1000:.2f}")
                print(f"    颜色转换: {avg_convert*1000:.2f}")
                print(f"    MediaPipe处理: {avg_mediapipe*1000:.2f}")
                print(f"    序列化关键点: {avg_serialize*1000:.2f}")
                print(f"    队列写入: {avg_queue_put*1000:.2f}")
                print(f"    总循环时间: {avg_loop*1000:.2f}")
                print(f"    理论FPS: {1/avg_loop:.1f}")
                print(f"    纯MediaPipe时间: {avg_mediapipe*1000:.2f}ms (对比无显示模式: 14.63ms)")
                print(f"    额外开销: {(avg_loop-avg_mediapipe)*1000:.2f}ms")
                print(f"    MediaPipe性能下降: {((avg_mediapipe-0.01463)/0.01463*100):.1f}%")
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
    
    # 停止显示进程
    stop_event.set()
    display_process.join(timeout=3)
    
    if display_process.is_alive():
        print("强制终止显示进程")
        display_process.terminate()
        display_process.join(timeout=1)
    
    print("优化多进程模式结束") 