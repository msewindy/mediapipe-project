#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe Holistic 视频处理演示
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

def create_test_video():
    """
    创建一个简单的测试视频（如果不存在的话）
    """
    if os.path.exists("video.mp4"):
        print("测试视频已存在")
        return
    
    print("创建测试视频...")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 20.0, (640, 480))
    
    # 创建一些测试帧
    for i in range(100):  # 5秒视频 (20fps * 5s)
        # 创建彩色背景
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加一些移动的彩色矩形模拟人体
        x = int(320 + 100 * np.sin(i * 0.1))
        y = int(240 + 50 * np.cos(i * 0.15))
        
        # 绘制"头部"
        cv2.circle(frame, (x, y - 80), 30, (255, 200, 200), -1)
        
        # 绘制"身体"
        cv2.rectangle(frame, (x - 40, y - 50), (x + 40, y + 100), (200, 255, 200), -1)
        
        # 绘制"手臂"
        cv2.rectangle(frame, (x - 60, y - 30), (x - 40, y + 30), (200, 200, 255), -1)
        cv2.rectangle(frame, (x + 40, y - 30), (x + 60, y + 30), (200, 200, 255), -1)
        
        # 绘制"腿部"
        cv2.rectangle(frame, (x - 30, y + 100), (x - 10, y + 180), (255, 255, 200), -1)
        cv2.rectangle(frame, (x + 10, y + 100), (x + 30, y + 180), (255, 255, 200), -1)
        
        # 添加帧号
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("测试视频创建完成: test_video.mp4")

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

def create_mediapipe_landmarks(landmarks_data):
    """
    将序列化的关键点数据转换回MediaPipe格式
    """
    if not landmarks_data:
        return None
    
    # 创建一个简单的关键点对象
    class SimpleLandmark:
        def __init__(self, x, y, z, visibility=None):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
    
    class SimpleLandmarkList:
        def __init__(self, landmarks):
            self.landmark = landmarks
    
    landmarks = []
    for lm_data in landmarks_data:
        if 'visibility' in lm_data:
            landmark = SimpleLandmark(lm_data['x'], lm_data['y'], lm_data['z'], lm_data['visibility'])
        else:
            landmark = SimpleLandmark(lm_data['x'], lm_data['y'], lm_data['z'])
        landmarks.append(landmark)
    
    return SimpleLandmarkList(landmarks)

def process_video(video_path, optimize_performance=False, no_delay=False):
    """
    处理视频并显示MediaPipe Holistic检测结果
    
    Args:
        video_path: 视频文件路径
        optimize_performance: 是否启用性能优化
        no_delay: 是否启用无延迟模式
    """
    print(f"开始处理视频: {video_path}")
    print(f"性能优化模式: {'开启' if optimize_performance else '关闭'}")
    print(f"无延迟模式: {'开启' if no_delay else '关闭'}")
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {total_frames}")
    
    # 创建窗口并设置大小
    window_name = 'MediaPipe Holistic Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    frame_count = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    
    # 性能统计变量
    total_read_time = 0
    total_convert_time = 0
    total_process_time = 0
    total_draw_time = 0
    total_display_time = 0
    total_loop_time = 0  # 添加循环总时间统计
    last_frame_time = time.time()  # 记录上一帧时间
    
    # 初始化MediaPipe绘图工具
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    
    print("操作说明:")
    print("- 按ESC键退出")
    print("- 按空格键暂停/继续")
    print("- 等待几秒钟让模型加载...")
    
    # 根据优化模式选择模型复杂度
    model_complexity = 0 if optimize_performance else 1  # 0=轻量级, 1=标准, 2=重型
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            # 记录循环开始时间
            loop_start = time.time()
            
            # 记录帧读取开始时间
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            
            if not ret:
                print("视频播放结束")
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # 计算帧间隔时间
            current_frame_time = time.time()
            frame_interval = current_frame_time - last_frame_time
            last_frame_time = current_frame_time
            
            # 性能优化：跳过部分帧处理
            if optimize_performance and frame_count % 2 == 0:
                # 跳过偶数帧的处理，只显示
                display_frame = frame.copy()
                cv2.putText(display_frame, f'Frame: {frame_count} (Skipped)', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f'Performance Mode: ON', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 计算实际FPS
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    actual_fps = fps_frame_count / (current_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = current_time
                else:
                    actual_fps = fps_frame_count / (current_time - fps_start_time) if (current_time - fps_start_time) > 0 else 0
                
                cv2.putText(display_frame, f'Actual FPS: {actual_fps:.2f}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                
                # 控制播放速度
                wait_time = max(1, int(1000 / fps))
                key = cv2.waitKey(wait_time) & 0xFF
                if key == 27:
                    break
                elif key == 32:
                    cv2.waitKey(0)
                continue
            
            # 计算实际FPS（每秒更新一次）
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                actual_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
                
                # 打印性能统计
                if frame_count > 1:
                    avg_read = total_read_time / (frame_count - 1)
                    avg_convert = total_convert_time / (frame_count - 1)
                    avg_process = total_process_time / (frame_count - 1)
                    avg_draw = total_draw_time / (frame_count - 1)
                    avg_display = total_display_time / (frame_count - 1)
                    avg_loop = total_loop_time / (frame_count - 1)
                    total_avg = avg_read + avg_convert + avg_process + avg_draw + avg_display
                    
                    print(f"\n=== 性能统计 (帧 {frame_count}) ===")
                    print(f"实际FPS: {actual_fps:.2f} (目标: {fps:.2f})")
                    print(f"帧间隔时间: {frame_interval*1000:.2f}ms")
                    print(f"循环总时间: {avg_loop*1000:.2f}ms")
                    print(f"帧读取: {avg_read*1000:.2f}ms")
                    print(f"颜色转换: {avg_convert*1000:.2f}ms")
                    print(f"MediaPipe处理: {avg_process*1000:.2f}ms")
                    print(f"绘制标记: {avg_draw*1000:.2f}ms")
                    print(f"显示帧: {avg_display*1000:.2f}ms")
                    print(f"处理总时间: {total_avg*1000:.2f}ms")
                    print(f"理论最大FPS: {1/total_avg:.2f}")
                    print(f"实际理论FPS: {1/avg_loop:.2f}")
                    print("=" * 40)
            else:
                actual_fps = fps_frame_count / (current_time - fps_start_time) if (current_time - fps_start_time) > 0 else 0
            
            # 计算进度
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            
            # 创建用于显示的帧副本
            display_frame = frame.copy()
            
            # 记录颜色转换开始时间
            convert_start = time.time()
            # 将BGR图像转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convert_time = time.time() - convert_start
            
            # 记录MediaPipe处理开始时间
            process_start = time.time()
            # 处理图像
            results = holistic.process(rgb_frame)
            process_time = time.time() - process_start
            
            # 添加检测结果日志（每100帧打印一次）
            if frame_count % 100 == 0:
                print(f"\n[Frame {frame_count}] 检测结果:")
                print(f"  Pose: {results.pose_landmarks is not None}")
                print(f"  Face: {results.face_landmarks is not None}")
                print(f"  Left Hand: {results.left_hand_landmarks is not None}")
                print(f"  Right Hand: {results.right_hand_landmarks is not None}")
                if results.pose_landmarks:
                    print(f"  Pose landmarks count: {len(results.pose_landmarks.landmark)}")
                if results.face_landmarks:
                    print(f"  Face landmarks count: {len(results.face_landmarks.landmark)}")
                if results.left_hand_landmarks:
                    print(f"  Left hand landmarks count: {len(results.left_hand_landmarks.landmark)}")
                if results.right_hand_landmarks:
                    print(f"  Right hand landmarks count: {len(results.right_hand_landmarks.landmark)}")
            
            # 记录绘制开始时间
            draw_start = time.time()
            # 将图像标记为可写
            display_frame.flags.writeable = True
            
            # 绘制检测结果
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            draw_time = time.time() - draw_start
            
            # 显示帧号
            cv2.putText(display_frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 显示检测状态
            status_text = []
            if results.pose_landmarks:
                status_text.append("Pose")
            if results.face_landmarks:
                status_text.append("Face")
            if results.left_hand_landmarks:
                status_text.append("L-Hand")
            if results.right_hand_landmarks:
                status_text.append("R-Hand")
            
            cv2.putText(display_frame, f'Detected: {", ".join(status_text) if status_text else "None"}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示实时帧率
            cv2.putText(display_frame, f'Actual FPS: {actual_fps:.2f}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示目标帧率
            cv2.putText(display_frame, f'Target FPS: {fps:.2f}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示进度
            cv2.putText(display_frame, f'Progress: {progress:.2f}%', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示优化模式
            if optimize_performance:
                cv2.putText(display_frame, f'Performance Mode: ON', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 记录显示开始时间
            display_start = time.time()
            # 显示处理后的帧
            cv2.imshow(window_name, display_frame)
            display_time = time.time() - display_start
            
            # 累计性能统计
            total_read_time += read_time
            total_convert_time += convert_time
            total_process_time += process_time
            total_draw_time += draw_time
            total_display_time += display_time
            
            # 记录循环总时间
            loop_time = time.time() - loop_start
            total_loop_time += loop_time
            
            # 根据模式选择等待策略
            if no_delay:
                # 无延迟模式：完全不等待，实现最大帧率
                # 注意：这可能导致CPU使用率很高
                pass  # 不调用waitKey
            else:
                # 正常模式：等待1ms
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break
                elif key == 32:  # 空格键
                    cv2.waitKey(0)  # 暂停直到按任意键
            
            # 无延迟模式下的按键检测（使用非阻塞方式）
            if no_delay:
                # 使用非阻塞方式检测按键
                try:
                    import msvcrt  # Windows
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\x1b':  # ESC键
                            break
                        elif key == b' ':  # 空格键
                            input("按回车继续...")
                except ImportError:
                    # Linux/Mac - 使用select检测stdin
                    import select
                    import sys
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        if key == '\x1b':  # ESC键
                            break
                        elif key == ' ':  # 空格键
                            input("按回车继续...")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印最终性能统计
        if frame_count > 0:
            print(f"\n=== 最终性能统计 ===")
            print(f"总处理帧数: {frame_count}")
            print(f"平均帧读取时间: {total_read_time/frame_count*1000:.2f}ms")
            print(f"平均颜色转换时间: {total_convert_time/frame_count*1000:.2f}ms")
            print(f"平均MediaPipe处理时间: {total_process_time/frame_count*1000:.2f}ms")
            print(f"平均绘制时间: {total_draw_time/frame_count*1000:.2f}ms")
            print(f"平均显示时间: {total_display_time/frame_count*1000:.2f}ms")
            total_avg_time = (total_read_time + total_convert_time + total_process_time + total_draw_time + total_display_time) / frame_count
            print(f"平均总处理时间: {total_avg_time*1000:.2f}ms")
            print(f"理论最大FPS: {1/total_avg_time:.2f}")
            print("=" * 40)

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
    
    if detection_process.is_alive():
        print("强制终止检测进程")
        detection_process.terminate()
        detection_process.join(timeout=1)
    
    print("多进程模式结束")

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

def process_video_headless(video_path, optimize_performance=False):
    """
    无显示模式处理视频，用于测试纯处理性能
    """
    print(f"开始无显示模式处理视频: {video_path}")
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
    
    frame_count = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    
    # 性能统计变量
    total_read_time = 0
    total_convert_time = 0
    total_process_time = 0
    total_loop_time = 0
    last_frame_time = time.time()
    
    # 检测统计
    pose_detected = 0
    face_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0
    
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
        
        while cap.isOpened():
            loop_start = time.time()
            
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            
            if not ret:
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            current_frame_time = time.time()
            frame_interval = current_frame_time - last_frame_time
            last_frame_time = current_frame_time
            
            # 跳过偶数帧（性能优化模式）
            if optimize_performance and frame_count % 2 == 0:
                continue
            
            convert_start = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convert_time = time.time() - convert_start
            
            process_start = time.time()
            results = holistic.process(rgb_frame)
            process_time = time.time() - process_start
            
            # 统计检测结果
            if results.pose_landmarks:
                pose_detected += 1
            if results.face_landmarks:
                face_detected += 1
            if results.left_hand_landmarks:
                left_hand_detected += 1
            if results.right_hand_landmarks:
                right_hand_detected += 1
            
            loop_time = time.time() - loop_start
            
            # 累计统计
            total_read_time += read_time
            total_convert_time += convert_time
            total_process_time += process_time
            total_loop_time += loop_time
            
            # 每秒打印一次统计
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                actual_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
                
                if frame_count > 1:
                    avg_read = total_read_time / frame_count
                    avg_convert = total_convert_time / frame_count
                    avg_process = total_process_time / frame_count
                    avg_loop = total_loop_time / frame_count
                    
                    print(f"\n=== 无显示模式统计 (帧 {frame_count}) ===")
                    print(f"实际FPS: {actual_fps:.2f}")
                    print(f"帧间隔时间: {frame_interval*1000:.2f}ms")
                    print(f"循环总时间: {avg_loop*1000:.2f}ms")
                    print(f"帧读取: {avg_read*1000:.2f}ms")
                    print(f"颜色转换: {avg_convert*1000:.2f}ms")
                    print(f"MediaPipe处理: {avg_process*1000:.2f}ms")
                    print(f"理论最大FPS: {1/avg_loop:.2f}")
                    print(f"检测统计 - Pose: {pose_detected}, Face: {face_detected}, L-Hand: {left_hand_detected}, R-Hand: {right_hand_detected}")
                    print("=" * 40)
        
        cap.release()
        
        # 最终统计
        if frame_count > 0:
            print(f"\n=== 最终无显示模式统计 ===")
            print(f"总处理帧数: {frame_count}")
            print(f"平均循环时间: {total_loop_time/frame_count*1000:.2f}ms")
            print(f"理论最大FPS: {1/(total_loop_time/frame_count):.2f}")
            print(f"检测率 - Pose: {pose_detected/frame_count*100:.1f}%, Face: {face_detected/frame_count*100:.1f}%, L-Hand: {left_hand_detected/frame_count*100:.1f}%, R-Hand: {right_hand_detected/frame_count*100:.1f}%")
            print("=" * 40)

def main():
    """
    主函数
    """
    print("MediaPipe Holistic 视频处理演示")
    print("=" * 40)
    
    # 创建测试视频
    create_test_video()
    
    # 询问是否使用测试视频
    use_test = input("是否使用测试视频? (y/n): ").lower().strip()
    
    if use_test == 'y':
        video_path = 'video.mp4'
    else:
        video_path = input("请输入视频文件路径: ").strip()
    
    # 询问是否启用性能优化
    optimize = input("是否启用性能优化模式? (y/n): ").lower().strip() == 'y'
    
    if optimize:
        print("\n性能优化模式说明:")
        print("- 使用轻量级模型 (model_complexity=0)")
        print("- 跳过偶数帧的MediaPipe处理")
        print("- 可以显著提高帧率，但可能影响检测精度")
        print()
    
    # 询问是否启用无延迟模式
    no_delay = input("是否启用无延迟模式? (y/n): ").lower().strip() == 'y'
    
    if no_delay:
        print("\n无延迟模式说明:")
        print("- 完全移除所有等待时间")
        print("- 实现最大可能的帧率")
        print("- CPU使用率会很高")
        print("- 按Ctrl+C退出程序")
        print()
    
    # 询问是否使用无显示模式（用于性能测试）
    headless = input("是否使用无显示模式进行性能测试? (y/n): ").lower().strip() == 'y'
    
    if headless:
        print("\n无显示模式说明:")
        print("- 不显示视频窗口，只进行纯处理")
        print("- 用于测试MediaPipe的真实性能")
        print("- 会显示检测统计信息")
        print()
        process_video_headless(video_path, optimize_performance=optimize)
    else:
        # 询问是否使用多线程模式
        multithread = input("是否使用多线程模式? (y/n): ").lower().strip() == 'y'
        
        if multithread:
            print("\n多线程模式说明:")
            print("- 分离检测和显示线程")
            print("- 检测线程独立运行，不受显示影响")
            print("- 显示线程按固定FPS播放")
            print("- 实现真正的实时处理")
            print("- 注意：受Python GIL限制，无法真正并行")
            print()
            process_video_multithread(video_path, optimize_performance=optimize)
        else:
            # 询问是否使用多进程模式
            multiprocess = input("是否使用多进程模式? (y/n): ").lower().strip() == 'y'
            
            if multiprocess:
                print("\n多进程模式说明:")
                print("- 使用独立进程进行检测")
                print("- 真正利用多核CPU")
                print("- 不受Python GIL限制")
                print("- 检测和显示完全分离")
                print("- 最佳性能表现")
                print()
                process_video_multiprocess(video_path, optimize_performance=optimize)
            else:
                # 处理视频
                process_video(video_path, optimize_performance=optimize, no_delay=no_delay)

if __name__ == "__main__":
    main() 