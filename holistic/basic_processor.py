#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础处理器：单线程处理模式
包含基本的视频处理和显示功能
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time


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