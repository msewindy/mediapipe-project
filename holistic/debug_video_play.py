#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试视频播放功能
"""

import cv2
import os

def debug_video_play(video_path):
    """
    简单的视频播放测试
    """
    print(f"尝试播放视频: {video_path}")
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"文件不存在: {video_path}")
        return
    
    # 获取文件大小
    file_size = os.path.getsize(video_path)
    print(f"文件大小: {file_size} 字节")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息:")
    print(f"  FPS: {fps}")
    print(f"  尺寸: {width}x{height}")
    print(f"  总帧数: {total_frames}")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束")
            break
        
        frame_count += 1
        
        # 显示帧号
        cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示帧
        cv2.imshow('Debug Video Player', frame)
        
        # 按ESC退出
        if cv2.waitKey(30) & 0xFF == 27:  # 30ms = 约33 FPS
            break
    
    print(f"总共播放了 {frame_count} 帧")
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("视频播放调试工具")
    print("=" * 30)
    
    # 检查测试视频是否存在
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        print(f"找到测试视频: {test_video}")
        use_test = input("是否使用测试视频? (y/n): ").strip().lower()
        
        if use_test == 'y':
            debug_video_play(test_video)
            return
    
    # 用户输入视频路径
    video_path = input("请输入视频文件路径: ").strip()
    if video_path:
        debug_video_play(video_path)
    else:
        print("未指定视频文件路径")

if __name__ == "__main__":
    main() 