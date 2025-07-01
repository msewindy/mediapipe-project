#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理模式选择器
包含所有视频处理模式的入口函数
"""

import os
import time
from basic_processor import process_video, process_video_headless
from multithread_processor import process_video_multithread
from multiprocess_processor import process_video_multiprocess
from optimized_multiprocess_processor import process_video_optimized_multiprocess
from ultra_performance_processor import process_video_ultra_performance


def run_basic_mode(video_path, optimize_performance=False, no_delay=False):
    """
    基础模式：单线程处理
    """
    print("\n=== 基础模式 ===")
    print("说明:")
    print("- 单线程处理，检测和显示在同一线程")
    print("- 简单直接，适合学习和调试")
    print("- 性能受限于单线程处理能力")
    print("- 按ESC键退出，按空格键暂停")
    print()
    
    process_video(video_path, optimize_performance=optimize_performance, no_delay=no_delay)


def run_multithread_mode(video_path, optimize_performance=False):
    """
    多线程模式：分离检测和显示线程
    """
    print("\n=== 多线程模式 ===")
    print("说明:")
    print("- 分离检测和显示线程")
    print("- 检测线程独立运行，不受显示影响")
    print("- 显示线程按固定FPS播放")
    print("- 实现真正的实时处理")
    print("- 注意：受Python GIL限制，无法真正并行")
    print("- 按ESC键退出，按空格键暂停")
    print()
    
    process_video_multithread(video_path, optimize_performance=optimize_performance)


def run_multiprocess_mode(video_path, optimize_performance=False):
    """
    多进程模式：使用独立进程进行检测
    """
    print("\n=== 多进程模式 ===")
    print("说明:")
    print("- 使用独立进程进行检测")
    print("- 真正利用多核CPU")
    print("- 不受Python GIL限制")
    print("- 检测和显示完全分离")
    print("- 最佳性能表现")
    print("- 按ESC键退出，按空格键暂停")
    print()
    
    process_video_multiprocess(video_path, optimize_performance=optimize_performance)


def run_optimized_multiprocess_mode(video_path, optimize_performance=False):
    """
    优化多进程模式：主进程负责读取和检测，子进程负责显示
    """
    print("\n=== 优化多进程模式 ===")
    print("说明:")
    print("- 主进程负责读取和检测，最大化检测性能")
    print("- 子进程负责显示，不阻塞检测流程")
    print("- 异步队列写入，最小化检测进程开销")
    print("- 相比传统多进程，检测性能更接近无显示模式")
    print("- 推荐用于实时检测场景")
    print("- 按ESC键退出，按空格键暂停")
    print()
    
    process_video_optimized_multiprocess(video_path, optimize_performance=optimize_performance)


def run_ultra_performance_mode(video_path, optimize_performance=False):
    """
    超高性能模式：单进程 + 线程池优化
    """
    print("\n=== 超高性能模式 ===")
    print("说明:")
    print("- 单进程 + 线程池架构，避免多进程开销")
    print("- 主线程专注MediaPipe检测，最大化CPU利用率")
    print("- 显示线程异步处理，不阻塞检测流程")
    print("- 最小化系统开销，接近无显示模式性能")
    print("- 推荐用于追求极致检测性能的场景")
    print("- 按ESC键退出，按空格键暂停")
    print()
    
    process_video_ultra_performance(video_path, optimize_performance=optimize_performance)


def run_headless_mode(video_path, optimize_performance=False):
    """
    无显示模式：纯性能测试
    """
    print("\n=== 无显示模式 ===")
    print("说明:")
    print("- 不显示视频窗口，只进行纯处理")
    print("- 用于测试MediaPipe的真实性能")
    print("- 会显示检测统计信息")
    print("- 按Ctrl+C退出程序")
    print()
    
    process_video_headless(video_path, optimize_performance=optimize_performance)


def run_performance_test():
    """
    性能测试模式：运行各种性能测试和分析
    """
    print("\n=== 性能测试模式 ===")
    print("说明:")
    print("- 运行各种性能测试和分析")
    print("- 包括线程性能、GIL分析等")
    print("- 帮助理解MediaPipe的性能特点")
    print("- 按Ctrl+C退出程序")
    print()
    
    try:
        from performance_tests import run_all_tests
        run_all_tests()
    except ImportError:
        print("错误：无法导入性能测试模块")
        print("请确保 performance_tests.py 文件在同一目录下")


def get_video_path():
    """
    获取视频文件路径
    """
    # 询问是否使用测试视频
    use_test = input("是否使用测试视频? (y/n): ").lower().strip()
    
    if use_test == 'y':
        video_path = 'video.mp4'
        if not os.path.exists(video_path):
            print(f"错误: 测试视频文件不存在: {video_path}")
            print("请先运行程序创建测试视频")
            return None
    else:
        video_path = input("请输入视频文件路径: ").strip()
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在: {video_path}")
            return None
    
    return video_path


def get_optimization_settings():
    """
    获取性能优化设置
    """
    # 询问是否启用性能优化
    optimize = input("是否启用性能优化模式? (y/n): ").lower().strip() == 'y'
    
    if optimize:
        print("\n性能优化模式说明:")
        print("- 使用轻量级模型 (model_complexity=0)")
        print("- 跳过偶数帧的MediaPipe处理")
        print("- 可以显著提高帧率，但可能影响检测精度")
        print()
    
    return optimize


def get_delay_settings():
    """
    获取延迟设置
    """
    # 询问是否启用无延迟模式
    no_delay = input("是否启用无延迟模式? (y/n): ").lower().strip() == 'y'
    
    if no_delay:
        print("\n无延迟模式说明:")
        print("- 完全移除所有等待时间")
        print("- 实现最大可能的帧率")
        print("- CPU使用率会很高")
        print("- 按Ctrl+C退出程序")
        print()
    
    return no_delay


def show_mode_selection():
    """
    显示模式选择菜单
    """
    print("\n请选择处理模式:")
    print("1. 基础模式 (单线程)")
    print("2. 多线程模式")
    print("3. 多进程模式")
    print("4. 优化多进程模式")
    print("5. 超高性能模式")
    print("6. 无显示模式 (性能测试)")
    print("7. 性能测试模式")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (0-7): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7']:
                return choice
            else:
                print("无效选择，请输入 0-7 之间的数字")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            return '0' 