#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe Holistic 视频处理演示
主入口文件
"""

import os
from utils import create_test_video


def main():
    """
    主函数
    """
    print("MediaPipe Holistic 视频处理演示")
    print("=" * 40)
    
    # 创建测试视频
    create_test_video()
    
    try:
        # 导入视频处理器模块
        from video_processors import (
            get_video_path,
            get_optimization_settings,
            get_delay_settings,
            show_mode_selection,
            run_basic_mode,
            run_multithread_mode,
            run_multiprocess_mode,
            run_optimized_multiprocess_mode,
            run_ultra_performance_mode,
            run_headless_mode,
            run_performance_test
        )
        
        # 获取视频路径
        video_path = get_video_path()
        if video_path is None:
            return
        
        # 获取优化设置
        optimize = get_optimization_settings()
        
        # 获取延迟设置
        no_delay = get_delay_settings()
        
        # 显示模式选择菜单
        choice = show_mode_selection()
        
        # 根据选择执行相应的处理模式
        if choice == '0':
            print("程序退出")
            return
        elif choice == '1':
            run_basic_mode(video_path, optimize_performance=optimize, no_delay=no_delay)
        elif choice == '2':
            run_multithread_mode(video_path, optimize_performance=optimize)
        elif choice == '3':
            run_multiprocess_mode(video_path, optimize_performance=optimize)
        elif choice == '4':
            run_optimized_multiprocess_mode(video_path, optimize_performance=optimize)
        elif choice == '5':
            run_ultra_performance_mode(video_path, optimize_performance=optimize)
        elif choice == '6':
            run_headless_mode(video_path, optimize_performance=optimize)
        elif choice == '7':
            run_performance_test()
        else:
            print("无效选择")
            
    except ImportError as e:
        print(f"错误：无法导入必要的模块: {e}")
        print("请确保所有处理器文件在同一目录下")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 