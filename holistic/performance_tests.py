#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe性能测试模块
包含各种性能测试和分析函数
"""

import threading
import time
import queue
import multiprocessing as mp
import math
import os

def test_threading_performance():
    """
    测试显示线程在计算密集型主线程下的表现
    """
    print("=== 线程性能测试 ===")
    print("测试显示线程在计算密集型主线程下的流畅性")
    
    # 测试参数
    test_duration = 10  # 测试10秒
    display_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    
    # 统计变量
    main_thread_count = 0
    display_thread_count = 0
    main_thread_times = []
    display_thread_times = []
    
    def compute_intensive_task(duration_ms):
        """
        真实的计算密集型任务
        使用数学计算来占用CPU，而不是sleep
        """
        start_time = time.time()
        target_time = duration_ms / 1000.0
        
        # 执行数学计算直到达到目标时间
        result = 0
        while time.time() - start_time < target_time:
            # 执行一些数学计算来占用CPU
            for i in range(1000):
                result += math.sin(i) * math.cos(i) + math.sqrt(i + 1)
        
        return result
    
    def display_worker():
        """显示线程：模拟显示操作"""
        nonlocal display_thread_count, display_thread_times
        
        while not stop_event.is_set():
            try:
                start_time = time.time()
                
                # 模拟显示操作（5ms）
                data = display_queue.get(timeout=0.1)
                
                # 模拟显示处理时间（真实的计算，不是sleep）
                compute_intensive_task(5)
                
                display_thread_count += 1
                display_thread_times.append(time.time() - start_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"显示线程错误: {e}")
                continue
    
    # 启动显示线程
    display_thread = threading.Thread(target=display_worker, daemon=True)
    display_thread.start()
    print("显示线程已启动")
    
    # 主线程：模拟计算密集型任务
    print("主线程开始执行计算密集型任务...")
    start_time = time.time()
    
    while time.time() - start_time < test_duration:
        loop_start = time.time()
        
        # 模拟MediaPipe检测（32ms）- 使用真实计算
        compute_intensive_task(32)
        
        # 模拟其他处理（5ms）- 使用真实计算
        compute_intensive_task(5)
        
        # 发送数据到显示队列
        try:
            display_queue.put_nowait(f"data_{main_thread_count}")
            main_thread_count += 1
        except queue.Full:
            pass  # 队列满了就跳过
        
        main_thread_times.append(time.time() - loop_start)
    
    # 停止测试
    stop_event.set()
    display_thread.join(timeout=3)
    
    # 分析结果
    total_time = time.time() - start_time
    main_fps = main_thread_count / total_time
    display_fps = display_thread_count / total_time
    
    avg_main_time = sum(main_thread_times) / len(main_thread_times) if main_thread_times else 0
    avg_display_time = sum(display_thread_times) / len(display_thread_times) if display_thread_times else 0
    
    print(f"\n=== 测试结果 ===")
    print(f"测试时长: {total_time:.1f}秒")
    print(f"主线程处理帧数: {main_thread_count}")
    print(f"显示线程处理帧数: {display_thread_count}")
    print(f"主线程FPS: {main_fps:.1f}")
    print(f"显示线程FPS: {display_fps:.1f}")
    print(f"主线程平均耗时: {avg_main_time*1000:.1f}ms")
    print(f"显示线程平均耗时: {avg_display_time*1000:.1f}ms")
    print(f"队列丢弃率: {((main_thread_count - display_thread_count) / main_thread_count * 100):.1f}%")
    
    # 分析显示流畅性
    if display_fps >= 20:
        print("✅ 显示线程运行流畅（FPS >= 20）")
    elif display_fps >= 15:
        print("⚠️  显示线程运行一般（15 <= FPS < 20）")
    else:
        print("❌ 显示线程运行不流畅（FPS < 15）")
    
    # 分析CPU竞争情况
    if display_fps / main_fps >= 0.8:
        print("✅ 显示线程CPU竞争较小（显示FPS/主线程FPS >= 80%）")
    elif display_fps / main_fps >= 0.5:
        print("⚠️  显示线程CPU竞争中等（50% <= 显示FPS/主线程FPS < 80%）")
    else:
        print("❌ 显示线程CPU竞争严重（显示FPS/主线程FPS < 50%）")
    
    print("=" * 40)

def test_cpu_intensive_vs_sleep():
    """
    对比测试：CPU密集型操作 vs sleep
    """
    print("=== CPU密集型 vs Sleep 对比测试 ===")
    
    def cpu_intensive_task(duration_ms):
        """CPU密集型任务"""
        start_time = time.time()
        target_time = duration_ms / 1000.0
        result = 0
        while time.time() - start_time < target_time:
            for i in range(1000):
                result += math.sin(i) * math.cos(i) + math.sqrt(i + 1)
        return result
    
    def sleep_task(duration_ms):
        """Sleep任务"""
        time.sleep(duration_ms / 1000.0)
    
    def worker_thread(task_func, task_name):
        """工作线程"""
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < 5:  # 运行5秒
            task_func(10)  # 10ms任务
            count += 1
        
        duration = time.time() - start_time
        fps = count / duration
        print(f"{task_name}线程: {count}次任务, {fps:.1f} FPS")
    
    # 测试1：两个sleep线程
    print("\n测试1：两个sleep线程（应该都能正常运行）")
    thread1 = threading.Thread(target=worker_thread, args=(sleep_task, "Sleep-1"))
    thread2 = threading.Thread(target=worker_thread, args=(sleep_task, "Sleep-2"))
    
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    
    # 测试2：两个CPU密集型线程
    print("\n测试2：两个CPU密集型线程（会竞争CPU资源）")
    thread3 = threading.Thread(target=worker_thread, args=(cpu_intensive_task, "CPU-1"))
    thread4 = threading.Thread(target=worker_thread, args=(cpu_intensive_task, "CPU-2"))
    
    thread3.start()
    thread4.start()
    thread3.join()
    thread4.join()
    
    # 测试3：一个CPU密集型 + 一个sleep
    print("\n测试3：CPU密集型 + Sleep线程（sleep线程应该不受影响）")
    thread5 = threading.Thread(target=worker_thread, args=(cpu_intensive_task, "CPU"))
    thread6 = threading.Thread(target=worker_thread, args=(sleep_task, "Sleep"))
    
    thread5.start()
    thread6.start()
    thread5.join()
    thread6.join()
    
    print("=" * 40)

def worker_process_for_mp(_):
    """工作进程：模拟MediaPipe内部的计算（用于多进程）"""
    start_time = time.time()
    result = 0
    for i in range(1000000):
        result += i * i + math.sqrt(i + 1)
    return result, time.time() - start_time

def analyze_gil_vs_mediapipe():
    """
    分析Python GIL vs MediaPipe并行化的区别
    """
    print("=== Python GIL vs MediaPipe并行化分析 ===")
    
    def python_compute_intensive():
        """Python计算密集型任务（受GIL限制）"""
        start_time = time.time()
        result = 0
        for i in range(1000000):
            result += i * i + math.sqrt(i + 1)
        return result, time.time() - start_time
    
    def simulate_mediapipe_parallel():
        """
        模拟MediaPipe的并行化行为
        使用多进程来模拟不受GIL限制的并行计算
        """
        # 使用多进程模拟MediaPipe内部并行化
        with mp.Pool(processes=4) as pool:
            results = pool.map(worker_process_for_mp, range(4))
        
        total_time = max(r[1] for r in results)  # 取最长的时间
        return total_time
    
    print("1. Python GIL对多线程的影响：")
    print("   - 同一时刻只有一个线程能执行Python字节码")
    print("   - 计算密集型任务无法真正并行")
    print("   - 多线程主要用于I/O密集型任务")
    
    print("\n2. MediaPipe内部并行化：")
    print("   - 使用C++实现，不受Python GIL限制")
    print("   - 内部使用多线程进行模型推理")
    print("   - 可以真正利用多个CPU核心")
    print("   - 可能使用SIMD指令和向量化计算")
    
    print("\n3. 对我们的测试的影响：")
    print("   - 我们的测试使用Python计算模拟MediaPipe")
    print("   - 测试中的'主线程'受GIL限制")
    print("   - 真实的MediaPipe检测不受GIL限制")
    print("   - 测试结果可能低估了MediaPipe的性能")
    
    print("\n4. 实际应用中的情况：")
    print("   - MediaPipe检测：不受GIL限制，可以并行")
    print("   - Python显示线程：受GIL限制")
    print("   - 两者竞争CPU资源，但竞争方式不同")
    
    # 实际测试对比
    print("\n=== 实际测试对比 ===")
    
    # 测试1：Python多线程（受GIL限制）
    print("\n测试1：Python多线程计算（受GIL限制）")
    start_time = time.time()
    
    def python_worker():
        result, duration = python_compute_intensive()
        return duration
    
    threads = []
    for i in range(4):
        thread = threading.Thread(target=python_worker)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    python_total_time = time.time() - start_time
    print(f"Python多线程总耗时: {python_total_time:.2f}秒")
    
    # 测试2：多进程（不受GIL限制）
    print("\n测试2：多进程计算（不受GIL限制，类似MediaPipe）")
    mp_start_time = time.time()
    mp_time = simulate_mediapipe_parallel()
    mp_total_time = time.time() - mp_start_time
    
    print(f"多进程总耗时: {mp_total_time:.2f}秒")
    print(f"并行化加速比: {python_total_time / mp_total_time:.2f}x")
    
    print("\n5. 结论：")
    print("   - 我们的测试使用Python计算，受GIL限制")
    print("   - 真实的MediaPipe不受GIL限制，性能更好")
    print("   - 测试结果可能低估了MediaPipe的并行能力")
    print("   - 但测试仍然有效，因为显示线程确实受GIL限制")
    print("   - 实际应用中，MediaPipe检测和Python显示会竞争CPU资源")
    
    print("=" * 40)

def compute_task_for_mp(_):
    """计算任务（用于多进程）"""
    start_time = time.time()
    result = 0
    for i in range(500000):  # 减少计算量以模拟MediaPipe检测
        result += i * i + math.sqrt(i + 1)
    return result, time.time() - start_time

def test_realistic_mediapipe_simulation():
    """
    真实MediaPipe模拟测试
    使用多进程模拟MediaPipe，Python线程模拟显示
    """
    print("=== 真实MediaPipe模拟测试 ===")
    print("使用多进程模拟MediaPipe，Python线程模拟显示")
    
    # 共享变量
    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    stats = {
        'mediapipe_frames': 0,
        'display_frames': 0,
        'start_time': time.time()
    }
    
    def display_worker():
        """显示线程：模拟显示处理"""
        print("显示线程已启动")
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
                stats['display_frames'] += 1
                # 模拟显示处理
                time.sleep(0.01)  # 10ms显示处理时间
            except queue.Empty:
                continue
    
    def mediapipe_simulation_worker():
        """MediaPipe模拟工作进程"""
        print("主线程开始模拟MediaPipe检测...")
        
        # 启动显示线程
        display_thread = threading.Thread(target=display_worker)
        display_thread.start()
        
        # 模拟MediaPipe检测循环
        while not stop_event.is_set():
            # 使用多进程模拟MediaPipe内部并行化
            with mp.Pool(processes=2) as pool:
                results = pool.map(compute_task_for_mp, range(2))
            
            # 模拟检测结果
            detection_result = sum(r[0] for r in results)
            
            # 将结果放入队列
            try:
                frame_queue.put_nowait(detection_result)
                stats['mediapipe_frames'] += 1
            except queue.Full:
                # 队列满了，丢弃帧
                pass
            
            # 模拟30FPS的检测频率
            time.sleep(1/30)
        
        display_thread.join()
    
    # 运行测试
    test_duration = 5.0  # 5秒测试
    timer = threading.Timer(test_duration, stop_event.set)
    timer.start()
    
    mediapipe_simulation_worker()
    
    # 计算统计结果
    total_time = time.time() - stats['start_time']
    mediapipe_fps = stats['mediapipe_frames'] / total_time
    display_fps = stats['display_frames'] / total_time
    
    print(f"\n=== 测试结果 ===")
    print(f"测试时长: {total_time:.1f}秒")
    print(f"MediaPipe模拟帧数: {stats['mediapipe_frames']}")
    print(f"显示帧数: {stats['display_frames']}")
    print(f"MediaPipe模拟FPS: {mediapipe_fps:.1f}")
    print(f"显示FPS: {display_fps:.1f}")
    
    # 分析结果
    if display_fps >= 20:
        print("✅ 显示线程运行流畅（FPS >= 20）")
    else:
        print("❌ 显示线程运行不流畅（FPS < 20）")
    
    if display_fps / mediapipe_fps >= 0.8:
        print("✅ 显示线程CPU竞争较小（显示FPS/MediaPipe FPS >= 80%）")
    else:
        print("❌ 显示线程CPU竞争较大（显示FPS/MediaPipe FPS < 80%）")
    
    print("=" * 40)

def run_all_tests():
    """
    运行所有性能测试
    """
    print("MediaPipe性能测试套件")
    print("=" * 50)
    
    while True:
        print("\n请选择要运行的测试：")
        print("1. 线程性能测试")
        print("2. CPU密集型 vs Sleep对比测试")
        print("3. GIL vs MediaPipe分析")
        print("4. 真实MediaPipe模拟测试")
        print("5. 运行所有测试")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == '0':
            print("退出测试")
            break
        elif choice == '1':
            test_threading_performance()
        elif choice == '2':
            test_cpu_intensive_vs_sleep()
        elif choice == '3':
            analyze_gil_vs_mediapipe()
        elif choice == '4':
            test_realistic_mediapipe_simulation()
        elif choice == '5':
            print("\n运行所有测试...")
            test_threading_performance()
            test_cpu_intensive_vs_sleep()
            analyze_gil_vs_mediapipe()
            test_realistic_mediapipe_simulation()
            print("\n所有测试完成！")
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    run_all_tests() 