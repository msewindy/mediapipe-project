#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试YouTube视频播放功能（不包含MediaPipe检测）
支持下载到本地播放
"""

import cv2
import yt_dlp
import subprocess
import threading
import queue
import time
import sys
import os
import tempfile
import numpy as np

class YouTubeVideoPlayer:
    def __init__(self):
        # 初始化视频处理队列
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        
        # 视频信息
        self.video_info = {}
    
    def get_youtube_stream_url(self, url):
        """
        获取YouTube视频流URL
        """
        print("正在获取YouTube视频流信息...")
        
        ydl_opts = {
            'format': 'best[height<=720]',  # 选择720p或更低分辨率
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                self.video_info = {
                    'title': title,
                    'duration': duration,
                    'url': stream_url
                }
                
                print(f"视频标题: {title}")
                print(f"视频时长: {duration}秒")
                print("成功获取视频流URL")
                
                return stream_url, title, duration
        except Exception as e:
            print(f"获取YouTube视频流失败: {e}")
            return None, None, None
    
    def download_video(self, url, output_filename=None):
        """
        下载YouTube视频到本地
        """
        if output_filename is None:
            # 生成安全的文件名
            safe_title = "".join(c for c in self.video_info.get('title', 'video') if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:50]  # 限制长度
            output_filename = f"{safe_title}.mp4"
        
        print(f"开始下载视频到: {output_filename}")
        
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': output_filename,
            'quiet': False,  # 显示下载进度
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if os.path.exists(output_filename):
                print(f"视频下载完成: {output_filename}")
                return output_filename
            else:
                print("下载失败，文件不存在")
                return None
                
        except Exception as e:
            print(f"下载视频时出错: {e}")
            return None
    
    def play_local_video(self, video_path):
        """
        播放本地视频文件
        """
        print(f"播放本地视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {total_frames}")
        
        # 创建窗口并设置大小
        window_name = 'YouTube Video Player (Local)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        
        frame_count = 0
        start_time = time.time()
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while cap.isOpened() and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束")
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # 计算实际FPS（每秒更新一次）
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                actual_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
            else:
                actual_fps = fps_frame_count / (current_time - fps_start_time) if (current_time - fps_start_time) > 0 else 0
            
            # 计算进度
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            
            # 在帧上显示信息
            cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Progress: {progress:.1f}%', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Target FPS: {fps:.1f}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Actual FPS: {actual_fps:.1f}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Size: {width}x{height}', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示帧
            cv2.imshow(window_name, frame)
            
            # 控制播放速度 - 根据原始FPS计算等待时间
            wait_time = max(1, int(1000 / fps))  # 转换为毫秒
            
            # 按ESC键退出，按空格键暂停/继续
            key = cv2.waitKey(wait_time) & 0xFF
            if key == 27:  # ESC键
                self.stop_event.set()
                break
            elif key == 32:  # 空格键
                cv2.waitKey(0)  # 暂停直到按任意键
        
        cap.release()
        cv2.destroyAllWindows()
    
    def start_ffmpeg_stream(self, stream_url):
        """
        启动ffmpeg进程来获取实时视频流
        """
        print("启动实时视频流处理...")
        
        # 创建临时管道
        temp_dir = tempfile.mkdtemp()
        pipe_path = os.path.join(temp_dir, "video_pipe")
        
        # 创建命名管道
        os.mkfifo(pipe_path)
        
        # ffmpeg命令：从YouTube流读取并输出到管道
        cmd = [
            'ffmpeg',
            '-i', stream_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',  # 禁用视频同步
            '-an',  # 禁用音频
            '-sn',  # 禁用字幕
            '-r', '30',  # 限制帧率
            '-s', '640x480',  # 限制分辨率
            '-y',  # 覆盖输出
            pipe_path
        ]
        
        print(f"FFmpeg命令: {' '.join(cmd)}")
        
        try:
            # 启动ffmpeg进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            return process, pipe_path, temp_dir
            
        except Exception as e:
            print(f"启动ffmpeg失败: {e}")
            return None, None, None
    
    def read_frames_from_pipe(self, pipe_path, frame_width=640, frame_height=480):
        """
        从管道读取视频帧
        """
        frame_size = frame_width * frame_height * 3  # BGR24格式
        
        print(f"开始从管道读取帧，帧大小: {frame_size} 字节")
        
        try:
            with open(pipe_path, 'rb') as pipe:
                frame_count = 0
                while not self.stop_event.is_set():
                    # 读取一帧的原始数据
                    raw_data = pipe.read(frame_size)
                    
                    if len(raw_data) != frame_size:
                        print(f"读取数据长度不匹配: {len(raw_data)} != {frame_size}")
                        break
                    
                    # 转换为numpy数组
                    frame = np.frombuffer(raw_data, dtype=np.uint8)
                    frame = frame.reshape((frame_height, frame_width, 3))
                    frame = frame.copy()  # 确保frame是可写的
                    
                    frame_count += 1
                    if frame_count % 30 == 0:  # 每30帧打印一次
                        print(f"已读取 {frame_count} 帧")
                    
                    # 将帧放入队列
                    try:
                        self.frame_queue.put(frame, timeout=1)
                    except queue.Full:
                        # 队列满了，丢弃最旧的帧
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, timeout=1)
                        except:
                            pass
                            
        except Exception as e:
            print(f"读取视频帧时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def play_frames(self):
        """
        播放视频帧
        """
        frame_count = 0
        start_time = time.time()
        
        print("开始播放视频帧...")
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取帧
                frame = self.frame_queue.get(timeout=1)
                frame_count += 1
                
                # 计算FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # 显示信息
                cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Queue Size: {self.frame_queue.qsize()}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示处理后的帧
                cv2.imshow('YouTube Video Player', frame)
                
                # 按ESC键退出，按空格键暂停/继续
                key = cv2.waitKey(33) & 0xFF  # 约30 FPS
                if key == 27:  # ESC键
                    self.stop_event.set()
                    break
                elif key == 32:  # 空格键
                    cv2.waitKey(0)  # 暂停直到按任意键
                    
            except queue.Empty:
                print("队列为空，等待帧...")
                continue
            except Exception as e:
                print(f"播放帧时出错: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 释放资源
        cv2.destroyAllWindows()
    
    def play_youtube_video(self, youtube_url, download_first=False):
        """
        播放YouTube视频的主函数
        """
        print("=" * 50)
        print("YouTube视频播放器")
        print("=" * 50)
        
        if download_first:
            # 先下载视频
            local_video_path = self.download_video(youtube_url)
            if local_video_path:
                print("开始播放本地视频...")
                print("操作说明:")
                print("- 按ESC键退出")
                print("- 按空格键暂停/继续")
                
                self.play_local_video(local_video_path)
                
                # 询问是否删除下载的文件
                delete_file = input("是否删除下载的视频文件? (y/n): ").strip().lower()
                if delete_file == 'y':
                    try:
                        os.remove(local_video_path)
                        print(f"已删除文件: {local_video_path}")
                    except:
                        print("删除文件失败")
            else:
                print("下载失败，无法播放")
        else:
            # 获取视频流URL
            stream_url, title, duration = self.get_youtube_stream_url(youtube_url)
            if not stream_url:
                print("无法获取视频流，程序退出")
                return
            
            # 启动ffmpeg进程
            process, pipe_path, temp_dir = self.start_ffmpeg_stream(stream_url)
            if not process:
                print("启动ffmpeg失败，程序退出")
                return
            
            try:
                print("开始播放视频...")
                print("操作说明:")
                print("- 按ESC键退出")
                print("- 按空格键暂停/继续")
                
                # 启动帧读取线程
                read_thread = threading.Thread(
                    target=self.read_frames_from_pipe, 
                    args=(pipe_path,)
                )
                read_thread.daemon = True
                read_thread.start()
                
                # 等待一下让ffmpeg开始输出
                print("等待ffmpeg开始输出...")
                time.sleep(3)
                
                # 开始播放帧
                self.play_frames()
                
            except Exception as e:
                print(f"播放视频时出错: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # 清理资源
                self.stop_event.set()
                
                # 终止ffmpeg进程
                if process:
                    print("终止ffmpeg进程...")
                    process.terminate()
                    process.wait()
                
                # 清理临时文件
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        os.remove(pipe_path)
                        os.rmdir(temp_dir)
                        print("临时文件已清理")
                    except:
                        pass

def main():
    # 检查依赖
    try:
        import yt_dlp
    except ImportError:
        print("缺少yt-dlp库，请安装: pip install yt-dlp")
        return
    
    # 检查ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("缺少ffmpeg，请安装ffmpeg")
        print("Ubuntu/Debian: sudo apt install ffmpeg")
        print("CentOS/RHEL: sudo yum install ffmpeg")
        return
    
    # 获取YouTube URL
    youtube_url = input("请输入YouTube视频链接 (或按回车使用默认链接): ").strip()
    
    if not youtube_url:
        youtube_url = "https://www.youtube.com/watch?v=g6D2rzuzdZs"
    
    print(f"播放YouTube视频: {youtube_url}")
    
    # 选择播放方式
    print("\n选择播放方式:")
    print("1. 实时流播放 (速度快)")
    print("2. 下载到本地播放 (正常速度)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    # 创建播放器并开始播放
    player = YouTubeVideoPlayer()
    
    if choice == "2":
        player.play_youtube_video(youtube_url, download_first=True)
    else:
        player.play_youtube_video(youtube_url, download_first=False)

if __name__ == "__main__":
    main() 