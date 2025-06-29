# MediaPipe Holistic 视频人体关键点检测

这个项目使用MediaPipe Holistic模型来检测视频中的人体关键点，包括面部、姿态和手部关键点。支持本地视频文件和YouTube视频链接。

## 功能特性

- 🎥 支持本地视频文件处理
- 🌐 支持YouTube视频链接处理
- 👤 检测面部关键点和网格
- 🏃 检测人体姿态关键点
- 🤚 检测左右手关键点
- 📊 实时显示检测结果和坐标信息
- ⏸️ 支持暂停/继续播放
- 🎮 交互式控制（ESC退出，空格暂停）
- 🔄 实时流处理（无需下载完整视频）

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install opencv-python mediapipe numpy yt-dlp
```

### 系统依赖

还需要安装ffmpeg：

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# macOS
brew install ffmpeg
```

## 使用方法

### 方法1：本地视频处理

运行演示脚本，它会自动创建一个测试视频：

```bash
python test_video_demo.py
```

### 方法2：YouTube视频处理（下载后处理）

处理YouTube视频，先下载再处理：

```bash
python youtube_holistic.py
```

### 方法3：YouTube实时流处理（推荐）

直接从YouTube获取实时视频流进行处理：

```bash
python youtube_stream_holistic.py
```

### 方法4：使用主程序

直接运行主程序：

```bash
python holistic.py
```

然后输入视频文件路径。

### 方法5：在代码中调用

```python
# 本地视频处理
from holistic import process_video
process_video("your_video.mp4")

# YouTube视频处理
from youtube_holistic import YouTubeHolisticProcessor
processor = YouTubeHolisticProcessor()
processor.process_youtube_video("https://www.youtube.com/watch?v=VIDEO_ID")

# YouTube实时流处理
from youtube_stream_holistic import YouTubeStreamProcessor
processor = YouTubeStreamProcessor()
processor.process_youtube_stream("https://www.youtube.com/watch?v=VIDEO_ID")
```

## 操作说明

- **ESC键**: 退出程序
- **空格键**: 暂停/继续播放
- **任意键**: 在暂停状态下继续播放

## 检测内容

程序会检测并显示以下关键点：

1. **面部关键点**: 468个面部关键点，包括眼睛、鼻子、嘴巴等
2. **姿态关键点**: 33个人体姿态关键点，包括头部、躯干、四肢
3. **手部关键点**: 每只手21个关键点，包括手指关节

## 输出信息

视频播放时会显示：

- 帧号
- 实时FPS
- 检测到的关键点类型
- 鼻子坐标（如果检测到姿态）
- 实时绘制的关键点连接线

## 支持的视频格式

### 本地视频
- MP4
- AVI
- MOV
- 其他OpenCV支持的格式

### YouTube视频
- 所有公开的YouTube视频
- 支持720p及以下分辨率
- 自动选择最佳可用格式

## 性能优化

- 使用`model_complexity=1`平衡性能和准确性
- 设置合适的检测和跟踪置信度阈值
- 图像处理优化（标记不可写以提高性能）
- 实时流处理减少内存占用
- 帧队列管理避免内存溢出

## 故障排除

### 常见问题

1. **无法打开视频文件**
   - 检查文件路径是否正确
   - 确保视频文件格式受支持

2. **YouTube视频无法处理**
   - 确保安装了yt-dlp: `pip install yt-dlp`
   - 确保安装了ffmpeg
   - 检查网络连接
   - 确认视频是公开的

3. **检测效果不佳**
   - 确保视频中有清晰的人体
   - 调整`min_detection_confidence`参数
   - 尝试不同的`model_complexity`设置

4. **性能问题**
   - 降低视频分辨率
   - 使用更低的`model_complexity`
   - 确保有足够的GPU内存
   - 使用实时流处理而不是下载完整视频

5. **ffmpeg相关错误**
   - 确保ffmpeg已正确安装
   - 检查ffmpeg版本是否支持所需功能

## 示例输出

### 本地视频处理
```
MediaPipe Holistic 视频处理演示
========================================
创建测试视频...
测试视频创建完成: test_video.mp4
是否使用测试视频? (y/n): y
开始处理视频: test_video.mp4
操作说明:
- 按ESC键退出
- 按空格键暂停/继续
- 等待几秒钟让模型加载...
视频信息: 640x480, 20 FPS
```

### YouTube视频处理
```
MediaPipe Holistic YouTube实时流处理
==================================================
正在获取YouTube视频流信息...
视频标题: Example Video
视频时长: 120秒
成功获取视频流URL
启动实时视频流处理...
开始实时处理视频...
操作说明:
- 按ESC键退出
- 按空格键暂停/继续
```

## 技术细节

- **模型**: MediaPipe Holistic
- **检测类型**: 面部、姿态、手部
- **关键点数量**: 
  - 面部: 468个
  - 姿态: 33个
  - 手部: 每只手21个
- **处理方式**: 
  - 本地视频: 逐帧处理
  - YouTube: 实时流处理
- **依赖库**:
  - OpenCV: 视频处理
  - MediaPipe: 人体关键点检测
  - yt-dlp: YouTube视频下载
  - ffmpeg: 视频流处理

## 文件说明

- `holistic.py`: 主程序，支持本地视频处理
- `test_video_demo.py`: 演示脚本，包含测试视频生成
- `youtube_holistic.py`: YouTube视频下载后处理
- `youtube_stream_holistic.py`: YouTube实时流处理（推荐）
- `requirements.txt`: 依赖包列表
- `README.md`: 项目说明文档

## 许可证

本项目基于MediaPipe开源项目，遵循相应的开源许可证。 