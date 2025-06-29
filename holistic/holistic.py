import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def process_video(video_path):
    """
    处理视频文件，检测人体关键点并显示结果
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS")
    
    # 初始化Holistic模型
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as holistic:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("视频读取完成或出错")
                break
            
            # 为了提高性能，将图像标记为不可写
            frame.flags.writeable = False
            # 将BGR图像转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理图像
            results = holistic.process(rgb_frame)
            
            # 将图像标记为可写
            frame.flags.writeable = True
            # 将RGB图像转换回BGR
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # 绘制面部关键点
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            
            # 绘制姿态关键点
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                
                # 显示鼻子坐标信息
                nose_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * width
                nose_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * height
                cv2.putText(frame, f'Nose: ({int(nose_x)}, {int(nose_y)})', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制左手关键点
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_hand_connections_style())
            
            # 绘制右手关键点
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_hand_connections_style())
            
            # 显示处理后的帧
            cv2.imshow('MediaPipe Holistic - 视频处理', frame)
            
            # 按ESC键退出，按空格键暂停/继续
            key = cv2.waitKey(1000 // fps) & 0xFF
            if key == 27:  # ESC键
                break
            elif key == 32:  # 空格键
                cv2.waitKey(0)  # 暂停直到按任意键
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 主程序
if __name__ == "__main__":
    # 指定视频文件路径
    video_path = input("请输入视频文件路径 (或按回车使用默认路径): ").strip()
    
    if not video_path:
        video_path = "test_video.mp4"  # 默认视频文件名
    
    print(f"开始处理视频: {video_path}")
    print("按ESC键退出，按空格键暂停/继续")
    
    try:
        process_video(video_path)
    except Exception as e:
        print(f"处理视频时出错: {e}")

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
      print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose, left and right hands, and face landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

""" # For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release() """