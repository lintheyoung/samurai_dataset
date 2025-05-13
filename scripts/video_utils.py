import os
import cv2
import numpy as np

def process_video(input_video_path, output_video_path, resolution="720P", target_fps=None, frame_skip=None):
    """
    处理视频：降低分辨率和抽帧（仅使用OpenCV实现）
    
    参数:
    input_video_path - 输入视频路径
    output_video_path - 输出视频路径
    resolution - 目标分辨率，可选值："480P", "720P", "1080P"
    target_fps - 目标帧率，如果指定将按此帧率抽帧
    frame_skip - 抽帧间隔，每frame_skip帧保留一帧（当target_fps未指定时使用）
    
    返回:
    处理后的视频路径、帧率、视频宽度、视频高度
    """
    # 读取原始视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{input_video_path}")
    
    # 获取原始视频属性
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 根据选择的分辨率设置长边的最大长度
    if resolution == "480P":
        max_length = 480
    elif resolution == "720P":
        max_length = 720
    elif resolution == "1080P":
        max_length = 1080
    else:
        max_length = 720  # 默认值
    
    # 判断视频是横屏还是竖屏
    is_landscape = orig_width >= orig_height
    
    # 按照长边计算新尺寸，保持原始宽高比
    if is_landscape:
        # 横屏，宽度是长边
        new_width = max_length
        new_height = int(orig_height * (new_width / orig_width))
    else:
        # 竖屏，高度是长边
        new_height = max_length
        new_width = int(orig_width * (new_height / orig_height))
    
    # 如果计算的新尺寸超过原始尺寸，则使用原始尺寸
    if new_width > orig_width or new_height > orig_height:
        new_width = orig_width
        new_height = orig_height
    
    # 计算抽帧策略
    actual_frame_skip = 1  # 默认不抽帧
    final_fps = orig_fps  # 默认保持原帧率
    
    # 如果指定了目标帧率
    if target_fps is not None and target_fps > 0:
        if target_fps >= orig_fps:
            # 如果目标帧率大于等于原始帧率，不需要抽帧
            print(f"目标帧率 {target_fps} fps 大于等于原始帧率 {orig_fps} fps，不会进行抽帧")
            actual_frame_skip = 1
            final_fps = orig_fps
        else:
            # 根据目标帧率计算跳帧数
            actual_frame_skip = max(1, round(orig_fps / target_fps))
            final_fps = orig_fps / actual_frame_skip
            print(f"根据目标帧率 {target_fps} fps 计算跳帧数: {actual_frame_skip}，实际帧率约为 {final_fps:.2f} fps")
    # 如果未指定目标帧率但指定了跳帧数
    elif frame_skip is not None and frame_skip > 1:
        actual_frame_skip = frame_skip
        final_fps = orig_fps / actual_frame_skip
        print(f"使用指定的跳帧数: {actual_frame_skip}，帧率将从 {orig_fps} fps 降至 {final_fps:.2f} fps")
    
    print(f"处理视频: 原始分辨率 {orig_width}x{orig_height}, 新分辨率 {new_width}x{new_height}")
    print(f"原始帧率: {orig_fps} fps, 处理后帧率: {final_fps:.2f} fps")
    print(f"原始总帧数: {total_frames}, 预计处理后帧数: {total_frames / actual_frame_skip:.0f}")
    
    # 创建视频写入器
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, final_fps, (new_width, new_height))
        if not out.isOpened():
            raise ValueError("无法创建输出视频文件")
    except Exception as e:
        print(f"创建视频写入器时发生错误: {str(e)}")
        # 尝试使用其他编码器
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, final_fps, (new_width, new_height))
        except Exception as e2:
            print(f"尝试备用编码器时发生错误: {str(e2)}")
            raise ValueError("无法创建视频写入器，请检查OpenCV安装和编码器支持")
    
    # 处理视频帧
    frame_count = 0
    processed_frames_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保留第一帧或者每actual_frame_skip帧保留一帧
        if frame_count == 0 or frame_count % actual_frame_skip == 0:
            # 调整帧的分辨率
            resized_frame = cv2.resize(frame, (new_width, new_height))
            out.write(resized_frame)
            processed_frames_count += 1
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"视频处理完成: 处理前 {frame_count} 帧, 处理后 {processed_frames_count} 帧")
    
    return output_video_path, final_fps, new_width, new_height