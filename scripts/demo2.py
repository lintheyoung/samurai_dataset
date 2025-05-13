# 导入部分：在文件前部添加导入语句
import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import requests
import tempfile
import shutil
import zipfile
import json
from urllib.parse import urlparse
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
# 在文件开头添加导入
import uuid
import re
# 导入修改后的数据集工具
from dataset_utils import save_frame_with_annotation, create_dataset_zip, save_video_settings, upload_to_r2, parallel_upload_to_r2
# 导入视频处理工具
from video_utils import process_video
# 导入数据集合并工具
from dataset_merge import merge_voc_datasets, zip_merged_dataset


color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def download_video(video_url):
    """从URL下载视频并返回本地临时文件路径"""
    try:
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        parsed_url = urlparse(video_url)
        file_name = os.path.basename(parsed_url.path)
        if not file_name.endswith('.mp4'):
            file_name = 'temp_video.mp4'
        
        temp_path = os.path.join(temp_dir, file_name)
        
        # 下载视频
        print(f"正在从 {video_url} 下载视频...")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()  # 确保请求成功
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"视频已下载到 {temp_path}")
        return temp_path
    except Exception as e:
        print(f"下载视频时出错: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise

def parse_bbox_params(bbox_str):
    """解析边界框参数字符串"""
    try:
        # 尝试分割参数字符串
        params = bbox_str.split(',')
        if len(params) == 4:
            # 如果是逗号分隔的字符串
            x, y, w, h = map(float, params)
        else:
            # 如果是空格分隔的字符串或单个参数
            params = bbox_str.split()
            if len(params) == 4:
                x, y, w, h = map(float, params)
            else:
                raise ValueError("边界框参数格式不正确")
        
        return int(x), int(y), int(w), int(h)
    except Exception as e:
        print(f"解析边界框参数时出错: {e}")
        raise ValueError(f"边界框参数格式不正确: {bbox_str}. 应该是四个数字，用逗号或空格分隔，例如 '79,85,66,108'")

def parse_multiple_bbox_params(bbox_list_str):
    """解析多个边界框参数字符串"""
    if not bbox_list_str.strip():
        return []
    bbox_list = bbox_list_str.split(';')
    return [parse_bbox_params(bbox) for bbox in bbox_list]

# 修改 parse_multiple_object_names 函数
def parse_multiple_object_names(object_names_str):
    """解析多个对象名称字符串"""
    if not object_names_str.strip():
        return []
    # 使用空格分隔对象名称
    return [name.strip() for name in object_names_str.split(',')]

def create_prompt_from_params(bbox_params):
    """从边界框参数创建提示字典"""
    x, y, w, h = bbox_params
    # 创建只有一帧(第0帧)的提示
    prompts = {0: ((x, y, x + w, y + h), 0)}
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("无法识别模型大小，路径中应包含large/base_plus/small/tiny")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("无效的视频路径格式。应为.mp4文件或图像帧目录。")

# 添加视频转换函数，使视频在网页端可播放
def convert_video_for_web(input_path):
    """
    转换视频格式，确保在web上可以播放
    """
    try:
        # 创建临时文件路径
        output_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        web_output_path = os.path.join(output_dir, f"{base_name}_web.mp4")
        
        # 尝试使用FFmpeg进行转换（如果安装了FFmpeg）
        try:
            import subprocess
            print(f"尝试使用FFmpeg转换视频为web格式...")
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-profile:v', 'baseline', '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y', web_output_path
            ]
            
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.returncode == 0:
                print(f"视频已成功转换为Web兼容格式: {web_output_path}")
                return web_output_path
            else:
                print(f"FFmpeg转换失败: {process.stderr.decode('utf-8')}")
                return input_path
        except (ImportError, FileNotFoundError):
            print("未安装FFmpeg或无法找到FFmpeg，跳过转换步骤")
            return input_path
    except Exception as e:
        print(f"视频转换过程中发生错误: {str(e)}")
        return input_path

# 添加将多个文件打包成zip的函数
def create_multiple_files_zip(files_dict, output_path):
    """
    将多个文件打包成一个zip文件
    :param files_dict: 字典，键为在zip中的文件名，值为文件路径
    :param output_path: 输出zip文件路径
    :return: 输出zip文件路径
    """
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for zip_path, file_path in files_dict.items():
                if os.path.exists(file_path):
                    zipf.write(file_path, zip_path)
                    print(f"已添加文件到zip: {file_path} -> {zip_path}")
                else:
                    print(f"警告: 文件不存在，跳过: {file_path}")
        
        print(f"已创建ZIP文件: {output_path}")
        return output_path
    except Exception as e:
        print(f"创建ZIP文件时出错: {str(e)}")
        return None

# 主函数修改部分，所有bbox渲染到同一个视频中
def main(args):
    # 如果提供了视频URL，下载视频
    if args.video.startswith("http"):
        video_path = download_video(args.video)
        # 确保在退出时删除临时文件
        temp_dir = os.path.dirname(video_path)
    else:
        # 如果是本地路径，直接使用
        video_path = args.video
        temp_dir = None
    
    try:
        # 解析多个边界框参数
        bbox_params_list = parse_multiple_bbox_params(args.txt)
        print(f"检测到 {len(bbox_params_list)} 个边界框参数")
        
        # 解析多个对象名称
        object_names_list = parse_multiple_object_names(args.object_name)
        print(f"检测到 {len(object_names_list)} 个对象名称")
        
        # 确保边界框和对象名称数量匹配
        if not object_names_list:
            # 如果没有提供对象名称列表，使用默认对象名称
            object_names_list = [args.object_name] * len(bbox_params_list)
            print(f"使用默认对象名称 '{args.object_name}' 应用于所有边界框")
        elif len(object_names_list) < len(bbox_params_list):
            # 如果对象名称不足，使用最后一个对象名称填充
            last_name = object_names_list[-1]
            object_names_list.extend([last_name] * (len(bbox_params_list) - len(object_names_list)))
            print(f"对象名称数量不足，使用最后一个名称 '{last_name}' 填充剩余边界框")
        elif len(object_names_list) > len(bbox_params_list):
            # 如果对象名称过多，截断
            object_names_list = object_names_list[:len(bbox_params_list)]
            print(f"对象名称数量过多，截断为与边界框数量相同")
        
        # 显示边界框和对象名称的对应关系
        for i, (bbox, obj_name) in enumerate(zip(bbox_params_list, object_names_list)):
            print(f"边界框 #{i+1}: {bbox} -> 对象名称: '{obj_name}'")
        
        # 如果启用了视频预处理
        if args.preprocess_video:
            print(f"开始预处理视频: 将分辨率降至{args.resolution}, 目标帧率: {args.target_fps if args.target_fps else '保持原帧率'}, 抽帧间隔: {args.frame_skip}")
            
            # 创建预处理视频的输出路径
            preprocessed_video_dir = os.path.join(os.path.dirname(args.output_dir), "preprocessed")
            os.makedirs(preprocessed_video_dir, exist_ok=True)
            
            # 生成随机UUID作为文件名一部分
            random_uuid = str(uuid.uuid4())[:8]
            preprocessed_video_path = os.path.join(preprocessed_video_dir, f"preprocessed_{random_uuid}.mp4")
            
            # 调用预处理函数
            video_path, frame_rate, width, height = process_video(
                input_video_path=video_path,
                output_video_path=preprocessed_video_path,
                resolution=args.resolution,
                target_fps=args.target_fps,
                frame_skip=args.frame_skip
            )
            print(f"视频预处理完成，保存到: {video_path}")
        
        # 加载视频帧，只需要加载一次
        loaded_frames = []
        # 获取视频帧率和尺寸
        if osp.isdir(video_path):
            frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
            frame_rate = 30  # 对于图像目录，默认帧率为30fps
        else:
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            
            if len(loaded_frames) == 0:
                raise ValueError("从视频中未加载到帧。")
            
            height, width = loaded_frames[0].shape[:2]
        
        # 创建随机的输出文件名
        base_random_uuid = str(uuid.uuid4())[:8]
        
        # 创建最终视频的输出路径
        final_video_output_path = args.video_output_path
        
        # 设置视频输出
        if args.save_to_video or args.upload_to_r2:
            # 尝试多种不同的编码器
            codecs_to_try = [
                ('mp4v', 'MPEG-4 编码器'),
                ('XVID', 'XVID 编码器'),
                ('MJPG', 'Motion JPEG 编码器'),
                ('H264', 'H.264 编码器')
            ]
            
            out = None
            for codec, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(final_video_output_path, fourcc, frame_rate, (width, height))
                    
                    if out.isOpened():
                        print(f"成功使用 {codec_name} 创建视频输出")
                        break
                    else:
                        out.release()
                        print(f"无法使用 {codec_name} 创建视频")
                except Exception as e:
                    print(f"尝试使用 {codec_name} 时出错: {str(e)}")
            
            if out is None or not out.isOpened():
                print("警告: 无法创建视频输出，尝试使用最基本的编码器")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 最基本的编码器
                out = cv2.VideoWriter(final_video_output_path, fourcc, frame_rate, (width, height))
                
                if not out.isOpened():
                    raise ValueError("无法创建视频输出，请检查OpenCV安装和编码器支持")
        
        # 初始化数据集目录
        dataset_dirs = []
        
        # 准备frames_or_path，使SAM2能处理视频
        frames_or_path = prepare_frames_or_path(video_path)
        
        # 加载模型（只需要加载一次）
        model_cfg = determine_model_cfg(args.model_path)
        predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
        
        # 创建一个数据结构来存储每一帧的所有bbox处理结果
        frame_results = [[] for _ in range(len(loaded_frames))]
        
        # 为每个边界框单独处理，但不单独创建视频
        for bbox_idx, (bbox_params, object_name) in enumerate(zip(bbox_params_list, object_names_list)):
            print(f"\n处理边界框 #{bbox_idx+1}/{len(bbox_params_list)}: {bbox_params} (对象: {object_name})")
            
            # 清理对象名称，确保文件名安全
            safe_object_name = re.sub(r'[^\w\-_]', '_', object_name)
            
            # 为当前bbox创建特定的输出目录（用于数据集生成）
            current_output_dir = os.path.join(args.output_dir, f"bbox_{bbox_idx+1}_{safe_object_name}")
            os.makedirs(current_output_dir, exist_ok=True)
            dataset_dirs.append(current_output_dir)
            
            # 保存视频设置
            save_video_settings(current_output_dir, width, height, frame_rate, len(loaded_frames))
            
            # 使用当前边界框参数创建提示
            prompts = create_prompt_from_params(bbox_params)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
                bbox, track_label = prompts[0]
                _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
                
                # 追踪视频中的物体
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    mask_to_vis = {}
                    bbox_to_vis = {}
                    
                    # 处理每个对象的掩码
                    for obj_id, mask in zip(object_ids, masks):
                        mask = mask[0].cpu().numpy()
                        mask = mask > 0.0
                        non_zero_indices = np.argwhere(mask)
                        
                        # 只有在有非零像素时才添加边界框和掩码
                        if len(non_zero_indices) > 0:  # 只在有物体时处理
                            y_min, x_min = non_zero_indices.min(axis=0).tolist()
                            y_max, x_max = non_zero_indices.max(axis=0).tolist()
                            box = [x_min, y_min, x_max - x_min, y_max - y_min]
                            bbox_to_vis[obj_id] = box
                            mask_to_vis[obj_id] = mask
                    
                    # 获取当前帧
                    img = loaded_frames[frame_idx].copy()
                    
                    # 提取边界框列表用于标注
                    bbox_list = [bbox for obj_id, bbox in bbox_to_vis.items()]
                    
                    # 保存帧和XML标注
                    if args.generate_dataset:
                        save_frame_with_annotation(
                            frame=img, 
                            frame_idx=frame_idx, 
                            output_dir=current_output_dir, 
                            bbox_list=bbox_list,
                            object_name=object_name  # 使用对应的对象名称
                        )
                    
                    # 存储当前帧的边界框、掩码和对象名称信息
                    if bbox_to_vis:
                        frame_results[frame_idx].append({
                            'bbox_idx': bbox_idx,
                            'object_name': object_name,
                            'bboxes': bbox_to_vis,
                            'masks': mask_to_vis,
                            'color_idx': bbox_idx % len(color)  # 为每个边界框使用不同的颜色
                        })
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # 清理SAM2模型资源
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        
        # 创建数据集ZIP文件列表
        dataset_zip_paths = []
        
        # 为每个bbox生成数据集ZIP文件
        if args.generate_dataset:
            for bbox_idx, (output_dir, object_name) in enumerate(zip(dataset_dirs, object_names_list)):
                # 清理对象名称，确保文件名安全
                safe_object_name = re.sub(r'[^\w\-_]', '_', object_name)
                
                # 如果指定了zip输出路径，为每个bbox创建唯一的zip名称
                zip_output_path = None
                if args.zip_output_path:
                    zip_dir = os.path.dirname(args.zip_output_path)
                    zip_base = os.path.splitext(os.path.basename(args.zip_output_path))[0]
                    zip_ext = os.path.splitext(args.zip_output_path)[1]
                    zip_output_path = os.path.join(zip_dir, f"{zip_base}_{safe_object_name}_{bbox_idx+1}{zip_ext}")
                
                # 创建数据集ZIP文件
                current_zip_path = create_dataset_zip(
                    output_dir, 
                    object_name,
                    zip_output_path
                )
                
                if current_zip_path:
                    print(f"边界框 #{bbox_idx+1} ({object_name}) 的数据集ZIP文件已创建在: {current_zip_path}")
                    dataset_zip_paths.append(current_zip_path)
                    
        # 合并数据集（如果启用）
        merged_dataset_zip_path = None
        if args.merge_datasets and args.generate_dataset and len(dataset_dirs) > 1:
            print("\n开始合并数据集...")
            
            # 设置合并数据集的输出目录
            merged_output_dir = args.merged_output_dir if args.merged_output_dir else os.path.join(args.output_dir, "merged_output")
            os.makedirs(merged_output_dir, exist_ok=True)
            
            # 合并数据集
            merged_path = merge_voc_datasets(args.output_dir, args.merged_tag, merged_output_dir)
            
            if merged_path:
                # 设置合并数据集的ZIP输出路径
                if args.merged_zip_output:
                    merged_zip_path = args.merged_zip_output
                else:
                    merged_zip_dir = os.path.dirname(args.output_dir)
                    merged_zip_name = f"merged_dataset_{base_random_uuid}.zip"
                    merged_zip_path = os.path.join(merged_zip_dir, merged_zip_name)
                
                # 打包合并数据集
                merged_dataset_zip_path = zip_merged_dataset(merged_path, merged_zip_path)
                
                if merged_dataset_zip_path:
                    print(f"合并数据集ZIP文件已创建在: {merged_dataset_zip_path}")
            else:
                print("警告: 数据集合并失败，将使用原始数据集ZIP文件")
        
        # 生成最终的视频，包含所有bbox
        if args.save_to_video or args.upload_to_r2:
            print(f"开始生成最终视频，包含所有 {len(bbox_params_list)} 个边界框...")
            
            for frame_idx, frame_result in enumerate(frame_results):
                if frame_idx % 100 == 0:
                    print(f"处理帧 {frame_idx}/{len(frame_results)}...")
                
                # 获取原始帧
                original_frame = loaded_frames[frame_idx].copy()
                
                # 如果当前帧没有检测到任何bbox，直接写入原始帧
                if not frame_result:
                    out.write(original_frame)
                    continue
                
                # 渲染所有bbox到当前帧
                for result in frame_result:
                    bbox_idx = result['bbox_idx']
                    object_name = result['object_name']
                    bboxes = result['bboxes']
                    masks = result['masks']
                    color_idx = result['color_idx']
                    
                    # 绘制掩码
                    for obj_id, mask in masks.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = color[color_idx]
                        original_frame = cv2.addWeighted(original_frame, 1, mask_img, 0.2, 0)
                    
                    # 绘制边界框
                    for obj_id, bbox in bboxes.items():
                        cv2.rectangle(original_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                    color[color_idx], 2)
                        
                        # 添加对象名称标签
                        label_position = (bbox[0], bbox[1] - 10 if bbox[1] > 20 else bbox[1] + bbox[3] + 10)
                        cv2.putText(original_frame, object_name, label_position, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[color_idx], 2)
                
                # 将当前帧写入视频
                out.write(original_frame)
            
            # 关闭视频输出
            out.release()
            print(f"最终视频已生成: {final_video_output_path}")
        
        # 创建所有数据集的合并ZIP (只在没有启用merge_datasets时创建)
        all_datasets_zip_path = None
        if dataset_zip_paths and args.generate_dataset and not args.merge_datasets:
            all_datasets_zip_dir = os.path.dirname(args.output_dir)
            all_datasets_zip_name = f"all_datasets_{base_random_uuid}.zip"
            all_datasets_zip_path = os.path.join(all_datasets_zip_dir, all_datasets_zip_name)
            
            # 创建要打包的文件字典
            dataset_files_dict = {
                f"dataset_{object_names_list[idx]}_{idx+1}.zip": path 
                for idx, path in enumerate(dataset_zip_paths)
            }
            
            all_datasets_zip_path = create_multiple_files_zip(dataset_files_dict, all_datasets_zip_path)
            print(f"所有数据集的合并ZIP文件已创建在: {all_datasets_zip_path}")
        
        # 准备上传文件
        files_to_upload = {}
        
        # 转换最终视频为Web兼容格式并准备上传
        if args.upload_to_r2 and os.path.exists(final_video_output_path):
            web_video_path = convert_video_for_web(final_video_output_path)
            files_to_upload['final_video'] = web_video_path
        
        # 添加数据集ZIP到上传列表
        # 如果启用了数据集合并并且合并成功，则仅上传合并后的数据集，否则上传所有数据集的打包文件
        if args.upload_to_r2:
            if args.merge_datasets and merged_dataset_zip_path:
                # 仅上传合并后的数据集
                files_to_upload['dataset'] = merged_dataset_zip_path
                print("仅上传合并后的数据集到R2")
            elif all_datasets_zip_path:
                # 上传所有数据集的打包文件（合并失败或未启用合并功能）
                files_to_upload['dataset'] = all_datasets_zip_path
                print("上传所有原始数据集的打包文件到R2")
        
        # 并行上传文件到R2
        urls = {}
        if files_to_upload and args.upload_to_r2:
            print("开始并行上传文件到R2...")
            urls = parallel_upload_to_r2(files_to_upload)
        
        # 最终结果
        final_result = {
            "message": f"成功处理了 {len(bbox_params_list)} 个边界框并生成了包含所有边界框的单个视频",
            "video_output_path": final_video_output_path if args.save_to_video or args.upload_to_r2 else None
        }
        
        # 添加数据集路径到结果
        if args.merge_datasets and merged_dataset_zip_path:
            final_result["merged_dataset_zip_path"] = merged_dataset_zip_path
        elif all_datasets_zip_path:
            final_result["all_datasets_zip_path"] = all_datasets_zip_path
        
        # 添加R2链接到结果
        if urls:
            final_result["upload_urls"] = urls
            
            # 添加数据集下载链接
            if 'dataset' in urls:
                if args.merge_datasets:
                    final_result["merged_dataset_download_url"] = urls['dataset']
                else:
                    final_result["dataset_download_url"] = urls['dataset']
                
            # 添加视频下载链接
            if 'final_video' in urls:
                final_result["video_download_url"] = urls['final_video']
        
        print("最终结果:", json.dumps(final_result, indent=2))
        return final_result
    
    finally:
        # 确保清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"已删除临时目录: {temp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改参数名称和帮助文本
    parser.add_argument("--video", required=True, help="视频URL或本地视频路径或帧目录路径。")
    parser.add_argument("--txt", required=True, help="边界框参数，格式为'x,y,width,height'。多个边界框用分号(;)分隔，例如: '79,85,66,108;120,200,80,90'")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="模型检查点路径。")
    parser.add_argument("--video_output_path", default="demo.mp4", help="输出视频保存路径。")
    parser.add_argument("--save_to_video", action="store_true", help="将结果保存为视频。")
    # 添加数据集生成相关参数
    parser.add_argument("--generate_dataset", action="store_true", help="生成VOC格式数据集。")
    parser.add_argument("--output_dir", default="output", help="保存数据集文件的目录。")
    parser.add_argument("--zip_output_path", default=None, help="输出ZIP文件的保存路径。")
    parser.add_argument("--object_name", default="object", help="要标注的对象名称。多个对象名称用逗号分隔，例如: 'car,person,dog'")
    # 添加R2上传相关参数
    parser.add_argument("--upload_to_r2", action="store_true", help="将ZIP文件和视频上传到Cloudflare R2。")
    
    # 添加视频预处理相关参数
    parser.add_argument("--preprocess_video", action="store_true", help="启用视频预处理（降低分辨率和抽帧）")
    parser.add_argument("--resolution", default="720P", choices=["480P", "720P", "1080P"], help="目标视频分辨率")
    parser.add_argument("--target_fps", type=float, default=None, help="目标视频帧率，如10fps")
    parser.add_argument("--frame_skip", type=int, default=None, help="抽帧间隔，每N帧保留一帧（当不设置target_fps时使用）")
    
    # 添加数据集合并相关参数
    parser.add_argument("--merge_datasets", action="store_true", help="合并生成的多个数据集，并且仅上传合并后的数据集")
    parser.add_argument("--merged_tag", default="merged", help="合并数据集的自定义标签")
    parser.add_argument("--merged_output_dir", default=None, help="合并数据集的输出目录，默认使用output_dir")
    parser.add_argument("--merged_zip_output", default=None, help="合并数据集的ZIP输出路径")
    
    args = parser.parse_args()
    
    main(args)