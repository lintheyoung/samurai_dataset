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
from urllib.parse import urlparse
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
# 导入数据集工具
from dataset_utils import save_frame_with_annotation, create_dataset_zip, save_video_settings, upload_to_r2

color = [(255, 0, 0)]

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

def create_prompt_from_params(bbox_params):
    """从边界框参数创建提示字典"""
    x, y, w, h = parse_bbox_params(bbox_params)
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
        model_cfg = determine_model_cfg(args.model_path)
        predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
        frames_or_path = prepare_frames_or_path(video_path)
        
        # 使用边界框参数创建提示，而不是从文件加载
        prompts = create_prompt_from_params(args.txt)
        
        # 创建输出目录
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        frame_rate = 30
        loaded_frames = []
        
        # 加载视频帧
        if osp.isdir(video_path):
            frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]
            
            if len(loaded_frames) == 0:
                raise ValueError("从视频中未加载到帧。")
        
        # 保存视频设置
        save_video_settings(output_dir, width, height, frame_rate, len(loaded_frames))
        
        # 设置视频输出（如果需要）
        if args.save_to_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))
        
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
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    bbox_to_vis[obj_id] = bbox
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
                        output_dir=output_dir, 
                        bbox_list=bbox_list,
                        object_name=args.object_name
                    )
                
                # 可视化处理
                if args.save_to_video:
                    vis_img = img.copy()
                    # 绘制掩码
                    for obj_id, mask in mask_to_vis.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = color[(obj_id + 1) % len(color)]
                        vis_img = cv2.addWeighted(vis_img, 1, mask_img, 0.2, 0)
                    
                    # 绘制边界框
                    for obj_id, bbox in bbox_to_vis.items():
                        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)
                    
                    out.write(vis_img)
            
            # 关闭视频输出
            if args.save_to_video:
                out.release()
            
            # 生成数据集ZIP文件
            zip_url = None
            if args.generate_dataset:
                zip_path = create_dataset_zip(output_dir, args.object_name, args.zip_output_path)
                print(f"数据集ZIP文件已创建在: {zip_path}")
                
                # 上传ZIP文件到Cloudflare R2
                if args.upload_to_r2:
                    zip_url = upload_to_r2(zip_path)
                    if zip_url:
                        print(f"数据集ZIP文件已上传到Cloudflare R2。下载URL: {zip_url}")
                    else:
                        print("上传ZIP文件到Cloudflare R2失败。")
        
        # 清理资源
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        
        # 返回结果
        result = {
            "message": "处理已成功完成",
        }
        
        if args.save_to_video:
            result["video_output_path"] = args.video_output_path
        
        if args.generate_dataset:
            result["dataset_zip_path"] = zip_path
            
            if zip_url:
                result["dataset_download_url"] = zip_url
        
        print("结果:", result)
        return result
    
    finally:
        # 确保清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"已删除临时目录: {temp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改参数名称和帮助文本
    parser.add_argument("--video", required=True, help="视频URL或本地视频路径或帧目录路径。")
    parser.add_argument("--txt", required=True, help="边界框参数，格式为'x,y,width,height'。")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="模型检查点路径。")
    parser.add_argument("--video_output_path", default="demo.mp4", help="输出视频保存路径。")
    parser.add_argument("--save_to_video", action="store_true", help="将结果保存为视频。")
    # 添加数据集生成相关参数
    parser.add_argument("--generate_dataset", action="store_true", help="生成VOC格式数据集。")
    parser.add_argument("--output_dir", default="output", help="保存数据集文件的目录。")
    parser.add_argument("--zip_output_path", default=None, help="输出ZIP文件的保存路径。")
    parser.add_argument("--object_name", default="object", help="要标注的对象名称。")
    # 添加R2上传相关参数
    parser.add_argument("--upload_to_r2", action="store_true", help="将ZIP文件上传到Cloudflare R2。")
    args = parser.parse_args()
    
    main(args)