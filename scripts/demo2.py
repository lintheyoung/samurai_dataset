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
import random
import time
from xml.etree import ElementTree
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

# 新增函数：根据分辨率变化按比例调整边界框参数
def scale_bbox_params(bbox_params_list, original_width, original_height, new_width, new_height):
    """
    根据分辨率变化按比例调整边界框参数
    :param bbox_params_list: 原始边界框参数列表 [(x, y, w, h), ...]
    :param original_width: 原始视频宽度
    :param original_height: 原始视频高度
    :param new_width: 新视频宽度
    :param new_height: 新视频高度
    :return: 调整后的边界框参数列表
    """
    width_ratio = new_width / original_width
    height_ratio = new_height / original_height
    
    print(f"应用分辨率缩放比例 - 宽度: {width_ratio:.4f}, 高度: {height_ratio:.4f}")
    
    scaled_bbox_params = []
    for x, y, w, h in bbox_params_list:
        new_x = int(x * width_ratio)
        new_y = int(y * height_ratio)
        new_w = int(w * width_ratio)
        new_h = int(h * height_ratio)
        scaled_bbox_params.append((new_x, new_y, new_w, new_h))
        print(f"调整边界框: 原始 ({x}, {y}, {w}, {h}) -> 新 ({new_x}, {new_y}, {new_w}, {new_h})")
    
    return scaled_bbox_params

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

# 添加用于递归查找特定类型文件的辅助函数
def find_files_recursive(directory, file_extension):
    """
    递归查找指定目录下所有特定扩展名的文件
    :param directory: 目录路径
    :param file_extension: 文件扩展名（如 '.xml'）
    :return: 找到的文件路径列表
    """
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(file_extension.lower()):
                    result.append(os.path.join(root, file))
    except Exception as e:
        print(f"递归查找文件时出错: {str(e)}")
    return result

# 打印目录结构的辅助函数
def print_directory_structure(directory, indent=0):
    """
    打印目录结构，便于调试
    :param directory: 要打印的目录
    :param indent: 缩进级别
    """
    try:
        print('  ' * indent + '+--' + os.path.basename(directory) + '/')
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                print_directory_structure(item_path, indent + 1)
            else:
                print('  ' * (indent + 1) + '+--' + item)
    except Exception as e:
        print(f"{'  ' * indent}无法访问目录 {directory}: {str(e)}")

# 打印ZIP文件内容的辅助函数
def print_zip_contents(zip_path):
    """
    打印ZIP文件中的内容列表，便于调试
    :param zip_path: ZIP文件路径
    """
    try:
        if not os.path.exists(zip_path):
            print(f"ZIP文件不存在: {zip_path}")
            return
        
        print(f"ZIP文件大小: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
        print(f"ZIP文件内容列表: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                print(f"  {file_info.filename} (大小: {file_info.file_size} 字节)")
    except Exception as e:
        print(f"打印ZIP文件内容时出错: {str(e)}")

# 添加VOC到YOLO转换功能，修改以支持不同的目录结构
def convert_voc_to_yolo(voc_dataset_path, output_yolo_path, zip_output_path=None):
    """
    将VOC格式数据集转换为YOLO格式，支持不同的目录结构
    :param voc_dataset_path: VOC数据集路径
    :param output_yolo_path: YOLO输出路径
    :param zip_output_path: YOLO ZIP文件输出路径（可选）
    :return: YOLO数据集ZIP文件路径
    """
    try:
        print(f"开始将VOC数据集转换为YOLO格式...")
        print(f"VOC数据集路径: {voc_dataset_path}")
        
        # 打印输入目录结构
        print(f"VOC数据集目录结构:")
        print_directory_structure(voc_dataset_path)
        
        # 创建YOLO数据集目录
        os.makedirs(output_yolo_path, exist_ok=True)
        images_dir = os.path.join(output_yolo_path, "images")
        labels_dir = os.path.join(output_yolo_path, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 递归查找所有XML文件
        all_xml_files = find_files_recursive(voc_dataset_path, '.xml')
        print(f"递归查找到 {len(all_xml_files)} 个XML文件")
        
        if not all_xml_files:
            print(f"错误：在 {voc_dataset_path} 及其子目录中未找到任何XML文件")
            return None
        
        # 尝试找出包含大多数XML文件的目录作为注释目录
        xml_dirs = {}
        for xml_path in all_xml_files:
            xml_dir = os.path.dirname(xml_path)
            xml_dirs[xml_dir] = xml_dirs.get(xml_dir, 0) + 1
        
        # 按文件数量排序目录
        sorted_xml_dirs = sorted(xml_dirs.items(), key=lambda x: x[1], reverse=True)
        annotations_dir = sorted_xml_dirs[0][0]
        print(f"选择的注释目录: {annotations_dir}，包含 {sorted_xml_dirs[0][1]} 个XML文件")
        
        # 递归查找所有图像文件
        all_image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            all_image_files.extend(find_files_recursive(voc_dataset_path, ext))
        print(f"递归查找到 {len(all_image_files)} 个图像文件")
        
        if not all_image_files:
            print(f"错误：在 {voc_dataset_path} 及其子目录中未找到任何图像文件")
            return None
        
        # 尝试找出包含大多数图像文件的目录作为图像目录
        img_dirs = {}
        for img_path in all_image_files:
            img_dir = os.path.dirname(img_path)
            img_dirs[img_dir] = img_dirs.get(img_dir, 0) + 1
        
        # 按文件数量排序目录
        sorted_img_dirs = sorted(img_dirs.items(), key=lambda x: x[1], reverse=True)
        jpegimages_dir = sorted_img_dirs[0][0]
        print(f"选择的图像目录: {jpegimages_dir}，包含 {sorted_img_dirs[0][1]} 个图像文件")
        
        # 创建文件名到路径的映射，以便快速查找
        image_file_map = {}
        for img_path in all_image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image_file_map[base_name] = img_path
        
        # 获取XML文件列表
        xml_files = [os.path.basename(f) for f in find_files_recursive(annotations_dir, '.xml')]
        
        if not xml_files:
            print(f"警告：在 {annotations_dir} 中未找到XML文件")
            return None
        
        print(f"找到 {len(xml_files)} 个XML文件进行处理")
        
        # 收集所有类别名称
        class_names = set()
        for xml_file in xml_files:
            xml_path = os.path.join(annotations_dir, xml_file)
            try:
                tree = ElementTree.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('.//object'):
                    name = obj.find('name').text
                    class_names.add(name)
            except Exception as e:
                print(f"解析 {xml_file} 时出错: {str(e)}")
        
        # 对类别名称排序以保持一致性
        class_names = sorted(list(class_names))
        print(f"找到以下类别: {class_names}")
        
        # 创建YOLO格式所需的names文件
        with open(os.path.join(output_yolo_path, "classes.names"), "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
                
        # 创建数据配置文件
        with open(os.path.join(output_yolo_path, "data.yaml"), "w") as f:
            f.write(f"train: ./train.txt\n")
            f.write(f"val: ./val.txt\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")
        
        # 开始转换
        converted_count = 0
        failed_count = 0
        for xml_file in xml_files:
            try:
                base_name = os.path.splitext(xml_file)[0]
                
                # 查找对应的图像文件
                img_path = image_file_map.get(base_name)
                
                if not img_path:
                    # 尝试其他可能的匹配
                    potential_matches = [k for k in image_file_map.keys() if k.startswith(base_name) or base_name.startswith(k)]
                    if potential_matches:
                        img_path = image_file_map[potential_matches[0]]
                        print(f"找到近似匹配: {base_name} -> {potential_matches[0]}")
                    else:
                        print(f"警告：找不到与 {xml_file} 对应的图像文件")
                        failed_count += 1
                        continue
                
                # 复制图像文件
                dest_img_path = os.path.join(images_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dest_img_path)
                
                # 解析XML文件
                xml_path = os.path.join(annotations_dir, xml_file)
                tree = ElementTree.parse(xml_path)
                root = tree.getroot()
                
                # 获取图像尺寸
                size_elem = root.find('.//size')
                if size_elem is not None:
                    width = int(size_elem.find('width').text)
                    height = int(size_elem.find('height').text)
                else:
                    # 如果XML中没有尺寸信息，从图像文件中获取
                    img = cv2.imread(img_path)
                    height, width = img.shape[:2]
                
                # 创建YOLO格式标注文件
                yolo_txt_path = os.path.join(labels_dir, base_name + ".txt")
                
                with open(yolo_txt_path, "w") as yolo_file:
                    objects_processed = 0
                    for obj in root.findall('.//object'):
                        try:
                            name = obj.find('name').text
                            class_id = class_names.index(name)
                            
                            bbox = obj.find('bndbox')
                            xmin = float(bbox.find('xmin').text)
                            ymin = float(bbox.find('ymin').text)
                            xmax = float(bbox.find('xmax').text)
                            ymax = float(bbox.find('ymax').text)
                            
                            # 转换为YOLO格式（中心点坐标和宽高，归一化为0-1）
                            x_center = (xmin + xmax) / 2.0 / width
                            y_center = (ymin + ymax) / 2.0 / height
                            w = (xmax - xmin) / width
                            h = (ymax - ymin) / height
                            
                            # 限制值在0-1范围内
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            w = max(0, min(1, w))
                            h = max(0, min(1, h))
                            
                            # 写入YOLO格式
                            yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                            objects_processed += 1
                        except Exception as e:
                            print(f"处理 {xml_file} 中的对象时出错: {str(e)}")
                    
                    if objects_processed == 0:
                        print(f"警告: {xml_file} 中没有处理任何对象")
                
                converted_count += 1
                if converted_count % 50 == 0 or converted_count == len(xml_files):
                    print(f"已处理 {converted_count}/{len(xml_files)} 个文件")
            except Exception as e:
                print(f"转换 {xml_file} 时出错: {str(e)}")
                failed_count += 1
        
        print(f"转换完成: 成功 {converted_count}/{len(xml_files)} 个文件, 失败 {failed_count} 个文件")
        
        # 创建训练集和验证集列表
        all_images = []
        for f in os.listdir(images_dir):
            if any(f.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"]):
                all_images.append(os.path.splitext(f)[0])
        
        if not all_images:
            print(f"警告: 未找到任何图像文件在 {images_dir} 中")
            return None
        
        # 随机分割训练集和验证集
        random.shuffle(all_images)
        val_count = max(1, int(len(all_images) * 0.2))  # 20%作为验证集
        val_set = all_images[:val_count]
        train_set = all_images[val_count:]
        
        print(f"划分数据集: 训练集 {len(train_set)} 个样本, 验证集 {len(val_set)} 个样本")
        
        # 获取图像文件的扩展名
        image_extensions = {}
        for img_file in os.listdir(images_dir):
            if any(img_file.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"]):
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                image_extensions[base_name] = ext
        
        # 创建训练集列表文件
        with open(os.path.join(output_yolo_path, "train.txt"), "w") as f:
            for name in train_set:
                ext = image_extensions.get(name, ".jpg")
                f.write(f"./images/{name}{ext}\n")
        
        # 创建验证集列表文件
        with open(os.path.join(output_yolo_path, "val.txt"), "w") as f:
            for name in val_set:
                ext = image_extensions.get(name, ".jpg")
                f.write(f"./images/{name}{ext}\n")
        
        # 打包为ZIP文件
        if zip_output_path:
            output_zip_path = zip_output_path
        else:
            output_zip_path = os.path.join(os.path.dirname(output_yolo_path), f"yolo_dataset_{str(uuid.uuid4())[:8]}.zip")
        
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_yolo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(output_yolo_path))
                    zipf.write(file_path, arcname)
        
        print(f"YOLO数据集已创建并打包为: {output_zip_path}")
        print(f"YOLO数据集ZIP文件大小: {os.path.getsize(output_zip_path) / (1024*1024):.2f} MB")
        return output_zip_path
    
    except Exception as e:
        print(f"转换VOC到YOLO数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 获取视频分辨率的函数
def get_video_resolution(video_path):
    """
    获取视频的分辨率
    :param video_path: 视频文件路径
    :return: (width, height) 元组，视频的宽度和高度
    """
    try:
        if os.path.isdir(video_path):
            # 如果是图像目录，获取第一个图像的大小
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                image_files.extend([f for f in os.listdir(video_path) if f.endswith(ext)])
            
            if not image_files:
                raise ValueError(f"目录 {video_path} 中没有找到图像文件")
            
            first_image_path = os.path.join(video_path, sorted(image_files)[0])
            img = cv2.imread(first_image_path)
            if img is None:
                raise ValueError(f"无法读取图像 {first_image_path}")
            
            height, width = img.shape[:2]
        else:
            # 如果是视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频 {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        
        return width, height
    except Exception as e:
        print(f"获取视频分辨率时出错: {str(e)}")
        raise

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
        
        # 在预处理视频前获取原始分辨率
        original_width, original_height = get_video_resolution(video_path)
        print(f"原始视频分辨率: {original_width}x{original_height}")
        
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
            video_path, frame_rate, new_width, new_height = process_video(
                input_video_path=video_path,
                output_video_path=preprocessed_video_path,
                resolution=args.resolution,
                target_fps=args.target_fps,
                frame_skip=args.frame_skip
            )
            print(f"视频预处理完成，保存到: {video_path}")
            print(f"新视频分辨率: {new_width}x{new_height}")
            
            # 根据分辨率变化按比例调整边界框参数
            bbox_params_list = scale_bbox_params(
                bbox_params_list, 
                original_width, 
                original_height, 
                new_width, 
                new_height
            )
        
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
        if args.merge_datasets and args.generate_dataset:
            print("\n开始处理数据集...")
            
            # 设置合并数据集的输出目录
            merged_output_dir = args.merged_output_dir if args.merged_output_dir else os.path.join(args.output_dir, "merged_output")
            os.makedirs(merged_output_dir, exist_ok=True)
            
            # 根据数据集数量选择处理方式
            if len(dataset_dirs) > 1:
                # 多数据集情况 - 执行合并
                print(f"合并 {len(dataset_dirs)} 个数据集...")
                merged_path = merge_voc_datasets(args.output_dir, args.merged_tag, merged_output_dir)
            else:
                # 单数据集情况 - 直接复制
                print("只有一个边界框数据集，直接复制为合并结果...")
                merged_path = merged_output_dir
                # 清理目标目录中的任何现有内容
                if os.path.exists(merged_path):
                    for item in os.listdir(merged_path):
                        item_path = os.path.join(merged_path, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                
                # 从单个数据集目录复制到合并目录
                source_dir = dataset_dirs[0]
                for item in os.listdir(source_dir):
                    source_path = os.path.join(source_dir, item)
                    dest_path = os.path.join(merged_path, item)
                    if os.path.isdir(source_path):
                        shutil.copytree(source_path, dest_path)
                    else:
                        shutil.copy2(source_path, dest_path)
                print(f"已复制数据集从 {source_dir} 到 {merged_path}")
            
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
                    # 打印ZIP文件内容
                    print_zip_contents(merged_dataset_zip_path)
            else:
                print("警告: 数据集处理失败，将使用原始数据集ZIP文件")
        
        # 添加VOC到YOLO的转换功能
        yolo_dataset_zip_path = None
        if args.convert_to_yolo and args.generate_dataset and merged_dataset_zip_path:
            print("\n开始将VOC数据集转换为YOLO格式...")
            
            # 设置YOLO输出目录
            yolo_output_dir = args.yolo_output_dir if args.yolo_output_dir else os.path.join(args.output_dir, "yolo_output")
            os.makedirs(yolo_output_dir, exist_ok=True)
            
            # 设置YOLO ZIP输出路径
            if args.yolo_zip_output:
                yolo_zip_output = args.yolo_zip_output
            else:
                yolo_zip_dir = os.path.dirname(args.output_dir)
                yolo_zip_name = f"yolo_dataset_{base_random_uuid}.zip"
                yolo_zip_output = os.path.join(yolo_zip_dir, yolo_zip_name)
            
            # 先解压合并后的VOC数据集到临时目录
            temp_merged_dir = os.path.join(args.output_dir, "temp_merged")
            os.makedirs(temp_merged_dir, exist_ok=True)
            
            try:
                # 检查合并后的ZIP文件是否存在
                if not os.path.exists(merged_dataset_zip_path):
                    print(f"错误：合并数据集ZIP文件不存在: {merged_dataset_zip_path}")
                    yolo_dataset_zip_path = None
                else:
                    print(f"解压合并数据集ZIP文件: {merged_dataset_zip_path}")
                    print(f"ZIP文件大小: {os.path.getsize(merged_dataset_zip_path) / (1024*1024):.2f} MB")
                    
                    # 使用zipfile查看ZIP文件的内容结构
                    print(f"合并数据集ZIP文件内容:")
                    with zipfile.ZipFile(merged_dataset_zip_path, 'r') as zipf:
                        for file_info in zipf.infolist():
                            print(f"  {file_info.filename} - {file_info.file_size} bytes")
                        
                        # 解压文件
                        print(f"正在解压文件到 {temp_merged_dir}...")
                        zipf.extractall(temp_merged_dir)
                    
                    # 打印解压后的目录结构
                    print("解压后的目录结构:")
                    print_directory_structure(temp_merged_dir)
                    
                    # 递归查找解压后目录中的XML文件
                    xml_files = find_files_recursive(temp_merged_dir, '.xml')
                    print(f"在解压目录中找到 {len(xml_files)} 个XML文件")
                    
                    if not xml_files:
                        print(f"警告: 在解压后的目录中未找到任何XML文件，可能是文件结构问题")
                        
                        # 尝试使用替代方法：直接使用合并路径而不是解压的临时目录
                        print(f"尝试使用合并数据集目录而不是解压的ZIP: {merged_path}")
                        if os.path.exists(merged_path):
                            print(f"使用 {merged_path} 代替解压的临时目录")
                            # 打印合并数据集目录结构
                            print("合并数据集目录结构:")
                            print_directory_structure(merged_path)
                            
                            # 递归查找合并目录中的XML文件
                            xml_files_in_merged = find_files_recursive(merged_path, '.xml')
                            print(f"在合并目录中找到 {len(xml_files_in_merged)} 个XML文件")
                            
                            if xml_files_in_merged:
                                print("使用合并数据集目录进行转换")
                                yolo_dataset_zip_path = convert_voc_to_yolo(
                                    merged_path, 
                                    yolo_output_dir,
                                    yolo_zip_output
                                )
                            else:
                                print("在合并数据集目录中也未找到XML文件，无法转换")
                        else:
                            print(f"合并数据集目录不存在: {merged_path}")
                    else:
                        # 调用转换函数
                        yolo_dataset_zip_path = convert_voc_to_yolo(
                            temp_merged_dir, 
                            yolo_output_dir,
                            yolo_zip_output
                        )
                    
                    if yolo_dataset_zip_path:
                        print(f"YOLO格式数据集ZIP文件已创建在: {yolo_dataset_zip_path}")
                    else:
                        print("转换失败，未能创建YOLO格式数据集")
                
            except Exception as e:
                print(f"转换VOC到YOLO格式时出错: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                # 清理临时目录
                if os.path.exists(temp_merged_dir):
                    shutil.rmtree(temp_merged_dir)
                    print(f"已删除临时目录: {temp_merged_dir}")
                    
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
        if args.upload_to_r2:
            if args.merge_datasets and merged_dataset_zip_path:
                # 上传合并后的VOC数据集
                files_to_upload['voc_dataset'] = merged_dataset_zip_path
                print("添加合并后的VOC数据集到上传列表")
                
                # 如果已经转换为YOLO格式，也上传YOLO数据集
                if args.convert_to_yolo and yolo_dataset_zip_path:
                    files_to_upload['yolo_dataset'] = yolo_dataset_zip_path
                    print("添加YOLO格式数据集到上传列表")
            elif all_datasets_zip_path:
                # 上传所有数据集的打包文件（合并失败或未启用合并功能）
                files_to_upload['dataset'] = all_datasets_zip_path
                print("上传所有原始数据集的打包文件到R2")
            elif len(dataset_zip_paths) == 1:
                # 如果只有一个数据集且未合并，直接上传它
                files_to_upload['voc_dataset'] = dataset_zip_paths[0]
                print("只有一个数据集，直接添加到上传列表")
        
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
            final_result["merged_voc_dataset_zip_path"] = merged_dataset_zip_path
        elif all_datasets_zip_path:
            final_result["all_datasets_zip_path"] = all_datasets_zip_path
        elif len(dataset_zip_paths) == 1:
            final_result["voc_dataset_zip_path"] = dataset_zip_paths[0]
        
        # 添加YOLO数据集路径到结果
        if args.convert_to_yolo and yolo_dataset_zip_path:
            final_result["yolo_dataset_zip_path"] = yolo_dataset_zip_path
        
        # 添加R2链接到结果
        if urls:
            final_result["upload_urls"] = urls
            
            # 添加VOC数据集下载链接
            if 'voc_dataset' in urls:
                final_result["voc_dataset_download_url"] = urls['voc_dataset']
            elif 'dataset' in urls:
                final_result["dataset_download_url"] = urls['dataset']
                
            # 添加YOLO数据集下载链接
            if 'yolo_dataset' in urls:
                final_result["yolo_dataset_download_url"] = urls['yolo_dataset']
                
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
    
    # 添加VOC到YOLO转换相关参数
    parser.add_argument("--convert_to_yolo", action="store_true", help="将合并后的VOC数据集转换为YOLO格式")
    parser.add_argument("--yolo_output_dir", default=None, help="YOLO数据集的输出目录")
    parser.add_argument("--yolo_zip_output", default=None, help="YOLO数据集的ZIP输出路径")
    
    args = parser.parse_args()
    
    main(args)