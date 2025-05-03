import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import zipfile
import random
from datetime import datetime
import shutil
import json

def create_voc_annotation(filename, size, bbox_list, object_name):
    """
    创建 Pascal VOC 格式的 XML 标注文件。
    :param filename: 图像文件名
    :param size: 图像大小 (width, height, depth)
    :param bbox_list: 边界框列表，每个元素是 (xmin, ymin, xmax, ymax)
    :param object_name: 物体名称
    """
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'images'

    filename_tag = ET.SubElement(annotation, 'filename')
    filename_tag.text = filename

    size_tag = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size_tag, 'width')
    width.text = str(size[0])
    height = ET.SubElement(size_tag, 'height')
    height.text = str(size[1])
    depth = ET.SubElement(size_tag, 'depth')
    depth.text = str(size[2])

    for bbox in bbox_list:
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = object_name
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bbox[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bbox[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bbox[0] + bbox[2])  # 注意这里是x_min + width
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bbox[1] + bbox[3])  # 注意这里是y_min + height

    return ET.ElementTree(annotation)

def save_frame_with_annotation(frame, frame_idx, output_dir, bbox_list, object_name):
    """
    保存帧图像和对应的XML标注文件
    """
    # 创建输出目录
    images_dir = os.path.join(output_dir, "images")
    xml_dir = os.path.join(output_dir, "xml")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)

    # 保存图像
    image_filename = f"{frame_idx:04d}.jpg"
    image_path = os.path.join(images_dir, image_filename)
    cv2.imwrite(image_path, frame)

    # 创建并保存XML标注
    height, width, depth = frame.shape
    xml_annotation = create_voc_annotation(
        filename=image_filename,
        size=(width, height, depth),
        bbox_list=bbox_list,
        object_name=object_name
    )
    
    xml_path = os.path.join(xml_dir, f"{frame_idx:04d}.xml")
    xml_annotation.write(xml_path)
    
    return image_path, xml_path

def create_dataset_zip(output_dir, object_name, zip_output_path=None):
    """
    创建数据集ZIP文件
    """
    # 生成当前日期（精确到毫秒）和四位随机数
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    random_number = random.randint(10000, 99999)
    
    # 清理物体名称以确保文件名安全
    object_name_sanitized = "".join(c if c.isalnum() else "_" for c in object_name)
    
    # 如果未指定ZIP输出路径，则使用默认名称
    if zip_output_path is None:
        zip_output_path = os.path.join(output_dir, f"dataset_{current_date}_{random_number}.zip")
    
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历 "images" 和 "xml" 文件夹
        for folder in ["images", "xml"]:
            folder_path = os.path.join(output_dir, folder)
            if not os.path.exists(folder_path):
                continue
                
            for file in os.listdir(folder_path):
                frame_idx_str = os.path.splitext(file)[0]
                try:
                    frame_idx = int(frame_idx_str)
                except ValueError:
                    continue
                
                # 确定文件扩展名
                ext = os.path.splitext(file)[1].lower()
                
                # 格式化帧编号为四位数
                frame_number = f"{frame_idx:04d}"
                
                # 构建新的文件名
                new_filename = f"{current_date}_{random_number:04d}_{frame_number}_{object_name_sanitized}{ext}"
                
                # 获取完整的文件路径
                file_path = os.path.join(folder_path, file)
                
                # 构建 ZIP 内的目标路径
                zip_target_path = os.path.join(folder, new_filename)
                
                # 将文件添加到 ZIP 中
                zipf.write(file_path, zip_target_path)
    
    return zip_output_path

def save_video_settings(output_dir, width, height, fps, total_frames):
    """
    保存视频设置到JSON文件
    """
    video_settings = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames
    }
    
    with open(os.path.join(output_dir, "video_settings.json"), "w") as f:
        json.dump(video_settings, f, indent=4)