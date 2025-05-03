import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import zipfile
import random
from datetime import datetime
import shutil
import json
import hashlib
# 添加R2客户端导入
from r2client.R2Client import R2Client as r2

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

# 添加上传到Cloudflare R2的函数
def upload_to_r2(file_path):
    """
    将文件上传到Cloudflare R2存储并返回下载URL
    :param file_path: 要上传的文件路径
    :return: 下载URL或None（如果上传失败）
    """
    def hash_time(time_string):
        """将时间字符串哈希处理，生成不可读的加密文件夹名称"""
        sha256 = hashlib.sha256()
        sha256.update(time_string.encode('utf-8'))
        return sha256.hexdigest()

    # 初始化 R2 客户端
    client = r2(
        access_key=os.environ.get('R2_ACCESS_KEY_ID', 'a16a270ce6ed98d1006fd9a80b6f84be'),
        secret_key=os.environ.get('R2_SECRET_KEY_ID', 'bd631ad197ce559d4341522df5d3bdfef6137b42da4c08721d6060d34ef9078d'),
        endpoint=f'https://{os.environ.get("R2_ACCOUNT_ID", "bf7302689d0dd0a365e5199aee2d3192")}.r2.cloudflarestorage.com'
    )

    bucket_name = os.environ.get('R2_BUCKET_NAME', 'dedemaker')
    base_folder_name = '3DA9EE71FB0F305B'  # 指定的顶级文件夹名称

    # 获取当前时间，并格式化为每小时的时间字符串
    current_time = datetime.now().strftime('%Y-%m-%d-%H')
    
    # 使用内部函数将时间加密为文件夹名称
    encrypted_folder = hash_time(current_time)

    file_name = os.path.basename(file_path)  # 获取文件名称
    r2_file_key = f"{base_folder_name}/{encrypted_folder}/{file_name}"  # 在指定的加密文件夹下存储文件

    # 上传文件到 R2
    try:
        client.upload_file(bucket_name, file_path, r2_file_key)
        print(f"File uploaded successfully to R2 at path: {r2_file_key}")
        # 返回自定义的下载 URL
        return f"https://pub-6c1e280a27614b05891bfd818585735e.r2.dev/{base_folder_name}/{encrypted_folder}/{file_name}"
    except Exception as e:
        print(f"Error occurred during file upload: {str(e)}")
        return None