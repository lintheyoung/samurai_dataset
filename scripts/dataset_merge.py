# dataset_merge.py - 修复版
import os
import xml.etree.ElementTree as ET
import shutil
import zipfile
import re

def extract_number(filename):
    """从文件名中提取帧号"""
    # 首先尝试匹配常规数字格式：0001.xml, frame_0001.xml 等
    match = re.search(r'(\d{4})', filename)
    if match:
        return int(match.group(1))
    
    # 如果没找到，尝试从复杂格式中提取
    parts = filename.split('_')
    if len(parts) >= 5:
        try:
            return int(parts[4])
        except (ValueError, IndexError):
            pass
    
    # 如果还是没找到，查找任何数字序列
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    
    return None

def merge_xml(files):
    """
    合并多个XML文件的object元素
    :param files: XML文件路径列表
    :return: 合并后的ElementTree
    """
    if not files:
        raise ValueError("无文件提供用于合并")
    
    # 解析第一个XML文件作为基础
    base_tree = ET.parse(files[0])
    base_root = base_tree.getroot()
    
    # 遍历其余XML文件，提取object元素并添加到基础XML
    for f in files[1:]:
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            # 找到所有object元素（表示标注对象）
            for obj in root.findall('object'):
                # 将它们添加到基础XML中
                base_root.append(obj)
        except Exception as e:
            print(f"处理文件 {f} 时出错: {str(e)}")
    
    return base_tree

def merge_voc_datasets(base_path, custom_tag, output_path):
    """
    合并多个VOC格式数据集
    :param base_path: 包含多个数据集文件夹的基础路径
    :param custom_tag: 自定义标签，用于生成输出文件名
    :param output_path: 输出路径
    :return: 合并后数据集的路径
    """
    print(f"开始合并数据集，基础路径: {base_path}")
    
    # 只获取以 "bbox_" 开头的目录作为有效类别
    categories = []
    for d in os.listdir(base_path):
        dir_path = os.path.join(base_path, d)
        if os.path.isdir(dir_path) and d.startswith('bbox_'):
            # 检查是否有xml子目录以确认是有效数据集
            if os.path.exists(os.path.join(dir_path, 'xml')):
                categories.append(d)
    
    print(f"找到以下有效类别目录: {categories}")
    
    if not categories:
        print("警告: 没有找到任何有效的bbox_*类别目录")
        return None
    
    # 创建输出目录结构
    merged_xml_dir = os.path.join(output_path, 'merged', 'xml')
    merged_images_dir = os.path.join(output_path, 'merged', 'images')
    os.makedirs(merged_xml_dir, exist_ok=True)
    os.makedirs(merged_images_dir, exist_ok=True)
    
    # 为每个类别获取并排序文件
    category_files = []
    for category in categories:
        annotation_dir = os.path.join(base_path, category, 'xml')
        if not os.path.exists(annotation_dir):
            print(f"警告: 标注目录 {annotation_dir} 不存在，跳过此类别。")
            continue
        
        files = os.listdir(annotation_dir)
        # 根据帧号排序文件
        sorted_files = sorted(files, key=lambda x: extract_number(x) if extract_number(x) is not None else -1)
        category_files.append(sorted_files)
    
    if not category_files:
        print("警告: 没有找到有效的XML文件")
        return None
    
    # 找到所有类别共有的最小文件数
    min_files = min(len(files) for files in category_files)
    if min_files == 0:
        print("警告: 某些类别没有XML文件")
        return None
    
    print(f"开始合并 {min_files} 个帧...")
    
    # 合并各类别的同帧XML文件
    merged_count = 0
    for i in range(min_files):
        try:
            # 构建要合并的同帧XML文件路径
            files_to_merge = []
            for ci in range(len(categories)):
                if i < len(category_files[ci]):
                    xml_path = os.path.join(base_path, categories[ci], 'xml', category_files[ci][i])
                    if os.path.exists(xml_path):
                        files_to_merge.append(xml_path)
            
            if not files_to_merge:
                print(f"警告: 帧 {i} 没有有效的XML文件可合并")
                continue
            
            # 合并XML
            merged_tree = merge_xml(files_to_merge)
            
            # 生成输出文件名并保存
            frame_number = extract_number(category_files[0][i])
            if frame_number is None:
                frame_number = i
            
            merged_xml_filename = f'{custom_tag}_merged_{frame_number:04d}.xml'
            merged_xml_path = os.path.join(merged_xml_dir, merged_xml_filename)
            merged_tree.write(merged_xml_path)
            
            # 复制对应的图像文件
            image_base_name = os.path.splitext(category_files[0][i])[0]
            original_image_path = None
            
            # 尝试不同的图像扩展名
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                img_path = os.path.join(base_path, categories[0], 'images', f"{image_base_name}{ext}")
                if os.path.exists(img_path):
                    original_image_path = img_path
                    break
            
            if original_image_path:
                # 使用目标扩展名
                target_ext = os.path.splitext(original_image_path)[1]
                merged_image_filename = f'{custom_tag}_merged_{frame_number:04d}{target_ext}'
                merged_image_path = os.path.join(merged_images_dir, merged_image_filename)
                shutil.copy2(original_image_path, merged_image_path)
                merged_count += 1
            else:
                print(f"警告: 无法找到帧 {i} 的图像文件，尝试寻找更广泛的匹配...")
                
                # 尝试直接按照帧号寻找图像
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    img_path = os.path.join(base_path, categories[0], 'images', f"{frame_number:04d}{ext}")
                    if os.path.exists(img_path):
                        original_image_path = img_path
                        break
                
                if original_image_path:
                    # 使用目标扩展名
                    target_ext = os.path.splitext(original_image_path)[1]
                    merged_image_filename = f'{custom_tag}_merged_{frame_number:04d}{target_ext}'
                    merged_image_path = os.path.join(merged_images_dir, merged_image_filename)
                    shutil.copy2(original_image_path, merged_image_path)
                    merged_count += 1
                else:
                    print(f"警告: 尝试更广泛的匹配后仍无法找到帧 {i} 的图像文件")
        
        except Exception as e:
            print(f"处理帧 {i} 时出错: {str(e)}")
    
    print(f"合并完成! 成功合并 {merged_count} 个帧。")
    return os.path.join(output_path, 'merged') if merged_count > 0 else None

def zip_merged_dataset(merged_path, output_zip):
    """
    将合并的数据集打包成ZIP文件
    :param merged_path: 合并数据集的路径
    :param output_zip: 输出ZIP文件路径
    :return: 输出ZIP文件路径
    """
    try:
        if not merged_path or not os.path.exists(merged_path):
            print(f"警告: 合并数据集路径不存在: {merged_path}")
            return None
            
        print(f"开始打包合并数据集: {merged_path} -> {output_zip}")
        
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(merged_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 计算相对路径保持目录结构
                    relative_path = os.path.relpath(file_path, os.path.dirname(merged_path))
                    zipf.write(file_path, relative_path)
        
        print(f"数据集打包成功: {output_zip}")
        return output_zip
    except Exception as e:
        print(f"打包数据集时出错: {str(e)}")
        return None