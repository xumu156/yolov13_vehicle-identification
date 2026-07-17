"""
Pascal VOC XML → YOLO 格式转换工具
从 Kaggle 数据集 (vehicles.v2-release.voc) 转换标注格式

用法:
    python preprocess/voc_to_yolo.py --input vehicles.v2-release.voc --output datasets
"""
import os
import xml.etree.ElementTree as ET
import shutil
import argparse

# 完整的12个车辆类别
categories = [
    'big truck', 'big car', 'bus-l-', 'bus-s-', 'car', 'mid truck',
    'null', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-'
]
category_to_index = {category: index for index, category in enumerate(categories)}

# 类别名称映射
category_mapping = {
    'big bus': 'big truck', 'big truck': 'big truck', 'big car': 'big car',
    'bus': 'bus-l-', 'buss': 'bus-l-', 'bus-I-buss': 'bus-l-',
    'bus-s-': 'bus-s-', 'car': 'car', 'mid truck': 'mid truck',
    'null': 'null', 'small bus': 'small truck', 'small truck': 'small truck',
    'truck': 'truck-l-', 'truck-I': 'truck-l-', 'truckmr': 'truck-m-',
    'truckre': 'truck-s-', 'truck-xl': 'truck-xl-', 'truck-xl-': 'truck-xl-'
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def copy_image_files(input_folder, output_folder, folder_name):
    """复制图像文件到输出文件夹"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    copied_count = 0
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"[{folder_name}] 复制图像: {filename}")
    return copied_count


def process_folder(input_folder, output_folder, folder_name):
    """处理单个文件夹的XML文件并复制图像"""
    os.makedirs(output_folder, exist_ok=True)
    total_files, total_objects = 0, 0

    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 {input_folder} 不存在，跳过")
        return total_files, total_objects

    print(f"[{folder_name}] 正在复制图像文件...")
    image_count = copy_image_files(input_folder, output_folder, folder_name)
    print(f"[{folder_name}] 复制了 {image_count} 个图像文件")

    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    if not xml_files:
        print(f"警告: 在 {input_folder} 中没有找到XML文件")
        return total_files, total_objects

    print(f"[{folder_name}] 找到 {len(xml_files)} 个XML文件")

    for filename in xml_files:
        xml_path = os.path.join(input_folder, filename)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text.strip()
                normalized_name = category_mapping.get(name, name)
                if normalized_name not in category_to_index:
                    print(f"警告: 未知类别 '{name}'，已跳过")
                    continue
                category_index = category_to_index[normalized_name]
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                x = max(0, min(1, (xmin + xmax) / 2.0 / width))
                y = max(0, min(1, (ymin + ymax) / 2.0 / height))
                w = max(0, min(1, (xmax - xmin) / width))
                h = max(0, min(1, (ymax - ymin) / height))
                objects.append(f"{category_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)
            with open(txt_path, 'w') as f:
                for obj in objects:
                    f.write(obj + '\n')

            total_files += 1
            total_objects += len(objects)
            print(f"[{folder_name}] 转换: {filename} -> {txt_filename} ({len(objects)} 目标)")

        except Exception as e:
            print(f"错误: 处理 {filename} 时异常 - {str(e)}")

    return total_files, total_objects


def create_data_yaml(output_dir):
    """创建YOLO数据配置文件"""
    data_yaml_content = f"""# YOLO 车辆检测数据集配置
# 数据集来源: Kaggle - vehicles.v2-release.voc
path: {output_dir}

train: train
val: valid
test: test

nc: {len(categories)}

names: {categories}
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Pascal VOC → YOLO 格式转换')
    parser.add_argument('--input', type=str, default=os.path.join(ROOT, 'vehicles.v2-release.voc'),
                        help='VOC 数据集目录')
    parser.add_argument('--output', type=str, default=os.path.join(ROOT, 'datasets'),
                        help='YOLO 格式输出目录')
    args = parser.parse_args()

    folder_mapping = [
        {'name': 'test', 'input': os.path.join(args.input, 'test'), 'output': os.path.join(args.output, 'test')},
        {'name': 'train', 'input': os.path.join(args.input, 'train'), 'output': os.path.join(args.output, 'train')},
        {'name': 'valid', 'input': os.path.join(args.input, 'valid'), 'output': os.path.join(args.output, 'valid')},
    ]

    print("开始批量处理数据集和图像文件...")
    print("=" * 60)

    total_stats = {}
    for folder_info in folder_mapping:
        folder_name = folder_info['name']
        print(f"\n开始处理 {folder_name} 文件夹...")
        print(f"输入: {folder_info['input']}")
        print(f"输出: {folder_info['output']}")
        files_count, objects_count = process_folder(folder_info['input'], folder_info['output'], folder_name)
        total_stats[folder_name] = {'files': files_count, 'objects': objects_count}
        print(f"{folder_name} 完成: {files_count} 文件, {objects_count} 目标")

    print("\n" + "=" * 60)
    print("处理完成统计:")
    total_files = sum(s['files'] for s in total_stats.values())
    total_objects = sum(s['objects'] for s in total_stats.values())
    for name, stats in total_stats.items():
        print(f"  {name:8}: {stats['files']:4} 文件, {stats['objects']:6} 目标")
    print(f"  {'总计':8}: {total_files:4} 文件, {total_objects:6} 目标")

    # 创建 classes.txt
    classes_path = os.path.join(args.output, 'classes.txt')
    with open(classes_path, 'w') as f:
        for category in categories:
            f.write(category + '\n')
    print(f"\n类别文件: {classes_path}")

    # 创建 data.yaml
    yaml_path = create_data_yaml(args.output)
    print(f"配置文件: {yaml_path}")

    print("\n类别索引对照表:")
    for index, category in enumerate(categories):
        print(f"  {index:2}: {category}")


if __name__ == "__main__":
    main()
