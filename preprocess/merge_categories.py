"""
类别合并工具 — 将 Pascal VOC 12类别合并为 8类别
从 vehicles.v2-release.voc 转换后，合并相似类别

用法:
    python preprocess/merge_categories.py --input datasets --output datasets_merged
"""
import os
import xml.etree.ElementTree as ET
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 合并映射
category_merging = {
    'big truck': 'big truck', 'big car': 'big car', 'car': 'car', 'null': 'null',
    'bus-l-': 'bus', 'bus-s-': 'bus',
    'mid truck': 'mid_truck', 'small truck': 'small_truck',
    'truck-l-': 'truck', 'truck-m-': 'truck', 'truck-s-': 'truck',
    'truck-xl-': 'big_truck'
}

new_categories = ['big truck', 'big car', 'bus', 'car', 'mid_truck', 'null', 'small_truck', 'truck']
new_category_to_index = {cat: idx for idx, cat in enumerate(new_categories)}


def create_data_yaml(categories_list, base_path):
    """创建数据集配置文件"""
    data_yaml_content = f"""# YOLO 车辆检测数据集配置（合并类别后）
# 数据集来源: Kaggle - vehicles.v2-release.voc
path: {base_path}

train: train
val: valid
test: test

nc: {len(categories_list)}

names: {categories_list}
"""
    yaml_path = os.path.join(base_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    print(f"data.yaml 已创建: {yaml_path}")
    print(f"合并后类别数: {len(categories_list)}")
    print("类别列表:", categories_list)


def merge_categories_in_dataset(input_base, output_base):
    """合并数据集类别"""
    folders = ['train', 'valid', 'test']

    for folder in folders:
        input_folder = os.path.join(input_base, folder)
        output_folder = os.path.join(output_base, folder)
        os.makedirs(output_folder, exist_ok=True)
        print(f"处理 {folder} 文件夹...")

        xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
        for xml_file in xml_files:
            xml_path = os.path.join(input_folder, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                objects = []
                for obj in root.findall('object'):
                    original_name = obj.find('name').text.strip()
                    new_name = category_merging.get(original_name, original_name)
                    if new_name in new_category_to_index:
                        idx = new_category_to_index[new_name]
                        bndbox = obj.find('bndbox')
                        xmin, ymin = int(bndbox.find('xmin').text), int(bndbox.find('ymin').text)
                        xmax, ymax = int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)

                        x = max(0, min(1, (xmin + xmax) / 2.0 / width))
                        y = max(0, min(1, (ymin + ymax) / 2.0 / height))
                        w = max(0, min(1, (xmax - xmin) / width))
                        h = max(0, min(1, (ymax - ymin) / height))
                        objects.append(f"{idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

                txt_filename = os.path.splitext(xml_file)[0] + '.txt'
                txt_path = os.path.join(output_folder, txt_filename)
                with open(txt_path, 'w') as f:
                    for obj in objects:
                        f.write(obj + '\n')

                # 复制图像
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_src = os.path.join(input_folder, os.path.splitext(xml_file)[0] + ext)
                    img_dst = os.path.join(output_folder, os.path.splitext(xml_file)[0] + ext)
                    if os.path.exists(img_src):
                        shutil.copy2(img_src, img_dst)
                        break

            except Exception as e:
                print(f"处理 {xml_file} 出错: {e}")

    create_data_yaml(new_categories, output_base)
    return new_categories


def main():
    parser = argparse.ArgumentParser(description='合并数据集类别')
    parser.add_argument('--input', type=str, default=os.path.join(ROOT, 'datasets'), help='输入目录')
    parser.add_argument('--output', type=str, default=os.path.join(ROOT, 'datasets_merged'), help='输出目录')
    args = parser.parse_args()

    merge_categories_in_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
