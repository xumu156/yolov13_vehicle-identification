"""
YOLOv13 车辆检测 - 模型检查工具
查看训练好的模型包含的类别信息

用法:
    python test_model.py --weights runs/detect/train/weights/best.pt
"""
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 模型信息查看')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    args = parser.parse_args()

    model = YOLO(args.weights)
    print("模型训练的类别：")
    for class_id, class_name in model.names.items():
        print(f"  类别ID {class_id}: {class_name}")


if __name__ == '__main__':
    main()
