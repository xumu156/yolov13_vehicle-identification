"""
YOLOv13 车辆检测 - 验证脚本
计算模型在测试集上的 mAP 指标

用法:
    python val.py --weights runs/detect/train/weights/best.pt --data data.yaml --device 0
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import argparse
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 模型验证')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data', type=str, default=os.path.join(ROOT, 'data.yaml'), help='数据集配置')
    parser.add_argument('--split', type=str, default='test', help='验证集划分')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--conf', type=float, default=0.001, help='置信度阈值')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程')
    parser.add_argument('--device', type=str, default='0', help='计算设备')
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device

    model = YOLO(args.weights)
    model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        iou=args.iou,
        conf=args.conf,
        workers=args.workers,
        device=device,
    )


if __name__ == '__main__':
    main()
