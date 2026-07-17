"""
YOLOv13 车辆检测 - 每日验证指标
计算并输出 mAP50 和 mAP50-95 指标

用法:
    python daily_val.py --weights runs/detect/train/weights/best.pt --device 0
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 每日验证')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--device', type=str, default='0', help='计算设备')
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device

    model = YOLO(args.weights)
    results = model.val(workers=0, device=device)

    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")


if __name__ == '__main__':
    main()
