"""
YOLOv13 车辆检测 - 主训练脚本
使用 SGD 优化器，适合从预训练权重开始训练

用法:
    python train.py --data data.yaml --epochs 400 --batch 8 --imgsz 416 --device 0
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import argparse
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 车辆检测训练')
    parser.add_argument('--weights', type=str, default='yolov13n.pt', help='预训练权重文件')
    parser.add_argument('--data', type=str, default=os.path.join(ROOT, 'data.yaml'), help='数据集配置')
    parser.add_argument('--epochs', type=int, default=400, help='训练轮次')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=416, help='训练图像尺寸')
    parser.add_argument('--workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--device', type=str, default='0', help='计算设备')
    parser.add_argument('--optimizer', type=str, default='SGD', help='优化器')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--amp', action='store_true', default=False, help='混合精度')
    parser.add_argument('--cache', action='store_true', default=False, help='缓存数据集')
    parser.add_argument('--close_mosaic', type=int, default=0, help='最后N轮关闭mosaic')
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device

    model = YOLO(args.weights)
    model.load(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        optimizer=args.optimizer,
        amp=args.amp,
        cache=args.cache,
        patience=args.patience,
        lr0=args.lr0,
        close_mosaic=args.close_mosaic,
    )
    print(f"训练完成! 模型保存至: {results.save_dir}")


if __name__ == '__main__':
    main()
