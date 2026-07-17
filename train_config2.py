"""
YOLOv13 车辆检测 - 训练配置2 (AdamW + 混合精度)
使用 AdamW 优化器，余弦退火学习率，适合精细调优

用法:
    python train_config2.py --weights yolov13s.pt --data data.yaml --epochs 250 --batch 8 --device 0
"""
import warnings
import os
import torch
import argparse

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 训练 (AdamW + AMP)')
    parser.add_argument('--weights', type=str, default='yolov13s.pt', help='预训练权重')
    parser.add_argument('--data', type=str, default=os.path.join(ROOT, 'data.yaml'), help='数据集配置')
    parser.add_argument('--epochs', type=int, default=250, help='训练轮次')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--workers', type=int, default=2, help='数据加载线程')
    parser.add_argument('--device', type=str, default='0', help='计算设备')
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=100,
        cos_lr=True,
        amp=True,
        cache='disk',
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.0,
        erasing=0.2,
        save_period=20,
        val=True,
        plots=False,
    )

    torch.cuda.empty_cache()

    # 读取训练日志
    train_dir = results.save_dir
    results_csv = os.path.join(train_dir, "results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        print("\n训练过程摘要:")
        print(f"总训练轮次: {len(df)}")
        print(f"最佳 mAP50: {df['metrics/mAP50(B)'].max():.4f}")
        print(f"最终 mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.4f}")

    # 最终验证
    torch.cuda.empty_cache()
    best_model = YOLO(os.path.join(train_dir, "weights", "best.pt"))
    val_results = best_model.val(workers=0, device=device, plots=False)
    print(f"\n最终模型性能:")
    print(f"mAP50:    {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
