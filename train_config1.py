"""
YOLOv13 车辆检测 - 训练配置1 (单图推理 + 保存)
用于训练后快速测试单张图片并保存结果

用法:
    python train_config1.py --weights runs/detect/train/weights/best.pt --source test.jpg
"""
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 快速推理保存')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='输入图像路径')
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, save=False, show=True)

    for r in results:
        im_array = r.plot()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(ROOT, 'runs', 'result')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'result_{timestamp}.jpg')
        cv2.imwrite(save_path, im_array)
        print(f"图片已保存到: {save_path}")

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
