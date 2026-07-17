"""
YOLOv13 车辆检测 - 单图预测
用于单张图片快速推理测试

用法:
    python predict_image.py --weights runs/detect/train/weights/best.pt --source test.jpg
"""
from ultralytics import YOLO
import cv2
import argparse
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 单图预测')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='输入图像路径')
    parser.add_argument('--save', action='store_true', default=False, help='保存结果')
    parser.add_argument('--show', action='store_true', default=True, help='显示结果')
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, save=args.save, show=args.show)

    if args.save:
        for r in results:
            im_array = r.plot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(ROOT, 'runs', 'result')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'result_{timestamp}.jpg')
            cv2.imwrite(save_path, im_array)
            print(f"结果已保存至: {save_path}")

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
