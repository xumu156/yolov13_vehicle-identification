"""
YOLOv13 实时摄像头检测
使用摄像头进行实时车辆检测，显示 FPS 和检测结果

用法:
    python predict_video.py --weights yolov13n.pt --source 0
"""
import cv2
from ultralytics import YOLO
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='YOLOv13 实时摄像头检测')
    parser.add_argument('--weights', type=str, default='yolov13n.pt', help='模型权重')
    parser.add_argument('--source', type=str, default='0', help='视频源 (0=摄像头, 或视频文件)')
    args = parser.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        start = time.perf_counter()
        result = model(frame)
        annotated_frame = result[0].plot()

        # 打印检测信息
        detections = result[0].boxes
        if len(detections) > 0:
            class_names = model.names
            for box in detections:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                conf = float(box.conf[0])
                print(f"目标: {class_name}, 置信度: {conf:.2f}")

        # FPS 显示
        end = time.perf_counter()
        fps = 1 / (end - start)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv13 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
