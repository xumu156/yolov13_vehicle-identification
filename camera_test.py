"""
YOLOv13 车辆检测 - 摄像头快速测试
简单的摄像头画面测试（不进行检测），验证摄像头可用性

用法:
    python camera_test.py
"""
import cv2


def main():
    cap = cv2.VideoCapture(0)
    print("摄像头已开启，按 'q' 退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
