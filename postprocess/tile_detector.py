"""YOLOv13 分块检测器 (Tile Detector)
将大尺寸图像切割为重叠小块逐一检测，再合并结果。适合高分辨率航拍/遥感车辆检测

用法:
    from postprocess.tile_detector import YOLOv13TileDetector
    detector = YOLOv13TileDetector('best.pt')
    results = detector.detect('large_image.jpg')
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch


class YOLOv13TileDetector:
    def __init__(self, model_path, tile_size=640, overlap=0.3, conf_threshold=0.25, iou_threshold=0.5):
        """
        初始化YOLOv13切割检测器
        Args:
            model_path: 模型路径
            tile_size: 切割块大小
            overlap: 重叠比例
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def split_image(self, image):
        """
        将图像切割成多个重叠的块
        """
        h, w = image.shape[:2]
        stride = int(self.tile_size * (1 - self.overlap))

        tiles = []
        locations = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + self.tile_size, w)
                y2 = min(y + self.tile_size, h)

                # 提取图块
                tile = image[y:y2, x:x2]

                # 如果图块尺寸不够，进行填充
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    padded_tile = np.full((self.tile_size, self.tile_size, 3), 114, dtype=np.uint8)
                    padded_tile[0:tile.shape[0], 0:tile.shape[1]] = tile
                    tile = padded_tile

                tiles.append(tile)
                locations.append((x, y, x2, y2))

        print(f"图像切割成 {len(tiles)} 个块")
        return tiles, locations

    def detect_tile(self, tile):
        """
        对单个图块进行检测
        """
        results = self.model(tile, conf=self.conf_threshold, verbose=False)

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 只保留car和truck类别（根据您的需求调整）
                class_name = self.model.names[cls_id]
                if class_name in ['car', 'truck']:
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })

        return detections

    def map_to_original_coords(self, detections, location):
        """
        将图块坐标映射回原始图像坐标
        """
        ox, oy, x2_max, y2_max = location  # 新增后两个变量
        original_detections = []  # 用来装映射后的结果

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # 先映射
            x1 += ox;
            y1 += oy;
            x2 += ox;
            y2 += oy
            # 再 clamp 到 tile 实际区域
            x1 = max(ox, min(x1, x2_max))
            y1 = max(oy, min(y1, y2_max))
            x2 = max(ox, min(x2, x2_max))
            y2 = max(oy, min(y2, y2_max))

            original_detections.append({
                'class': det['class'],
                'confidence': det['confidence'],
                'bbox': [x1, y1, x2, y2]
            })

        return original_detections

    def apply_nms(self, detections):
        """
        应用非极大值抑制
        """
        if len(detections) == 0:
            return []

        # 转换为numpy数组
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        classes = np.array([det['class'] for det in detections])

        # 按类别分别应用NMS
        unique_classes = np.unique(classes)
        final_detections = []

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_detections = [detections[i] for i in range(len(detections)) if classes[i] == cls]

            if len(cls_boxes) == 0:
                continue

            # 使用OpenCV的NMS
            indices = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(),
                cls_scores.tolist(),
                self.conf_threshold,
                self.iou_threshold
            )

            if len(indices) > 0:
                for i in indices.flatten():
                    final_detections.append(cls_detections[i])

        return final_detections

    def detect_image(self, image_path, output_path=None):
        """
        主检测函数
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        original_h, original_w = image.shape[:2]
        print(f"原始图像尺寸: {original_w} x {original_h}")

        # 1. 切割图像
        tiles, locations = self.split_image(image)

        # 2. 对每个图块进行检测
        all_detections = []
        for i, (tile, location) in enumerate(zip(tiles, locations)):
            print(f"检测图块 {i + 1}/{len(tiles)}...")
            tile_detections = self.detect_tile(tile)
            original_detections = self.map_to_original_coords(tile_detections, location)
            all_detections.extend(original_detections)

        print(f"切割检测总共找到 {len(all_detections)} 个检测结果")

        # 3. 对整个图像进行原始检测（用于对比）
        print("进行原始整图检测...")
        original_results = self.model(image, conf=self.conf_threshold, verbose=False)
        original_detections = []
        if len(original_results) > 0 and original_results[0].boxes is not None:
            for box in original_results[0].boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = self.model.names[cls_id]
                if class_name in ['car', 'truck']:
                    original_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })

        print(f"原始检测找到 {len(original_detections)} 个检测结果")

        # 4. 合并两种检测结果
        combined_detections = all_detections + original_detections

        # 5. 应用全局NMS
        final_detections = self.apply_nms(combined_detections)
        print(f"NMS后剩余 {len(final_detections)} 个检测结果")

        # 6. 绘制结果
        result_image = image.copy()
        for det in final_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            # 选择颜色
            color = (0, 255, 0) if det['class'] == 'car' else (255, 0, 0)  # 绿色-car, 蓝色-truck

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{det['class']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 7. 保存结果
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"结果保存至: {output_path}")

        return final_detections, result_image


# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    detector = YOLOv13TileDetector(
        model_path=r"E:\yolo13_program\yolov13-main\yolov13n.pt",
        tile_size=512,  # 与您训练尺寸一致
        overlap=0.4,  # 40%重叠，确保小物体不被切割
        conf_threshold=0.6,  # 较低阈值以检测小物体
        iou_threshold=0.5  # 较宽松的NMS
    )

    # 进行检测
    input_image = r"C:\Users\xu\Desktop\R-C.jpg"  # 替换为您的图像路径
    output_image = r"E:\yolo13_program\yolov13-main\Result Processing\result_p\4（n）.jpg"

    detections, result_img = detector.detect_image(input_image, output_image)

    # 打印检测结果
    print("\n最终检测结果:")
    for i, det in enumerate(detections):
        print(f"{i + 1}: {det['class']} {det['confidence']:.3f}")