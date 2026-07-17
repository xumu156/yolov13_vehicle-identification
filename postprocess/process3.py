"""后处理工具3 — 分块检测变体
YOLOv13TileDetector 优化版，调整了重叠率和后处理参数

用法:
    from postprocess.process3 import YOLOv13TileDetector
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime


class YOLOv13TileDetector:
    def __init__(self,
                 model_path,
                 tile_size=640,
                 overlap=0.3,
                 conf_threshold=0.25,
                 iou_threshold=0.5,
                 debug=False):
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.debug = debug

        # 自动创建调试目录
        if self.debug:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = rf"E:\yolo13_program\yolov13-main\Result Processing\result_p\{run_id}"
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[Debug] tile 图片将保存在: {self.debug_dir}")

    # ---------------- 1. 切图 ----------------
    def split_image(self, image):
        h, w = image.shape[:2]
        stride = int(self.tile_size * (1 - self.overlap))
        tiles, locations = [], []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2, y2 = min(x + self.tile_size, w), min(y + self.tile_size, h)
                tile = image[y:y2, x:x2]
                tiles.append(tile)
                locations.append((x, y, x2, y2))
        print(f"图像切割成 {len(tiles)} 个块")
        return tiles, locations

    # ---------------- 2. 单 tile 检测 + 面积过滤 ----------------
    def detect_tile(self, tile):
        h_tile, w_tile = tile.shape[:2]
        effective = tile[:h_tile, :w_tile]
        results = self.model(effective, conf=self.conf_threshold, verbose=False)

        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = self.model.names[cls_id]
                if class_name not in ['car', 'truck']:
                    continue

                w = x2 - x1
                h = y2 - y1
                if w * h > 0.6 * self.tile_size**2 or max(w, h) > 0.9 * self.tile_size:
                    continue
                detections.append({'class': class_name,
                                   'confidence': conf,
                                   'bbox': [x1, y1, x2, y2]})
        return detections

    # ---------------- 3. 坐标映射 + clamp ----------------
    def map_to_original_coords(self, detections, location):
        ox, oy, x2_max, y2_max = location
        original_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1 += ox; y1 += oy; x2 += ox; y2 += oy
            x1 = max(ox, min(x1, x2_max))
            y1 = max(oy, min(y1, y2_max))
            x2 = max(ox, min(x2, x2_max))
            y2 = max(oy, min(y2, y2_max))
            original_detections.append({**det, 'bbox': [x1, y1, x2, y2]})
        return original_detections

    # ---------------- 4. NMS + 全局面积过滤 ----------------
    def apply_nms(self, detections, img_shape):
        if not detections:
            return []
        original_h, original_w = img_shape[:2]
        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d['confidence'] for d in detections], dtype=np.float32)
        classes = np.array([d['class'] for d in detections])

        unique_classes = np.unique(classes)
        final_detections = []
        for cls in unique_classes:
            mask = classes == cls
            cls_boxes = boxes[mask]
            cls_scores = scores[mask]
            cls_dets = [detections[i] for i in np.where(mask)[0]]

            # 软 NMS：线性加权衰减
            keep = self.soft_nms(cls_boxes, cls_scores, sigma=0.5, Nt=0.1, threshold=0.15, method=1)
            final_detections.extend([cls_dets[i] for i in keep])

        # 面积过滤
        img_area = original_w * original_h
        final_detections = [
            d for d in final_detections
            if (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) < 0.6 * img_area
        ]
        return final_detections

    def soft_nms(self, boxes, scores, sigma=0.5, Nt=0.1, threshold=0.15, method=1):
        """安全版 Soft-NMS，返回保留索引"""
        boxes = boxes.copy().tolist()
        scores = scores.copy().tolist()
        inds = list(range(len(boxes)))
        keep = []

        while boxes:
            # 当前最高分
            max_idx = np.argmax(scores)
            keep.append(inds[max_idx])
            max_box = boxes[max_idx]

            # 计算其余框与最大框的 IoU
            # 计算其余框与最大框的 IoU
            left = np.maximum(max_box[0], [b[0] for b in boxes])
            top = np.maximum(max_box[1], [b[1] for b in boxes])
            right = np.minimum(max_box[2], [b[2] for b in boxes])
            bottom = np.minimum(max_box[3], [b[3] for b in boxes])
            w = np.maximum(0, right - left)
            h = np.maximum(0, bottom - top)
            inter = w * h
            area1 = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            union = area1 + np.array(areas) - inter  # ★ 修复这里
            iou = inter / (union + 1e-7)

            # 衰减分数
            if method == 1:
                weight = np.where(iou < Nt, 1., 1. - iou)
            else:
                weight = np.exp(-(iou * iou) / sigma)
            scores = [s * w for s, w in zip(scores, weight)]

            # 剔除已处理框 & 低分框
            remain = [(b, s, i) for b, s, i in zip(boxes, scores, inds)
                      if i != inds[max_idx] and s > threshold]
            if not remain:
                break
            boxes, scores, inds = map(list, zip(*remain))

        return keep

    # ---------------- 5. 主流程 ----------------
    def detect_image(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        original_h, original_w = image.shape[:2]

        tiles, locations = self.split_image(image)
        all_detections = []
        for i, (tile, loc) in enumerate(zip(tiles, locations)):
            dets = self.detect_tile(tile)

            # ★ 调试：保存 tile 调试图
            if self.debug:
                tile_vis = tile.copy()
                for det in dets:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    cv2.rectangle(tile_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(tile_vis,
                                f"{det['class']} {det['confidence']:.2f}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                save_path = os.path.join(self.debug_dir, f"tile{i:03d}_{loc}.jpg")
                cv2.imwrite(save_path, tile_vis)
                print(f"[Debug] 已保存 {save_path}")

            all_detections.extend(self.map_to_original_coords(dets, loc))

        # 整图再检
        whole_detections = []
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = self.model.names[cls_id]
                if class_name in ['car', 'truck']:
                    whole_detections.append({'class': class_name,
                                             'confidence': conf,
                                             'bbox': [x1, y1, x2, y2]})

        combined = all_detections + whole_detections
        final_detections = self.apply_nms(combined, image.shape)

        # 绘制最终大图
        result_image = image.copy()
        for det in final_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            color = (0, 255, 0) if det['class'] == 'car' else (255, 0, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"结果保存至: {output_path}")
        return final_detections, result_image


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    detector = YOLOv13TileDetector(
        model_path=r"E:\yolo13_program\yolov13-main\runs\detect\train16\weights\best.pt",
        tile_size=512,
        overlap=0.4,
        conf_threshold=0.2,
        iou_threshold=0.1,
        debug=True  # ★ 打开调试
    )

    input_image = r"C:\Users\xu\Desktop\cv\A-1.jpg"
    output_image = r"E:\yolo13_program\yolov13-main\Result Processing\result_p\9（n）.jpg"

    detections, result_img = detector.detect_image(input_image, output_image)

    print("\n最终检测结果:")
    for i, det in enumerate(detections, 1):
        print(f"{i}: {det['class']} {det['confidence']:.3f}")