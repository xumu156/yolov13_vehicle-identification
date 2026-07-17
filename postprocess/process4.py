"""后处理工具4 — 分块检测变体 (完整注释版)
YOLOv13TileDetector 的详细注释版本，含完整参数说明和调试信息

用法:
    from postprocess.process4 import YOLOv13TileDetector
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime


class YOLOv13TileDetector:
    # 初始化类
    def __init__(self,
                 model_path,  # YOLO模型文件路径
                 tile_size=640,  # 切块大小，默认640x640像素
                 overlap=0.3,  # 切块重叠率，默认30%
                 conf_threshold=0.25,  # 置信度阈值，默认0.25
                 iou_threshold=0.5,  # IoU阈值，默认0.5
                 debug=False):  # 调试模式开关，默认关闭
        self.model = YOLO(model_path)  # 加载YOLO模型
        self.tile_size = tile_size  # 设置切块大小
        self.overlap = overlap  # 设置重叠率
        self.conf_threshold = conf_threshold  # 设置置信度阈值
        self.iou_threshold = iou_threshold  # 设置IoU阈值
        self.debug = debug  # 设置调试模式

        # 自动创建调试目录
        if self.debug:  # 如果开启调试模式
            # 生成唯一运行ID，使用当前时间戳
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 调试目录路径
            self.debug_dir = rf"E:\yolo13_program\yolov13-main\Result Processing\result_p\{run_id}"
            os.makedirs(self.debug_dir, exist_ok=True)  # 创建目录（如果不存在）
            print(f"[Debug] tile 图片将保存在: {self.debug_dir}")  # 打印调试信息

    # ---------------- 1. 切图 ----------------
    def split_image(self, image):
        h, w = image.shape[:2]  # 获取图像高度和宽度
        stride = int(self.tile_size * (1 - self.overlap))  # 计算滑动步长 = 切块大小*(1-重叠率)
        tiles, locations = [], []  # 初始化存储切块和位置的列表

        # 双重循环遍历图像
        for y in range(0, h, stride):  # y方向遍历
            for x in range(0, w, stride):  # x方向遍历
                # 计算切块的右下角坐标（不超过图像边界）
                x2, y2 = min(x + self.tile_size, w), min(y + self.tile_size, h)
                # 提取图像切块
                tile = image[y:y2, x:x2]
                tiles.append(tile)  # 添加到切块列表
                locations.append((x, y, x2, y2))  # 保存切块位置信息（左上角和右下角坐标）
        print(f"图像切割成 {len(tiles)} 个块")  # 打印切块数量
        return tiles, locations  # 返回切块和位置信息

    # ---------------- 2. 单 tile 检测 + 面积过滤 ----------------
    def detect_tile(self, tile):
        h_tile, w_tile = tile.shape[:2]  # 获取切块的高度和宽度
        effective = tile[:h_tile, :w_tile]  # 提取有效区域（实际上就是整个切块）
        # 使用YOLO模型检测切块，设置置信度阈值，不显示详细输出
        results = self.model(effective, conf=self.conf_threshold, verbose=False)

        detections = []  # 初始化检测结果列表
        if results[0].boxes is not None:  # 如果有检测结果
            for box in results[0].boxes:  # 遍历每个检测框
                cls_id = int(box.cls.item())  # 获取类别ID
                conf = box.conf.item()  # 获取置信度
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框坐标（左上角x,y,右下角x,y）
                class_name = self.model.names[cls_id]  # 获取类别名称
                if class_name not in ['car', 'truck']:  # 只保留'car'和'truck'类别
                    continue  # 跳过其他类别

                # 计算检测框的宽度和高度
                w = x2 - x1
                h = y2 - y1
                # 面积过滤：如果检测框面积大于切块面积的60% 或者 最长边大于切块边长的90%，则跳过
                if w * h > 0.6 * self.tile_size ** 2 or max(w, h) > 0.9 * self.tile_size:
                    continue  # 跳过过大检测框
                # 将有效检测结果添加到列表
                detections.append({'class': class_name,
                                   'confidence': conf,
                                   'bbox': [x1, y1, x2, y2]})
        return detections  # 返回检测结果

    # ---------------- 3. 坐标映射 + clamp ----------------
    def map_to_original_coords(self, detections, location):
        ox, oy, x2_max, y2_max = location  # 解包位置信息：左上角x,y和右下角x,y（切块边界）
        original_detections = []  # 初始化映射后的检测结果列表
        for det in detections:  # 遍历每个检测结果
            x1, y1, x2, y2 = det['bbox']  # 获取检测框坐标
            # 将切块内的相对坐标映射回原图的绝对坐标
            x1 += ox;
            y1 += oy;
            x2 += ox;
            y2 += oy
            # Clamp操作：确保坐标不超出切块边界
            x1 = max(ox, min(x1, x2_max))
            y1 = max(oy, min(y1, y2_max))
            x2 = max(ox, min(x2, x2_max))
            y2 = max(oy, min(y2, y2_max))
            # 添加映射后的检测结果（保持其他属性不变）
            original_detections.append({**det, 'bbox': [x1, y1, x2, y2]})
        return original_detections  # 返回映射后的检测结果

    # ---------------- 4. NMS + 全局面积过滤 ----------------
    def apply_nms(self, detections, img_shape):
        if not detections:  # 如果没有检测结果
            return []  # 返回空列表

        original_h, original_w = img_shape[:2]  # 获取原图高度和宽度
        # 提取所有检测框坐标，转换为numpy数组
        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        # 提取所有置信度分数
        scores = np.array([d['confidence'] for d in detections], dtype=np.float32)
        # 提取所有类别
        classes = np.array([d['class'] for d in detections])

        unique_classes = np.unique(classes)  # 获取唯一类别
        final_detections = []  # 初始化最终检测结果列表

        # 对每个类别分别处理
        for cls in unique_classes:
            mask = classes == cls  # 创建当前类别的掩码
            cls_boxes = boxes[mask]  # 提取当前类别的检测框
            cls_scores = scores[mask]  # 提取当前类别的置信度
            # 提取当前类别的原始检测结果
            cls_dets = [detections[i] for i in np.where(mask)[0]]

            # 软NMS：线性加权衰减，返回保留的索引
            keep = self.soft_nms(cls_boxes, cls_scores, sigma=0.5, Nt=0.1, threshold=0.15, method=1)
            # 将保留的检测结果添加到最终列表
            final_detections.extend([cls_dets[i] for i in keep])

        # 全局面积过滤
        img_area = original_w * original_h  # 计算图像总面积
        # 过滤掉面积大于图像面积60%的检测框
        final_detections = [
            d for d in final_detections
            if (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) < 0.6 * img_area
        ]
        return final_detections  # 返回最终检测结果

    def soft_nms(self, boxes, scores, sigma=0.5, Nt=0.1, threshold=0.15, method=1):
        """安全版 Soft-NMS，返回保留索引"""
        boxes = boxes.copy().tolist()  # 复制并转换为列表
        scores = scores.copy().tolist()  # 复制并转换为列表
        inds = list(range(len(boxes)))  # 创建索引列表
        keep = []  # 初始化保留索引列表

        while boxes:  # 循环直到所有框都被处理
            # 找到当前最高分的框
            max_idx = np.argmax(scores)
            keep.append(inds[max_idx])  # 将其索引添加到保留列表
            max_box = boxes[max_idx]  # 获取最高分框的坐标

            # 计算其余框与最大框的 IoU
            # 计算交集区域的左、上、右、下边界
            left = np.maximum(max_box[0], [b[0] for b in boxes])
            top = np.maximum(max_box[1], [b[1] for b in boxes])
            right = np.minimum(max_box[2], [b[2] for b in boxes])
            bottom = np.minimum(max_box[3], [b[3] for b in boxes])
            w = np.maximum(0, right - left)  # 计算交集宽度（确保非负）
            h = np.maximum(0, bottom - top)  # 计算交集高度（确保非负）
            inter = w * h  # 计算交集面积

            # 计算最大框的面积
            area1 = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
            # 计算所有框的面积
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            # 计算并集面积 = 最大框面积 + 其他框面积 - 交集面积
            union = area1 + np.array(areas) - inter  # ★ 修复这里（使用numpy数组运算）
            # 计算IoU = 交集面积 / 并集面积（加微小值避免除零）
            iou = inter / (union + 1e-7)

            # 衰减分数：根据方法选择不同的权重计算方式
            if method == 1:  # 线性加权
                weight = np.where(iou < Nt, 1., 1. - iou)  # IoU小于阈值Nt时权重为1，否则为1-IoU
            else:  # 高斯加权
                weight = np.exp(-(iou * iou) / sigma)  # 使用高斯函数计算权重
            # 应用权重衰减分数
            scores = [s * w for s, w in zip(scores, weight)]

            # 剔除已处理框 & 低分框
            remain = [(b, s, i) for b, s, i in zip(boxes, scores, inds)
                      if i != inds[max_idx] and s > threshold]  # 保留非最大框且分数高于阈值的框
            if not remain:  # 如果没有剩余框
                break  # 结束循环
            # 解包剩余框的信息
            boxes, scores, inds = map(list, zip(*remain))

        return keep  # 返回保留的索引列表

    # ---------------- 5. 主流程 ----------------
    def detect_image(self, image_path, output_path=None):
        image = cv2.imread(image_path)  # 读取输入图像
        if image is None:  # 检查图像是否读取成功
            raise ValueError(f"无法读取图像: {image_path}")  # 抛出错误
        original_h, original_w = image.shape[:2]  # 获取原图尺寸

        # 步骤1：切图
        tiles, locations = self.split_image(image)
        all_detections = []  # 初始化所有检测结果列表

        # 遍历每个切块
        for i, (tile, loc) in enumerate(zip(tiles, locations)):
            # 步骤2：检测单个切块
            dets = self.detect_tile(tile)

            # ★ 调试：保存 tile 调试图
            if self.debug:  # 如果开启调试模式
                tile_vis = tile.copy()  # 复制切块用于可视化
                for det in dets:  # 在切块上绘制检测框
                    x1, y1, x2, y2 = map(int, det['bbox'])  # 转换为整数坐标
                    cv2.rectangle(tile_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色矩形框
                    # 添加类别和置信度标签
                    cv2.putText(tile_vis,
                                f"{det['class']} {det['confidence']:.2f}",
                                (x1, y1 - 5),  # 位置在框上方
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 字体设置
                # 保存调试图
                save_path = os.path.join(self.debug_dir, f"tile{i:03d}_{loc}.jpg")
                cv2.imwrite(save_path, tile_vis)
                print(f"[Debug] 已保存 {save_path}")

            # 步骤3：坐标映射
            all_detections.extend(self.map_to_original_coords(dets, loc))

        # 整图再检：对整个图像进行一次检测（补充可能漏检的目标）
        whole_detections = []
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = self.model.names[cls_id]
                if class_name in ['car', 'truck']:  # 只保留'car'和'truck'类别
                    whole_detections.append({'class': class_name,
                                             'confidence': conf,
                                             'bbox': [x1, y1, x2, y2]})

        # 合并切块检测结果和整图检测结果
        combined = all_detections + whole_detections
        # 步骤4：应用NMS和过滤
        final_detections = self.apply_nms(combined, image.shape)

        # 绘制最终大图：在原图上绘制最终检测结果
        result_image = image.copy()  # 复制原图
        for det in final_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])  # 获取坐标并转换为整数
            # 确保坐标不超出图像边界
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            # 根据类别选择颜色：car为绿色，truck为蓝色
            color = (0, 255, 0) if det['class'] == 'car' else (255, 0, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)  # 绘制边界框
            label = f"{det['class']} {det['confidence']:.2f}"  # 创建标签文本
            # 获取文本尺寸用于绘制背景框
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            # 绘制文本背景框（半透明效果）
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)  # -1表示填充矩形
            # 绘制文本
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 保存结果图像
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"结果保存至: {output_path}")
        return final_detections, result_image  # 返回检测结果和可视化图像


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 创建检测器实例
    detector = YOLOv13TileDetector(
        model_path=r"E:\yolo13_program\yolov13-main\runs\detect\train16\weights\best.pt",  # 模型路径
        tile_size=512,  # 切块大小512x512
        overlap=0.4,  # 重叠率40%
        conf_threshold=0.2,  # 置信度阈值0.2
        iou_threshold=0.1,  # IoU阈值0.1
        debug=True  # ★ 打开调试模式
    )

    # 输入输出路径
    input_image = r"C:\Users\xu\Desktop\cv\A-1.jpg"
    output_image = r"E:\yolo13_program\yolov13-main\Result Processing\result_p\9（n）.jpg"

    # 执行检测
    detections, result_img = detector.detect_image(input_image, output_image)

    # 打印检测结果
    print("\n最终检测结果:")
    for i, det in enumerate(detections, 1):
        print(f"{i}: {det['class']} {det['confidence']:.3f}")