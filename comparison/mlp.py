"""MLP 目标检测器 — YOLOv13 对比基线模型
使用多层感知机进行车辆检测回归+分类，作为最简基线与 YOLOv13 对比

用法:
    python comparison/mlp.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime

# ====================== 全局配置 ======================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据集根路径
DATA_ROOT = os.path.join(ROOT, 'datasets_merged')
# 结果保存根路径
RESULT_ROOT = os.path.join(ROOT, 'comparison', 'result')
# 创建结果目录（自动创建子目录：mlp/cnn/rnn）
os.makedirs(os.path.join(RESULT_ROOT, "mlp"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "cnn"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "rnn"), exist_ok=True)


# ====================== 数据集加载类 ======================
class CarDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=640):
        self.img_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)
        self.img_size = img_size
        self.img_paths = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取图像
        img_name = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 读取标注
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        label = np.zeros(6)  # [x1,y1,x2,y2,car,truck]

        if os.path.exists(label_path):
            annot = np.loadtxt(label_path).reshape(-1, 5)
            if len(annot) > 0:
                cls_id, x_c, y_c, w, h = annot[0]
                x1 = x_c - w / 2
                y1 = y_c - h / 2
                x2 = x_c + w / 2
                y2 = y_c + h / 2
                car = 1.0 if cls_id == 0 else 0.0
                truck = 1.0 if cls_id == 1 else 0.0
                label = np.array([x1, y1, x2, y2, car, truck])

        return img_tensor, torch.from_numpy(label).float(), img_name  # 新增返回图像名

# ====================== MLP模型定义与训练 ======================
class MLPDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(640 * 640 * 3, 2048), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        x = x.flatten(1)
        out = self.fc(x)
        return torch.cat([torch.sigmoid(out[:, :4]), torch.softmax(out[:, 4:], dim=1)], dim=1)


def train_mlp():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载数据集
    train_dataset = CarDataset(DATA_ROOT, split="train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 初始化模型
    model = MLPDetector().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)

    # 训练日志记录
    train_log = []
    epochs = 50
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        start_time = datetime.now()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 计算本轮指标
        avg_loss = total_loss / len(train_loader)
        epoch_time = (datetime.now() - start_time).total_seconds()
        train_log.append({"epoch": epoch + 1, "loss": avg_loss, "time(s)": epoch_time})
        print(f"MLP Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    # 1. 保存模型权重
    torch.save(model.state_dict(), os.path.join(RESULT_ROOT, "mlp", "mlp_car_detector.pth"))
    # 2. 保存训练日志
    pd.DataFrame(train_log).to_csv(os.path.join(RESULT_ROOT, "mlp", "train_log.csv"), index=False)
    # 3. 测试并保存检测结果
    test_mlp(model, device)


def test_mlp(model, device):
    # 加载测试集（用train集替代，若无val集）
    test_dataset = CarDataset(DATA_ROOT, split="train")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    result_list = []
    with torch.no_grad():
        for imgs, labels, img_names in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            # 解析结果
            bbox = outputs[0, :4].cpu().numpy()
            cls = torch.argmax(outputs[0, 4:]).item()
            cls_name = "car" if cls == 0 else "truck"
            conf = outputs[0, 4 + cls].item()

            # 保存检测结果图像
            img_path = os.path.join(DATA_ROOT, "images", "train", img_names[0])
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = map(int, [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(RESULT_ROOT, "mlp", f"det_{img_names[0]}"), img)

            # 记录检测结果
            result_list.append({
                "image": img_names[0],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class": cls_name, "confidence": conf
            })

    # 保存检测结果表格
    pd.DataFrame(result_list).to_csv(os.path.join(RESULT_ROOT, "mlp", "detection_result.csv"), index=False)
    print("MLP检测结果已保存至:", os.path.join(RESULT_ROOT, "mlp"))