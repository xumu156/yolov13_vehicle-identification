"""CNN 目标检测器 — YOLOv13 对比基线模型
使用 5层卷积+池化的轻量级 CNN 进行车辆检测，作为 YOLOv13 的性能对比基线

用法:
    python comparison/cnn.py
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
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ====================== 全局配置 =======================
import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据集路径 (可通过 --data 参数覆盖)
DATA_ROOT = os.path.join(ROOT, 'datasets')
# 训练结果保存路径 (可通过 --output 参数覆盖)
RESULT_ROOT = os.path.join(ROOT, 'comparison', 'result', 'cnn')
# 图像尺寸（显存不足可改为416/320）
IMG_SIZE = 640
# 训练超参数
EPOCHS = 50  # 训练轮次
BATCH_SIZE = 8  # 批次大小（GPU不足改4/2/1）
LEARNING_RATE = 0.001  # 初始学习率
WEIGHT_DECAY = 0.0005  # 权重衰减（防止过拟合）
# 设备配置（自动识别GPU/CPU）
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 自动创建结果目录
os.makedirs(os.path.join(RESULT_ROOT, "weights"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "detections"), exist_ok=True)


# ====================== 核心功能：扫描所有标注文件，检测所有类别ID =======================
def scan_all_class_ids():
    """
    扫描train/valid/test所有标注文件，返回：
    1. 所有出现的类别ID（去重）
    2. 自动生成的类别名称映射（默认ID: "class_XXX"，可手动修改）
    """
    print("=" * 60)
    print("          开始扫描所有标注文件，检测类别ID          ")
    print("=" * 60)

    # 要扫描的数据集子集
    splits = ["train", "valid", "test"]
    all_class_ids = set()  # 存储所有出现的类别ID（去重）
    label_file_count = 0  # 统计标注文件数量

    for split in splits:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir):
            print(f"⚠️ 未找到{split}目录，跳过")
            continue

        # 获取所有标注文件
        label_files = [
            f for f in os.listdir(split_dir)
            if f.endswith('.txt') and os.path.isfile(os.path.join(split_dir, f))
        ]
        if len(label_files) == 0:
            print(f"⚠️ {split}目录下无标注文件，跳过")
            continue

        # 解析每个标注文件
        for label_file in label_files:
            label_path = os.path.join(split_dir, label_file)
            try:
                # 读取标注内容（每行：class_id xc yc w h）
                annots = np.loadtxt(label_path).reshape(-1, 5)
                # 提取所有类别ID并去重
                cls_ids = annots[:, 0].astype(int)
                for cls_id in cls_ids:
                    all_class_ids.add(cls_id)
                label_file_count += 1
            except Exception as e:
                print(f"⚠️ 解析标注文件{label_file}失败：{str(e)}，跳过")

    # 结果汇总
    all_class_ids = sorted(list(all_class_ids))
    print(f"\n✅ 扫描完成！共解析{label_file_count}个标注文件")
    print(f"📌 检测到的所有类别ID：{all_class_ids}")

    # 自动生成类别名称映射（默认class_0, class_1...，可手动修改）
    CLASS_MAP = {cls_id: f"class_{cls_id}" for cls_id in all_class_ids}
    # 手动补充已知类别名称（如0=car，3=other）
    manual_cls_names = {0: "car", 3: "other"}  # 可根据实际情况修改
    for cls_id, cls_name in manual_cls_names.items():
        if cls_id in CLASS_MAP:
            CLASS_MAP[cls_id] = cls_name
    print(f"📌 自动生成的类别映射：{CLASS_MAP}")

    # 确认是否继续
    user_input = input("\n是否确认使用以上类别映射开始训练？(y/n)：")
    if user_input.lower() != 'y':
        print("❌ 训练终止，请手动调整类别映射后重新运行")
        exit()

    # 生成ID到索引的映射
    CLASS_ID_TO_INDEX = {cls_id: idx for idx, cls_id in enumerate(all_class_ids)}
    NUM_CLASSES = len(CLASS_MAP)

    print(f"\n📌 最终配置：")
    print(f"   - 类别总数：{NUM_CLASSES}")
    print(f"   - ID→索引映射：{CLASS_ID_TO_INDEX}")
    print("=" * 60)

    return CLASS_MAP, CLASS_ID_TO_INDEX, NUM_CLASSES


# ====================== 1. 数据集加载类（适配自动检测的类别ID）======================
class YoloDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=IMG_SIZE, class_map=None, class_id_to_index=None):
        self.data_dir = os.path.join(root_dir, split)
        self.img_size = img_size
        self.class_map = class_map
        self.class_id_to_index = class_id_to_index

        # 校验目录
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据集目录不存在：{self.data_dir}")

        # 获取所有图像文件
        self.img_paths = [
            f for f in os.listdir(self.data_dir)
            if f.endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(self.data_dir, f))
        ]
        assert len(self.img_paths) > 0, f"{self.data_dir}下无有效图像文件"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 读取并预处理图像
        img_name = self.img_paths[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像：{img_path}")

        # 保存原始尺寸（用于后续检测框还原）
        orig_h, orig_w = img.shape[:2]
        # 图像预处理
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # 2. 解析YOLO标注
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.data_dir, label_name)
        label = np.zeros(4 + len(self.class_map))  # [x1,y1,x2,y2, cls1, cls2...]

        if os.path.exists(label_path):
            try:
                annots = np.loadtxt(label_path).reshape(-1, 5)
                if len(annots) > 0:
                    cls_id, xc, yc, w, h = annots[0]
                    cls_id = int(cls_id)
                    # 归一化坐标转像素坐标
                    x1 = (xc - w / 2) * self.img_size
                    y1 = (yc - h / 2) * self.img_size
                    x2 = (xc + w / 2) * self.img_size
                    y2 = (yc + h / 2) * self.img_size

                    # 类别ID映射到数组索引
                    if cls_id in self.class_id_to_index:
                        cls_idx = self.class_id_to_index[cls_id]
                        cls_onehot = np.zeros(len(self.class_map))
                        cls_onehot[cls_idx] = 1.0
                    else:
                        print(f"⚠️ 图像{img_name}包含未扫描到的类别ID：{cls_id}，已忽略")
                        cls_onehot = np.zeros(len(self.class_map))

                    label = np.array([x1, y1, x2, y2] + list(cls_onehot))
            except Exception as e:
                print(f"⚠️ 解析图像{img_name}的标注失败：{str(e)}")

        return img_tensor, torch.from_numpy(label).float(), img_name, (orig_w, orig_h)


# ====================== 2. CNN检测模型 =======================
class CNNDetector(nn.Module):
    def __init__(self, num_classes, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size

        # 特征提取网络（5层卷积+池化）
        self.features = nn.Sequential(
            # 卷积块1：3→16
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # 卷积块2：16→32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # 卷积块3：32→64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # 卷积块4：64→128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # 卷积块5：128→256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 全连接预测层
        self.fc_input_size = 256 * (self.img_size // 32) * (self.img_size // 32)
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4 + num_classes)
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 扁平化
        x = x.view(x.size(0), -1)
        # 预测
        out = self.classifier(x)
        # 输出处理：坐标归一化+类别softmax
        coords = torch.sigmoid(out[:, :4]) * self.img_size
        cls_probs = torch.softmax(out[:, 4:], dim=1)
        return torch.cat([coords, cls_probs], dim=1)


# ====================== 3. 损失函数定义 =======================
def detection_loss(pred, target):
    """
    pred: 模型输出 (batch_size, 4+num_classes)
    target: 真实标注 (batch_size, 4+num_classes)
    """
    # 坐标损失（MSE）
    coord_loss = nn.MSELoss()(pred[:, :4], target[:, :4])
    # 类别损失（交叉熵）
    cls_loss = nn.CrossEntropyLoss()(pred[:, 4:], torch.argmax(target[:, 4:], dim=1))
    # 总损失（坐标损失权重更高）
    return coord_loss * 5 + cls_loss


# ====================== 4. 训练函数 =======================
def train_cnn(CLASS_MAP, CLASS_ID_TO_INDEX, NUM_CLASSES):
    print("=" * 60)
    print("          开始训练CNN目标检测模型          ")
    print("=" * 60)
    print(f"设备：{DEVICE} | 轮次：{EPOCHS} | 批次：{BATCH_SIZE} | 学习率：{LEARNING_RATE}")
    print(f"数据集：{DATA_ROOT} | 类别数：{NUM_CLASSES} | 图像尺寸：{IMG_SIZE}")
    print(f"类别映射：{CLASS_MAP}")
    print("=" * 60)

    # 1. 加载数据集
    # 训练集
    train_dataset = YoloDataset(
        DATA_ROOT,
        split="train",
        class_map=CLASS_MAP,
        class_id_to_index=CLASS_ID_TO_INDEX
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows建议设0
        drop_last=True
    )
    # 验证集（无valid则用train）
    val_split = "valid" if os.path.exists(os.path.join(DATA_ROOT, "valid")) else "train"
    val_dataset = YoloDataset(
        DATA_ROOT,
        split=val_split,
        class_map=CLASS_MAP,
        class_id_to_index=CLASS_ID_TO_INDEX
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    print(f"✅ 训练集样本数：{len(train_dataset)} | 验证集样本数：{len(val_dataset)}")

    # 2. 初始化模型/优化器/学习率调度器
    model = CNNDetector(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 3. 训练日志初始化
    train_log = []
    best_val_loss = float('inf')
    start_time = datetime.now()

    # 4. 训练循环
    for epoch in range(EPOCHS):
        # 训练模式
        model.train()
        train_total_loss = 0.0
        epoch_start = datetime.now()

        # 遍历训练集
        for batch_idx, (imgs, labels, img_names, orig_sizes) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # 前向传播+计算损失+反向传播
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = detection_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累加损失
            train_total_loss += loss.item() * imgs.size(0)

            # 打印批次进度
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # 5. 计算训练集平均损失
        avg_train_loss = train_total_loss / len(train_dataset)

        # 6. 验证集评估
        avg_val_loss = validate_cnn(model, val_loader, detection_loss, DEVICE)

        # 7. 学习率更新
        scheduler.step()

        # 8. 记录日志
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        train_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "lr": round(optimizer.param_groups[0]['lr'], 6),
            "time(s)": round(epoch_time, 2)
        })

        # 9. 打印轮次结果
        print("-" * 60)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.2f}s")
        print("-" * 60)

        # 10. 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'class_map': CLASS_MAP,
                'class_id_to_index': CLASS_ID_TO_INDEX
            }, os.path.join(RESULT_ROOT, "weights", "cnn_best.pth"))
            print(f"✅ 保存最优模型（Val Loss: {best_val_loss:.4f}）")

    # 11. 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_map': CLASS_MAP,
        'class_id_to_index': CLASS_ID_TO_INDEX
    }, os.path.join(RESULT_ROOT, "weights", "cnn_final.pth"))

    # 12. 保存训练日志
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(os.path.join(RESULT_ROOT, "logs", "train_log.csv"), index=False)

    # 13. 绘制损失曲线
    plot_loss_curve(train_log)

    # 14. 训练总结
    total_time = (datetime.now() - start_time).total_seconds()
    print("=" * 60)
    print("          训练完成！          ")
    print("=" * 60)
    print(f"总耗时：{total_time / 60:.2f}分钟 | 最优验证损失：{best_val_loss:.4f}")
    print(f"模型保存：{os.path.join(RESULT_ROOT, 'weights')}")
    print(f"日志保存：{os.path.join(RESULT_ROOT, 'logs')}")
    print("=" * 60)

    return model


# ====================== 5. 验证函数 =======================
def validate_cnn(model, val_loader, loss_fn, device):
    model.eval()
    val_total_loss = 0.0
    with torch.no_grad():
        for imgs, labels, img_names, orig_sizes in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            val_total_loss += loss.item() * imgs.size(0)
    return val_total_loss / len(val_loader.dataset)


# ====================== 6. 损失曲线绘制 =======================
def plot_loss_curve(train_log):
    epochs = [log["epoch"] for log in train_log]
    train_losses = [log["train_loss"] for log in train_log]
    val_losses = [log["val_loss"] for log in train_log]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2, color="#1f77b4")
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2, color="#ff7f0e")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("CNN Training & Validation Loss Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULT_ROOT, "logs", "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 损失曲线已保存：{os.path.join(RESULT_ROOT, 'logs', 'loss_curve.png')}")


# ====================== 7. 检测可视化 =======================
def detect_visualize(model, CLASS_MAP, CLASS_ID_TO_INDEX):
    print("\n" + "=" * 60)
    print("          开始检测结果可视化          ")
    print("=" * 60)

    # 加载验证集
    val_split = "valid" if os.path.exists(os.path.join(DATA_ROOT, "valid")) else "train"
    val_dataset = YoloDataset(
        DATA_ROOT,
        split=val_split,
        class_map=CLASS_MAP,
        class_id_to_index=CLASS_ID_TO_INDEX
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        # 反向映射：索引→原始类别ID
        index_to_cls_id = {v: k for k, v in CLASS_ID_TO_INDEX.items()}

        for idx, (imgs, labels, img_names, orig_sizes) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            orig_w, orig_h = orig_sizes[0]
            img_name = img_names[0]

            # 模型预测
            outputs = model(imgs)

            # 解析预测结果
            pred_coords = outputs[0, :4].cpu().numpy()
            # 坐标还原到原始图像尺寸
            pred_x1 = int(pred_coords[0] * orig_w / IMG_SIZE)
            pred_y1 = int(pred_coords[1] * orig_h / IMG_SIZE)
            pred_x2 = int(pred_coords[2] * orig_w / IMG_SIZE)
            pred_y2 = int(pred_coords[3] * orig_h / IMG_SIZE)

            # 解析类别
            pred_cls_idx = torch.argmax(outputs[0, 4:]).item()
            pred_cls_id = index_to_cls_id[pred_cls_idx]
            pred_cls_name = CLASS_MAP[pred_cls_id]
            pred_conf = outputs[0, 4 + pred_cls_idx].item()

            # 读取原始图像
            img_path = os.path.join(DATA_ROOT, val_split, img_name)
            img = cv2.imread(img_path)

            # 绘制检测框
            cv2.rectangle(img, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{pred_cls_name} {pred_conf:.2f}",
                (pred_x1, pred_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # 保存可视化结果
            save_path = os.path.join(RESULT_ROOT, "detections", f"det_{img_name}")
            cv2.imwrite(save_path, img)

            # 打印进度
            if (idx + 1) % 10 == 0:
                print(f"✅ 已处理 {idx + 1}/{len(val_loader)} 张图像 | 保存至：{save_path}")

            # 处理全部图像（注释下面两行即可检测所有图像）
            # if idx >= 49:
            #     break

    print(f"✅ 检测可视化完成！结果保存至：{os.path.join(RESULT_ROOT, 'detections')}")


# ====================== 8. 主函数（先扫描ID，再训练）======================
if __name__ == "__main__":
    # 第一步：扫描所有标注文件，检测所有类别ID
    CLASS_MAP, CLASS_ID_TO_INDEX, NUM_CLASSES = scan_all_class_ids()

    # 第二步：训练模型
    trained_model = train_cnn(CLASS_MAP, CLASS_ID_TO_INDEX, NUM_CLASSES)

    # 第三步：可视化检测结果（检测全部图像）
    detect_visualize(trained_model, CLASS_MAP, CLASS_ID_TO_INDEX)

    print("\n🎉 所有任务完成！训练结果汇总：")
    print(f"📌 模型权重：{os.path.join(RESULT_ROOT, 'weights')}")
    print(f"📌 训练日志：{os.path.join(RESULT_ROOT, 'logs')}")
    print(f"📌 检测结果：{os.path.join(RESULT_ROOT, 'detections')}")