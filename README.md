# YOLOv13 车辆检测系统 (Vehicle Detection)

基于 YOLOv13 的 8 类车辆目标检测系统。项目已精检重构：删除数据集和模型权重、重组目录结构、修复硬编码路径为相对路径+argparse、添加 .gitignore/README/requirements，从 9.5GB 精简至 161KB。

> 📌 数据集来源: [Kaggle - vehicles.v2-release.voc](https://www.kaggle.com/) | 原始 12 类别 → 合并为 8 类

## 检测类别 (8类)

| ID | 类别 | 说明 |
|----|------|------|
| 0 | big truck | 大型卡车 |
| 1 | big car | 大型汽车 |
| 2 | bus | 巴士/公交车 |
| 3 | car | 小汽车 |
| 4 | mid_truck | 中型卡车 |
| 5 | null | 无效目标 |
| 6 | small_truck | 小型卡车 |
| 7 | truck | 卡车 |

## 项目结构

```
yolov13_vehicle-identification/
├── train.py                 # 主训练脚本 (SGD 优化器)
├── train_config2.py         # 训练配置2 (AdamW + 混合精度)
├── train_config1.py         # 训练配置1 (单图快速推理)
├── val.py                   # 模型验证脚本
├── test_model.py            # 模型信息查看
├── daily_val.py             # 日常验证指标
├── predict_image.py         # 单张图片预测
├── predict_video.py         # 实时摄像头检测
├── camera_test.py           # 摄像头可用性测试
├── data.yaml                # 数据集配置文件
├── requirements.txt         # 依赖包列表
├── preprocess/              # 数据预处理工具
│   ├── voc_to_yolo.py       # Pascal VOC → YOLO 格式转换
│   ├── merge_categories.py  # 12类合并为8类
│   └── category_mapping.py  # 类别映射定义
├── comparison/              # 对比模型 (基线实验)
│   ├── cnn.py               # CNN 检测器
│   ├── mlp.py               # MLP 检测器
│   └── rnn.py               # RNN 检测器
└── postprocess/             # 后处理工具
    ├── tile_detector.py     # 分块检测器
    ├── process2.py          # 处理脚本2
    ├── process3.py          # 处理脚本3
    └── process4.py          # 处理脚本4
```

## 数据集

> 📌 **数据集来源**: [Kaggle - vehicles.v2-release.voc](https://www.kaggle.com/)
>
> 原始数据集包含 12 个车辆类别，经过预处理后合并为 8 个类别。

### 数据准备

1. 从 Kaggle 下载 `vehicles.v2-release.voc` 数据集
2. 解压到项目根目录下的 `vehicles.v2-release.voc/` 文件夹
3. 运行预处理脚本：

```bash
# 步骤1: Pascal VOC XML → YOLO 格式 (12类)
python preprocess/voc_to_yolo.py --input vehicles.v2-release.voc --output datasets

# 步骤2: 合并相似类别 (12类 → 8类)
python preprocess/merge_categories.py --input datasets --output datasets_merged
```

### 数据集目录结构

```
datasets_merged/
├── data.yaml          # 数据集配置
├── train/             # 训练集
│   ├── img_001.jpg
│   ├── img_001.txt    # YOLO格式标注
│   └── ...
├── valid/             # 验证集
└── test/              # 测试集
```

## 环境安装

```bash
# 创建虚拟环境
conda create -n yolov13 python=3.10
conda activate yolov13

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
# 基础训练 (SGD, 小尺寸快速训练)
python train.py --weights yolov13n.pt --data data.yaml --epochs 400 --batch 8 --imgsz 416

# 精细训练 (AdamW, 混合精度, 大尺寸)
python train_config2.py --weights yolov13s.pt --data data.yaml --epochs 250 --batch 8 --imgsz 640
```

### 模型验证

```bash
python val.py --weights runs/detect/train/weights/best.pt --data data.yaml
```

### 实时检测

```bash
# 摄像头实时检测
python predict_video.py --weights runs/detect/train/weights/best.pt --source 0

# 单张图片预测
python predict_image.py --weights runs/detect/train/weights/best.pt --source test.jpg
```

## 训练建议

- **显存不足**: 减小 `--batch` 和 `--imgsz` 参数
- **过拟合**: 增加数据增强，减小 `--epochs`，增大 `--patience`
- **精度不够**: 尝试更大的模型 (`yolov13s.pt`, `yolov13m.pt`)
- **训练加速**: 使用 `--amp --cache` 参数

## 依赖

- Python >= 3.8
- PyTorch >= 2.0
- Ultralytics >= 8.0
- OpenCV, NumPy, Pandas, Matplotlib
