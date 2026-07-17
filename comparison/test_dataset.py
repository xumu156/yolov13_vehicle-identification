"""
数据集加载测试脚本
验证数据集是否正确加载

用法:
    python comparison/test_dataset.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'datasets_merged')

# 测试数据集加载 - 需要先运行对应的训练脚本导入 CarDataset 类
print(f"数据集路径: {DATA_ROOT}")
print("请从 cnn.py / mlp.py / rnn.py 中导入 CarDataset 类进行测试")
print(f"train 目录: {os.path.join(DATA_ROOT, 'train')}")
print(f"valid 目录: {os.path.join(DATA_ROOT, 'valid')}")
print(f"test 目录: {os.path.join(DATA_ROOT, 'test')}")
