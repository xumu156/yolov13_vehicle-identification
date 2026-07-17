"""类别映射定义 — 12类 → 8类合并方案
将 Kaggle vehicles.v2-release.voc 的原始 12 个车辆类别合并为 8 类
"""

# 新的类别合并方案
category_merging = {
    # 保持不变的类别
    'big truck': 'big truck',
    'big car': 'big car',
    'car': 'car',
    'null': 'null',

    # 合并的类别
    'bus-l-': 'bus',  # 合并所有bus类型
    'bus-s-': 'bus',  # 合并所有bus类型
    'mid truck': 'mid_truck',  # 保持独立
    'small truck': 'small_truck',  # 保持独立
    'truck-l-': 'truck',  # 合并各种truck
    'truck-m-': 'truck',  # 合并各种truck
    'truck-s-': 'truck',  # 合并各种truck
    'truck-xl-': 'big_truck'  # 并入big truck
}

# 新的类别列表（合并后）
new_categories = [
    'big truck', 'big car', 'bus', 'car', 'mid_truck',
    'null', 'small_truck', 'truck'
]