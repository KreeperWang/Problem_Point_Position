train_blip.py--only use blip2-2.7b model
train_blip_enhanced.py --use enhanced frame 
utils.py -- for enhanced frame
config.py -- for both 2 method


# PGDP5K 几何图形点位置识别数据集

![Dataset Banner](https://img.shields.io/badge/Dataset-PGDP5K-blue) ![Task](https://img.shields.io/badge/Task-Point%20Position%20Detection-green) ![Language](https://img.shields.io/badge/Language-Python-orange)

## 📖 数据集简介

**PGDP5K (Plane Geometry Diagram Parsing 5K)** 是一个几何图形形式化数据集。该数据集包含5000张几何示意图，每张图都标注了图中关键点的精确坐标位置。

### 🎯 任务目标
- **主要任务**: 从几何示意图中准确识别并输出指定点的坐标位置
- **应用场景**: 几何教学辅助、自动批改系统、几何图形分析
- **技术挑战**: 多模态理解、精确定位、点名映射

## 📊 数据集统计

| 项目 | 数值 |
|------|------|
| **总图像数量** | 5,000张 |
| **训练集** | 4,000张 |
| **测试集** | 1,000张 |
| **图像格式** | PNG |
| **标注格式** | JSON |
| **点坐标精度** | 像素级 |

## 📁 文件结构

```
Problem_Point_Position/
├── README.md              # 本文档
├── train.json            # 训练集标注文件 (4,000条)
├── test.json             # 测试集标注文件 (1,000条)
├── cal_acc.py            # 评估脚本
└── Images/               # 图像文件夹
    ├── 0.png
    ├── 1.png
    ├── ...
    └── 4999.png
```

## 📋 数据格式说明

### 图像文件
- **命名规则**: `{ID}.png`，其中ID为0-4999的整数
- **图像内容**: 包含各种几何图形的示意图，如三角形、四边形、圆形等
- **图像质量**: 清晰的几何图形，点标记明显

### 标注文件格式

JSON文件采用以下格式：

```json
{
  "图像ID": {
    "点名": [x坐标, y坐标],
    "点名": [x坐标, y坐标],
    ...
  }
}
```

#### 示例
```json
{
  "4238": {
    "Y": [711.36, 85.40397350993378],
    "R": [126.72, 183.12582781456953],
    "S": [180.0, 344.9006622516556],
    "X": [764.0228571428571, 247.17880794701986],
    "T": [446.4, 127.28476821192052],
    "Z": [443.88, 302.1473509933775]
  }
}
```

### 数据特点
- **坐标系**: 图像坐标系，原点(0,0)位于图像左上角
- **点命名**: 使用大写英文字母(A-Z)命名几何图形中的关键点
- **坐标精度**: 支持浮点数坐标，精确到小数点后多位
- **点数量**: 每张图像包含2-10个不等的标注点

## 🚀 快速开始

### 环境要求
```bash
Python >= 3.7
numpy
matplotlib
json
```

### 数据加载示例

```python
import json
import os

# 加载训练数据
def load_dataset(annotation_file, image_dir):
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    dataset = []
    for image_id, points in annotations.items():
        image_path = os.path.join(image_dir, f"{image_id}.png")
        if os.path.exists(image_path):
            dataset.append({
                'image_id': image_id,
                'image_path': image_path,
                'points': points
            })
    
    return dataset

# 使用示例
train_data = load_dataset('train.json', 'Images/')
test_data = load_dataset('test.json', 'Images/')

print(f"训练集样本数: {len(train_data)}")
print(f"测试集样本数: {len(test_data)}")
```

## 📏 评估方法

我们提供了完整的评估脚本 `cal_acc.py`，支持多种评估指标：

### 核心评估指标

1. **欧几里得距离误差 (Euclidean Distance Error)**
   - 计算预测点与真实点之间的像素距离
   - 公式: `distance = √[(x₁-x₂)² + (y₁-y₂)²]`

2. **平均绝对误差 (MAE)**
   - 所有匹配点的平均距离误差
   - 反映模型整体定位精度

3. **阈值准确率 (Accuracy@Threshold)**
   - 在不同像素阈值下的准确率
   - 默认阈值: 5px, 10px, 15px, 20px

4. **点检测率曲线 (PDR Curve)**
   - Point Detection Rate vs Distance Threshold
   - 可视化模型在不同精度要求下的性能

5. **归一化欧几里得距离误差 (Normalized Euclidean Distance Error)**
   - 计算预测点与真实点之间的距离，并除以图像对角线长度，实现尺度无关的误差评估
   - 公式: `normalized_distance = distance / diag_len`，其中`diag_len = sqrt(width^2 + height^2)`
   - 输出为所有点的平均归一化距离误差，范围0~1，越小越好

6. **相对点间距离误差 (Relative Point Distance Error)**
   - 计算所有点对之间的预测距离与真实距离的平均误差，反映模型对点间相对关系的把握
   - 公式: `mean(|d_pred(i,j) - d_gt(i,j)|)`，其中`d_pred(i,j)`为预测点i和j的距离，`d_gt(i,j)`为真实距离
   - 输出为所有图像的平均相对点间距离误差，单位为像素，越小越好

### 使用评估脚本

```python
from cal_acc import PointAccuracyCalculator

# 创建评估器
calculator = PointAccuracyCalculator(
    gt_file='test.json',           # 真实标注
    pred_file='predictions.json',  # 模型预测结果
    image_size=(800, 600)         # 图像尺寸(可选)
)

# 计算所有指标
results = calculator.calculate_all_metrics(
    thresholds=[5, 10, 15, 20, 25]
)

# 打印结果
calculator.print_results()

# 绘制PDR曲线
calculator.plot_pdr_curve()

# 保存结果
calculator.save_results('evaluation_results.json')

# 获取归一化误差和相对点间距离误差
print('归一化平均距离误差:', results.get('mean_normalized_error'))
print('相对点间距离误差:', results.get('relative_point_distance_error'))
```

### 预测结果格式要求

模型的预测结果需要保存为与标注文件相同格式的JSON文件：

```json
{
  "图像ID": {
    "点名": [预测x坐标, 预测y坐标],
    "点名": [预测x坐标, 预测y坐标]
  }
}
```

## 🏆 基准性能

### 评估基准
- **优秀**: MAE < 5px, Accuracy@10px > 90%
- **良好**: MAE < 10px, Accuracy@15px > 85%
- **及格**: MAE < 20px, Accuracy@20px > 70%

### 评估注意事项
1. **点名匹配**: 预测结果中的点名必须与标注文件中的点名完全一致
2. **缺失处理**: 如果模型未预测某个点，该点将被计为最大误差
3. **额外点**: 预测文件中的额外点（标注中不存在）将被忽略
4. **坐标范围**: 坐标应在合理的图像范围内

### 🎓 致营员的话

这个数据集是专门为您的多模态大模型微调实践而准备的。通过这个任务，您将学会：

1. **多模态理解**: 如何让AI同时理解图像和文本信息
2. **精确定位**: 如何训练模型进行像素级的精确定位
3. **几何推理**: 如何让AI理解几何图形的空间关系
4. **评估方法**: 如何设计合理的评估指标来衡量模型性能

建议您从简单的几何图形开始，逐步提升模型的复杂度。记住，好的结果需要耐心的调试和优化！

**祝您学习顺利！🚀**
