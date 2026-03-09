# JPEG Image Enhancement Module

这是一个专注于 **JPEG 图像去压缩伪影** 的增强模块，属于 FAST-JPEG 项目的核心部分。该模块利用深度学习技术（PyTorch）提升经过 JPEG 压缩后的图像质量。

##  项目特性

* **深度学习驱动**：基于 PyTorch 实现的高性能图像修复算法。
* **完整工作流**：从原始数据生成到模型训练、测试的端到端支持。
* **灵活扩展**：核心模型代码位于 `sci_enhancer.py`，方便进行架构微调。

##  目录结构说明

```text
.
├── dataset.py          # 自定义数据加载脚本
├── sci_enhancer.py     # 核心增强模型定义
├── train.py            # 模型训练主程序
├── test.py             # 模型测试与性能评估
├── generate_data.py    # 训练/测试数据预处理脚本
├── data_split.py       # 数据集划分工具
├── data/               # (Git 忽略) 存放训练与测试数据集
├── checkpoints/        # (Git 忽略) 存放训练好的模型权重
└── results/            # (Git 忽略) 存放增强后的对比结果图
```
##  环境要求
Python: 3.8+

框架: PyTorch

领域知识: 视频压缩与图像编码相关技术

##  快速开始
1. 数据准备
首先，你需要生成成对的（原图 vs 压缩图）训练数据：

```python
python generate_data.py
python data_split.py
```
2. 模型训练
启动训练过程，权重将默认保存至 checkpoints/ 文件夹：

```python
python train.py 
```

3. 性能评估
在测试集上运行推理并查看指标结果：

```python
python test.py
```
 
##  备注
本模块目前作为独立子项目在 GitHub 维护，后续将与 FAST-JPEG-MAIN 主框架进行深度整合。
Author: yangjingwen

Current Date: 2026-03-09

License: MIT
