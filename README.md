# 情感分析（Sentiment Analysis）

基于 **RNN / LSTM / GRU** 的中文购物书评情感分类，是一个二分类任务（正面 / 负面）。

## 项目结构

```
.
├── train.py          # 核心脚本：模型 + 训练 + 评估 + 模型对比
├── utils.py          # 数据加载与预处理（jieba 分词、词表、序列定长）
├── results.txt       # 三个模型的测试准确率对比结果
└── data/             # 数据目录（需自行放入）
    ├── pos.csv       # 正面评论（10696 条）
    └── neg.csv       # 负面评论（10428 条）
```

## 环境依赖

- Python 3.x
- PyTorch
- jieba

```bash
pip install torch jieba
```

## 运行方法

```bash
python3 train.py
```

脚本会自动完成：加载数据 → 分别训练 RNN / LSTM / GRU 三个模型 → 保存每个模型的最佳权重（`model_*.pth`）→ 输出对比表并写入 `results.txt`。

## 实验结果

| 模型 | 测试准确率 |
| --- | --- |
| RNN | 91.39% |
| LSTM | 92.66% |
| GRU | **92.95%** |

## 模型设计

- **Embedding**：词表 50938 → 128 维词向量
- **编码器**：RNN / LSTM / GRU（单层，隐维度 128）
- **句子编码**：对时间步做平均池化
- **分类头**：全连接 → 2 类
- **训练**：CrossEntropyLoss + Adam（lr=1e-3）+ 梯度裁剪
