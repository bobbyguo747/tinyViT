# 代码讲解与 Shape 数据流

## 1. 建议阅读顺序

1. `config.yaml`：理解实验参数；
2. `src/dataset.py`：理解数据划分；
3. `src/model.py`：理解 TinyViT 和 CNN；
4. `src/inspect_model.py`：核对所有 shape；
5. `src/train.py`：理解 loss、梯度、验证和 checkpoint；
6. `src/evaluate.py`：理解独立测试；
7. `src/plot_results.py`：理解图表来源。

## 2. 数据读取

`src/dataset.py` 使用 `FashionMNIST` 自动下载数据，通过 `ToTensor` 转为 `[1,28,28]`，再使用均值 0.2860、标准差 0.3530 标准化。

`stratified_split_indices` 按类别分层划分 54,000/6,000，固定 seed 42，并显式检查训练和验证索引零重叠。训练 Loader 打乱样本，验证和测试 Loader 保持固定顺序。

## 3. Patch Embedding

`PatchEmbedding` 使用 `kernel_size=stride=4` 的卷积同时完成不重叠 patch 切分和线性投影：

```text
[B,1,28,28]
→ Conv2d(1,64,4,4)
→ [B,64,7,7]
→ flatten + transpose
→ [B,49,64]
```

28 能被 4 整除，因此每边 7 个 patch，共 49 个。代码根据尺寸计算 patch 数，没有散落硬编码。

## 4. CLS 和位置编码

可学习 CLS 的初始形状为 `[1,1,64]`，扩展到 batch 后与 patch token 拼接：

```text
[B,1,64] + [B,49,64] → [B,50,64]
```

完整 TinyViT 再加上 `[1,50,64]` 的可学习位置编码。无位置编码模型将该参数注册为 `None`，前向传播跳过加法。这一开关构成控制变量实验。

## 5. 显式 Q/K/V 多头自注意力

`MultiHeadSelfAttention` 将 `[B,50,64]` 一次投影为 `[B,50,192]`，再切成 Q、K、V：

```text
Q/K/V：[B,50,64]
→ 4 个头：[B,4,50,16]
```

注意力计算为：

```text
Softmax(QKᵀ / √16)V
```

注意力分数和权重均为 `[B,4,50,50]`，表示每个头中 50 个 token 两两比较。上下文重新合并成 `[B,50,64]`，与输入维度一致，便于残差连接。

## 6. Encoder Block

项目采用 Pre-LayerNorm：

```python
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

MLP 为 `64 → 256 → 64`，使用 GELU 和 Dropout。两个残差连接保留原始信息并改善梯度传播。项目堆叠两层 Encoder。

## 7. 分类输出

两层 Encoder 后执行最终 LayerNorm，取第 0 个 CLS token：

```text
[B,50,64] → [B,64] → Linear(64,10) → [B,10]
```

输出是原始 logits。训练使用 `CrossEntropyLoss`，所以分类器末尾不能手动加入 Softmax；推理展示概率时才使用 Softmax。

## 8. CNN 基线

`SimpleCNN` 包含两个卷积块：

```text
[B,1,28,28]
→ [B,32,14,14]
→ [B,64,7,7]
→ [B,3136]
→ [B,10]
```

CNN 的局部连接和权重共享适合 Fashion-MNIST 的边缘与轮廓，因此在当前实验中优于 TinyViT。

## 9. 单轮训练

`train_one_epoch` 对每个 batch 执行：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

`backward` 根据链式法则计算梯度，`step` 使用 AdamW 更新参数。图片、标签和模型均移动到同一设备。

## 10. 验证与最佳模型

验证函数使用 `model.eval()` 和 `torch.no_grad()`，不会更新参数。正式训练以验证 Accuracy 为主要选择指标，Accuracy 相同时比较 Macro-F1，并保存包含模型权重、配置、类别和环境信息的 `best.pt`。

训练脚本不遍历测试集。`evaluate.py` 在模型选择结束后加载 `best.pt`，独立处理 10,000 张测试图片。

## 11. 输出文件

- `history.csv`：逐 epoch 的训练和验证指标；
- `training_summary.json`：最佳 epoch 和训练时间；
- `test_metrics.json`：总体及每类测试指标；
- `test_predictions.csv`：10,000 条标签、预测和 logits；
- `confusion_matrix.csv`：原始 10×10 混淆矩阵；
- `outputs/comparison/`：训练曲线、模型比较和错误样本图。

