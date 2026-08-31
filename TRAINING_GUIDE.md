# 训练与复现指南

本文说明项目的运行顺序、每条命令的用途，以及四项训练之间的关系。

## 1. 四项训练分别解决什么问题

| 训练 | 目的 | 是否属于正式对比 |
|---|---|---|
| 256 样本 TinyViT 过拟合诊断 | 检查模型、loss、梯度和优化器是否能学习 | 否 |
| 完整 TinyViT | 完成 Transformer 图像分类主实验 | 是 |
| 无位置编码 TinyViT | 消融位置编码，研究空间信息的作用 | 是 |
| CNN | 为 TinyViT 提供传统视觉模型基线 | 是 |

过拟合诊断不是预训练。正式实验会重新随机初始化模型，不会继承诊断权重。

## 2. 虚拟环境

`.venv/` 是项目专用的 Python 3.11 环境，包含 Python、PyTorch、torchvision 和 CUDA 运行依赖。它约 3.3 GB，不属于源码，也不提交到 GitHub。

本项目直接指定虚拟环境中的解释器，因此不需要先激活：

```powershell
.\.venv\Scripts\python.exe .\src\train.py
```

在新电脑上重建环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

## 3. 进入项目并检查 GPU

```powershell
cd "项目所在路径\tinyViT"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本机验证环境为 Python 3.11.9、PyTorch 2.12.1+cu130 和 RTX 5070 Ti Laptop GPU。CUDA 应显示 `True`。

## 4. 下载并检查数据

```powershell
.\.venv\Scripts\python.exe .\src\dataset.py --config .\config.yaml
```

代码自动下载 Fashion-MNIST，并从官方 60,000 张训练图片中按类别分层划分：

- Train：54,000，用于更新参数；
- Validation：6,000，用于选择最佳模型；
- Test：10,000，只用于最终独立测试。

固定随机种子为 42，训练和验证索引重叠为 0。一个 batch 的图片形状为 `[128,1,28,28]`。

## 5. 检查 TinyViT shape

```powershell
.\.venv\Scripts\python.exe .\src\inspect_model.py
.\.venv\Scripts\python.exe .\src\inspect_model.py --no-position-encoding
```

主要 shape：

```text
[B,1,28,28]  输入
→ [B,64,7,7] Patch Embedding
→ [B,49,64]  Patch Tokens
→ [B,50,64]  加入 CLS
→ [B,50,64]  两层 Encoder 输出
→ [B,64]     CLS 向量
→ [B,10]     分类 logits
```

脚本还会显示 Q/K/V 的 `[B,4,50,16]` 和注意力矩阵的 `[B,4,50,50]`。

## 6. 不更新参数的链路检查

```powershell
.\.venv\Scripts\python.exe .\src\train.py --mode formal --model tinyvit --check-only
```

它只执行训练与验证 batch 的前向传播和交叉熵，不调用 `backward()` 或优化器，也不会遍历测试集。

## 7. 小数据过拟合诊断

```powershell
.\.venv\Scripts\python.exe .\src\train.py --mode overfit --model tinyvit --overfit-samples 256 --epochs 100 --target-accuracy 0.98
```

- `--overfit-samples 256`：从训练集分层抽取 256 张；
- `--epochs 100`：最多训练 100 轮；
- `--target-accuracy 0.98`：同一子集达到 98% 后停止。

实际在第 71 轮达到 99.22%，耗时约 3.90 秒。它只证明训练链路能够记忆小数据，不代表测试集性能。

## 8. 正式训练完整 TinyViT

```powershell
.\.venv\Scripts\python.exe .\src\train.py --mode formal --model tinyvit
```

配置来自 `config.yaml`：20 epoch、batch size 128、AdamW、学习率 0.001、权重衰减 0.0001。每轮使用训练集更新参数，然后在验证集计算 Accuracy 和 Macro-F1。验证 Accuracy 更高时保存 `best.pt`，并列时再比较 Macro-F1。

实际最佳模型位于第 18 轮：验证 Accuracy 88.02%，独立测试 Accuracy 87.78%，Macro-F1 0.8777。

## 9. 正式训练无位置编码 TinyViT

```powershell
.\.venv\Scripts\python.exe .\src\train.py --mode formal --model tinyvit_no_pos
```

与完整 TinyViT 的唯一关键差别是 `use_position_embedding=False`。Patch、CLS、Q/K/V、Encoder、数据和训练配置保持不变。

实际最佳模型位于第 16 轮，测试 Accuracy 83.09%，比完整 TinyViT 低 4.69 个百分点。

## 10. 正式训练 CNN 基线

```powershell
.\.venv\Scripts\python.exe .\src\train.py --mode formal --model cnn
```

实际最佳模型位于第 20 轮，测试 Accuracy 91.92%，Macro-F1 0.9183。CNN 在当前小型灰度图任务上更强，但不能推广为 CNN 永远优于 Transformer。

## 11. 独立测试

```powershell
.\.venv\Scripts\python.exe .\src\evaluate.py --checkpoint .\checkpoints\tinyvit_seed42\best.pt
.\.venv\Scripts\python.exe .\src\evaluate.py --checkpoint .\checkpoints\tinyvit_no_pos_seed42\best.pt
.\.venv\Scripts\python.exe .\src\evaluate.py --checkpoint .\checkpoints\cnn_seed42\best.pt
```

评估脚本只加载验证集选出的 `best.pt`，在官方 10,000 张测试图片上生成 Accuracy、Macro-F1、每类指标、原始预测和混淆矩阵。不要根据测试结果反复调参。

## 12. 图表、预测和验收

```powershell
.\.venv\Scripts\python.exe .\src\plot_results.py
.\.venv\Scripts\python.exe .\src\predict.py --checkpoint .\checkpoints\tinyvit_seed42\best.pt --index 0 --output .\outputs\prediction_demo.png
.\.venv\Scripts\python.exe .\src\verify_project.py
```

- `plot_results.py`：从已保存的 CSV/JSON 重新生成图表，不重新训练；
- `predict.py`：演示单样本预测，测试集第 0 张 Ankle boot 的预测概率为 94.09%；
- `verify_project.py`：检查文件、数据划分、模型参数量、checkpoint、10,000 条预测和 PNG 完整性。

额外验证从空目录自动下载数据：

```powershell
.\.venv\Scripts\python.exe .\src\verify_project.py --fresh-download
```

## 13. 重跑注意事项

再次运行同名训练会覆盖对应的 `checkpoints/.../best.pt`、`history.csv` 和 `training_summary.json`。如果只是阅读或演示，不需要重新训练；亲自重跑前应备份当前 `checkpoints/` 和 `outputs/`。

