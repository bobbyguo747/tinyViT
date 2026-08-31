# TinyViT 图像分类学习项目

本项目使用 PyTorch 在公共数据集 Fashion-MNIST 上实现一个容易解释的 TinyViT，目标是清楚展示 Transformer 在图像分类中的数据流和核心机制，而不是堆砌功能或复现大型论文模型。

## 小数据过拟合诊断

该诊断只使用训练集中的 256 个固定、分层样本，用来检查模型与训练链路能否主动记忆小数据。它不是正式实验，其准确率不能当作测试集结果。

先进行不更新参数的检查：

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe .\src\train.py --check-only
```

确认无误后运行过拟合诊断：

```powershell
.\.venv\Scripts\python.exe .\src\train.py --overfit-samples 256 --epochs 100 --target-accuracy 0.98
```

达到 98% 子集准确率后会自动提前停止。诊断文件单独保存在 `checkpoints/overfit_tinyvit_seed42/` 和 `outputs/overfit_tinyvit_seed42/`。

本机实际诊断结果：第 71 轮达到 99.22% 子集准确率并提前停止，用时约 3.90 秒。该结果只证明训练链路能够记忆小数据，不代表模型的泛化性能。

## 正式 TinyViT 训练与评估

正式训练只使用 54,000 个训练样本更新参数，并使用 6,000 个验证样本选择最佳 checkpoint。`train.py` 不遍历测试集。

```powershell
.\.venv\Scripts\python.exe .\src\train.py --mode formal
```

训练完成后，独立评估脚本只从验证集选出的 `best.pt` 重建模型，并在官方 10,000 个测试样本上计算最终指标：

```powershell
.\.venv\Scripts\python.exe .\src\evaluate.py --checkpoint .\checkpoints\tinyvit_seed42\best.pt
```

不要根据测试集结果修改超参数后反复测试；模型选择和调参只能依据验证集表现。

本机实际结果（RTX 5070 Ti Laptop GPU，seed=42）：20 轮训练用时约 101.98 秒，验证集在第 18 轮达到最佳 Accuracy 88.02%、Macro-F1 0.8803；对应 `best.pt` 在独立测试集上得到 Accuracy 87.78%、Macro-F1 0.8777。验证与测试表现接近，测试集结果未用于模型选择。

## CNN 基线与位置编码消融

三组实验使用相同的数据划分、随机种子、训练轮数、优化器、学习率和权重衰减。参数量分别为：TinyViT 105,098、无位置编码 TinyViT 101,898、CNN 96,362。

```powershell
.\.venv\Scripts\python.exe .\src\train.py --mode formal --model cnn
.\.venv\Scripts\python.exe .\src\train.py --mode formal --model tinyvit_no_pos
```

训练完成后分别评估验证集选出的最佳 checkpoint：

```powershell
.\.venv\Scripts\python.exe .\src\evaluate.py --checkpoint .\checkpoints\cnn_seed42\best.pt
.\.venv\Scripts\python.exe .\src\evaluate.py --checkpoint .\checkpoints\tinyvit_no_pos_seed42\best.pt
```

本机三组实际结果：

| 模型              | 参数量   | 最佳轮次 | 验证 Accuracy | 测试 Accuracy | 测试 Macro-F1 | 训练时间 |
|---                |---:     |---:     |---:           |---:           |---:          |---:     |
| CNN               | 96,362  | 20      | 92.55%        | 91.92%        | 0.9183       | 101.41s |
| TinyViT           | 105,098 | 18      | 88.02%        | 87.78%        | 0.8777       | 101.98s |
| TinyViT 无位置编码 | 101,898 | 16      | 83.20%        | 83.09%        | 0.8298       | 103.42s |

在相同训练条件下，CNN 在该小型灰度图数据集上表现最好；移除位置编码后 TinyViT 测试准确率下降 4.69 个百分点，说明空间位置信息对当前模型确实重要。该结论仅针对本项目配置，不推广为所有 CNN 与 Transformer 的普遍结论。

完整图表和结果讨论见 [EXPERIMENT_ANALYSIS.md](EXPERIMENT_ANALYSIS.md)。全部图表可由以下命令从已保存的 CSV/JSON 重新生成，不会重新训练或测试模型：

```powershell
.\.venv\Scripts\python.exe .\src\plot_results.py
```

## 文档

- [EXPERIMENT_ANALYSIS.md](EXPERIMENT_ANALYSIS.md)：真实指标、混淆矩阵、错误案例和实验局限。

## 仓库内容

```text
tinyViT/
├─ src/                 PyTorch 源码
├─ checkpoints/         实际训练得到的最佳权重
├─ outputs/             训练记录、测试预测和图表
├─ config.yaml          统一实验配置
├─ requirements.txt     Python 依赖
├─ README.md            项目概览
└─ EXPERIMENT_ANALYSIS.md 实验分析
```

`.venv/` 和 `data/` 不提交：虚拟环境应由 `requirements.txt` 重建，Fashion-MNIST 会由代码自动下载。
