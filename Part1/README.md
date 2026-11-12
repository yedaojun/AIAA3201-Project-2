# Part 1：模板基线与自编码器扩展

此文件夹只放 Part 1 的入口和自编码器实现，不包含 AnomalyCLIP。
在本仓库根目录运行下面的命令。

| 内容 | 实现或入口 | 结果目录 |
|---|---|---|
| 必做：模板差分 | `run_baseline.py`；完整已有实现 `../anomaly_baseline/` | `../results/part1/` |
| 选做：卷积自编码器 | `autoencoder/model.py`、`autoencoder/run.py` | `../results/part1_autoencoder/` |
| 模板 Demo | `demo_baseline.py` | `../results/demo_part1/` |
| 自编码器独立 Demo | `demo_autoencoder.py` | `../results/demo_part1_autoencoder/` |

## 运行

```powershell
# 只运行 Part 1 必做模板基线
.\.venv-part2\Scripts\python.exe Part1/run_baseline.py

# 只训练/评估 Part 1 自编码器；已完成且配置一致时复用记录
.\.venv-part2\Scripts\python.exe Part1/run_autoencoder.py

# 两个独立窗口，按需要启动一个
.\.venv-part2\Scripts\python.exe Part1/demo_baseline.py
.\.venv-part2\Scripts\python.exe Part1/demo_autoencoder.py
```

自编码器只学习正常训练图的重建，每类一个模型，各 20 轮；总计 2813 张拟合图、
312 张独立正常验证图，每类另有 20 张正常校准图。它不是 few-shot 实验。

详细方法见 [扩展说明](../docs/optional_extensions.md)。共享的 `experiment_utils/`
仅提供数据路径、数值评估、CSV 和窗口组件，不包含 AnomalyCLIP 或模型融合逻辑。
本次目录调整保留了模型/训练源码的原始字节和校验值，原有权重与结果继续有效。
