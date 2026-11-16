# Part 2：AnomalyDINO 复现与 AnomalyCLIP 扩展

此文件夹只放 Part 2 的入口和 AnomalyCLIP 实现，不包含自编码器训练。
在本仓库根目录运行下面的命令。

| 内容 | 实现或入口 | 结果目录 |
|---|---|---|
| 必做：官方 AnomalyDINO | `run_anomalydino.py`；完整已有适配 `../anomaly_dino/` | `../results/part2/` |
| 选做：AnomalyCLIP | `anomalyclip/model.py`、`anomalyclip/run.py` | `../results/part2_anomalyclip/` |
| AnomalyDINO Demo | `demo_anomalydino.py` | `../results/demo_part2/` |
| AnomalyCLIP 独立 Demo | `demo_anomalyclip.py` | `../results/demo_part2_anomalyclip/` |

## 运行

```powershell
# 只运行 Part 2 必做 AnomalyDINO
.\.venv-part2\Scripts\python.exe Part2/run_anomalydino.py

# 只运行 Part 2 AnomalyCLIP；已完成且配置一致时复用记录
.\.venv-part2\Scripts\python.exe Part2/run_anomalyclip.py

# 两个独立窗口，按需要启动一个
.\.venv-part2\Scripts\python.exe Part2/demo_anomalydino.py
.\.venv-part2\Scripts\python.exe Part2/demo_anomalyclip.py
```

AnomalyDINO 使用与 Part 1 模板基线相同的 1/2/4/8 张正常参考图，各三个参考组。
AnomalyCLIP 则使用官方在外部 MVTec AD 上训练的提示权重，VisA 训练与参考图数量为 0，
只用每类 20 张正常图设置阈值；其 448 方形缩放与必做方法保持纵横比的几何处理不同。
它是独立的补充实验，不与自编码器合成一个模型。

详细复现说明见 [AnomalyDINO](../docs/part2_method.md) 和 [AnomalyCLIP 扩展](../docs/optional_extensions.md)。
共享的 `experiment_utils/` 仅负责通用数据、评估、日志与窗口组件。

## Demo 阈值与官方复现的区别

**规则来源说明：**“前 20 张正常训练图 + 第 95 百分位”明确写在任务书 §5.1 Part 1 的 **Demo Threshold** 中。§5.2 Part 2 明确要求参考采样时排除这些校准图，但没有再次指定 AnomalyDINO 必须采用 95 分位阈值。我们为 Part 2 Demo 的 OK/NG 输出额外沿用同一正常校准策略；这是本项目的决策层设计，不是官方 AnomalyDINO 评分方法或 Part 2 明文规定的阈值。校准不修改官方原始分数或异常图；Image/Pixel AUROC 仍使用阈值前的分数，Accuracy、FP/FN 则是在本项目该决策策略下报告。
