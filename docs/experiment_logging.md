# 实验记录与逐张比较

## 已有结果保存位置

`results/part1` 是完整 Part 1 的结果根目录。每个 `<category>/shot_<k>/seed_<s>/` 内有原始 `predictions.csv`、`calibration.csv`、`metrics.json`、全部测试图的 `maps/`，以及用于复核 Pixel AUROC 的直方图。

结果根目录中保留 `run_config.json`、`metrics_all.json`、`summary_by_category.csv`、`summary_macro.csv`、`report.html`、`RESULTS.md`。参考名单在 `configs/reference_manifest.json`。原始和整理后的数据均在 `data/`。

模板基线没有需要训练的神经网络权重；复现它需要代码、方法配置、参考图、校准图和已经记录的阈值。

## 四份统一 CSV

运行：

```powershell
python tools/export_prediction_logs.py
```

| 文件 | 一行代表什么 | 用途 |
|---|---|---|
| `results/part1/predictions_all.csv` | 某方法、类别、shot、seed 对一张测试图片的预测 | 全部 11,520 次预测；逐张定位正确和错误 |
| `results/part1/errors_only.csv` | 上述明细中的一条 FP 或 FN | 只看误报、漏报 |
| `results/part1/runs.csv` | 一个完整实验配置 | 72 行，方法参数、参考图/校准图、阈值、AUROC、混淆矩阵和耗时 |
| `results/part1/input_images.csv` | 某个实验配置使用的一张参考图或校准图 | 1,710 行；role 区分 reference 和 calibration |

所有 CSV 使用 UTF-8 BOM，方便 Excel 打开中文。文件路径以项目根目录为起点，避免换电脑后盘符变化使记录失效。
`csv_export_metadata.json` 保存导出时间、行数、路径基准和 CSV 的 SHA256 校验值。
导出读取已经完成的预测，不重新预测，也不根据测试标签修改阈值。

今后运行 `python run_part1.py` 会自动依次准备、运行/续跑和导出上述四份 CSV；新一轮可用 `--output results/part1_round2` 指定独立目录。

## 预测明细的重要字段

- `method_id`、`method_name`：方法标识和名称，例如 `baseline_rgb_template`。
- `dataset`、`split`、`category`、`shots`、`seed`：数据和配置。
- `run_id`、`run_signature`：本轮实验标识和实现/配置指纹。
- `sample_id`：跨方法可配对的测试样本标识；同一张图在不同 shot/seed 下会有多行，不是错误重复。
- `image_file`、`image_sha256`：测试输入路径及内容校验值。
- `reference_images_json`：本组实际使用的全部正常参考图；`closest_template` 是本张图最终选中的模板。
- `ground_truth_label`、`ground_truth_name`：真实标签，0/good=正常，1/bad=异常。
- `ground_truth_mask`：真实像素标注路径；正常图留空，`normal_mask_definition` 说明评估时使用全零掩码。
- `score`、`threshold`：原始异常分数和本组正常校准阈值。
- `predicted_label`、`predicted_decision`：预测，0/OK=正常，1/NG=异常。
- `outcome`、`outcome_zh`、`is_correct`：TP/TN/FP/FN、中文解释、是否正确（1/0）。
- `anomaly_map`：与这条预测对应的原始异常图路径。
- `reference_set_sha256`、`calibration_set_sha256`、`dataset_split_sha256`：保证比较使用同一套输入和划分。

## TP、TN、FP、FN

这里将“异常”定义为正类 positive。

| 真实标签 | 预测 | outcome | 意义 |
|---|---|---|---|
| bad | NG | TP | 异常正确检出 |
| good | OK | TN | 正常正确通过 |
| good | NG | FP | 正常误报为异常 |
| bad | OK | FN | 异常漏报为正常 |

NG/OK 只有预测含义，不能单独判断预测是否正确。你提供的 `candle/test/bad/000.JPG` 在 4-shot/seed 0 下属于 FN。
同一张图后续如果变为 NG，就属于 `FN->TP`，漏报被纠正。正常图从 NG 变为 OK 属于 `FP->TN`，误报被纠正。
同时也要统计 `TP->FN` 或 `TN->FP`，避免只挑有提升的样例。

## 后续方法怎样保存和比较

每个方法/轮次使用独立输出目录，保持上述 schema，不覆盖原 baseline。不同参考数量和 seed 作为不同配置保存。
Part 2 或改进方法实现时，应为其结果补齐同样字段；现有导出器使用当前项目的标准结果布局，官方 AnomalyDINO 的原始输出还需要适配，不能直接改个文件名就比较。

两种方法都有统一预测 CSV 后，可以运行：

```powershell
python tools/compare_prediction_logs.py --before results/part1/predictions_all.csv --after results/part2/predictions_all.csv --output results/comparisons/part1_vs_part2.csv
```

比较器按数据集、划分、类别、shot、seed、sample_id 配对，并拒绝：缺失样本、重复键、参考图/校准图不一致、图像内容或真实标注不一致、输入几何参数不一致。

输出包括每张图的 `before_outcome`、`after_outcome`、`transition` 和 `change`：

- `error_fixed`：错误被纠正，如 FN->TP、FP->TN。
- `regression`：原本正确变成错误，如 TP->FN、TN->FP。
- `both_correct`：两种方法都正确。
- `both_wrong`：两种方法都错误。

旁边的 `.summary.json` 汇总所有转变数量。各方法分数不一定在同一尺度上，不能仅用原始 Score 大小评价提升。
原始预测的总 AUROC 和每类 AUROC 仍保存在各自的指标文件中，需要结合逐图变化一起分析。

## Part 2 记录

`results/part2` 使用同样的四份 CSV。`method_id=official_anomalydino`，另有骨干名称、权重 SHA256 和官方代码 commit 等字段。
Part 2 每条记录的 `anomaly_map` 指向原始浮点 patch `.npy`，`anomaly_map_format` 说明重建方式。不能把它当成 PNG 直接打开；使用官方 `dists2map` 恢复全分辨率评分图，图形 Demo 和对比报告已执行这一步。
每个 patch 分数都保持原始精度，PNG 颜色图只用于显示。

原始批量计时保留官方边界；`results/comparisons/timing_same_inputs.csv` 则是在两轮完整评估之后，串行测量相同 120 张输入、相同 4-shot/seed 0 和同一完整推理边界的时间。
某些完整评估配置曾分目录并行计算，因此正式速度比较以这份串行 benchmark 为准。
