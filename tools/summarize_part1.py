"""Audit completed outputs and write a concise, measured Part 1 analysis."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/part1'
DATA = ROOT / 'data/VisA_pytorch/1cls'
manifest = json.loads((ROOT / 'configs/reference_manifest.json').read_text())
metrics = json.loads((OUTPUT / 'metrics_all.json').read_text())
assert len(metrics) == 72, f'Expected 72 configurations, got {len(metrics)}'
assert len({(m['category'], m['shot'], m['seed']) for m in metrics}) == 72
assert len({m['run_signature'] for m in metrics}) == 1
prediction_count = 0
for metric in metrics:
    category, shot, seed = metric['category'], metric['shot'], metric['seed']
    folder = OUTPUT / category / f'shot_{shot}' / f'seed_{seed}'
    selection = manifest['categories'][category]
    assert metric['references'] == selection['references'][str(shot)][str(seed)]
    assert metric['calibration_files'] == selection['calibration']
    assert not set(metric['references']).intersection(selection['calibration'])
    with (folder / 'predictions.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    labels = np.array([int(row['label']) for row in rows])
    scores = np.array([float(row['score']) for row in rows])
    p, n = scores[labels == 1, None], scores[labels == 0][None, :]
    independent_auc = float(np.mean((p > n) + .5 * (p == n)))
    assert abs(independent_auc - metric['image_auroc']) < 1e-12
    assert sum(labels) == 100
    assert int((labels == 0).sum()) == manifest['counts'][category]['test/good']
    assert int((labels == 1).sum()) == manifest['counts'][category]['test/bad']
    assert len(rows) == metric['normal_test_images'] + metric['anomalous_test_images']
    with (folder / 'calibration.csv').open(encoding='utf-8-sig', newline='') as file:
        calibration = list(csv.DictReader(file))
    assert len(calibration) == 20
    assert {row['file'] for row in calibration} == set(selection['calibration'])
    threshold = np.percentile([float(row['score']) for row in calibration], 95)
    assert abs(threshold - metric['threshold']) < 1e-12
    expected_pixels = 0
    for row in rows:
        assert (row['decision'] == 'NG') == (float(row['score']) > threshold)
        assert (folder / row['anomaly_map']).is_file()
        with Image.open(DATA / category / row['file']) as image:
            expected_pixels += image.width * image.height
    counts = np.load(folder / 'pixel_score_histograms.npz')
    positive, negative = counts['positive'], counts['negative']
    assert int(positive.sum() + negative.sum()) == expected_pixels == metric['pixel_count']
    # Independent reversed-rank derivation: one minus probability of wrong ordering.
    wrong = np.sum(negative * (np.cumsum(positive) - .5 * positive)) / (positive.sum() * negative.sum())
    assert abs(1 - wrong - metric['pixel_auroc']) < 1e-12
    prediction_count += len(rows)
assert prediction_count == 11520

with (OUTPUT / 'summary_macro.csv').open(encoding='utf-8-sig', newline='') as file:
    macro = list(csv.DictReader(file))
with (OUTPUT / 'summary_by_category.csv').open(encoding='utf-8-sig', newline='') as file:
    per_category = list(csv.DictReader(file))
lines = ['# Part 1 实测结果', '',
         'RGB 模板差分基线；短边 448；高斯 sigma=2；最高 1% 工作像素均值作为图像分数。参数在完整测试前固定。', '',
         '## 六类等权平均', '',
         '每个 seed 先对六类取等权平均，再报告三个 seed 的均值 ± 样本标准差。', '',
         '| Shots | Image AUROC (%) | Pixel AUROC (%) | 平均推理时间 (ms/image) |',
         '|---:|---:|---:|---:|']
for row in macro:
    lines.append(f'| {row["shot"]} | {100*float(row["image_auroc_mean"]):.2f} ± {100*float(row["image_auroc_std"]):.2f} | '
                 f'{100*float(row["pixel_auroc_mean"]):.2f} ± {100*float(row["pixel_auroc_std"]):.2f} | {float(row["inference_ms_mean_mean"]):.1f} |')
lines += ['', '## 4-shot 各类别结果', '',
          '| 类别 | Image AUROC (%) | Pixel AUROC (%) |', '|---|---:|---:|']
for row in per_category:
    if row['shot'] == '4':
        lines.append(f'| {row["category"]} | {100*float(row["image_auroc_mean"]):.2f} ± {100*float(row["image_auroc_std"]):.2f} | '
                     f'{100*float(row["pixel_auroc_mean"]):.2f} ± {100*float(row["pixel_auroc_std"]):.2f} |')
lines += ['', '## Demo 阈值的实际误报与漏报', '',
          '以下固定为 4-shot / seed 0；阈值只来自各类 20 张独立正常校准图。AUROC 与该阈值无关。', '',
          '| 类别 | 正常测试数 | 误报数 FP | 异常测试数 | 漏报数 FN |', '|---|---:|---:|---:|---:|']
for metric in metrics:
    if metric['shot'] == 4 and metric['seed'] == 0:
        lines.append(f'| {metric["category"]} | {metric["normal_test_images"]} | {metric["false_positives"]} | '
                     f'{metric["anomalous_test_images"]} | {metric["false_negatives"]} |')
lines += ['', '## 解释与局限', '',
          '- 模板差分直接比较固定空间位置。姿态、物体布局、轮廓和背景变化都会产生差异，分数不等同于缺陷概率。',
          '- 每类第 95 百分位阈值仅由 20 张正常图估计，不保证测试误报率为 5%；漏报和误报均如实列出。',
          '- 增加参考图并不保证 AUROC 单调提高：最接近的模板由全图平均差选择，而图像异常分数取局部最高差异；两者目标不同。',
          '- 热力图可能主要响应物体轮廓或背景，而遗漏细小缺陷。请结合 report.html 的真实困难案例判断。',
          '- 未使用配准、旋转增强、前景掩码或学习特征；这些不是隐藏的测试后处理。',
          '- 本文仅报告 Part 1；Part 2 的实测对比见 results/part2/report.html。不同方法必须复用当前 manifest 并采用一致评估定义。', '',
          '## 完整性审计', '',
          f'- 72/72 配置完成，共 {prediction_count:,} 次测试预测，所有测试异常图均保存。',
          '- 每组恰好 20 张校准图，且与参考图互斥；名单与 manifest 完全一致。',
          '- 所有图像 AUROC 经独立逐正负样本配对公式复核；所有像素 AUROC 经反向秩公式复核。',
          '- 像素统计总数与全部正常、异常原图面积之和一致，未漏掉正常图。',
          '- 推理使用 CPU，方法 GPU 显存占用为 0；计时边界见 README。',
          '- 结果不包括 Part 3，不能作为完整项目最终提交。', '']
(OUTPUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
(OUTPUT / 'audit.json').write_text(json.dumps({'passed': True, 'configurations': 72,
    'test_predictions': prediction_count, 'independent_image_auc_check': True,
    'pixel_area_and_auc_check': True, 'calibration_reference_disjoint': True}, indent=2), encoding='utf-8')
print('\n'.join(lines), flush=True)
