"""Analysis-only supplement: matched templates, score distributions and TP/TN/FP/FN."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from anomaly_baseline.core import CATEGORIES, load_image

OUTCOME_ZH = {'TP': '异常正确检出', 'TN': '正常正确通过', 'FP': '正常误报为异常', 'FN': '异常漏报为正常'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_panel(row, data_root, destination, short_edge):
    image_path, reference_path = PROJECT / row['image_file'], PROJECT / row['closest_template']
    anomaly_path = PROJECT / row['anomaly_map']
    with Image.open(image_path) as source:
        original = source.convert('RGB')
    with Image.open(anomaly_path) as source:
        anomaly = np.array(source)
    query, original_size = load_image(image_path, short_edge)
    reference, _ = load_image(reference_path, short_edge)
    if reference.shape[:2] != query.shape[:2]:
        reference = cv2.resize(reference, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_LINEAR)
    displayed_template = Image.fromarray(cv2.resize(reference, original_size, interpolation=cv2.INTER_LINEAR))
    if row['ground_truth_mask']:
        with Image.open(PROJECT / row['ground_truth_mask']) as source:
            mask = np.asarray(source) > 0
    else:
        mask = np.zeros(anomaly.shape, dtype=bool)
    assert anomaly.shape == mask.shape == (original.height, original.width)
    heat = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(anomaly, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
    overlay = Image.blend(original, heat, .45)
    canvas = Image.new('RGB', (1500, 410), '#f8fafc')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    title = (f'{row["category"]} | {row["outcome"]} | true={row["ground_truth_name"]}, predicted={row["predicted_decision"]} | '
             f'score={float(row["score"]):.4f}, threshold={float(row["threshold"]):.4f} | 4-shot / seed 0')
    draw.text((14, 10), title, font=font, fill='#142c43')
    panels = [('Test image\n' + image_path.name, original),
              ('Matched normal template\n' + reference_path.name, displayed_template),
              ('Ground truth\nwhite = labeled defect', Image.fromarray(mask.astype(np.uint8) * 255).convert('RGB')),
              ('Anomaly heatmap\nfixed scale: 0..255', heat), ('Overlay', overlay)]
    for index, (label, panel) in enumerate(panels):
        panel = panel.copy()
        panel.thumbnail((280, 280), Image.Resampling.NEAREST if index == 2 else Image.Resampling.LANCZOS)
        draw.multiline_text((index * 300 + 12, 44), label, font=font, fill='#334e68', spacing=2)
        canvas.paste(panel, (index * 300 + (300 - panel.width) // 2, 90 + (280 - panel.height) // 2))
    draw.text((14, 385), 'Template shown after query-size preprocessing; no registration. Ground truth is used for evaluation only. Colors are not rescaled per image.', font=small, fill='#526577')
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return {'defect_area_percent': float(mask.mean() * 100),
            'heat_mean_in_defect': float(anomaly[mask].mean()) if mask.any() else '',
            'heat_mean_outside_defect': float(anomaly[~mask].mean())}


def distributions(stats, values, destination):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(top=.80, bottom=.09, left=.055, right=.985, hspace=.37, wspace=.24)
    bins = np.linspace(0, 1, 41)
    styles = [('calibration', 'Calibration normal', '#9b6c11', ':'),
              ('normal_test', 'Test normal', '#2864a0', '-'),
              ('anomalous_test', 'Test anomaly', '#c44b38', '-')]
    for axis, category in zip(axes.flat, CATEGORIES):
        category_values = values[category]
        for key, label, color, linestyle in styles:
            axis.hist(category_values[key], bins=bins, density=True, histtype='step', linewidth=1.6,
                      label=label, color=color, linestyle=linestyle)
        metric = next(s for s in stats if s['category'] == category and s['shots'] == 4 and s['seed'] == 0)
        axis.axvline(metric['threshold'], color='#202e3b', linestyle='--', linewidth=1.7, label='Calibration threshold')
        axis.set(title=category, xlabel='Image anomaly score', ylabel='Density', xlim=(0, 1))
        axis.text(.02, .96, f'T={metric["threshold"]:.3f} | FP={metric["FP"]}/{metric["normal_test_count"]} | FN={metric["FN"]}/100',
                  transform=axis.transAxes, va='top', fontsize=9,
                  bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': 2})
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
    fig.suptitle('Part 1 score distributions — 4-shot / seed 0\nThresholds fixed from 20 normal calibration images per category', fontsize=14, y=.99)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc='upper center', bbox_to_anchor=(.5, .90), ncol=4, frameon=False)
    for extension in ('png', 'svg'):
        fig.savefig(destination.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=PROJECT / 'results/part1')
    parser.add_argument('--data-root', type=Path, default=PROJECT / 'data/VisA_pytorch/1cls')
    args = parser.parse_args()
    root = args.results.resolve()
    destination = root / 'supplement'
    destination.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    metrics = json.loads((root / 'metrics_all.json').read_text())
    if len(metrics) != 72:
        raise ValueError('This supplement expects the completed 72-configuration baseline')
    source_files = [root / name for name in ('metrics_all.json', 'run_config.json', 'predictions_all.csv', 'runs.csv')]
    for metric in metrics:
        group = root / metric['category'] / f'shot_{metric["shot"]}' / f'seed_{metric["seed"]}'
        source_files.extend(group / name for name in ('metrics.json', 'predictions.csv', 'calibration.csv'))
    before = {p.relative_to(root).as_posix(): digest(p) for p in source_files}
    rows = read_csv(root / 'predictions_all.csv')
    config = json.loads((root / 'run_config.json').read_text())['config']
    stats, values = [], {}
    for metric in metrics:
        category, shot, seed = metric['category'], metric['shot'], metric['seed']
        group = root / category / f'shot_{shot}' / f'seed_{seed}'
        predictions = read_csv(group / 'predictions.csv')
        calibration = read_csv(group / 'calibration.csv')
        normal = np.array([float(p['score']) for p in predictions if p['label'] == '0'])
        abnormal = np.array([float(p['score']) for p in predictions if p['label'] == '1'])
        cal = np.array([float(p['score']) for p in calibration])
        assert abs(float(np.percentile(cal, 95)) - metric['threshold']) < 1e-12
        stats.append({'category': category, 'shots': shot, 'seed': seed,
                      'calibration_count': len(cal), 'normal_test_count': len(normal), 'anomalous_test_count': len(abnormal),
                      'calibration_median': float(np.median(cal)), 'threshold': metric['threshold'],
                      'normal_test_median': float(np.median(normal)), 'anomalous_test_median': float(np.median(abnormal)),
                      'normal_test_min': float(normal.min()), 'normal_test_max': float(normal.max()),
                      'anomalous_test_min': float(abnormal.min()), 'anomalous_test_max': float(abnormal.max()),
                      'FP': metric['false_positives'], 'FN': metric['false_negatives'],
                      'false_positive_rate': metric['false_positives'] / len(normal),
                      'false_negative_rate': metric['false_negatives'] / len(abnormal)})
        if shot == 4 and seed == 0:
            values[category] = {'calibration': cal, 'normal_test': normal, 'anomalous_test': abnormal}
    write_csv(destination / 'score_distribution_summary.csv', stats)
    distributions(stats, values, destination / 'score_distributions_4shot_seed0')

    chosen, availability = [], []
    # Fixed settings and first filename in each outcome; no score-based aesthetic selection.
    for category in CATEGORIES:
        category_rows = [r for r in rows if r['category'] == category and r['shots'] == '4' and r['seed'] == '0']
        counts = Counter(row['outcome'] for row in category_rows)
        availability.append({'category': category, 'shots': 4, 'seed': 0,
                             **{key: counts[key] for key in ('TP', 'TN', 'FP', 'FN')}})
        for outcome in ('TP', 'TN', 'FP', 'FN'):
            candidates = sorted((r for r in category_rows if r['outcome'] == outcome), key=lambda r: r['sample_id'])
            if not candidates:
                continue
            row = candidates[0]
            name = f'template_comparisons/{category}_{outcome}.png'
            descriptive = compare_panel(row, args.data_root, destination / name, config['short_edge'])
            chosen.append({**{key: row[key] for key in ('category', 'shots', 'seed', 'sample_id', 'image_file',
                              'closest_template', 'ground_truth_mask', 'anomaly_map', 'score', 'threshold', 'outcome',
                              'ground_truth_name', 'predicted_decision', 'run_id')},
                           **descriptive, 'outcome_zh': OUTCOME_ZH[outcome], 'figure': name,
                           'selection_rule': 'first lexicographic sample_id for each outcome in each category, 4-shot seed 0'})
    write_csv(destination / 'case_index.csv', chosen)
    write_csv(destination / 'case_availability.csv', availability)

    lines = ['# Part 1 补充分析', '',
             '本补充读取已经保存的预测和异常图，不重新设定参数或阈值。展示统一固定为 4-shot / seed 0。', '',
             '## 1. 测试图、匹配模板与真实缺陷对照', '',
             '每张对比图依次显示：测试原图、实际选中的正常模板、Ground truth、预测热力图、叠加图。',
             '模板按推理时相同的尺寸处理显示，没有额外配准；所有预测热力图使用固定 0–255 色阶。', '',
             '## 2. Score 分布与校准阈值', '',
             '![六类 Score 分布](score_distributions_4shot_seed0.png)', '',
             '三组曲线分别来自 20 张正常校准图、全部正常测试图、全部异常测试图。纵轴为密度，每组独立归一化，便于比较不同样本数量的分布。',
             '虚线是原先用正常校准分数计算的第 95 百分位阈值；测试标签只用于分析分组，不用于重新选择阈值。', '',
             '| 类别 | 校准正常中位数 | 测试正常中位数 | 测试异常中位数 | 原阈值 | FP / 正常测试数 | FN / 异常测试数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    fixed_stats = [s for s in stats if s['shots'] == 4 and s['seed'] == 0]
    for row in fixed_stats:
        lines.append(f'| {row["category"]} | {row["calibration_median"]:.4f} | {row["normal_test_median"]:.4f} | '
                     f'{row["anomalous_test_median"]:.4f} | {row["threshold"]:.4f} | {row["FP"]}/{row["normal_test_count"]} | {row["FN"]}/100 |')
    reversed_medians = [r['category'] for r in fixed_stats if r['anomalous_test_median'] < r['normal_test_median']]
    lines += ['', f'在该固定配置下，异常中位分数低于正常中位分数的类别：{", ".join(reversed_medians) or "无"}。',
              '分布重叠或排序反向说明仅改变阈值无法同时消除误报与漏报。该图用于解释现有结果，不提供基于测试集调参的新阈值。', '',
              '## 3. TP / TN / FP / FN 案例', '',
              '每类、每种 outcome 按 sample_id 排序取第一张，规则固定且可复现。缺少某种错误的类别明确标为 0，不人为制造案例。', '',
              '| 类别 | TP | TN | FP | FN |', '|---|---:|---:|---:|---:|']
    for row in availability:
        lines.append(f'| {row["category"]} | {row["TP"]} | {row["TN"]} | {row["FP"]} | {row["FN"]} |')
    lines += ['', f'共生成 {len(chosen)} 张模板对比图；详细文件名、选择规则和分数保存在 case_index.csv。', '']
    lines += ['### 从实际对比图可以观察到什么', '',
              '- candle 的首张 FN 正是之前检查的 000.JPG。热力图主要突出圆形轮廓，真实标注只占很小区域；这与像素模板差对轮廓偏移敏感的局限一致。',
              '- capsules 的首张 FP 是正常图。对照正常模板能看到胶囊的位置和排列不同，热力图在多个正常胶囊区域响应，提示布局差异会造成误报。',
              '- TP 表示整图判断正确，并不自动保证热力图精确覆盖真实缺陷；TN 表示分数未超过阈值，也不要求热力图完全为零。',
              '- FN 可能同时涉及局部表征弱和正常校准阈值较高。以下缺陷面积、缺陷内外平均热度可作为补充证据，不能只凭一张图判定所有错误的原因。', '']
    for row in chosen:
        lines += [f'### {row["category"]} — {row["outcome"]}（{row["outcome_zh"]}）', '',
                  f'输入：`{row["image_file"]}`；匹配模板：`{row["closest_template"]}`。',
                  f'Score={float(row["score"]):.4f}，阈值={float(row["threshold"]):.4f}。', '',
                  f'![{row["category"]} {row["outcome"]}]({row["figure"]})', '']
        if row['ground_truth_name'] == 'bad':
            lines += [f'真实缺陷占图像 {row["defect_area_percent"]:.4f}%；缺陷内平均热度 {row["heat_mean_in_defect"]:.2f}/255，'
                      f'非缺陷区平均热度 {row["heat_mean_outside_defect"]:.2f}/255。该数值仅辅助描述，不能替代定位 AUROC。', '']
    (destination / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')

    escape = html.escape
    table_header = '<tr><th>类别</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr>'
    table_rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(row[k]))}</td>' for k in ('category', 'TP', 'TN', 'FP', 'FN')) + '</tr>' for row in availability)
    gallery = ''
    for category in CATEGORIES:
        figures = ''.join(f'<figure><figcaption><b>{r["outcome"]} — {r["outcome_zh"]}</b> | '
                          f'{escape(r["sample_id"])} | Score={float(r["score"]):.4f}, T={float(r["threshold"]):.4f}</figcaption>'
                          f'<img loading="lazy" src="supplement/{r["figure"]}" alt="{r["category"]} {r["outcome"]} template comparison"></figure>'
                          for r in chosen if r['category'] == category)
        gallery += f'<details><summary>{category}：点击查看模板对比与分类案例</summary>{figures}</details>'
    block = f'''<!-- PART1_SUPPLEMENT_START -->
<section id="part1-supplement"><h2>补充分析：模板、分数分布与分类案例</h2>
<p>固定展示 4-shot / seed 0。原始分数、阈值和评估结果保持原值。<a href="supplement/analysis.md">详细中文分析</a> · <a href="supplement/case_index.csv">案例 CSV</a> · <a href="supplement/score_distribution_summary.csv">72 组分布统计</a></p>
<h3>Score 分布与校准阈值</h3><p>棕色点线：20 张正常校准图；蓝色：正常测试图；红色：异常测试图；黑色虚线：原校准阈值。密度曲线分别归一化，测试标签只用于展示分析。</p>
<img src="supplement/score_distributions_4shot_seed0.png" alt="六类分数分布与原校准阈值">
<h3>成功、误报和漏报案例</h3><p>每类每种 outcome 按 sample_id 排序取第一张；缺少某种 outcome 时不生成假案例。对比图依次为测试原图、匹配正常模板、真实缺陷、热力图、叠加图。</p>
<table>{table_header}{table_rows}</table>{gallery}</section>
<!-- PART1_SUPPLEMENT_END -->'''
    report_path = root / 'report.html'
    text = report_path.read_text(encoding='utf-8')
    if '<!-- PART1_SUPPLEMENT_START -->' in text:
        text = re.sub(r'<!-- PART1_SUPPLEMENT_START -->.*?<!-- PART1_SUPPLEMENT_END -->', lambda _: block, text, flags=re.S)
    else:
        text = text.replace('<h2>Method and evaluation settings</h2>', block + '<h2>Method and evaluation settings</h2>')
    report_path.write_text(text, encoding='utf-8')
    standalone = block.replace('src="supplement/', 'src="').replace('href="supplement/', 'href="')
    (destination / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Part 1 补充分析</title><style>body{font:16px system-ui,"Microsoft YaHei";max-width:1300px;margin:35px auto;padding:0 20px;color:#172332}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px 16px}summary{cursor:pointer;padding:14px;background:#eef3f8}figure{margin:20px 0}figcaption{margin:8px 0}</style>' + standalone + '</html>', encoding='utf-8')
    overview = root / 'RESULTS.md'
    overview_text = overview.read_text(encoding='utf-8') if overview.exists() else '# Part 1 实测结果\n\n详见 report.html。\n'
    link = '\n## 模板对照、Score 分布和四类案例补充\n\n已补充 22 张模板对比图、六类 Score 分布与阈值图、72 组分布统计，以及 TP/TN/FP/FN 案例。见 [补充分析](supplement/analysis.md) 或 [图文页面](supplement/index.html)。\n'
    if '## 模板对照、Score 分布和四类案例补充' not in overview_text:
        overview.write_text(overview_text + link, encoding='utf-8')
    after = {p.relative_to(root).as_posix(): digest(p) for p in source_files}
    assert before == after, 'Analysis unexpectedly altered source results'
    counts = dict(Counter(row['outcome'] for row in chosen))
    audit = {'passed': True, 'original_results_unchanged': before == after,
             'original_result_sha256': before, 'fixed_shots': 4, 'fixed_seed': 0,
             'case_count': len(chosen), 'case_outcomes': counts,
             'distribution_summary_rows': len(stats),
             'selection_rule': 'First sample_id for each category and outcome at 4-shot / seed 0',
             'thresholds_recalibrated': False}
    (destination / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('passed', 'original_results_unchanged', 'case_count', 'case_outcomes', 'distribution_summary_rows')}, indent=2))
    print('Report:', report_path)


if __name__ == '__main__':
    main()
