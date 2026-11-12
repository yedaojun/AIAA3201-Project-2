"""Measured Part 2 report and fixed-sample comparison with the original baseline."""
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from anomaly_baseline.core import CATEGORIES
from anomaly_dino.official import dense_map

BASE = ROOT / 'results/part1'
OUTPUT = ROOT / 'results/part2'
COMPARE = ROOT / 'results/comparisons'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel(before, after, destination):
    with Image.open(ROOT / before['image_file']) as source:
        image = source.convert('RGB')
    if before['ground_truth_mask']:
        with Image.open(ROOT / before['ground_truth_mask']) as source:
            mask = (np.array(source) > 0).astype(np.uint8) * 255
    else:
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
    baseline = np.array(Image.open(ROOT / before['anomaly_map']))
    dino = dense_map(ROOT / after['anomaly_map'], (image.height, image.width, 3))
    b_heat = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(baseline, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
    d_heat = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(np.clip(dino * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
    canvas = Image.new('RGB', (1500, 410), '#f8fafc')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    draw.text((14, 10), f'{after["category"]} | {Path(after["image_file"]).name} | {before["outcome"]} -> {after["outcome"]} | 4-shot / seed 0', font=font, fill='#142c43')
    labels = ['Input image', 'Ground truth', f'Baseline: {before["outcome"]}\ns={float(before["score"]):.4f} T={float(before["threshold"]):.4f}',
              f'AnomalyDINO: {after["outcome"]}\ns={float(after["score"]):.4f} T={float(after["threshold"]):.4f}', 'AnomalyDINO overlay']
    for index, (label, displayed) in enumerate(zip(labels, [image, Image.fromarray(mask).convert('RGB'), b_heat, d_heat, Image.blend(image, d_heat, .45)])):
        displayed = displayed.copy()
        displayed.thumbnail((280, 280), Image.Resampling.NEAREST if index == 1 else Image.Resampling.LANCZOS)
        draw.multiline_text((index * 300 + 12, 44), label, font=font, fill='#334e68', spacing=2)
        canvas.paste(displayed, (index * 300 + (300 - displayed.width) // 2, 90 + (280 - displayed.height) // 2))
    draw.text((14, 385), 'Fixed display scales: baseline 0..255; AnomalyDINO 0..1. Scores have method-specific scales and thresholds. Ground truth is not used for prediction.', font=small, fill='#526577')
    canvas.save(destination)
    if mask.any():
        ys, xs = np.where(mask > 0)
        width = min(image.width, max(160, int((xs.max() - xs.min() + 1) * 2)))
        height = min(image.height, max(160, int((ys.max() - ys.min() + 1) * 2)))
        left = max(0, min(image.width - width, int((xs.min() + xs.max() - width) / 2)))
        top = max(0, min(image.height - height, int((ys.min() + ys.max() - height) / 2)))
        crop = (left, top, left + width, top + height)
        zoom = Image.new('RGB', (1200, 365), '#f8fafc')
        draw_zoom = ImageDraw.Draw(zoom)
        draw_zoom.text((12, 8), 'GT-guided zoom for analysis only; original inference and scores are unchanged.', font=font, fill='#142c43')
        for index, (label, displayed) in enumerate(zip(['Input crop', 'Ground-truth crop', 'Baseline crop', 'AnomalyDINO crop'],
                [image, Image.fromarray(mask).convert('RGB'), b_heat, d_heat])):
            displayed = displayed.crop(crop)
            displayed.thumbnail((280, 280), Image.Resampling.NEAREST if index == 1 else Image.Resampling.LANCZOS)
            draw_zoom.text((index * 300 + 12, 40), label, font=font, fill='#334e68')
            zoom.paste(displayed, (index * 300 + (300 - displayed.width) // 2, 72 + (280 - displayed.height) // 2))
        zoom_path = destination.with_name(destination.stem + '_zoom.png')
        zoom.save(zoom_path)
        return 'examples/' + zoom_path.name
    return ''


def main():
    OUTPUT.mkdir(exist_ok=True)
    figures, examples = OUTPUT / 'figures', OUTPUT / 'examples'
    figures.mkdir(exist_ok=True)
    examples.mkdir(exist_ok=True)
    baseline_macro, dino_macro = read_csv(BASE / 'summary_macro.csv'), read_csv(OUTPUT / 'summary_macro.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for axis, metric, title in zip(axes, ('image_auroc', 'pixel_auroc'), ('Image detection', 'Pixel localization')):
        for data, label, color in [(baseline_macro, 'RGB template', '#7b8794'), (dino_macro, 'Official AnomalyDINO', '#137c69')]:
            shots = [int(r['shot']) for r in data]
            axis.errorbar(shots, [float(r[metric + '_mean']) for r in data],
                          yerr=[float(r[metric + '_std']) for r in data], label=label, color=color,
                          marker='o', linewidth=2, capsize=4)
        axis.set(xscale='log', xticks=shots, xticklabels=list(map(str, shots)), ylim=(0, 1),
                 xlabel='Normal reference images', ylabel='AUROC', title=title)
        axis.minorticks_off()
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=8, loc='lower right')
    fig.suptitle('Shared references and full official test split\nSix-category macro average; mean ± sample SD across three seeds', fontsize=12)
    for extension in ('png', 'svg'):
        fig.savefig(figures / f'baseline_vs_dino.{extension}', dpi=180)
    plt.close(fig)

    before_rows, after_rows = read_csv(BASE / 'predictions_all.csv'), read_csv(OUTPUT / 'predictions_all.csv')
    def key(row):
        return row['category'], row['shots'], row['seed'], row['sample_id']
    before, after = {key(r): r for r in before_rows}, {key(r): r for r in after_rows}
    assert before.keys() == after.keys() and len(after) == 11520
    # Compare exactly the 22 already published Part 1 case IDs, rather than selecting only improvements.
    fixed_cases = read_csv(BASE / 'supplement/case_index.csv')
    case_rows = []
    for index, original in enumerate(fixed_cases, 1):
        identity = original['category'], original['shots'], original['seed'], original['sample_id']
        old, new = before[identity], after[identity]
        name = f'fixed_{index:02d}_{new["category"]}_{old["outcome"]}_to_{new["outcome"]}.png'
        zoom = panel(old, new, examples / name)
        case_rows.append({'sample_id': new['sample_id'], 'category': new['category'], 'shots': 4, 'seed': 0,
                          'before_outcome': old['outcome'], 'after_outcome': new['outcome'],
                          'transition': old['outcome'] + '->' + new['outcome'],
                          'before_score': old['score'], 'before_threshold': old['threshold'],
                          'after_score': new['score'], 'after_threshold': new['threshold'],
                          'figure': 'examples/' + name, 'zoom_figure': zoom, 'selection_rule': 'exactly the previously selected 22 Part 1 case IDs'})
    write_csv(OUTPUT / 'fixed_case_comparison.csv', case_rows)

    difficult = []
    # One difficult example per category, at most five. Actual errors first; correct borderline fallback.
    for category in CATEGORIES[:5]:
        candidates = [r for r in after_rows if r['category'] == category and r['shots'] == '4' and r['seed'] == '0']
        candidates.sort(key=lambda r: (r['is_correct'] == '1',
            -abs(float(r['score']) - float(r['threshold'])) if r['is_correct'] == '0' else abs(float(r['score']) - float(r['threshold'])), r['sample_id']))
        chosen = candidates[0]
        old = before[key(chosen)]
        name = f'difficult_{category}_{chosen["outcome"]}.png'
        zoom = panel(old, chosen, examples / name)
        difficult.append({'category': category, 'sample_id': chosen['sample_id'], 'outcome': chosen['outcome'],
                          'description': 'actual error' if chosen['is_correct'] == '0' else 'borderline correct prediction',
                          'score': chosen['score'], 'threshold': chosen['threshold'], 'figure': 'examples/' + name, 'zoom_figure': zoom})
    write_csv(OUTPUT / 'difficult_cases.csv', difficult)

    # A score-distribution plot for the fixed, common 4-shot / seed-0 configuration.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(top=.80, bottom=.09, left=.055, right=.985, hspace=.37, wspace=.24)
    for axis, category in zip(axes.flat, CATEGORIES):
        subset = [r for r in after_rows if r['category'] == category and r['shots'] == '4' and r['seed'] == '0']
        cal = read_csv(OUTPUT / category / 'shot_4/seed_0/calibration.csv')
        for values, label, color, linestyle in [([float(r['score']) for r in cal], 'Calibration normal', '#9b6c11', ':'),
                ([float(r['score']) for r in subset if r['ground_truth_label'] == '0'], 'Test normal', '#2864a0', '-'),
                ([float(r['score']) for r in subset if r['ground_truth_label'] == '1'], 'Test anomaly', '#c44b38', '-')]:
            axis.hist(values, bins=np.linspace(0, 1, 51), density=True, histtype='step', linewidth=1.6, color=color, linestyle=linestyle, label=label)
        threshold = float(subset[0]['threshold'])
        axis.axvline(threshold, color='#202e3b', linestyle='--', linewidth=1.7, label='Calibration threshold')
        axis.set(title=category, xlabel='Image anomaly score', ylabel='Density', xlim=(0, 1))
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .90), ncol=4, frameon=False)
    fig.suptitle('Official AnomalyDINO score distributions — 4-shot / seed 0\nOriginal thresholds from 20 independent normal calibration images', y=.99, fontsize=14)
    for extension in ('png', 'svg'):
        fig.savefig(figures / f'score_distributions.{extension}', dpi=170)
    plt.close(fig)

    lines = ['# Part 2：AnomalyDINO 实测与基线对比', '',
             '固定 DINOv2 ViT-S/14、短边 448、官方 agnostic、CPU FAISS，权重冻结。复用 Part 1 的全部参考图和校准图名单。', '',
             '## 六类等权平均', '', '均值 ± 样本标准差来自三个 seed；AUROC 以下以百分数表示。', '',
             '| Shots | Baseline Image AUROC | AnomalyDINO Image AUROC | Baseline Pixel AUROC | AnomalyDINO Pixel AUROC |', '|---:|---:|---:|---:|---:|']
    macro_html = ''
    for old, new in zip(baseline_macro, dino_macro):
        assert old['shot'] == new['shot']
        cells = [new['shot']]
        for data, metric in [(old, 'image_auroc'), (new, 'image_auroc'), (old, 'pixel_auroc'), (new, 'pixel_auroc')]:
            cells.append(f'{100*float(data[metric+"_mean"]):.2f} ± {100*float(data[metric+"_std"]):.2f}')
        lines.append('| ' + ' | '.join(cells) + ' |')
        macro_html += '<tr>' + ''.join(f'<td>{html.escape(cell)}</td>' for cell in cells) + '</tr>'
    lines += ['', '![宏平均对比](figures/baseline_vs_dino.png)', '',
              '## 保存和比较', '',
              '- `predictions_all.csv` 保存 11,520 条正式测试预测，`errors_only.csv` 保存全部 FP/FN。',
              '- `runs.csv` 保存 72 组参数、参考名单、阈值、混淆矩阵、原始计时和显存。',
              '- `../comparisons/part1_vs_part2.csv` 对每张图记录 before/after outcome、修复和退步。',
              '- `fixed_case_comparison.csv` 与 Part 1 既有 22 张案例严格对齐；并非重新挑选提升最大的图片。',
              '- `.npy` 保存原始浮点 patch 图，通过官方函数恢复全分辨率图；彩色 PNG 只用于展示。', '',
              '## 分数和阈值', '', '![分数分布](figures/score_distributions.png)', '',
              '阈值仍由 20 张正常校准图决定，未用异常测试图调节。AUROC 与单个阈值无关；不同方法的原始 Score 不宜直接比大小。', '',
              '## 残余困难案例', '']
    category_summary = read_csv(OUTPUT / 'summary_by_category.csv')
    category_table = '<table><tr><th>类别</th><th>Shots</th><th>Image AUROC (%)</th><th>Pixel AUROC (%)</th></tr>'
    for row in category_summary:
        category_table += f'<tr><td>{row["category"]}</td><td>{row["shot"]}</td><td>{100*float(row["image_auroc_mean"]):.2f} ± {100*float(row["image_auroc_std"]):.2f}</td><td>{100*float(row["pixel_auroc_mean"]):.2f} ± {100*float(row["pixel_auroc_std"]):.2f}</td></tr>'
    category_table += '</table>'
    transition_summary = json.loads((COMPARE / 'part1_vs_part2.summary.json').read_text())
    transitions = transition_summary['transition_counts']
    transition_text = (f'在 11,520 条配对预测记录中：FN→TP {transitions.get("FN->TP",0)} 条，'
                       f'FP→TN {transitions.get("FP->TN",0)} 条，TP→FN {transitions.get("TP->FN",0)} 条，'
                       f'TN→FP {transitions.get("TN->FP",0)} 条。同一测试图在不同 shot/seed 下各占一条记录。')
    lines += ['## 逐图转变汇总', '', transition_text, '',
              '全部类别、shot 和 seed 的指标见 summary_by_category.csv；所有新增错误与纠正样例均保留在配对 CSV 中。', '']
    for row in difficult:
        lines += [f'### {row["category"]} — {row["outcome"]} ({row["description"]})', '',
                  f'`{row["sample_id"]}`，score={float(row["score"]):.4f}，T={float(row["threshold"]):.4f}。', '',
                  f'![{row["category"]}]({row["figure"]})', '']
        if row['zoom_figure']:
            lines += [f'![GT-guided zoom]({row["zoom_figure"]})', '', '局部框只用于展示，未重新裁图推理或调整分数。', '']
    lines += ['## 解释与局限', '',
              '- 预训练 patch 特征和最近邻检索允许相似正常结构跨空间位置匹配，减少对逐像素对齐的依赖；实际提升应结合本表和逐图转变看。',
              '- 14×14 输入 patch 在原图中对应更大的区域，细小缺陷可能被正常结构稀释；PCA foreground mask 也可能排除有用区域。',
              '- 即使 Pixel AUROC 较高，也不保证每张图的 OK/NG 都正确，仍须查看漏报和误报。',
              '- 存在修复，也可能有新增错误；完整比较 CSV 保留两者。不要只挑改善案例。',
              '- 查询特征每次新提取，参考旋转特征可精确复用；正式完整测试没有用缓存查询来伪装 GPU 推理速度。',
              '- 逐图官方计时不含后续分数聚合/全图恢复，并行类别运行会影响延迟；正式速度对比使用另行串行测得的相同输入、相同端到端边界 benchmark。',
              '- Part 3 仍未完成，当前不代表整个课程项目最终提交。', '']
    timing_file = COMPARE / 'timing_same_inputs.summary.json'
    if timing_file.exists():
        timing = json.loads(timing_file.read_text())
        lines += ['## 相同输入的端到端耗时', '',
                  '固定 4-shot / seed 0，每类首 10 张正常和首 10 张异常图，共每方法 120 张；包含读取、预处理、表征、匹配、评分和原图热力图恢复，不含建库和输出文件 I/O。', '',
                  f'- Baseline：平均 {timing["baseline_rgb_template"]["mean_ms"]:.1f} ms/image。',
                  f'- AnomalyDINO：平均 {timing["official_anomalydino"]["mean_ms"]:.1f} ms/image。', '']
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    def zoom_tag(row):
        return f'<img loading="lazy" src="{row["zoom_figure"]}" alt="GT-guided analysis-only zoom">' if row['zoom_figure'] else ''
    gallery = ''.join(f'<figure><figcaption>{html.escape(r["sample_id"])} — {r["description"]}, {r["outcome"]}</figcaption><img src="{r["figure"]}">{zoom_tag(r)}</figure>' for r in difficult)
    fixed_gallery = ''.join(f'<figure><figcaption>{html.escape(r["sample_id"])} — {r["transition"]}</figcaption><img loading="lazy" src="{r["figure"]}">{zoom_tag(r)}</figure>' for r in case_rows)
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Part 2 AnomalyDINO</title>
<style>body{{font:16px system-ui,"Microsoft YaHei";max-width:1300px;margin:35px auto;padding:0 20px;color:#172332}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:10px}}figure{{margin:24px 0}}summary{{cursor:pointer;padding:14px;background:#eef3f8}}pre{{white-space:pre-wrap}}</style>
<h1>Part 2：官方 AnomalyDINO 复现</h1><p>六类、1/2/4/8-shot、三个 seed；复用 Part 1 的参考和校准名单。冻结 DINOv2 ViT-S/14，448，agnostic，CPU FAISS。</p>
<p><a href="RESULTS.md">中文结果分析</a> · <a href="predictions_all.csv">全部逐图记录</a> · <a href="errors_only.csv">错误样本</a> · <a href="../comparisons/part1_vs_part2.csv">逐图前后对比</a> · <a href="../../docs/part2_method.md">方法与复现细节</a></p>
<img src="../../docs/figures/part2_pipeline.png" alt="AnomalyDINO pipeline">
<h2>六类平均指标</h2><p>AUROC (%)，三个 seed 的均值 ± 样本标准差。</p><table><tr><th>Shots</th><th>Baseline Image</th><th>AnomalyDINO Image</th><th>Baseline Pixel</th><th>AnomalyDINO Pixel</th></tr>{macro_html}</table>
<img src="figures/baseline_vs_dino.png"><h2>分数分布和原校准阈值</h2><img src="figures/score_distributions.png">
<details><summary>查看全部类别与 shot 的指标（24 行）</summary>{category_table}</details><h2>逐图变化</h2><p>{transition_text}</p>
<h2>五个实际困难样例</h2><p>固定 4-shot / seed 0；存在真实错误时展示错误，否则明确标注为接近阈值的正确预测。</p>{gallery}
<details><summary>查看与 Part 1 原来 22 张案例的完整对比</summary>{fixed_gallery}</details>
<h2>需要注意</h2><p>高 AUROC 不保证没有误报或漏报。细小缺陷、粗 patch 网格和 PCA mask 都可能影响结果。不同方法分数尺度不同，各自使用独立正常校准阈值。原始 patch 图保持浮点精度，展示图固定色阶。完整运行计时边界和串行 benchmark 见中文分析。</p></html>'''
    (OUTPUT / 'report.html').write_text(text, encoding='utf-8')
    print('\n'.join(lines[:20]), flush=True)
    print('REPORT:', OUTPUT / 'report.html', flush=True)


if __name__ == '__main__':
    main()
