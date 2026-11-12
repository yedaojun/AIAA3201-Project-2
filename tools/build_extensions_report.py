"""Combined optional-method results, fixed-case visual comparisons and CSV transitions."""
import csv
from decimal import Decimal, ROUND_HALF_UP
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
from PIL import Image, ImageDraw

from anomaly_baseline.core import CATEGORIES
from anomaly_baseline.run import save_csv, get_mask
from experiment_utils.common import DATA, restore_map, write_json

OUT = ROOT / 'results/extensions'


def percent(value):
    return str((Decimal(str(value)) * 100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def heatmap(array, upper):
    return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(array / upper, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))


def add_panel(canvas, panel, x, y, width, height, title):
    draw = ImageDraw.Draw(canvas)
    draw.text((x + 8, y + 6), title, fill='#1b283a')
    panel = panel.copy()
    panel.thumbnail((width - 16, height - 35))
    canvas.paste(panel, (x + (width - panel.width) // 2, y + 30 + (height - 35 - panel.height) // 2))


def case_panel(method, row, destination, zoom=False):
    path = ROOT / row['image_file']
    with Image.open(path) as source:
        image = source.convert('RGB')
    dense = restore_map(np.load(ROOT / row['anomaly_map'])['map'], image.size)
    mask = get_mask(DATA, row['category'], 'bad' if int(row['label']) else 'good', path, dense.shape)
    upper = .20 if method == 'autoencoder' else 4.
    heat = heatmap(dense, upper)
    overlay = Image.blend(image, heat, .45)
    if method == 'autoencoder':
        recon_path = ROOT / 'results/part1_autoencoder' / row['category'] / 'reconstructions' / path.parent.name / (path.stem + '.png')
        with Image.open(recon_path) as source:
            reconstruction = source.convert('RGB').resize(image.size, Image.Resampling.BILINEAR)
    else:
        reconstruction = None
    gt = Image.fromarray(mask.astype(np.uint8) * 255).convert('RGB')
    pictures = [(image, 'Input'), (gt, 'Ground truth')]
    if reconstruction is not None:
        pictures.append((reconstruction, 'AE reconstruction'))
    pictures += [(heat, f'Heatmap fixed 0-{upper:g}'), (overlay, 'Overlay')]
    if zoom and mask.any():
        ys, xs = np.where(mask)
        pad = max(30, int(max(xs.max() - xs.min(), ys.max() - ys.min()) * .7))
        box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
               min(image.width, int(xs.max()) + pad + 1), min(image.height, int(ys.max()) + pad + 1))
        pictures = [(p.crop(box), title + ' (GT zoom)') for p, title in pictures]
    canvas = Image.new('RGB', (280 * len(pictures), 350), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'{method} | {row["category"]} {path.parent.name}/{path.name} | {row["outcome"]} | Score {float(row["score"]):.4f} T {float(row["threshold"]):.4f}', fill='#132238')
    for i, (picture, title) in enumerate(pictures):
        add_panel(canvas, picture, i * 280, 33, 280, 310, title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def draw_pipeline():
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.8))
    lines = [
        ('Part 1 optional: normal-only reconstruction', ['Normal fit images\n+ separate validation', 'Compact CNN\nautoencoder', 'RGB reconstruction\nresidual', 'Top 1% score\n+ residual map']),
        ('Part 2 optional: external-source prompts, zero VisA references', ['Frozen CLIP\n+ MVTec-trained prompts', 'Image / patch\ntext similarity', 'Global probability\n+ 4 layer maps', 'Score + heatmap\n20 normals set threshold']),
    ]
    for axis, (title, boxes) in zip(axes, lines):
        axis.axis('off')
        axis.set_title(title, loc='left', fontsize=13, fontweight='bold')
        for i, text in enumerate(boxes):
            x = .02 + i * .25
            axis.text(x, .45, text, transform=axis.transAxes, va='center', fontsize=10,
                      bbox=dict(boxstyle='round,pad=.75', fc='#edf4fa', ec='#91a7bc'))
            if i < 3:
                axis.annotate('', xy=(x + .245, .45), xytext=(x + .205, .45), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='#385776', lw=1.5))
    fig.tight_layout(pad=2)
    fig.savefig(OUT / 'figures/optional_pipelines.png', dpi=170)
    plt.close(fig)


def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    metrics, records = {}, {}
    for method, folder in [('autoencoder', 'part1_autoencoder'), ('anomalyclip', 'part2_anomalyclip')]:
        source = ROOT / 'results' / folder
        metrics[method] = json.loads((source / 'metrics_all.json').read_text())
        records[method] = read_csv(source / 'predictions_all.csv')
        assert len(metrics[method]) == 6 and len(records[method]) == 960
    rows = []
    for title, folder in [('RGB template (4-shot, 3 groups)', 'part1'), ('AnomalyDINO (4-shot, 3 groups)', 'part2')]:
        macro = next(r for r in read_csv(ROOT / 'results' / folder / 'summary_macro.csv') if int(r['shot']) == 4)
        rows.append({'method': title, 'image_auroc': f'{float(macro["image_auroc_mean"])*100:.2f} +/- {float(macro["image_auroc_std"])*100:.2f}',
                     'pixel_auroc': f'{float(macro["pixel_auroc_mean"])*100:.2f} +/- {float(macro["pixel_auroc_std"])*100:.2f}',
                     'target_fit': '0', 'normal_references_per_category': '4 per run',
                     'supervision': 'normal references; frozen / no task training'})
    for method in ('autoencoder', 'anomalyclip'):
        selected = metrics[method]
        rows.append({'method': 'CNN autoencoder (one seed)' if method == 'autoencoder' else 'AnomalyCLIP (zero target references)',
                     'image_auroc': percent(float(np.mean([r['image_auroc'] for r in selected]))),
                     'pixel_auroc': percent(float(np.mean([r['pixel_auroc'] for r in selected]))),
                     'target_fit': '2813 fit + 312 normal validation' if method == 'autoencoder' else '0',
                     'normal_references_per_category': 'not a few-shot method' if method == 'autoencoder' else '0',
                     'supervision': 'normal-only reconstruction' if method == 'autoencoder' else 'external MVTec normal/anomaly labels; frozen prompts'})
    save_csv(OUT / 'methods_context.csv', rows)
    base_records = {}
    for method, folder in [('template', 'part1'), ('dino', 'part2')]:
        base_records[method] = {r['sample_id']: r for r in read_csv(ROOT / 'results' / folder / 'predictions_all.csv') if int(r['shots']) == 4 and int(r['seed']) == 0}
    transitions = []
    for method, selected in records.items():
        for r in selected:
            for before, baseline in base_records.items():
                b = baseline[r['sample_id']]
                assert b['image_sha256'] == r['image_sha256']
                assert b['ground_truth_mask_sha256'] == r['ground_truth_mask_sha256']
                assert int(b['ground_truth_label']) == int(r['label'])
                old, new = b['outcome'], r['outcome']
                transitions.append({'sample_id': r['sample_id'], 'category': r['category'],
                                    'before_method': before + '_4shot_seed0', 'after_method': method,
                                    'before_outcome': old, 'after_outcome': new,
                                    'error_fixed': int(old in ('FP', 'FN') and new in ('TP', 'TN')),
                                    'regression': int(old in ('TP', 'TN') and new in ('FP', 'FN')),
                                    'before_score': b['score'], 'after_score': r['score'],
                                    'before_threshold': b['threshold'], 'after_threshold': r['threshold'],
                                    'protocol_note': 'same test input, different fitting/supervision; not a matched few-shot gain'})
    save_csv(OUT / 'paired_predictions.csv', transitions)
    case_index, figures = [], []
    with (ROOT / 'results/part1/supplement/case_index.csv').open(encoding='utf-8-sig', newline='') as file:
        original_cases = list(csv.DictReader(file))
    # Reuse the original Part 1 case set; add the first actual remaining optional-method error per class.
    for method, selected in records.items():
        chosen = {}
        featured = set()
        for case in original_cases:
            file_name = case.get('file') or case.get('test_file') or case.get('image_file')
            category = case['category']
            matches = [r for r in selected if r['category'] == category and r['file'] == file_name]
            if not matches:
                # The original fixed case manifest uses source_file in some schema versions.
                matches = [r for r in selected if r['category'] == category and any(r['file'] == str(v) or r['sample_id'] == str(v) for v in case.values())]
            if len(matches) != 1:
                raise ValueError(f'Cannot resolve original fixed case: {case}')
            chosen[matches[0]['sample_id']] = (matches[0], 'original Part 1 fixed case')
        for case in read_csv(ROOT / 'results/part2/difficult_cases.csv'):
            row = next(r for r in selected if r['sample_id'] == case['sample_id'])
            chosen.setdefault(row['sample_id'], (row, 'original Part 2 difficult case'))
            featured.add(row['sample_id'])
        for category in CATEGORIES:
            errors = sorted([r for r in selected if r['category'] == category and not int(r['is_correct'])], key=lambda r: r['sample_id'])
            if errors:
                chosen.setdefault(errors[0]['sample_id'], (errors[0], 'first actual optional-method error by sample_id'))
                featured.add(errors[0]['sample_id'])
        gallery = []
        for i, (row, reason) in enumerate(chosen.values()):
            relative = f'examples/{method}_{i:02d}_{row["category"]}.png'
            case_panel(method, row, OUT / relative)
            path = ROOT / row['image_file']
            with Image.open(path) as image:
                mask = get_mask(DATA, row['category'], 'bad' if int(row['label']) else 'good', path, (image.height, image.width))
            zoom = ''
            if mask.any() and np.mean(mask) < .01:
                zoom = relative.replace('.png', '_zoom.png')
                case_panel(method, row, OUT / zoom, zoom=True)
            dense = restore_map(np.load(ROOT / row['anomaly_map'])['map'], (mask.shape[1], mask.shape[0]))
            case_index.append({'method': method, 'sample_id': row['sample_id'], 'outcome': row['outcome'],
                               'score': row['score'], 'threshold': row['threshold'], 'selection_reason': reason,
                               'defect_area_percent': float(mask.mean() * 100),
                               'mean_score_in_defect': float(dense[mask].mean()) if mask.any() else '',
                               'mean_score_outside_defect': float(dense[~mask].mean()),
                               'figure': relative, 'zoom': zoom, 'anomaly_map': row['anomaly_map']})
            if row['sample_id'] in featured or (row['category'] == 'candle' and path.name == '000.JPG'):
                gallery.append((row, relative, zoom))
        figures.append((method, gallery))
    save_csv(OUT / 'case_index.csv', case_index)
    # Training curves: every epoch, selected solely by normal validation loss.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for axis, category in zip(axes.flat, CATEGORIES):
        history = read_csv(ROOT / 'results/part1_autoencoder' / category / 'training_history.csv')
        for field, label in [('train_mse', 'normal fit'), ('validation_mse', 'normal validation')]:
            axis.plot([int(r['epoch']) for r in history], [float(r[field]) for r in history], label=label)
        axis.set_yscale('log')
        axis.set_title(category)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('RGB MSE (log)')
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'figures/autoencoder_training.png', dpi=160)
    plt.close(fig)
    # All category metrics, including weak results, on common axes.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(6)
    for j, metric in enumerate(('image_auroc', 'pixel_auroc')):
        for i, method in enumerate(('autoencoder', 'anomalyclip')):
            lookup = {m['category']: m for m in metrics[method]}
            axes[j].bar(x + (i - .5) * .36, [lookup[c][metric] * 100 for c in CATEGORIES], width=.36, label=method)
        axes[j].set_xticks(x, CATEGORIES, rotation=22)
        axes[j].set_ylim(0, 100)
        axes[j].set_ylabel(('Image AUROC' if j == 0 else 'Pixel AUROC') + ' (%)')
        axes[j].grid(axis='y', alpha=.2)
        axes[j].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / 'figures/category_metrics.png', dpi=170)
    plt.close(fig)
    for method, folder in [('autoencoder', 'part1_autoencoder'), ('anomalyclip', 'part2_anomalyclip')]:
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2))
        for axis, category in zip(axes.flat, CATEGORIES):
            selected = [r for r in records[method] if r['category'] == category]
            calibration = read_csv(ROOT / 'results' / folder / category / 'calibration.csv')
            groups = [[float(r['score']) for r in calibration],
                      [float(r['score']) for r in selected if not int(r['label'])],
                      [float(r['score']) for r in selected if int(r['label'])]]
            low, high = min(map(min, groups)), max(map(max, groups))
            bins = np.linspace(low, high + max(1e-6, (high-low) * .001), 22)
            for values, label, color in zip(groups, ['calibration normal (20)', 'test normal', 'test anomalous'], ['#54708b', '#187a8f', '#c14b43']):
                density, _ = np.histogram(values, bins=bins, density=True)
                axis.stairs(density, bins, label=label, color=color, linewidth=1.6)
            axis.axvline(float(selected[0]['threshold']), color='#333', linestyle='--', label='calibration threshold')
            axis.set_title(category)
            axis.set_xlabel('Raw image score')
            axis.set_ylabel('Density')
            axis.grid(alpha=.15)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(method + ': fixed calibration threshold; test labels used for analysis only')
        fig.tight_layout()
        fig.savefig(OUT / f'figures/{method}_score_distributions.png', dpi=160)
        plt.close(fig)
    draw_pipeline()
    table_headers = ['方法', 'Image AUROC (%)', 'Pixel AUROC (%)', '目标域训练图', '正常参考图/类', '监督来源']
    table = '<table><tr>' + ''.join(f'<th>{h}</th>' for h in table_headers) + '</tr>'
    for row in rows:
        table += '<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row.values()) + '</tr>'
    table += '</table>'
    body = ''
    for method, gallery in figures:
        body += f'<h2>{method}：真实错误与固定案例</h2>'
        for row, relative, zoom in gallery:
            body += f'<p>{html.escape(row["sample_id"])}：{row["outcome"]}，Score={float(row["score"]):.4f}，阈值={float(row["threshold"]):.4f}。</p><img src="{relative}">'
            if zoom:
                body += f'<p>GT 定位的局部放大，仅用于展示，未重新裁图推理。</p><img src="{zoom}">'
    timing = ''
    if (OUT / 'timing_summary.csv').exists():
        timing_rows = read_csv(OUT / 'timing_summary.csv')
        timing = '<h2>同一批 120 张图的独立串行计时</h2><table><tr><th>方法</th><th>平均 ms/图</th><th>峰值显存 MiB</th></tr>' + ''.join(f'<tr><td>{r["method_id"]}</td><td>{float(r["mean_ms"]):.2f}</td><td>{float(r["gpu_peak_memory_mb"]):.1f}</td></tr>' for r in timing_rows) + '</table>'
    training_counts = sum(m['training']['fit_images'] for m in metrics['autoencoder'])
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Part 1 + Part 2 可选扩展</title>
<style>body{{font:16px 'Segoe UI','Microsoft YaHei',sans-serif;max-width:1320px;margin:36px auto;padding:0 24px;color:#1a293c;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border:1px solid #dbe2e9;text-align:left}}th{{background:#edf3f9}}img{{width:100%;height:auto;margin:10px 0 22px}}.note{{padding:16px;background:#fff5dc;border-left:4px solid #bd8d29}}a{{color:#185da6}}code{{background:#edf1f6;padding:2px 5px}}</style>
<h1>Part 1 与 Part 2：两个可选扩展</h1>
<p>完整官方六类测试集：360 张正常图 + 600 张异常图。必做方法的旧结果保持原样。AUROC 不是阈值下的准确率。</p>
<div class="note">这些是不同监督条件的补充实验：自编码器实际拟合 {training_counts} 张正常图，另有 312 张正常验证图；AnomalyCLIP 使用在外部 MVTec AD 正常/异常标注上训练的提示权重，VisA 参考图为 0。两者各额外使用每类 20 张正常图校准阈值。AnomalyCLIP 使用 448 方形缩放，必做方法保持纵横比。不能将本表解读为严格同条件 few-shot 排名，也不能将扩展算作 Part 3 改进。</div>
<h2>完整结果与条件</h2>{table}
<p>必做方法报告三组参考的均值和标准差；两个可选方法只运行一个预先固定的 seed，没有虚构重复实验或标准差。</p>
<img src="figures/category_metrics.png"><h2>方法流程</h2><img src="figures/optional_pipelines.png">
<h2>自编码器：训练是否完成</h2><p>六个独立小模型，342,539 参数/模型，20 轮正常图重建训练。按独立正常验证集的最低 MSE 选择 checkpoint。训练损失下降只说明重建改善，不保证缺陷检测改善。</p><img src="figures/autoencoder_training.png">
{timing}<h2>解释结果时要区分什么</h2>
<p>校准、正常测试与异常测试的原始图像分数分布如下。虚线是预先由 20 张正常校准图得到的阈值；没有按测试分布重新选择阈值。</p><img src="figures/autoencoder_score_distributions.png"><img src="figures/anomalyclip_score_distributions.png">
<p>模板差分容易响应产品位置、轮廓与排列变化；自编码器用训练得到的重建代替单张模板，但正常边缘的重建误差也可能很高，细微缺陷又可能被一起重建。AnomalyCLIP 的全图文本异常概率和局部热力图来自不同读出，局部亮点不自动保证整图 NG。这些是需要结合实际案例检验的机制，不把所有错误归结为同一原因。</p>
<p>自编码器 cashew 的 Image AUROC 为 95.98%，但 Pixel AUROC 仅为 31.56%，低于随机排序的 50%：整图排序好不代表定位正确。实际 bad/063.JPG 中，重建图将粗糙背景明显平滑，残差大量落在正常背景，整体判为 TP 仍不能证明找到了真实缺陷。该低分原样保留，没有反转异常图或使用测试标签改动参数。</p>
<p>所有阈值只来自正常校准图。固定显示色阶：AE 0–0.20，CLIP 0–4；显示裁剪不参与指标。下方保留真实 FP/FN，完整固定案例与所有错误均可追溯。</p>
{body}<h2>每张图的记录和可复现文件</h2>
<ul><li><a href="../part1_autoencoder/predictions_all.csv">自编码器逐图预测 CSV</a> / <a href="../part1_autoencoder/errors_only.csv">全部错误</a> / <a href="../part1_autoencoder/input_images.csv">训练、验证、校准用图与校验值</a></li>
<li><a href="../part2_anomalyclip/predictions_all.csv">AnomalyCLIP 逐图预测 CSV</a> / <a href="../part2_anomalyclip/errors_only.csv">全部错误</a> / <a href="../part2_anomalyclip/run_config.json">权重来源和配置</a></li>
<li><a href="paired_predictions.csv">与固定 4-shot/seed-0 的同图 TP/TN/FP/FN 转变</a>（监督条件不同）</li><li><a href="case_index.csv">全部固定案例与放大图索引</a> / <a href="audit.json">独立核验</a> / <a href="../../docs/optional_extensions.md">完整复现说明</a></li></ul>
<p>Part 1 独立 Demo：<code>.\\.venv-part2\\Scripts\\python.exe Part1/demo_autoencoder.py</code><br>Part 2 独立 Demo：<code>.\\.venv-part2\\Scripts\\python.exe Part2/demo_anomalyclip.py</code></p>
<p>来源：<a href="https://github.com/zqhang/AnomalyCLIP">AnomalyCLIP 官方仓库</a>；<a href="https://github.com/openai/CLIP">OpenAI CLIP</a>；VisA 数据集与课程官方划分。</p></html>'''
    (OUT / 'report.html').write_text(text, encoding='utf-8')
    md = '# Part 1 与 Part 2 可选扩展实测\n\n两个扩展均完成六类全部 960 张正式测试图。不同监督条件，不作为同条件 few-shot 排名。\n\n'
    md += '| 方法 | Image AUROC (%) | Pixel AUROC (%) |\n|---|---:|---:|\n'
    md += '\n'.join(f'| {r["method"]} | {r["image_auroc"]} | {r["pixel_auroc"]} |' for r in rows)
    md += '\n\n自编码器：2813 张拟合 + 312 张正常验证，6 模型各 20 轮，342539 参数/模型。AnomalyCLIP：MVTec 源域监督提示权重，目标域 0 参考图，448 方形输入（官方默认 518）。均额外使用每类 20 张正常图校准。\n\n'
    md += '每张图的真实标签、原始分数、阈值、TP/TN/FP/FN、错误纠正及退步均已保存。`report.html` 提供训练曲线、方法图、完整指标和真实失败案例；`case_index.csv` 包含原先 Part 1 的全部 22 张固定案例在两种扩展下的输出。\n'
    (OUT / 'RESULTS.md').write_text(md, encoding='utf-8')
    print('Report:', OUT / 'report.html', flush=True)


if __name__ == '__main__':
    main()
