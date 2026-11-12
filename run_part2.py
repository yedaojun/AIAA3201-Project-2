"""Complete local Part 2 workflow, with logs and strict paired baseline comparison."""
import subprocess
from pathlib import Path
import argparse
import csv
import json
import time

root = Path(__file__).resolve().parent
python = root / '.venv-part2/Scripts/python.exe'
if not python.exists():
    raise SystemExit('Create .venv-part2 with the documented CUDA environment first.')
parser = argparse.ArgumentParser()
parser.add_argument('--postprocess-only', action='store_true')
parser.add_argument('--wait', action='store_true', help='Wait for this running experiment to finish before postprocessing')
args = parser.parse_args()
steps = [
    ['-u', '-m', 'anomaly_dino'],
    ['tools/export_prediction_logs.py', '--results', 'results/part2', '--method-id', 'official_anomalydino'],
    ['tools/compare_prediction_logs.py', '--before', 'results/part1/predictions_all.csv', '--after', 'results/part2/predictions_all.csv', '--output', 'results/comparisons/part1_vs_part2.csv'],
    ['tools/audit_part2.py'],
    ['tools/benchmark_part1_part2.py'],
    ['tools/draw_part2_pipeline.py'],
    ['tools/build_part2_report.py'],
]
if args.postprocess_only:
    steps = steps[1:]
if args.wait:
    last_count = -1
    while True:
        output = root / 'results/part2'
        count = len(list(output.glob('*/shot_*/seed_*/metrics.json')))
        if count != last_count:
            print(f'Waiting for complete experiment: {count}/72 configurations', flush=True)
            last_count = count
        try:
            completed = json.loads((output / 'metrics_all.json').read_text())
            with (output / 'summary_macro.csv').open(encoding='utf-8-sig', newline='') as file:
                summary = list(csv.DictReader(file))
            if count == len(completed) == 72 and len(summary) == 4 and all(r['category_count'] == '6' and r['seed_count'] == '3' for r in summary):
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(5)
for step in steps:
    print('Running:', ' '.join(step), flush=True)
    subprocess.run([str(python)] + step, cwd=root, check=True)
