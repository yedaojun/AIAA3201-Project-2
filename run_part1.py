"""Run the full baseline and always export standardized experiment logs."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/part1')
    args = parser.parse_args()
    steps = [
        ['-m', 'anomaly_baseline.prepare'],
        ['-u', '-m', 'anomaly_baseline.run', '--output', args.output],
        ['tools/export_prediction_logs.py', '--results', args.output],
        ['tools/build_part1_supplement.py', '--results', args.output],
    ]
    for step in steps:
        print('Running:', ' '.join([sys.executable] + step), flush=True)
        subprocess.run([sys.executable] + step, cwd=root, check=True)


if __name__ == '__main__':
    main()
