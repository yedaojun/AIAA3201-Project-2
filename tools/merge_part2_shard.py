"""Atomically collect completed, disjoint category results without overwriting work."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--shard', type=Path, default=ROOT / 'results/part2_shard')
parser.add_argument('--destination', type=Path, default=ROOT / 'results/part2')
args = parser.parse_args()
shard, destination = args.shard.resolve(), args.destination.resolve()
boundary = (ROOT / 'results').resolve()
assert shard != destination and shard.is_relative_to(boundary) and destination.is_relative_to(boundary)
if json.loads((shard / 'run_config.json').read_text()) != json.loads((destination / 'run_config.json').read_text()):
    raise ValueError('Shard configuration differs; refusing to merge')
expected = {(shot, seed) for shot in (1, 2, 4, 8) for seed in (0, 1, 2)}
signatures = {json.loads(path.read_text())['run_signature'] for path in destination.glob('*/shot_*/seed_*/metrics.json')}
assert len(signatures) == 1
merged = []
for source in sorted(shard.iterdir()):
    if not source.is_dir():
        continue
    metrics = [json.loads(path.read_text()) for path in source.glob('shot_*/seed_*/metrics.json')]
    if {(m['shot'], m['seed']) for m in metrics} != expected:
        print('Not yet complete:', source.name)
        continue
    if any(m['run_signature'] not in signatures or m['category'] != source.name for m in metrics):
        raise ValueError('Shard signature/category mismatch')
    target = destination / source.name
    if target.exists():
        print('Destination category already exists; leaving independent shard untouched:', source.name)
        continue
    # Both resolved absolute paths remain inside this project's results directory.
    assert source.resolve().is_relative_to(boundary) and target.resolve().is_relative_to(boundary)
    source.rename(target)
    merged.append(source.name)
    print('Collected complete category:', target)
print(json.dumps({'merged_categories': merged}))
