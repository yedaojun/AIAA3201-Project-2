"""Fetch pinned upstream sources without overwriting existing local changes."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ('AnomalyDINO', 'https://github.com/dammsi/AnomalyDINO.git', 'b9d1c2648e3a5247437d4d953d907a8f3d994457'),
    ('dinov2', 'https://github.com/facebookresearch/dinov2.git', '7764ea0f912e53c92e82eb78a2a1631e92725fc8'),
    ('AnomalyCLIP', 'https://github.com/zqhang/AnomalyCLIP.git', '3911738c0867544f545a076ad78f3f11d9ecbfdf'),
)


def main():
    git = shutil.which('git')
    if git is None:
        raise SystemExit('Install Git and make it available on PATH.')
    (ROOT / 'third_party').mkdir(exist_ok=True)
    for name, url, commit in SOURCES:
        target = ROOT / 'third_party' / name
        if target.exists():
            actual = subprocess.check_output([git, '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != commit:
                raise RuntimeError(f'{target} already exists at {actual}; expected {commit}. No existing files were changed.')
            print(f'Already pinned: {name} {commit}', flush=True)
            continue
        subprocess.run([git, 'clone', url, str(target)], check=True)
        subprocess.run([git, '-C', str(target), 'checkout', commit], check=True)
        print(f'Pinned: {name} {commit}', flush=True)


if __name__ == '__main__':
    main()
