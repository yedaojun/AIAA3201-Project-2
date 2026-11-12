"""Portable entry point; git lookup does not change the detector/evaluation code."""
from pathlib import Path
import shutil
import subprocess

from . import run


def portable_commit(folder):
    git = shutil.which('git')
    if git is None:
        fallback = Path('C:/Users/Ziqing/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe')
        git = str(fallback) if fallback.is_file() else None
    if git is None:
        raise RuntimeError('Git is required on PATH to record the pinned upstream versions.')
    return subprocess.check_output([git, '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()


run.git_commit = portable_commit
run.main()
