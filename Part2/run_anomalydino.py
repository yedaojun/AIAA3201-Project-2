"""Part 2 required official AnomalyDINO entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
if __name__ == '__main__':
    runpy.run_module('anomaly_dino', run_name='__main__')
