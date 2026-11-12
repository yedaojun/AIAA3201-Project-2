"""Part 2 optional AnomalyCLIP demo entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Part2.anomalyclip.demo import main
if __name__ == '__main__':
    main()
