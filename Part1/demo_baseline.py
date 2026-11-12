"""Part 1 required template baseline demo entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anomaly_baseline.demo import main
if __name__ == '__main__':
    main()
