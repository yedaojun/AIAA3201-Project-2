"""Part 1 required template baseline entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anomaly_baseline.run import main
if __name__ == '__main__':
    main()
