"""Part 1 optional autoencoder demo entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Part1.autoencoder.demo import main
if __name__ == '__main__':
    main()
