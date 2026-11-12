"""Part 1 optional autoencoder entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Part1.autoencoder.run import main
if __name__ == '__main__':
    main()
