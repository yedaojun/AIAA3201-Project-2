"""Part 1 autoencoder-only prediction; no method selector."""
from experiment_utils.common import ROOT
from experiment_utils.prediction import run_cli
from .model import AutoencoderPredictor

RESULTS = ROOT / 'results/part1_autoencoder'


def load_predictor(category):
    return AutoencoderPredictor(RESULTS / category / 'best.pt')


def main():
    run_cli('autoencoder', load_predictor, RESULTS, ROOT / 'results/demo_part1_autoencoder')


if __name__ == '__main__':
    main()
