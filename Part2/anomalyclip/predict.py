"""Part 2 AnomalyCLIP-only prediction; no autoencoder import or method selector."""
from experiment_utils.common import ROOT
from experiment_utils.prediction import run_cli
from .model import AnomalyCLIPPredictor

RESULTS = ROOT / 'results/part2_anomalyclip'


def load_predictor(category):
    return AnomalyCLIPPredictor()


def main():
    run_cli('anomalyclip', load_predictor, RESULTS, ROOT / 'results/demo_part2_anomalyclip')


if __name__ == '__main__':
    main()
