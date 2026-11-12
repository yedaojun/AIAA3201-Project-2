"""Independent Part 1 autoencoder demo."""
from experiment_utils.common import ROOT
from experiment_utils.demo_window import run_window
from .predict import RESULTS, load_predictor


def main():
    run_window(title='Part 1 optional - Convolutional Autoencoder', method='autoencoder',
               loader=load_predictor, results=RESULTS, output=ROOT / 'results/demo_part1_autoencoder',
               upper=.20, reconstruction=True, category_specific=True)


if __name__ == '__main__':
    main()
