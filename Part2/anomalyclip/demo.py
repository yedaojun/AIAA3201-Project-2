"""Independent Part 2 AnomalyCLIP demo."""
from experiment_utils.common import ROOT
from experiment_utils.demo_window import run_window
from .predict import RESULTS, load_predictor


def main():
    run_window(title='Part 2 optional - AnomalyCLIP', method='anomalyclip',
               loader=load_predictor, results=RESULTS, output=ROOT / 'results/demo_part2_anomalyclip',
               upper=4, reconstruction=False, category_specific=False)


if __name__ == '__main__':
    main()
