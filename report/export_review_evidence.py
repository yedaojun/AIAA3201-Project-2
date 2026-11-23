"""Copy compact audit records into the report package (no raw image data)."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/evidence'
FILES = {
    'reference_manifest.json': 'configs/reference_manifest.json',
    'all_shots_category_seed.csv': 'results/part3/multiscale_fusion/all_shots_category_seed.csv',
    'all_shots_macro_by_seed.csv': 'results/part3/multiscale_fusion/all_shots_macro_by_seed.csv',
    'all_shots_summary.json': 'results/part3/multiscale_fusion/all_shots_summary.json',
    'all_shots_per_image_transitions.csv':
        'results/part3/multiscale_fusion/all_shots_per_image_transitions.csv',
    'all_shots_transition_counts.json':
        'results/part3/multiscale_fusion/all_shots_transition_counts.json',
    'multiscale_pixel_metrics_all_shots_3seeds.json':
        'results/part3/multiscale_fusion/pixel_metrics_all_shots_3seeds.json',
    'a_anyup_pixel_ap_4shot_3seeds.json':
        'results/part3/a_anyup_pixel_ap_4shot_3seeds.json',
    'defect_size_image_summary.csv': 'results/part3/defect_size_audit/summary.csv',
    'defect_size_pixel_ap_4shot_3seeds.csv':
        'results/part3/defect_size_audit/pixel_ap_by_size_4shot.csv',
    'normal_threshold_estimator_summary.csv':
        'results/part3/calibration_methods_4shot_3seeds/summary.csv',
    'normal_threshold_estimator_category_seed.csv':
        'results/part3/calibration_methods_4shot_3seeds/category_seed.csv',
    'unified_benchmark_4shot_seed0.json':
        'results/part3/unified_benchmark_4shot_seed0/summary.json',
    'calibrated_aggregation_summary.csv':
        'results/part3/calibrated_aggregation_audit/summary.csv',
    'calibrated_aggregation_cases.csv':
        'results/part3/calibrated_aggregation_audit/cases.csv',
    'pure_dino_risk_coverage.csv':
        'results/part3/pure_dino_reliability_4shot_3seeds/risk_coverage.csv',
    'pure_dino_operating_points.csv':
        'results/part3/pure_dino_reliability_4shot_3seeds/operating_points.csv',
}


def main():
    missing = [relative for relative in FILES.values() if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError('Missing complete experiment files: ' + ', '.join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, relative in FILES.items():
        shutil.copyfile(ROOT / relative, OUT / name)
    print(f'Exported {len(FILES)} compact evidence files to {OUT}', flush=True)


if __name__ == '__main__':
    main()
