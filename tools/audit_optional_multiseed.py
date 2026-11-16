"""Audit completed three-seed experiments against fixed data and saved predictions."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES=('candle','capsules','cashew','chewinggum','fryum','pipe_fryum')
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def audit():
    report={}
    for folder in ('part1_autoencoder_multiseed','part2_anomalyclip_multiseed'):
        base=ROOT/'results'/folder
        summary=json.loads((base/'summary.json').read_text())
        assert summary['seed_count']==3 and summary['total_test_predictions']==2880
        score_arrays=[];map_digest=[];initial_checks=[]
        for seed in (0,1,2):
            perseed=[];digest=hashlib.sha256()
            for c in CATEGORIES:
                path=base/f'seed_{seed}'/c
                metric=json.loads((path/'metrics.json').read_text())
                assert metric['experiment_seed']==seed
                predictions=read(path/'predictions.csv')
                labels=[int(r['label']) for r in predictions];scores=[float(r['score']) for r in predictions]
                assert abs(roc_auc_score(labels,scores)-metric['image_auroc'])<1e-12
                assert all(r['decision']==('NG' if float(r['score'])>metric['threshold'] else 'OK') for r in predictions)
                assert sum(metric[k] for k in ('TP','TN','FP','FN'))==len(predictions)
                for r in predictions:
                    with np.load(path/r['anomaly_map']) as a:
                        assert np.isfinite(a['map']).all()
                        digest.update(a['map'].tobytes())
                perseed.extend(scores)
                if folder.startswith('part1'):
                    images=read(path/'training_images.csv')
                    roles={role:{r['file'] for r in images if r['role']==role} for role in ('fit','validation','calibration')}
                    assert not roles['fit']&roles['validation'] and not (roles['fit']|roles['validation'])&roles['calibration']
                    if seed:
                        previous=read(base/f'seed_0/{c}/training_images.csv')
                        assert [(r['file'],r['role'],r['sha256']) for r in images]==[(r['file'],r['role'],r['sha256']) for r in previous]
                    config=json.loads((path/'training_config.json').read_text())['config']
                    assert config['training_seed']==seed and config['split_seed']==111
                    history=read(path/'training_history.csv')
                    assert len(history)==20 and history[-1]['epoch']=='20'
                    initial_checks.append((c,seed,history[0]['train_mse']))
            assert len(perseed)==960
            score_arrays.append(np.array(perseed));map_digest.append(digest.hexdigest())
        item={'actual_seeds':[0,1,2],'category_runs':18,'test_predictions':2880,'maps_finite':True,
              'maximum_score_difference':float(max(np.max(np.abs(score_arrays[0]-a)) for a in score_arrays[1:])),
              'identical_map_values_across_seeds':len(set(map_digest))==1}
        if initial_checks:
            assert any(len({v for category,s,v in initial_checks if category==c})>1 for c in CATEGORIES)
            item['distinct_initial_training_losses']=True
            item['fixed_fit_validation_calibration_files']=True
        report[folder]=item
    target=ROOT/'results/optional_multiseed_audit.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))
if __name__=='__main__':audit()
