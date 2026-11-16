"""Run both optional extensions at seeds 0, 1, 2, preserving the earlier seed111 runs."""
import csv,json,subprocess,sys,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from anomaly_baseline.run import save_csv
from experiment_utils.common import write_json

def aggregate(folder):
    group_rows=[];runs=[];predictions=[];inputs=[]
    for seed in (0,1,2):
        path=folder/f'seed_{seed}'
        metrics=json.loads((path/'metrics_all.json').read_text())
        assert len(metrics)==6 and sum(m['test_images'] for m in metrics)==960
        group={'seed':seed,'category_count':6,'test_images':960}
        for key in ('image_auroc','pixel_auroc','accuracy','image_ap'):
            group[key]=statistics.mean(m[key] for m in metrics)
        group_rows.append(group)
        for m in metrics:
            assert m['experiment_seed']==seed
            runs.append({k:v for k,v in m.items() if k not in ('training','calibration_files')})
        for filename,rows in [('predictions_all.csv',predictions),('input_images.csv',inputs)]:
            with (path/filename).open(encoding='utf-8-sig',newline='') as f:
                rows.extend({**r,'experiment_seed':seed} for r in csv.DictReader(f))
    summary={'method_id':metrics[0]['method_id'],'seed_count':3,'seeds':[0,1,2],
             'category_count':6,'unique_test_images':960,'total_test_predictions':len(predictions),
             'aggregation':'equal mean over six categories within each seed, then mean and sample SD over three seeds',
             'protocol_note':'Optional extension with its own training/supervision; not the matched few-shot reference-group experiment.'}
    for key in ('image_auroc','pixel_auroc','accuracy','image_ap'):
        summary[key+'_mean']=statistics.mean(r[key] for r in group_rows)
        summary[key+'_std']=statistics.stdev(r[key] for r in group_rows)
    if summary['method_id']=='official_anomalyclip':
        aligned=[]
        for seed in (0,1,2):
            aligned.append({r['sample_id']:(float(r['score']),r['decision'],float(r['threshold'])) for r in predictions if r['experiment_seed']==seed})
        assert all(len(x)==960 for x in aligned)
        summary['identical_scores_and_decisions_across_seeds']=aligned[0]==aligned[1]==aligned[2]
        summary['maximum_score_difference']=max(abs(aligned[0][k][0]-aligned[s][k][0]) for k in aligned[0] for s in (1,2))
    save_csv(folder/'summary_by_seed.csv',group_rows)
    save_csv(folder/'runs.csv',runs)
    save_csv(folder/'predictions_all.csv',predictions)
    save_csv(folder/'errors_only.csv',[r for r in predictions if int(r['is_correct'])==0])
    save_csv(folder/'input_images.csv',inputs)
    save_csv(folder/'summary_macro.csv',[{k:v for k,v in summary.items() if k!='seeds'}])
    write_json(folder/'summary.json',summary)
    print(json.dumps(summary),flush=True)

def main():
    for module,name in [('Part1.autoencoder.run','part1_autoencoder_multiseed'),('Part2.anomalyclip.run','part2_anomalyclip_multiseed')]:
        folder=ROOT/'results'/name;folder.mkdir(parents=True,exist_ok=True)
        for seed in (0,1,2):
            print(f'START {module}, seed={seed}',flush=True)
            subprocess.run([sys.executable,'-m',module,'--seed',str(seed),'--output',str(folder/f'seed_{seed}')],cwd=ROOT,check=True)
        aggregate(folder)
    print('All optional three-seed experiments finished.',flush=True)

if __name__=='__main__':main()
