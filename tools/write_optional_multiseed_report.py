"""Write a compact, traceable report only after both three-seed studies finish."""
import csv,json
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def percent(x):return str((Decimal(str(x))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=[];summary={}
for method,name in [('part1_autoencoder_multiseed','Autoencoder'),('part2_anomalyclip_multiseed','AnomalyCLIP')]:
    base=ROOT/'results'/method
    s=json.loads((base/'summary.json').read_text());assert s['seed_count']==3
    summary[name]=s
    if name=='Autoencoder':
        history=[]
        for seed in (0,1,2):
            for path in sorted((base/f'seed_{seed}').glob('*/training_history.csv')):
                history.extend({'category':path.parent.name,'experiment_seed':seed,**r} for r in read(path))
        assert len(history)==360
        with (base/'training_history_all.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    for r in read(base/'summary_by_seed.csv'):
        rows.append(f'| {name} | {r["seed"]} | {percent(r["image_auroc"])} | {percent(r["pixel_auroc"])} |')
    rows.append(f'| **{name} mean ± sample SD** | **0, 1, 2** | **{percent(s["image_auroc_mean"])} ± {percent(s["image_auroc_std"])}** | **{percent(s["pixel_auroc_mean"])} ± {percent(s["pixel_auroc_std"])}** |')
audit=json.loads((ROOT/'results/optional_multiseed_audit.json').read_text())
clip=audit['part2_anomalyclip_multiseed']
report='# Optional extensions: seeds 0, 1, 2\n\nBoth optional methods completed three actual runs on the same six VisA categories and all 960 official test images. The original seed111 experiments remain separate and do not enter the new mean or standard deviation.\n\n| Method | Seed | Image AUROC (%) | Pixel AUROC (%) |\n| --- | --- | ---: | ---: |\n'+'\n'.join(rows)+'\n\n'
report+='Each seed first receives an equal six-category macro average. The mean and sample standard deviation then use those three macro values (`ddof=1`). Standard deviations in this table are percentage points, not confidence intervals.\n\n'
report+='The autoencoder uses the same 2,813 normal fitting images, 312 normal validation images and 120 calibration images across all seeds. Each seed trains six separate category models for 20 epochs, so the new study trained 18 models. Initialization and training order change. The fit/validation split stays fixed at split_seed111. Checkpoint selection uses normal validation MSE only.\n\n'
report+='AnomalyCLIP repeats inference using the same verified CLIP backbone and external MVTec-supervised prompt checkpoint. It does not train prompts or a backbone on VisA. It uses no VisA reference bank, but its decision thresholds still use the same 20 normal calibration images per category. It uses square448 geometry, whereas the required methods preserve aspect ratio. These are separately reported supplementary conditions, not matched few-shot trials.\n\n'
report+=f'The cross-seed audit found a maximum AnomalyCLIP image-score difference of {clip["maximum_score_difference"]:.10g}. Identical saved anomaly-map values across the three seeds: **{clip["identical_map_values_across_seeds"]}**. These are deterministic inference repeats, not a few-shot stability test. Zero observed variation does not establish robustness to new images or other datasets.\n\n'
report+='Each optional study contains 18 category-level runs and 2,880 test predictions. Combined with the 144 required-method runs, the presentation evidence contains 180 category-level runs and 28,800 test predictions on 960 unique test images.\n\n'
report+='## Reproduce\n\n```powershell\n.\\.venv-part2\\Scripts\\python.exe tools\\run_optional_multiseed.py\n.\\.venv-part2\\Scripts\\python.exe tools\\audit_optional_multiseed.py\n```\n\n'
report+='Use a new output directory if method settings or source fingerprints differ. The run entry points reject changed settings before overwriting saved experiment metadata. Individual commands also accept `--seed 0`, `--seed 1` or `--seed 2` under `Part1/run_autoencoder.py` and `Part2/run_anomalyclip.py`.\n\n'
report+='## Saved records\n\nEach parent study directory has `summary_by_seed.csv`, `summary_macro.csv`, `runs.csv`, `input_images.csv`, `predictions_all.csv` and `errors_only.csv`. Each `seed_N/category/` keeps its own metrics, calibration scores, per-image predictions and maps. The autoencoder also keeps checkpoints, the 20-row training history and input-image roles with hashes.\n'
(ROOT/'docs/optional_multiseed.md').write_text(report,encoding='utf8')
(ROOT/'results/optional_multiseed_report.md').write_text(report,encoding='utf8')
print(json.dumps(summary))
