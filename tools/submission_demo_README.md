# AIAA 3201 Project 2 — Anomaly Demo

`anomaly_demo.mp4` is a short walkthrough rendered from three **real saved runs** of the single-image Part 3 CLI. It is not a screen capture or a new benchmark. The three `examples/*.png` files show the input product image, its pixel-level anomaly heatmap, and an overlay. `example_results.csv` provides the unrounded score and threshold used for the displayed OK/NG decision.

The first two examples are a normal product accepted as OK and a defective product flagged NG. The third example is a documented false negative, included to show a current limitation. The test folder name was **not** read by the detector to make its decision.

## Run the interactive single-image demo

The source code and setup instructions are in the public repository:

https://github.com/yedaojun/AIAA3201-Project-2

After installing the repository dependencies, preparing the official six-category VisA split, and running the four-reference/group-0 two-layer configuration, choose any product image of the matching category:

```powershell
.\.venv-part2\Scripts\python.exe -m part3.demo --image data/VisA_pytorch/1cls/candle/test/good/0011.JPG --category candle --shots 4 --seed 0 --mode multilayer --map-restoration official --output results/submission_demo/good_candle
```

`--image` accepts another candle product image path. Change `--category` to the appropriate one of `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`, or `pipe_fryum` for other products. The command writes `prediction.json` (score, threshold, decision, reference filenames) and `prediction.png` (input, anomaly heatmap, overlay). The image score uses the raw patch distances; the map display uses one fixed 0–0.4 scale.

The three examples in the MP4 were generated with:

```powershell
.\.venv-part2\Scripts\python.exe -m part3.demo --image data/VisA_pytorch/1cls/candle/test/good/0011.JPG --category candle --shots 4 --seed 0 --mode multilayer --map-restoration official --output results/submission_demo/good_candle
.\.venv-part2\Scripts\python.exe -m part3.demo --image data/VisA_pytorch/1cls/candle/test/bad/001.JPG --category candle --shots 4 --seed 0 --mode multilayer --map-restoration official --output results/submission_demo/bad_detected_candle
.\.venv-part2\Scripts\python.exe -m part3.demo --image data/VisA_pytorch/1cls/candle/test/bad/000.JPG --category candle --shots 4 --seed 0 --mode multilayer --map-restoration official --output results/submission_demo/missed_candle
```

The normal-only threshold comes from 20 disjoint normal training images. Neither the ground-truth mask nor the test label is used for inference; ground truth is only used to describe whether a saved example was correct.
