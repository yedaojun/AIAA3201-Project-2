"""Package real, already-run demo predictions as the required MP4 and ZIP.

This script performs no model inference and reads only saved single-image demo
outputs. Run the three commands in submission_demo_README.md first if needed.
"""

import csv
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'results/submission_demo'
OUT = ROOT / 'output/submission'
CASES = (
    ('good_candle', 'Normal product', 'OK', 'A normal candle is accepted.'),
    ('bad_detected_candle', 'Visible defect', 'NG', 'A defective candle is flagged.'),
    ('missed_candle', 'Remaining failure', 'OK', 'This defective candle is missed at the normal-only threshold.'),
)
WIDTH, HEIGHT, FPS = 1280, 720, 20


def font(size, bold=False):
    path = Path('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path), size)


def canvas(title, subtitle, panel=None, footer=''):
    image = Image.new('RGB', (WIDTH, HEIGHT), '#f5f7fa')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 12), fill='#197a72')
    draw.text((66, 44), title, font=font(41, True), fill='#17324a')
    draw.text((68, 108), subtitle, font=font(23), fill='#385366')
    if panel is not None:
        with Image.open(panel) as source:
            source = source.convert('RGB')
            source.thumbnail((1170, 430), Image.Resampling.LANCZOS)
            x = (WIDTH - source.width) // 2
            y = 178 + (430 - source.height) // 2
            image.paste(source, (x, y))
        draw.rounded_rectangle((62, 633, 1218, 682), radius=11, fill='#e4efee')
        draw.text((80, 644), footer, font=font(22, True), fill='#174c47')
    else:
        draw.rounded_rectangle((64, 197, 1216, 556), radius=18, fill='white',
                               outline='#d7e1e6', width=2)
        for i, line in enumerate(footer.split('\n')):
            draw.text((92, 239 + i * 60), line, font=font(25, i == 0), fill='#203b50')
    return np.asarray(image)[:, :, ::-1].copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'tools/submission_demo_README.md', OUT / 'README.md')
    examples = OUT / 'examples'
    examples.mkdir(exist_ok=True)
    rows = []
    for folder, label, expected, _ in CASES:
        path = SAVED / folder
        with (path / 'prediction.json').open(encoding='utf-8') as stream:
            result = json.load(stream)
        if result['dino_decision'] != expected or result['category'] != 'candle':
            raise ValueError(f'Unexpected saved prediction: {folder}')
        if not (path / 'prediction.png').is_file():
            raise FileNotFoundError(path / 'prediction.png')
        shutil.copyfile(path / 'prediction.png', examples / f'{folder}.png')
        rows.append({'case': label, 'input_file': Path(result['image']).name,
                     'reference_count': result['shots'], 'reference_group': result['seed'],
                     'score': f'{result["dino_score"]:.6f}',
                     'threshold': f'{result["dino_threshold"]:.6f}',
                     'decision': result['dino_decision'],
                     'heatmap': f'examples/{folder}.png'})
    with (OUT / 'example_results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    video = OUT / 'anomaly_demo.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'avc1'), FPS,
                             (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError('OpenCV MP4 encoder could not start')

    def hold(frame, seconds):
        for _ in range(round(seconds * FPS)):
            writer.write(frame)

    try:
        hold(canvas('Industrial anomaly inspection', 'AIAA 3201 Project 2 | four normal references',
                    footer='Recorded output walkthrough from the runnable Part 3 CLI demo.\n'
                           'Input image  →  score and threshold  →  OK/NG  →  heatmap'), 3)
        hold(canvas('Run the demo on a product image', 'The image path is supplied by the user.',
                    footer='python -m part3.demo --image <product.JPG> --category candle\n'
                           '--shots 4 --seed 0 --mode multilayer --map-restoration official\n'
                           'The command saves prediction.json and prediction.png.'), 4)
        for folder, label, _, note in CASES:
            row = next(r for r in rows if r['case'] == label)
            relation = '>' if float(row['score']) > float(row['threshold']) else '<'
            footer = (f'Score {float(row["score"]):.4f} {relation} threshold '
                      f'{float(row["threshold"]):.4f}  |  {row["decision"]}  |  {note}')
            hold(canvas(label, 'Input image, raw anomaly map, and overlay',
                        examples / f'{folder}.png', footer), 5)
        hold(canvas('Try another image', 'The same command accepts any matching product image.',
                    footer='Use the public GitHub repository to run the CLI demo.\n'
                           'The examples also show one remaining false negative.\n'
                           'See README.md for setup and exact reproduction commands.'), 3)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((codec_value >> (8 * index)) & 255) for index in range(4))
    capture.release()
    if frames < 450 or codec.lower() not in ('h264', 'avc1'):
        raise RuntimeError(f'Incomplete or non-H.264 demo video: {frames} frames, {codec}')

    archive = ROOT / 'anomaly_demo.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=7) as output:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(OUT).as_posix())
    with ZipFile(archive) as check:
        bad = check.testzip()
        if bad:
            raise RuntimeError(f'Corrupt ZIP member: {bad}')
    print(f'Built {archive} ({frames} frames; {archive.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
