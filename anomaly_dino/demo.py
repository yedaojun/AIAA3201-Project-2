"""Live desktop demo for Part 2, with lazy model loading and explicit score scales."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from anomaly_baseline.core import CATEGORIES
from anomaly_baseline.demo import category_from_dataset_path


def main():
    project = Path(__file__).resolve().parents[1]
    data, results = project / 'data/VisA_pytorch/1cls', project / 'results/part2'
    window = tk.Tk()
    window.title('Part 2 — Official AnomalyDINO Inspection')
    window.geometry('1160x650')
    window.minsize(1050, 570)
    frame = ttk.Frame(window, padding=20)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Part 2 · Official AnomalyDINO', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='Live inference with a frozen DINOv2 backbone. Dataset category is detected from the folder.').pack(anchor='w', pady=(6, 18))
    controls = ttk.Frame(frame)
    controls.pack(fill='x')
    category, shot, seed = tk.StringVar(value='candle'), tk.StringVar(value='4'), tk.StringVar(value='0')
    selectors = []
    for column, (name, variable, values) in enumerate([
            ('Category', category, CATEGORIES), ('References', shot, ('1', '2', '4', '8')),
            ('Reference group', seed, ('0', '1', '2'))]):
        ttk.Label(controls, text=name).grid(row=0, column=column * 2, padx=(0, 8))
        selector = ttk.Combobox(controls, textvariable=variable, values=values, state='readonly', width=14)
        selector.grid(row=0, column=column * 2 + 1, padx=(0, 16))
        selectors.append(selector)
    status = ttk.Label(frame, text='Ready', font=('Segoe UI', 14, 'bold'))
    status.pack(anchor='w', pady=(20, 6))
    details = ttk.Label(frame, text='', wraplength=1080)
    details.pack(anchor='w')
    previews = ttk.Frame(frame)
    previews.pack(fill='both', expand=True, pady=16)
    labels = []
    for column, title in enumerate(('Input image', 'Anomaly heatmap · fixed scale 0–1', 'Overlay')):
        cell = ttk.Frame(previews)
        cell.grid(row=0, column=column, padx=8, sticky='nsew')
        previews.columnconfigure(column, weight=1)
        ttk.Label(cell, text=title).pack()
        label = ttk.Label(cell)
        label.pack(pady=8)
        labels.append(label)
    state = {'model': None, 'detector': None, 'key': None, 'path': None}
    retained = []

    def show(path):
        try:
            inferred = category_from_dataset_path(path, (project / 'data/VisA', data))
            if inferred:
                category.set(inferred)
            key = (category.get(), shot.get(), seed.get())
            metric_file = results / key[0] / f'shot_{key[1]}' / f'seed_{key[2]}' / 'metrics.json'
            if not metric_file.exists():
                raise ValueError('This configuration has not finished yet. Wait for the corresponding experiment to complete.')
            metric = json.loads(metric_file.read_text())
            status.configure(text='Loading model / running live inference…')
            window.update_idletasks()
            from threadpoolctl import threadpool_limits
            import faiss
            import torch
            from .official import load_model, FeatureArchive
            from .predict import OfficialPredictor
            cv2.setNumThreads(1)
            torch.set_num_threads(4)
            faiss.omp_set_num_threads(4)
            threadpool_limits(4)
            if state['model'] is None:
                state['model'] = load_model()
                state['archive'] = FeatureArchive(state['model'])
            if key != state['key']:
                state['detector'] = OfficialPredictor(state['model'],
                    [data / key[0] / 'train/good' / name for name in metric['references']], int(key[2]))
                state['key'] = key
            score, dense, _, _ = state['detector'].predict(path)
            decision = 'NG' if score > metric['threshold'] else 'OK'
            comparison = '>' if decision == 'NG' else '<='
            status.configure(text=f'{decision} | Score {score:.4f} {comparison} threshold {metric["threshold"]:.4f}',
                             foreground='#a73232' if decision == 'NG' else '#14785d')
            details.configure(text=f'Category: {key[0]} | References: {key[1]} | Group: {key[2]}\n{Path(path).resolve()}')
            with Image.open(path) as source:
                image = source.convert('RGB')
            display = np.clip(dense * 255, 0, 255).astype(np.uint8)
            heat = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(display, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
            retained.clear()
            for label, panel in zip(labels, (image, heat, Image.blend(image, heat, .45))):
                panel.thumbnail((340, 340))
                rendered = ImageTk.PhotoImage(panel)
                retained.append(rendered)
                label.configure(image=rendered)
            state['path'] = Path(path)
        except Exception as error:
            state['path'] = None
            status.configure(text='No current prediction', foreground='#334e68')
            for label in labels:
                label.configure(image='')
            retained.clear()
            messagebox.showerror('Cannot inspect image', str(error))

    def choose():
        path = filedialog.askopenfilename(initialdir=data / category.get() / 'test',
            filetypes=[('Images', '*.JPG *.jpg *.jpeg *.png *.bmp'), ('All files', '*.*')])
        if path:
            show(Path(path))

    def example(event=None):
        paths = sorted((data / category.get() / 'test/good').glob('*.JPG'))
        if paths:
            show(paths[0])

    def refresh(event=None):
        if state['path'] is not None:
            show(state['path'])
        else:
            example()

    selectors[0].bind('<<ComboboxSelected>>', example)
    for selector in selectors[1:]:
        selector.bind('<<ComboboxSelected>>', refresh)
    ttk.Button(controls, text='Choose image…', command=choose).grid(row=0, column=6)
    window.after(100, example)
    window.mainloop()


if __name__ == '__main__':
    main()
