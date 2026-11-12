"""Desktop file-picker demo; uses saved normal-only calibration, never labels."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .core import CATEGORIES, TemplateDetector


def category_from_dataset_path(image_path, dataset_roots):
    """Read product category metadata only; never infer the OK/NG label."""
    path = Path(image_path).resolve()
    for root in dataset_roots:
        root = Path(root).resolve()
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] not in CATEGORIES:
            continue
        if 'ground_truth' in relative.parts or 'Masks' in relative.parts:
            raise ValueError('This file is a ground-truth mask. Please choose a product image from Images or test/good/test/bad.')
        return relative.parts[0]
    return None


def main():
    cv2.setNumThreads(1)
    project = Path(__file__).resolve().parents[1]
    results = project / 'results/part1'
    data = project / 'data/VisA_pytorch/1cls'
    window = tk.Tk()
    window.title('Part 1 — Product Anomaly Inspection')
    window.geometry('1140x600')
    window.minsize(1020, 550)
    window.configure(background='#f4f7fa')
    style = ttk.Style()
    style.configure('TFrame', background='#f4f7fa')
    style.configure('TLabel', background='#f4f7fa', font=('Segoe UI', 11))
    frame = ttk.Frame(window, padding=20)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Product anomaly inspection', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='Dataset category is detected from the folder. Changing settings refreshes the prediction.').pack(anchor='w', pady=(4, 18))
    controls = ttk.Frame(frame)
    controls.pack(fill='x')
    category, shot, seed = tk.StringVar(value='cashew'), tk.StringVar(value='4'), tk.StringVar(value='0')
    selectors = []
    for column, (label, variable, values, width) in enumerate([
            ('Category', category, CATEGORIES, 16), ('References', shot, ('1', '2', '4', '8'), 8),
            ('Reference group', seed, ('0', '1', '2'), 8)]):
        ttk.Label(controls, text=label).grid(row=0, column=column * 2, padx=(0, 8))
        selector = ttk.Combobox(controls, textvariable=variable, values=values, state='readonly', width=width)
        selector.grid(row=0, column=column * 2 + 1, padx=(0, 20))
        selectors.append(selector)
    status = ttk.Label(frame, text='Ready', font=('Segoe UI', 14, 'bold'))
    status.pack(anchor='w', pady=(20, 4))
    details = ttk.Label(frame, text='')
    details.configure(wraplength=1060)
    details.pack(anchor='w')
    previews = ttk.Frame(frame)
    previews.pack(fill='both', expand=True, pady=15)
    labels = []
    for column, title in enumerate(('Input image', 'Anomaly heatmap · fixed 0–255', 'Overlay')):
        cell = ttk.Frame(previews)
        cell.grid(row=0, column=column, sticky='nsew', padx=5)
        previews.columnconfigure(column, weight=1)
        ttk.Label(cell, text=title).pack()
        label = ttk.Label(cell)
        label.pack(pady=8)
        labels.append(label)
    retained_images = []
    current_path = None

    def show_image(path):
        nonlocal current_path
        try:
            path = Path(path).resolve()
            inferred_category = category_from_dataset_path(path, (project / 'data/VisA', data))
            if inferred_category:
                category.set(inferred_category)
            metric = json.loads((results / category.get() / f'shot_{shot.get()}' / f'seed_{seed.get()}' / 'metrics.json').read_text())
            config = json.loads((results / 'run_config.json').read_text())['config']
            detector = TemplateDetector([data / category.get() / 'train/good' / name for name in metric['references']], config)
            score, anomaly, _, _ = detector.predict(path)
            decision = 'NG' if score > metric['threshold'] else 'OK'
            comparison = '>' if decision == 'NG' else '<='
            status.configure(text=f'{decision}  |  Score {score:.4f} {comparison} threshold {metric["threshold"]:.4f}',
                             foreground='#a73232' if decision == 'NG' else '#14785d')
            details.configure(text=f'Category: {category.get()} | References: {shot.get()} | Group: {seed.get()}\n{path}')
            with Image.open(path) as source:
                image = source.convert('RGB')
            heat = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(anomaly, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
            retained_images.clear()
            for label, panel in zip(labels, (image, heat, Image.blend(image, heat, .45))):
                panel.thumbnail((330, 330))
                rendered = ImageTk.PhotoImage(panel)
                retained_images.append(rendered)
                label.configure(image=rendered)
            current_path = path
        except Exception as error:
            current_path = None
            status.configure(text='No current prediction', foreground='#334e68')
            details.configure(text='Choose a product image to inspect.')
            for label in labels:
                label.configure(image='')
            retained_images.clear()
            messagebox.showerror('Cannot inspect image', str(error))

    def choose():
        path = filedialog.askopenfilename(title='Select product image',
            initialdir=data / category.get() / 'test',
            filetypes=[('Images', '*.jpg *.JPG *.jpeg *.png *.bmp'), ('All files', '*.*')])
        if path:
            show_image(path)

    def load_example(event=None):
        # Fixed first test filename, not selected using its prediction or label score.
        paths = sorted((data / category.get() / 'test/good').glob('*'))
        if paths:
            show_image(paths[0])

    def refresh(event=None):
        if current_path is not None:
            show_image(current_path)
        else:
            load_example()

    selectors[0].bind('<<ComboboxSelected>>', load_example)
    for selector in selectors[1:]:
        selector.bind('<<ComboboxSelected>>', refresh)
    ttk.Button(controls, text='Choose image…', command=choose).grid(row=0, column=6)
    window.after(100, load_example)
    window.mainloop()


if __name__ == '__main__':
    main()
