"""A reusable single-method image viewer; imports no anomaly model implementation."""
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch

from anomaly_baseline.core import CATEGORIES
from .common import DATA, configure
from .prediction import infer_category, save_prediction


class InspectionWindow:
    def __init__(self, window, *, title, method, loader, results, output, upper,
                 reconstruction=False, category_specific=True):
        self.window, self.executor = window, ThreadPoolExecutor(max_workers=1)
        self.method, self.loader, self.results, self.output = method, loader, results, output
        self.upper, self.reconstruction, self.category_specific = upper, reconstruction, category_specific
        self.predictor, self.key, self.future, self.path = None, None, None, None
        window.title(title)
        window.geometry('1260x740' if reconstruction else '1100x740')
        self.category = tk.StringVar(value=CATEGORIES[0])
        self.message = tk.StringVar(value='Choose an image to inspect.')
        ttk.Label(window, text=title, font=('Arial', 22, 'bold')).pack(anchor='w', padx=20, pady=16)
        bar = ttk.Frame(window)
        bar.pack(fill='x', padx=20)
        ttk.Label(bar, text='Category').pack(side='left', padx=(0, 8))
        box = ttk.Combobox(bar, textvariable=self.category, values=CATEGORIES, state='readonly', width=20)
        box.pack(side='left', padx=(0, 20))
        box.bind('<<ComboboxSelected>>', self.selection_changed)
        button = ttk.Button(bar, text='Choose image...', command=self.choose)
        button.pack(side='left')
        self.controls = [box, button]
        ttk.Label(window, textvariable=self.message, wraplength=1200, font=('Arial', 12)).pack(anchor='w', padx=20, pady=16)
        panels = ttk.Frame(window)
        panels.pack(fill='both', expand=True, padx=12)
        self.labels, self.photos = [], []
        titles = ['Input image'] + (['AE reconstruction'] if reconstruction else []) + ['Anomaly heatmap', 'Overlay']
        for i, label_title in enumerate(titles):
            column = ttk.Frame(panels)
            column.grid(row=0, column=i, sticky='nsew', padx=5)
            panels.columnconfigure(i, weight=1, uniform='panel')
            ttk.Label(column, text=label_title).pack(pady=12)
            label = ttk.Label(column, anchor='center')
            label.pack(fill='both', expand=True)
            self.labels.append(label)
        ttk.Label(window, text=f'Heatmap fixed display range: 0-{upper:g}. Predictions saved to {output}.', wraplength=1200).pack(anchor='w', padx=20, pady=14)
        window.after(300, self.default_image)

    def default_image(self):
        self.path = sorted((DATA / self.category.get() / 'test/good').glob('*.JPG'))[0]
        self.run()

    def selection_changed(self, event=None):
        if self.future is not None:
            return
        inferred = infer_category(self.path) if self.path is not None else None
        if self.path is None or (inferred is not None and inferred != self.category.get()):
            self.default_image()
        else:
            self.run()

    def choose(self):
        selected = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.JPG *.jpeg *.png *.bmp'), ('All files', '*.*')])
        if not selected:
            return
        try:
            category = infer_category(selected)
            if category:
                self.category.set(category)
            self.path = Path(selected)
            self.run()
        except Exception as error:
            self.message.set(str(error))

    def worker(self, category, path):
        key = category if self.category_specific else 'shared frozen model'
        if self.key != key:
            self.predictor = None
            gc.collect()
            torch.cuda.empty_cache()
            self.predictor = self.loader(category)
            self.key = key
        return save_prediction(self.method, category, path, self.predictor, self.results, self.output)

    def run(self):
        if self.future is not None:
            return
        self.message.set('Loading model / inspecting image...')
        for control in self.controls:
            control.configure(state='disabled')
        self.future = self.executor.submit(self.worker, self.category.get(), self.path)
        self.window.after(100, self.poll)

    def poll(self):
        if not self.future.done():
            self.window.after(100, self.poll)
            return
        try:
            result, dense, reconstruction = self.future.result()
            self.message.set(f'{result["decision"]} | Score {result["score"]:.5f} | Threshold {result["threshold"]:.5f} | {result["inference_ms"]:.1f} ms\n{result["image"]}')
            with Image.open(result['image']) as source:
                image = source.convert('RGB')
            heat = cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(dense / self.upper, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            pictures = [image]
            if self.reconstruction:
                pictures.append(Image.fromarray(np.rint(np.clip(reconstruction, 0, 1) * 255).astype(np.uint8)))
            pictures += [Image.fromarray(heat), Image.blend(image, Image.fromarray(heat), .45)]
            self.photos = []
            for label, picture in zip(self.labels, pictures):
                picture.thumbnail((285 if self.reconstruction else 330, 450))
                photo = ImageTk.PhotoImage(picture)
                self.photos.append(photo)
                label.configure(image=photo)
        except Exception as error:
            self.message.set(f'Unable to inspect: {error}')
        self.future = None
        for control in self.controls:
            control.configure(state='readonly' if isinstance(control, ttk.Combobox) else 'normal')


def run_window(**kwargs):
    configure()
    window = tk.Tk()
    InspectionWindow(window, **kwargs)
    window.mainloop()
