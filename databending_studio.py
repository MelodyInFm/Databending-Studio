import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageOps
import numpy as np
import random
import os
import json
import random
from pathlib import Path

class GlitchCore:
    @staticmethod
    def _prepare_mask(mask_img, img_size):
        if mask_img is None:
            return None
        mask_img = mask_img.resize(img_size[::-1], Image.NEAREST)
        return np.array(mask_img) > 0

    @staticmethod
    def _apply_with_mask_and_mix(img, effect_result, mix, mask, mask_inside):
        if mix <= 0:
            return img.copy()
        if mix >= 1:
            result = effect_result
        else:
            result = (img * (1 - mix) + effect_result * mix).astype(np.uint8)
        mask_arr = GlitchCore._prepare_mask(mask, (img.shape[1], img.shape[0])) if mask else None
        if mask_arr is None:
            return result
        out = img.copy()
        region = mask_arr if mask_inside else ~mask_arr
        out[region] = result[region]
        return out
    
    @staticmethod
    def noise(img, intensity=0.1, mix=1.0, mask=None, mask_inside=True):
        if intensity <= 0 or mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        noise_img = np.random.randint(0, 256, img.shape, dtype=np.uint8)
        if mix >= 1:
            result = noise_img
        else:
            result = (img * (1 - mix) + noise_img * mix).astype(np.uint8)
        mask_arr = GlitchCore._prepare_mask(mask, (img.shape[1], img.shape[0])) if mask else None
        if mask_arr is None:
            return result
        out = img.copy()
        region = mask_arr if mask_inside else ~mask_arr
        out[region] = result[region]
        return out

    @staticmethod
    def bit_shift(img, bits=1, channel='all', direction='left', mix=1.0, mask=None, mask_inside=True):
        bits = max(0, min(7, bits))
        if bits == 0 or mix <= 0:
            return img.copy()
        arr = img.astype(np.uint16)
        if direction == 'left':
            shifted = (arr << bits) & 0xFF
        else:
            shifted = (arr >> bits) & 0xFF
        shifted = shifted.astype(np.uint8)
        effect = img.copy()
        if channel == 'all':
            effect = shifted
        elif channel == 'red':
            effect[:, :, 0] = shifted[:, :, 0]
        elif channel == 'green':
            effect[:, :, 1] = shifted[:, :, 1]
        elif channel == 'blue':
            effect[:, :, 2] = shifted[:, :, 2]
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def channel_swap(img, order=(1,2,0), mix=1.0, mask=None, mask_inside=True):
        effect = img[:, :, order]
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def pixel_sort_rows(img, fraction=0.3, reverse=False, metric='luma', mix=1.0, mask=None, mask_inside=True):
        if fraction <= 0 or mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        if metric == 'luma':
            value = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        elif metric == 'red':
            value = img[:,:,0]
        elif metric == 'green':
            value = img[:,:,1]
        else:
            value = img[:,:,2]
        result = img.copy()
        rows = np.random.choice(h, size=int(h*fraction), replace=False)
        for r in rows:
            order = np.argsort(value[r, :])
            if reverse:
                order = order[::-1]
            result[r, :, :] = img[r, order, :]
        return GlitchCore._apply_with_mask_and_mix(img, result, mix, mask, mask_inside)

    @staticmethod
    def pixel_sort_cols(img, fraction=0.3, reverse=False, metric='luma', mix=1.0, mask=None, mask_inside=True):
        if fraction <= 0 or mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        if metric == 'luma':
            value = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        elif metric == 'red':
            value = img[:,:,0]
        elif metric == 'green':
            value = img[:,:,1]
        else:
            value = img[:,:,2]
        result = img.copy()
        cols = np.random.choice(w, size=int(w*fraction), replace=False)
        for c in cols:
            order = np.argsort(value[:, c])
            if reverse:
                order = order[::-1]
            result[:, c, :] = img[order, c, :]
        return GlitchCore._apply_with_mask_and_mix(img, result, mix, mask, mask_inside)

    @staticmethod
    def rgb_shift(img, shift=5, axis='horizontal', mix=1.0, mask=None, mask_inside=True):
        if shift == 0 or mix <= 0:
            return img.copy()
        effect = img.copy()
        if axis == 'horizontal':
            effect[:, :, 0] = np.roll(effect[:, :, 0], shift, axis=1)
            effect[:, :, 2] = np.roll(effect[:, :, 2], -shift, axis=1)
        else:
            effect[:, :, 0] = np.roll(effect[:, :, 0], shift, axis=0)
            effect[:, :, 2] = np.roll(effect[:, :, 2], -shift, axis=0)
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def block_shift(img, block_w=20, block_h=20, shift_x=5, shift_y=3, mix=1.0, mask=None, mask_inside=True):
        if mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        effect = img.copy()
        num_blocks = int((h // block_h) * (w // block_w) * 0.3)
        for _ in range(num_blocks):
            x = random.randint(0, w - block_w)
            y = random.randint(0, h - block_h)
            dx = random.randint(-shift_x, shift_x)
            dy = random.randint(-shift_y, shift_y)
            block = img[y:y+block_h, x:x+block_w, :].copy()
            nx = max(0, min(w - block_w, x + dx))
            ny = max(0, min(h - block_h, y + dy))
            effect[y:y+block_h, x:x+block_w, :] = 0
            effect[ny:ny+block_h, nx:nx+block_w, :] = block
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def bit_plane_slice(img, plane=0, channel='all', mix=1.0, mask=None, mask_inside=True):
        if mix <= 0:
            return img.copy()
        bit = plane % 8
        mask_val = 1 << bit
        effect = np.zeros_like(img)
        if channel == 'all':
            effect = (img & mask_val) * (255 // mask_val)
        elif channel == 'red':
            effect[:,:,0] = (img[:,:,0] & mask_val) * (255 // mask_val)
            effect[:,:,1:] = img[:,:,1:]
        elif channel == 'green':
            effect[:,:,1] = (img[:,:,1] & mask_val) * (255 // mask_val)
            effect[:,:,0] = img[:,:,0]
            effect[:,:,2] = img[:,:,2]
        else:
            effect[:,:,2] = (img[:,:,2] & mask_val) * (255 // mask_val)
            effect[:,:,:2] = img[:,:,:2]
        effect = effect.astype(np.uint8)
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def invert(img, channel='all', mix=1.0, mask=None, mask_inside=True):
        if mix <= 0:
            return img.copy()
        effect = img.copy()
        if channel == 'all':
            effect = 255 - effect
        elif channel == 'red':
            effect[:,:,0] = 255 - effect[:,:,0]
        elif channel == 'green':
            effect[:,:,1] = 255 - effect[:,:,1]
        else:
            effect[:,:,2] = 255 - effect[:,:,2]
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def stripes(img, stripe_width=4, shift=2, mix=1.0, mask=None, mask_inside=True):
        if mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        effect = img.copy()
        for x in range(0, w, stripe_width):
            dx = random.randint(-shift, shift)
            if dx != 0:
                seg = effect[:, x:x+stripe_width, :].copy()
                effect[:, x:x+stripe_width, :] = np.roll(seg, dx, axis=1)
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def posterize(img, levels=4, mix=1.0, mask=None, mask_inside=True):
        if levels < 2 or mix <= 0:
            return img.copy()
        factor = 255 / (levels - 1)
        effect = (np.round(img / factor) * factor).astype(np.uint8)
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def brightness_contrast(img, brightness=0, contrast=1.0, mix=1.0, mask=None, mask_inside=True):
        if mix <= 0:
            return img.copy()
        arr = img.astype(np.float32)
        effect = (arr * contrast + brightness).clip(0, 255).astype(np.uint8)
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def saturation(img, factor=1.0, mix=1.0, mask=None, mask_inside=True):
        if mix <= 0:
            return img.copy()
        pil = Image.fromarray(img)
        hsv = pil.convert('HSV')
        h, s, v = hsv.split()
        s_arr = np.array(s, dtype=np.float32)
        s_arr = np.clip(s_arr * factor, 0, 255).astype(np.uint8)
        s_new = Image.fromarray(s_arr)
        effect = np.array(Image.merge('HSV', (h, s_new, v)).convert('RGB'))
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def pixelate(img, block_size=8, mix=1.0, mask=None, mask_inside=True):
        """Пикселизация (уменьшение разрешения)"""
        if block_size < 2 or mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        temp = img.copy()
        small = Image.fromarray(temp).resize((w//block_size, h//block_size), Image.NEAREST)
        effect = np.array(small.resize((w, h), Image.NEAREST))
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def wave(img, amplitude=10, frequency=0.1, axis='horizontal', mix=1.0, mask=None, mask_inside=True):
        """Синусоидальная деформация пикселей"""
        if amplitude == 0 or mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        effect = np.zeros_like(img)
        if axis == 'horizontal':
            for y in range(h):
                offset = int(amplitude * np.sin(2 * np.pi * frequency * y))
                effect[y, :, :] = np.roll(img[y, :, :], offset, axis=0)
        else:
            for x in range(w):
                offset = int(amplitude * np.sin(2 * np.pi * frequency * x))
                effect[:, x, :] = np.roll(img[:, x, :], offset, axis=0)
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def edge_detect(img, mix=1.0, mask=None, mask_inside=True):
        """Детектор границ (преобразование в чёрно-белые линии)"""
        if mix <= 0:
            return img.copy()
        gray = Image.fromarray(img).convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        effect = np.array(Image.merge('RGB', (edges, edges, edges)))
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

    @staticmethod
    def halftone(img, dot_size=4, mix=1.0, mask=None, mask_inside=True):
        """Простейший халфтон (имитация растровой точки)"""
        if dot_size < 2 or mix <= 0:
            return img.copy()
        h, w, _ = img.shape
        effect = img.copy()
        for y in range(0, h, dot_size):
            for x in range(0, w, dot_size):
                patch = img[y:min(y+dot_size, h), x:min(x+dot_size, w)]
                if patch.size == 0:
                    continue
                avg = patch.mean(axis=(0,1)).astype(np.uint8)
                effect[y:min(y+dot_size, h), x:min(x+dot_size, w)] = avg
        return GlitchCore._apply_with_mask_and_mix(img, effect, mix, mask, mask_inside)

EFFECTS_DB = [
    {'name': 'Noise', 'func': GlitchCore.noise,
     'params': [{'name': 'intensity', 'min': 0, 'max': 1, 'default': 0.2, 'type': float},
                {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]},
    {'name': 'Bit Shift', 'func': GlitchCore.bit_shift,
     'params': [
         {'name': 'bits', 'min': 1, 'max': 7, 'default': 2, 'type': int},
         {'name': 'channel', 'choices': ['all','red','green','blue'], 'default': 'all', 'type': str},
         {'name': 'direction', 'choices': ['left','right'], 'default': 'left', 'type': str},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Channel Swap', 'func': GlitchCore.channel_swap,
     'params': [{'name': 'order', 'choices': ['GBR','BRG','RBG','GRB','BGR','random'], 'default': 'random', 'type': str},
                {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]},
    {'name': 'Pixel Sort Rows', 'func': GlitchCore.pixel_sort_rows,
     'params': [
         {'name': 'fraction', 'min': 0, 'max': 1, 'default': 0.3, 'type': float},
         {'name': 'reverse', 'choices': [False, True], 'default': False, 'type': bool},
         {'name': 'metric', 'choices': ['luma','red','green','blue'], 'default': 'luma', 'type': str},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Pixel Sort Columns', 'func': GlitchCore.pixel_sort_cols,
     'params': [
         {'name': 'fraction', 'min': 0, 'max': 1, 'default': 0.3, 'type': float},
         {'name': 'reverse', 'choices': [False, True], 'default': False, 'type': bool},
         {'name': 'metric', 'choices': ['luma','red','green','blue'], 'default': 'luma', 'type': str},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'RGB Shift', 'func': GlitchCore.rgb_shift,
     'params': [
         {'name': 'shift', 'min': 0, 'max': 50, 'default': 5, 'type': int},
         {'name': 'axis', 'choices': ['horizontal','vertical'], 'default': 'horizontal', 'type': str},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Block Shift', 'func': GlitchCore.block_shift,
     'params': [
         {'name': 'block_w', 'min': 5, 'max': 60, 'default': 20, 'type': int},
         {'name': 'block_h', 'min': 5, 'max': 60, 'default': 20, 'type': int},
         {'name': 'shift_x', 'min': 0, 'max': 30, 'default': 5, 'type': int},
         {'name': 'shift_y', 'min': 0, 'max': 30, 'default': 3, 'type': int},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Bit Plane Slice', 'func': GlitchCore.bit_plane_slice,
     'params': [
         {'name': 'plane', 'min': 0, 'max': 7, 'default': 0, 'type': int},
         {'name': 'channel', 'choices': ['all','red','green','blue'], 'default': 'all', 'type': str},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Invert', 'func': GlitchCore.invert,
     'params': [{'name': 'channel', 'choices': ['all','red','green','blue'], 'default': 'all', 'type': str},
                {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]},
    {'name': 'Stripes', 'func': GlitchCore.stripes,
     'params': [
         {'name': 'stripe_width', 'min': 2, 'max': 30, 'default': 4, 'type': int},
         {'name': 'shift', 'min': 0, 'max': 15, 'default': 2, 'type': int},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Posterize', 'func': GlitchCore.posterize,
     'params': [{'name': 'levels', 'min': 2, 'max': 256, 'default': 8, 'type': int},
                {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]},
    {'name': 'Brightness/Contrast', 'func': GlitchCore.brightness_contrast,
     'params': [
         {'name': 'brightness', 'min': -100, 'max': 100, 'default': 0, 'type': int},
         {'name': 'contrast', 'min': 0.0, 'max': 3.0, 'default': 1.0, 'type': float},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Saturation', 'func': GlitchCore.saturation,
     'params': [{'name': 'factor', 'min': 0.0, 'max': 3.0, 'default': 1.0, 'type': float},
                {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]},
    {'name': 'Pixelate', 'func': GlitchCore.pixelate,
     'params': [{'name': 'block_size', 'min': 2, 'max': 32, 'default': 8, 'type': int},
                {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]},
    {'name': 'Wave', 'func': GlitchCore.wave,
     'params': [
         {'name': 'amplitude', 'min': 0, 'max': 50, 'default': 10, 'type': int},
         {'name': 'frequency', 'min': 0.01, 'max': 0.5, 'default': 0.1, 'type': float},
         {'name': 'axis', 'choices': ['horizontal','vertical'], 'default': 'horizontal', 'type': str},
         {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}
     ]},
    {'name': 'Edge Detect', 'func': GlitchCore.edge_detect,
     'params': [{'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]},
    {'name': 'Halftone', 'func': GlitchCore.halftone,
     'params': [{'name': 'dot_size', 'min': 2, 'max': 16, 'default': 4, 'type': int},
                {'name': 'mix', 'min': 0, 'max': 1, 'default': 1.0, 'type': float}]}
]

EFFECTS_BY_NAME = {e['name']: e for e in EFFECTS_DB}

class EffectItem:
    def __init__(self, name, params=None):
        self.name = name
        self.params = {}
        self.mask = None
        self.mask_inside = True
        info = EFFECTS_BY_NAME[name]
        for p in info['params']:
            self.params[p['name']] = p['default']
        if params:
            self.params.update(params)

    def apply(self, img):
        info = EFFECTS_BY_NAME[self.name]
        kwargs = {}
        for p in info['params']:
            val = self.params[p['name']]
            if p['type'] == bool:
                val = bool(val)
            elif p['type'] == int:
                val = int(val)
            elif p['type'] == float:
                val = float(val)
            if self.name == "Channel Swap" and p['name'] == 'order':
                order_map = {
                    'GBR': (1,2,0),
                    'BRG': (2,0,1),
                    'RBG': (0,2,1),
                    'GRB': (1,0,2),
                    'BGR': (2,1,0),
                    'random': tuple(random.sample([0,1,2], 3))
                }
                val = order_map.get(val, (1,2,0))
            kwargs[p['name']] = val
        kwargs['mask'] = self.mask
        kwargs['mask_inside'] = self.mask_inside
        return info['func'](img, **kwargs)

class MaskDialogs:
    @staticmethod
    def draw_mask(parent, img, callback):
        win = tk.Toplevel(parent)
        win.title("Draw Mask - Rectangle (R) / Polygon (P) - Finish (F) / Clear (C)")
        win.geometry("900x700")
        win.configure(bg='#0a0a0a')
        
        disp = img.copy()
        disp.thumbnail((800, 600), Image.LANCZOS)
        photo = ImageTk.PhotoImage(disp)
        canvas = tk.Canvas(win, width=disp.width, height=disp.height, bg='#1a1a1a', highlightthickness=0)
        canvas.pack(pady=10)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        
        mode = "rect"
        shapes = []
        current_poly = []
        rect_start = None
        rect_id = None
        poly_id = None
        
        def update_preview():
            overlay = Image.new('RGBA', img.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            for sh in shapes:
                if isinstance(sh, tuple) and len(sh)==4:
                    draw.rectangle(sh, fill=(255,0,0,100))
                elif isinstance(sh, list) and len(sh)>=3:
                    draw.polygon(sh, fill=(255,0,0,100))
            overlay_small = overlay.resize((disp.width, disp.height), Image.NEAREST)
            overlay_photo = ImageTk.PhotoImage(overlay_small)
            canvas.delete("mask_overlay")
            canvas.create_image(0, 0, anchor=tk.NW, image=overlay_photo, tags="mask_overlay")
            canvas.overlay_img = overlay_photo
        
        def add_shape(shape):
            shapes.append(shape)
            update_preview()
        
        def finish_mask():
            final_mask = Image.new('L', img.size, 0)
            draw = ImageDraw.Draw(final_mask)
            for sh in shapes:
                if isinstance(sh, tuple) and len(sh)==4:
                    draw.rectangle(sh, fill=255)
                elif isinstance(sh, list) and len(sh)>=3:
                    draw.polygon(sh, fill=255)
            win.destroy()
            callback(final_mask)
        
        def clear_all():
            nonlocal shapes, current_poly, rect_start, rect_id, poly_id
            shapes.clear()
            current_poly = []
            rect_start = None
            if rect_id:
                canvas.delete(rect_id)
                rect_id = None
            if poly_id:
                canvas.delete(poly_id)
                poly_id = None
            update_preview()
        
        def on_click(e):
            nonlocal rect_start, rect_id, current_poly, poly_id, mode
            x, y = e.x, e.y
            if mode == "rect":
                rect_start = (x, y)
                if rect_id:
                    canvas.delete(rect_id)
                rect_id = canvas.create_rectangle(x, y, x, y, outline='red', width=2)
                def on_drag(ev):
                    if rect_id:
                        canvas.coords(rect_id, rect_start[0], rect_start[1], ev.x, ev.y)
                def on_release(ev):
                    nonlocal rect_start, rect_id
                    if rect_start and rect_id:
                        x1, y1 = rect_start
                        x2, y2 = ev.x, ev.y
                        rect_start = None
                        sx = img.width / disp.width
                        sy = img.height / disp.height
                        x1i, x2i = sorted([int(x1*sx), int(x2*sx)])
                        y1i, y2i = sorted([int(y1*sy), int(y2*sy)])
                        add_shape((x1i, y1i, x2i, y2i))
                        if rect_id:
                            canvas.delete(rect_id)
                            rect_id = None
                    canvas.unbind('<B1-Motion>')
                    canvas.unbind('<ButtonRelease-1>')
                canvas.bind('<B1-Motion>', on_drag)
                canvas.bind('<ButtonRelease-1>', on_release)
            else:  # poly
                current_poly.append((x, y))
                if poly_id:
                    canvas.delete(poly_id)
                if len(current_poly) >= 2:
                    poly_id = canvas.create_polygon(current_poly, outline='red', fill='', width=2)
                else:
                    poly_id = canvas.create_oval(x-3, y-3, x+3, y+3, fill='red')
                def finish_poly(ev):
                    nonlocal current_poly, poly_id
                    if len(current_poly) >= 3:
                        sx = img.width / disp.width
                        sy = img.height / disp.height
                        scaled = [(int(xx*sx), int(yy*sy)) for xx,yy in current_poly]
                        add_shape(scaled)
                    current_poly = []
                    if poly_id:
                        canvas.delete(poly_id)
                        poly_id = None
                    canvas.unbind('<Double-1>')
                canvas.bind('<Double-1>', finish_poly)
        
        canvas.bind("<Button-1>", on_click)
        
        def set_mode_rect():
            nonlocal mode
            mode = "rect"
            win.title("Mask: Rectangle mode (R)")
        def set_mode_poly():
            nonlocal mode
            mode = "poly"
            win.title("Mask: Polygon mode (P) - click points, double-click to finish")
        
        def key_handler(event):
            if event.char == 'r':
                set_mode_rect()
            elif event.char == 'p':
                set_mode_poly()
            elif event.char == 'f':
                finish_mask()
            elif event.char == 'c':
                clear_all()
        win.bind("<Key>", key_handler)
        
        btn_frame = tk.Frame(win, bg='#0a0a0a')
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Rectangle (R)", command=set_mode_rect, bg='#e63946', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Polygon (P)", command=set_mode_poly, bg='#e63946', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Finish Mask (F)", command=finish_mask, bg='#28a745', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear All (C)", command=clear_all, bg='#dc3545', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=win.destroy, bg='#555', fg='white').pack(side=tk.LEFT, padx=5)
        
        instr = tk.Label(win, text="Rect: click-drag-release | Polygon: click points, double-click to finish | Press F to finish mask",
                         bg='#0a0a0a', fg='#aaa')
        instr.pack()

class DatabendingStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Databending Studio")
        self.root.geometry("1300x750")
        self.root.minsize(1000, 600)
        self._setup_style()
        self.current_path = None
        self.orig_img = None
        self.orig_arr = None
        self.result_arr = None
        self.effects = []
        self.selected_idx = None
        self.png_files = sorted(Path.cwd().glob("*.png"))
        self.save_dir = Path.cwd()
        self._build_gui()
        self._refresh_effects_list()
        if self.png_files:
            self._load_img(self.png_files[0])
        else:
            messagebox.showwarning("No PNG", "No PNG in script folder. Use Open.")

    def _setup_style(self):
        self.root.configure(bg='#050505')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#0a0a0a', foreground='#e0e0e0', fieldbackground='#1a1a1a')
        style.configure('TLabel', background='#0a0a0a', foreground='#e0e0e0')
        style.configure('TLabelframe', background='#0a0a0a', foreground='#e63946', bordercolor='#e63946')
        style.configure('TLabelframe.Label', background='#0a0a0a', foreground='#e63946')
        style.configure('TButton', background='#1a1a1a', foreground='#e63946', borderwidth=1, font=('Consolas',9,'bold'))
        style.map('TButton', background=[('active', '#ff6b6b'), ('pressed', '#e63946')],
                  foreground=[('active', 'black'), ('pressed', 'white')])
        style.configure('Accent.TButton', background='#e63946', foreground='white')
        style.map('Accent.TButton', background=[('active', '#ff6b6b'), ('pressed', '#b32d3a')])
        style.configure('TCombobox', fieldbackground='#1a1a1a', background='#1a1a1a')
        style.configure('TScale', background='#0a0a0a', troughcolor='#1a1a1a', slidercolor='#e63946')
        style.configure('TEntry', fieldbackground='#f0f0f0', foreground='black')
        style.configure('Vertical.TScrollbar', background='#1a1a1a', troughcolor='#0a0a0a', arrowcolor='#e63946')

    def _build_gui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_container = ttk.Frame(main_paned)
        main_paned.add(left_container, weight=1)
        left_canvas = tk.Canvas(left_container, bg='#0a0a0a', highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_scrollable = ttk.Frame(left_canvas)
        left_scrollable.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0,0), window=left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # File
        frm = ttk.LabelFrame(left_scrollable, text="[ DATA SOURCE ]", padding=5)
        frm.pack(fill=tk.X, pady=5)
        ttk.Label(frm, text="PNG:").grid(row=0, column=0, sticky=tk.W)
        self.file_cb = ttk.Combobox(frm, values=[p.name for p in self.png_files], state="readonly", width=35)
        self.file_cb.grid(row=0, column=1, padx=5)
        self.file_cb.bind("<<ComboboxSelected>>", lambda e: self._load_img(next(p for p in self.png_files if p.name == self.file_cb.get())))
        btnf = ttk.Frame(frm)
        btnf.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(btnf, text="⟳ REFRESH", command=self._refresh_file_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(btnf, text="📂 OPEN", command=self._open_manual).pack(side=tk.LEFT, padx=2)

        # Add effect
        add_frm = ttk.LabelFrame(left_scrollable, text="[ ADD EFFECT ]", padding=5)
        add_frm.pack(fill=tk.X, pady=5)
        self.effect_cb = ttk.Combobox(add_frm, values=list(EFFECTS_BY_NAME.keys()), state="readonly", width=38)
        self.effect_cb.pack(side=tk.LEFT, padx=5)
        ttk.Button(add_frm, text="+ ADD", command=self._add_effect, style='Accent.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(add_frm, text="🎲 RANDOM CHAIN", command=self._random_stack, style='Accent.TButton').pack(side=tk.LEFT, padx=2)

        # Effects list
        lst_frm = ttk.LabelFrame(left_scrollable, text="[ EFFECT STACK ]", padding=5)
        lst_frm.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox = tk.Listbox(lst_frm, bg='#1a1a1a', fg='#e0e0e0', selectbackground='#e63946',
                                  selectforeground='white', font=('Consolas',9), height=12)
        scrol = ttk.Scrollbar(lst_frm, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrol.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrol.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        # Effect controls
        ctrl = ttk.Frame(left_scrollable)
        ctrl.pack(fill=tk.X, pady=5)
        ttk.Button(ctrl, text="↑ UP", command=self._move_up, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="↓ DOWN", command=self._move_down, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="✖ REMOVE", command=self._remove_effect, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="🗑 CLEAR", command=self._clear_stack, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="🎭 MASK", command=self._edit_mask, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="🔄 IN/OUT", command=self._toggle_mask_inside, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="❌ CLR MASK", command=self._clear_mask, width=8).pack(side=tk.LEFT, padx=2)

        # Preset buttons
        preset_frame = ttk.Frame(left_scrollable)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Button(preset_frame, text="💾 SAVE PRESET", command=self._save_preset, style='Accent.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="📂 LOAD PRESET", command=self._load_preset, style='Accent.TButton').pack(side=tk.LEFT, padx=2)

        # Parameters panel
        self.params_frame = ttk.LabelFrame(left_scrollable, text="[ NO EFFECT SELECTED ]", padding=5)
        self.params_frame.pack(fill=tk.X, pady=5)
        self.param_widgets = {}

        # Right preview
        right = ttk.Frame(main_paned)
        main_paned.add(right, weight=2)
        paned = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        orig_f = ttk.LabelFrame(paned, text="[ ORIGINAL ]", padding=5)
        paned.add(orig_f, weight=1)
        self.orig_lbl = ttk.Label(orig_f, background='#050505')
        self.orig_lbl.pack(fill=tk.BOTH, expand=True)
        glitch_f = ttk.LabelFrame(paned, text="[ GLITCHED ]", padding=5)
        paned.add(glitch_f, weight=1)
        self.glitch_lbl = ttk.Label(glitch_f, background='#050505')
        self.glitch_lbl.pack(fill=tk.BOTH, expand=True)

        # Bottom
        bot = ttk.Frame(self.root)
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(bot, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(bot, text="💾 SAVE GLITCH (choose folder)", command=self._save,
                  bg='#e63946', fg='white', font=('Consolas',10,'bold'), padx=10, pady=4).pack(side=tk.RIGHT, padx=10)
        
    def _refresh_file_list(self):
        self.png_files = sorted(Path.cwd().glob("*.png"))
        self.file_cb['values'] = [p.name for p in self.png_files]
        if self.png_files and (not self.current_path or self.current_path not in self.png_files):
            self._load_img(self.png_files[0])

    def _open_manual(self):
        p = filedialog.askopenfilename(filetypes=[("PNG","*.png")])
        if p:
            self._load_img(Path(p))
            if Path(p) not in self.png_files:
                self.png_files.append(Path(p))
                self.file_cb['values'] = [p.name for p in self.png_files]
            self.file_cb.set(Path(p).name)

    def _load_img(self, path):
        self.current_path = path
        self.orig_img = Image.open(path).convert("RGB")
        self.orig_arr = np.array(self.orig_img)
        self._apply_all()
        self._update_status(f"Loaded: {path.name} ({self.orig_arr.shape[1]}x{self.orig_arr.shape[0]})")
        self.file_cb.set(path.name)

    def _apply_all(self):
        if self.orig_arr is None:
            return
        arr = self.orig_arr.copy()
        for eff in self.effects:
            try:
                arr = eff.apply(arr)
            except Exception as e:
                self._update_status(f"Error in {eff.name}: {e}")
                return
        self.result_arr = arr
        self._update_previews()

    def _update_previews(self):
        if self.orig_img:
            o = self.orig_img.copy()
            o.thumbnail((500,500), Image.LANCZOS)
            self.orig_photo = ImageTk.PhotoImage(o)
            self.orig_lbl.config(image=self.orig_photo)
            self.orig_lbl.image = self.orig_photo
        if self.result_arr is not None:
            g = Image.fromarray(self.result_arr)
            g.thumbnail((500,500), Image.LANCZOS)
            self.glitch_photo = ImageTk.PhotoImage(g)
            self.glitch_lbl.config(image=self.glitch_photo)
            self.glitch_lbl.image = self.glitch_photo
        else:
            self.glitch_lbl.config(image='')

    def _update_status(self, msg):
        self.status_var.set(msg)

    def _refresh_effects_list(self):
        self.listbox.delete(0, tk.END)
        for i, e in enumerate(self.effects):
            short = ", ".join(f"{k}={v}" for k,v in list(e.params.items())[:2])
            if len(e.params)>2:
                short+="..."
            mask_icon = "🎭 " if e.mask is not None else "   "
            self.listbox.insert(tk.END, f"{mask_icon}{i+1:2d}. {e.name} [{short}]")
        if self.selected_idx is not None and self.selected_idx < len(self.effects):
            self.listbox.selection_set(self.selected_idx)
        else:
            self.selected_idx = None
            self._clear_params()

    def _clear_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.params_frame.config(text="[ NO EFFECT SELECTED ]")

    def _on_select(self, ev):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.selected_idx = sel[0]
        eff = self.effects[self.selected_idx]
        self._show_params(eff)

    def _show_params(self, effect):
        for w in self.params_frame.winfo_children():
            w.destroy()
        info = EFFECTS_BY_NAME[effect.name]
        self.params_frame.config(text=f" PARAMETERS: {effect.name} ")
        row = 0
        self.param_vars = {}
        for p in info['params']:
            label = ttk.Label(self.params_frame, text=p['name']+":", foreground='#e63946')
            label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
            val = effect.params[p['name']]
            
            if p['type'] in (int, float):
                var = tk.DoubleVar(value=val) if p['type']==float else tk.IntVar(value=val)
                scale = ttk.Scale(self.params_frame, from_=p['min'], to=p['max'], variable=var, orient=tk.HORIZONTAL, length=180)
                scale.grid(row=row, column=1, padx=5, sticky=tk.W)
                entry = ttk.Entry(self.params_frame, textvariable=var, width=8)
                entry.grid(row=row, column=2, padx=5)
                def cb(v=var, param=p['name'], e=effect):
                    e.params[param] = v.get() if p['type']==float else int(round(v.get()))
                    self._apply_all()
                    self._refresh_effects_list()
                var.trace_add('write', lambda *a, cb=cb: cb())
                self.param_vars[p['name']] = var
                
            elif p['type'] == bool:
                var = tk.BooleanVar(value=val)
                cb = ttk.Checkbutton(self.params_frame, variable=var)
                cb.grid(row=row, column=1, sticky=tk.W)
                def cb2(v=var, param=p['name'], e=effect):
                    e.params[param] = v.get()
                    self._apply_all()
                    self._refresh_effects_list()
                var.trace_add('write', lambda *a, cb=cb2: cb2())
                self.param_vars[p['name']] = var
                
            elif p['type'] == str:
                var = tk.StringVar(value=val)
                if 'choices' in p:
                    combo = ttk.Combobox(self.params_frame, textvariable=var, values=p['choices'], state="readonly", width=15)
                    combo.grid(row=row, column=1, sticky=tk.W, padx=5)
                else:
                    entry = ttk.Entry(self.params_frame, textvariable=var, width=20)
                    entry.grid(row=row, column=1, sticky=tk.W, padx=5)
                def cb3(v=var, param=p['name'], e=effect):
                    e.params[param] = v.get()
                    self._apply_all()
                    self._refresh_effects_list()
                var.trace_add('write', lambda *a, cb=cb3: cb3())
                self.param_vars[p['name']] = var
            row += 1
        self.params_frame.columnconfigure(1, weight=1)

    def _add_effect(self):
        name = self.effect_cb.get()
        if not name:
            return
        self.effects.append(EffectItem(name))
        self.selected_idx = len(self.effects)-1
        self._refresh_effects_list()
        self.listbox.selection_set(self.selected_idx)
        self._show_params(self.effects[self.selected_idx])
        self._apply_all()

    def _remove_effect(self):
        if self.selected_idx is not None and self.selected_idx < len(self.effects):
            del self.effects[self.selected_idx]
            self.selected_idx = None
            self._refresh_effects_list()
            self._clear_params()
            self._apply_all()

    def _move_up(self):
        if self.selected_idx is not None and self.selected_idx > 0:
            i = self.selected_idx
            self.effects[i], self.effects[i-1] = self.effects[i-1], self.effects[i]
            self.selected_idx = i-1
            self._refresh_effects_list()
            self.listbox.selection_set(self.selected_idx)
            self._apply_all()

    def _move_down(self):
        if self.selected_idx is not None and self.selected_idx < len(self.effects)-1:
            i = self.selected_idx
            self.effects[i], self.effects[i+1] = self.effects[i+1], self.effects[i]
            self.selected_idx = i+1
            self._refresh_effects_list()
            self.listbox.selection_set(self.selected_idx)
            self._apply_all()

    def _clear_stack(self):
        self.effects.clear()
        self.selected_idx = None
        self._refresh_effects_list()
        self._clear_params()
        self._apply_all()

    def _random_stack(self):
        self.effects.clear()
        num = random.randint(2, 10)
        names = list(EFFECTS_BY_NAME.keys())
        for _ in range(num):
            name = random.choice(names)
            eff = EffectItem(name)
            info = EFFECTS_BY_NAME[name]
            for p in info['params']:
                if p['type'] == int:
                    eff.params[p['name']] = random.randint(p['min'], p['max'])
                elif p['type'] == float:
                    eff.params[p['name']] = random.uniform(p['min'], p['max'])
                elif p['type'] == bool:
                    eff.params[p['name']] = random.choice([True, False])
                elif p['type'] == str and 'choices' in p:
                    eff.params[p['name']] = random.choice(p['choices'])
            self.effects.append(eff)
        self.selected_idx = None
        self._refresh_effects_list()
        self._clear_params()
        self._apply_all()
        self._update_status(f"🎲 Random chain with {num} effects")

    def _edit_mask(self):
        if self.selected_idx is None:
            messagebox.showinfo("No effect", "Select an effect first.")
            return
        if self.orig_img is None:
            return
        MaskDialogs.draw_mask(self.root, self.orig_img, self._set_mask)

    def _set_mask(self, mask_img):
        if self.selected_idx is not None:
            self.effects[self.selected_idx].mask = mask_img
            self._apply_all()
            self._refresh_effects_list()
            self._update_status("Mask applied")

    def _clear_mask(self):
        if self.selected_idx is not None:
            self.effects[self.selected_idx].mask = None
            self._apply_all()
            self._refresh_effects_list()
            self._update_status("Mask cleared")

    def _toggle_mask_inside(self):
        if self.selected_idx is not None:
            eff = self.effects[self.selected_idx]
            eff.mask_inside = not eff.mask_inside
            self._apply_all()
            self._update_status(f"Mask mode: {'inside' if eff.mask_inside else 'outside'}")

    def _save(self):
        if self.result_arr is None:
            messagebox.showwarning("No result", "Apply effects first.")
            return
        folder = filedialog.askdirectory(title="Select save folder", initialdir=self.save_dir)
        if not folder:
            return
        self.save_dir = Path(folder)
        base = os.path.splitext(os.path.basename(self.current_path))[0] if self.current_path else "glitched"
        out = self.save_dir / f"{base}_glitched.png"
        cnt = 1
        while out.exists():
            out = self.save_dir / f"{base}_glitched_{cnt}.png"
            cnt += 1
        Image.fromarray(self.result_arr).save(out, "PNG")
        self._update_status(f"Saved: {out}")
        messagebox.showinfo("Saved", f"Image saved to\n{out}")

    def _save_preset(self):
        if not self.effects:
            messagebox.showwarning("No effects", "Nothing to save.")
            return
        preset_data = []
        for eff in self.effects:
            preset_data.append({
                'name': eff.name,
                'params': eff.params,
                'mask_inside': eff.mask_inside
            })
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            self._update_status(f"Preset saved: {filepath}")

    def _load_preset(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            self.effects.clear()
            for item in preset_data:
                name = item['name']
                name_mapping = {
                    "Channel Swap (перестановка каналов)": "Channel Swap",
                    "RGB Shift (хроматическая аберрация)": "RGB Shift",
                    "Bit Plane Slice (битовый слой)": "Bit Plane Slice",
                    "Posterize (пастеризация)": "Posterize",
                    "Brightness/Contrast": "Brightness/Contrast",
                    "Noise (шум)": "Noise",
                    "Bit Shift (битовый сдвиг)": "Bit Shift",
                    "Pixel Sort Rows (сортировка строк)": "Pixel Sort Rows",
                    "Pixel Sort Columns (сортировка столбцов)": "Pixel Sort Columns",
                    "Block Shift (сдвиг блоков)": "Block Shift",
                    "Invert (инверсия)": "Invert",
                    "Stripes (полосы со сдвигом)": "Stripes",
                    "Saturation (насыщенность)": "Saturation"
                }
                if name in name_mapping:
                    name = name_mapping[name]
                if name not in EFFECTS_BY_NAME:
                    raise ValueError(f"Unknown effect: {name}")
                
                params = item.get('params', {})
                if 'mix' not in params:
                    params['mix'] = 1.0
                if name == "Channel Swap" and 'order' in params:
                    old_order = params['order']
                    if '->' in old_order:
                        params['order'] = old_order.split('->')[1]
                
                eff = EffectItem(name, params)
                eff.mask_inside = item.get('mask_inside', True)
                self.effects.append(eff)
            self.selected_idx = None
            self._refresh_effects_list()
            self._clear_params()
            self._apply_all()
            self._update_status(f"Preset loaded: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset:\n{str(e)}")
            self._update_status(f"Load error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DatabendingStudio(root)
    root.mainloop()