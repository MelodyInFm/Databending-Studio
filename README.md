# Databending Studio

Desktop application for glitch art with real-time preview. Stack effects, paint masks, save presets.

## Features

- **16 glitch effects** — noise, bit shift, channel swap, pixel sorting, RGB shift, block shift, wave, halftone, edge detect, and more
- **Non-destructive stack** — reorder, add/remove, tweak parameters live
- **Masking** — draw rectangle/polygon masks per effect; toggle inside/outside
- **Presets** — save/load effect chains as JSON
- **Real-time preview** — changes apply instantly
- **Batch orientation** — auto-detects PNG files in current folder

## Requirements

- Python 3.8+
- Packages: `Pillow`, `numpy`

## Quick Start

    # Install dependencies
    pip install Pillow numpy

    # Run
    python databending_studio.py

Place PNG files in the same folder or open them via **📂 OPEN**.

## Usage

1. **Load image** – select from dropdown or open manually
2. **Add effect** – pick from the combobox and press **+ ADD** (or use **🎲 RANDOM CHAIN**)
3. **Tune parameters** – select effect in the stack, adjust sliders/dropdowns
4. **Reorder** – ↑ ↓ buttons
5. **Mask** – **🎭 MASK** opens a drawing window (R – rectangle, P – polygon, F – finish, C – clear)
6. **Export** – **💾 SAVE GLITCH** chooses folder and writes PNG

## Effects reference

| Effect | Key parameters |
|--------|----------------|
| Noise | intensity, mix |
| Bit Shift | bits, channel, direction, mix |
| Channel Swap | order (GBR, BRG, RBG, GRB, BGR, random), mix |
| Pixel Sort Rows | fraction, reverse, metric (luma/red/green/blue), mix |
| Pixel Sort Columns | fraction, reverse, metric, mix |
| RGB Shift | shift (px), axis (horizontal/vertical), mix |
| Block Shift | block size, shift amount, mix |
| Bit Plane Slice | plane (0-7), channel, mix |
| Invert | channel, mix |
| Stripes | stripe width, shift, mix |
| Posterize | levels, mix |
| Brightness/Contrast | brightness, contrast, mix |
| Saturation | factor, mix |
| Pixelate | block size, mix |
| Wave | amplitude, frequency, axis, mix |
| Edge Detect | mix |
| Halftone | dot size, mix |
