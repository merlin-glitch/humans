"""
map_color_tools.py - Utilities to extract colors from a PNG map and
interactively assign semantic categories (food, terrain, obstacles, ...).

Usage (CLI):
    python map_color_tools.py --map images/big_map.png --max-colors 24 \
        --save images/big_map_color_map.json

Programmatic:
    from map_color_tools import extract_colors_and_prompt
    mapping = extract_colors_and_prompt("images/big_map.png")
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from matplotlib.colors import CSS4_COLORS  # provides human color names


RGB = Tuple[int, int, int]


def _to_hex(color: RGB) -> str:
    return "#%02x%02x%02x" % color


def _hex_to_rgb(hex_str: str) -> RGB:
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_distance_sq(a: RGB, b: RGB) -> int:
    dr = a[0] - b[0]
    dg = a[1] - b[1]
    db = a[2] - b[2]
    return dr*dr + dg*dg + db*db


def _rgb_to_css4_name(color: RGB) -> Tuple[str, int]:
    """
    Return nearest CSS4 color name and squared distance for the given RGB.
    """
    best_name: str = _to_hex(color)
    best_dist: int = 256*256*3
    for name, hex_code in CSS4_COLORS.items():
        c = _hex_to_rgb(hex_code)
        d = _rgb_distance_sq(color, c)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist


def _unique_colors_rgb(img: Image.Image, *, ignore_alpha_below: int = 1) -> List[RGB]:
    """
    Return unique RGB colors present in the image. If the image has an alpha
    channel, ignore pixels where alpha < ignore_alpha_below.
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    pixels = img.getdata()
    uniq: Counter[RGB] = Counter()

    if img.mode == "RGBA":
        for r, g, b, a in pixels:  # type: ignore[misc]
            if a >= ignore_alpha_below:
                uniq[(r, g, b)] += 1
    else:  # RGB
        for r, g, b in pixels:  # type: ignore[misc]
            uniq[(r, g, b)] += 1

    # Sort by frequency desc, then by color for determinism
    return [c for c, _ in uniq.most_common()]


def _quantize_image(img: Image.Image, max_colors: int) -> Image.Image:
    """
    Quantize the image down to at most `max_colors` using an adaptive palette.
    Keeps alpha by compositing on opaque background if necessary.
    """
    # Work in RGB for palette quantization
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    if img.mode == "RGBA":
        # Composite over opaque background to avoid introducing extra colors
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        base = bg
    else:
        base = img.convert("RGB")

    return base.quantize(colors=max_colors, method=Image.Quantize.ADLAB).convert("RGB")


def _collect_palette(img: Image.Image) -> List[RGB]:
    """Collect unique palette colors used in an RGB image, ordered by frequency."""
    return _unique_colors_rgb(img)


def extract_colors_and_prompt(
    map_path: str,
    *,
    max_colors: int = 32,
    ignore_alpha_below: int = 1,
    categories: Optional[Sequence[str]] = None,
    default_category: Optional[str] = None,
    save_path: Optional[str] = None,
    prompt_fn = input,
) -> Dict[str, str]:
    """
    Load a PNG map, extract unique colors (quantized if needed), and interactively
    ask the user to assign each color to a semantic category.

    Returns a dict mapping human-friendly color name (CSS4, e.g. "forestgreen")
    to the chosen category. If multiple distinct RGBs map to the same name, the
    key is disambiguated as "name (#rrggbb)".

    Args:
        map_path: Path to the input image.
        max_colors: If the image has more unique colors, quantize to this count first.
        ignore_alpha_below: If RGBA, ignore pixels with alpha < this threshold.
        categories: Available categories; defaults to a sensible set.
        default_category: If provided, hitting Enter assigns this category.
        save_path: If provided, write the mapping to JSON at this path.
        prompt_fn: Function to read user input (defaults to built-in input).
    """
    if categories is None:
        categories = (
            "food",
            "terrain",
            "obstacle",
            "water",
            "house",
            "spawn",
            "ignore",
            "other",
        )

    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"Map not found: {map_path}")

    img = Image.open(map_path)
    base_colors = _unique_colors_rgb(img, ignore_alpha_below=ignore_alpha_below)

    if len(base_colors) > max_colors:
        q = _quantize_image(img, max_colors=max_colors)
        colors = _collect_palette(q)
    else:
        colors = base_colors

    # Presentation header
    print(f"Found {len(colors)} colors to categorize (max_colors={max_colors}).")
    print("Available categories:")
    print("  " + ", ".join(categories))
    if default_category:
        print(f"Press Enter for default: {default_category}")
    print()

    mapping: Dict[str, str] = {}
    used_names: set[str] = set()
    for idx, color in enumerate(colors, start=1):
        hex_color = _to_hex(color)
        name_guess, _ = _rgb_to_css4_name(color)
        display_name = name_guess
        # ensure uniqueness if same CSS name appears multiple times
        key_name = display_name
        if key_name in used_names:
            key_name = f"{display_name} ({hex_color})"
        used_names.add(key_name)

        prompt = (
            f"[{idx}/{len(colors)}] Assign category for {display_name} {hex_color} "
            f"(R,G,B={color}): "
        )
        while True:
            answer = prompt_fn(prompt).strip()
            if answer == "" and default_category is not None:
                answer = default_category

            if answer in categories:
                mapping[key_name] = answer
                break
            elif answer:
                print(f"Invalid category '{answer}'. Choose one of: {', '.join(categories)}")
            else:
                # Allow skipping by empty if no default; mark as ignore
                print("No input; marking as 'ignore'.")
                mapping[key_name] = "ignore"
                break

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"map": os.path.abspath(map_path), "mapping": mapping}, f, indent=2)
        print(f"Saved color mapping to {save_path}")

    return mapping


def _infer_default_save_path(map_path: str) -> str:
    base, _ = os.path.splitext(map_path)
    return f"{base}_color_map.json"


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Extract colors from a PNG and categorize them.")
    parser.add_argument("--map", dest="map_path", required=True, help="Path to the PNG map (e.g., images/big_map.png)")
    parser.add_argument("--max-colors", dest="max_colors", type=int, default=32, help="Max colors after quantization")
    parser.add_argument("--default", dest="default_category", default=None, help="Default category when pressing Enter")
    parser.add_argument("--save", dest="save_path", default=None, help="Path to save JSON mapping (default: map_path + _color_map.json)")

    args = parser.parse_args(argv)

    save_path = args.save_path or _infer_default_save_path(args.map_path)
    extract_colors_and_prompt(
        args.map_path,
        max_colors=args.max_colors,
        default_category=args.default_category,
        save_path=save_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


