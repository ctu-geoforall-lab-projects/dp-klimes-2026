import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from osgeo import gdal
import argparse


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

import matplotlib
matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size":       11,
    "text.usetex":     False,
})

# Colors for each class (0=clear, 1=cloud, 2=shadow)
CLASS_COLORS = {
    0: (0.05, 0.05, 0.05),  # clear  – black
    1: (1.00, 1.00, 1.00),  # cloud  – white
    2: (0.55, 0.55, 0.55),  # shadow – grey
}
CLASS_NAMES = {
    0: "Clear",
    1: "Cloud",
    2: "Shadow",
}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def read_rgb(image_path, gamma=1.0):
    """Read R=band3, G=band2, B=band1 from a multiband GeoTIFF and return uint8 array."""
    ds = gdal.Open(str(image_path))
    r = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
    g = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    b = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None

    def normalize(band):
        p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
        band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
        if gamma != 1.0:
            band = np.power(band, 1.0 / gamma)
        return (band * 255).astype(np.uint8)

    return np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)


def read_mask(mask_path):
    ds = gdal.Open(str(mask_path))
    mask = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return mask


def mask_to_rgb(mask):
    """Convert integer class mask to an RGB image using CLASS_COLORS."""
    h, w = mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    for class_idx, color in CLASS_COLORS.items():
        m = mask == class_idx
        rgb[m] = color
    return rgb


def make_legend(class_indices):
    return [
        mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i], edgecolor="black", linewidth=0.8)
        for i in sorted(class_indices)
    ]


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def detect_ignore_border(mask, ignore_value=255):
    """Detect uniform ignore-value padding on right and bottom edges."""
    rows, cols = mask.shape
    crop_r, crop_b = 0, 0
    for c in range(cols - 1, -1, -1):
        if np.all(mask[:, c] == ignore_value):
            crop_r += 1
        else:
            break
    for r in range(rows - 1, -1, -1):
        if np.all(mask[r, :] == ignore_value):
            crop_b += 1
        else:
            break
    return crop_b, crop_r  # rows to remove from bottom, cols from right


def visualize_tile(pred_mask_path, data_base_dir, output_path=None, title=None, gamma=1.0):
    """
    pred_mask_path : path to the predicted mask .tif
    data_base_dir  : base directory containing _images/ and _masks/ subfolders
    output_path    : where to save the output .png
    """
    pred_mask_path = Path(pred_mask_path)
    data_base_dir  = Path(data_base_dir)

    if output_path is None:
        output_path = pred_mask_path.parent / "vis" / (pred_mask_path.stem + "_vis.pdf")
    output_path = Path(output_path)

    stem = pred_mask_path.stem

    image_path = data_base_dir / "val_images" / pred_mask_path.name
    gt_path    = data_base_dir / "val_masks"  / pred_mask_path.name

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT mask not found: {gt_path}")

    rgb       = read_rgb(image_path, gamma=gamma)
    gt_mask   = read_mask(gt_path)
    pred_mask = read_mask(pred_mask_path)

    gt_rgb   = mask_to_rgb(np.where(gt_mask   == 255, 0, gt_mask))
    pred_rgb = mask_to_rgb(np.where(pred_mask == 255, 0, pred_mask))

    # auto-detect and crop ignore padding on right and bottom
    crop_b, crop_r = detect_ignore_border(gt_mask)
    if crop_b > 0 or crop_r > 0:
        rb = -crop_b if crop_b > 0 else None
        rc = -crop_r if crop_r > 0 else None
        rgb      = rgb     [:rb, :rc]
        gt_rgb   = gt_rgb  [:rb, :rc]
        pred_rgb = pred_rgb[:rb, :rc]

    # legend — only classes present in either mask
    present_classes = set(np.unique(gt_mask[gt_mask != 255])) | \
                      set(np.unique(pred_mask[pred_mask != 255]))

    # crop 3-pixel ignore padding on right and bottom
    rgb       = rgb[:-3, :-3]
    gt_rgb    = gt_rgb[:-3, :-3]
    pred_rgb  = pred_rgb[:-3, :-3]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=11, fontweight="bold")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground truth", fontsize=11, fontweight="bold")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    # shared legend below
    legend = make_legend(present_classes)
    fig.legend(handles=legend, loc="lower center", ncol=len(present_classes),
               fontsize=10, frameon=True, edgecolor="black", bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RGB + GT + Prediction for a tile.")
    parser.add_argument("--pred_mask",      required=True, help="Path to predicted mask .tif, or folder if --batch is set")
    parser.add_argument("--data_base_dir",  required=True, help="Base directory with val_images/ and val_masks/ subfolders")
    parser.add_argument("--batch",  action="store_true", help="Process all .tif files in pred_mask folder")
    parser.add_argument("--gamma",  type=float, default=1.0,
                        help="Gamma correction for RGB brightness (e.g. 1.5 brightens, default: 1.0)")
    parser.add_argument("--title",  help="Optional figure title (single mode only)", default=None)
    parser.add_argument("--output", help="Output path (default: pred_mask dir/vis/ with _vis.pdf suffix)", default=None)
    args = parser.parse_args()

    if args.batch:
        pred_dir = Path(args.pred_mask)
        tifs = sorted(pred_dir.glob("*.tif"))
        if not tifs:
            print(f"No .tif files found in {pred_dir}")
        for tif in tifs:
            try:
                visualize_tile(tif, args.data_base_dir, gamma=args.gamma)
            except FileNotFoundError as e:
                print(f"  Skipped {tif.name}: {e}")
    else:
        visualize_tile(args.pred_mask, args.data_base_dir, output_path=args.output,
                       title=args.title, gamma=args.gamma)