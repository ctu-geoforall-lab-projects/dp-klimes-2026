import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from osgeo import gdal
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, TENSORS, SCALARS


# ─────────────────────────────────────────────
#  TensorBoard parsing
# ─────────────────────────────────────────────

def parse_tensorboard_logs(log_base_dir, val_tag="val_loss", train_tag="train_loss"):
    """
    Reads TensorBoard event files from <log_base_dir>/validation and <log_base_dir>/train.
    Returns:
        best_epoch      – step/epoch with lowest val loss
        best_val_loss   – lowest val loss value
        final_val_loss  – val loss at last recorded step
        train_steps     – list of (step, loss) for training
        val_steps       – list of (step, loss) for validation
    """
    val_dir   = Path(log_base_dir) / "validation"
    train_dir = Path(log_base_dir) / "train"

    def load_scalars(folder, tag):
        ea = EventAccumulator(str(folder), size_guidance={SCALARS: 0, TENSORS: 0})
        ea.Reload()

        # TF1-style scalars
        scalar_tags = ea.Tags().get("scalars", [])
        if tag in scalar_tags:
            return [(e.step, e.value) for e in ea.Scalars(tag)]

        # TF2 stores scalars as tensors — decode with tf.make_ndarray
        tensor_tags = ea.Tags().get("tensors", [])
        if tag in tensor_tags:
            return [
                (e.step, float(tf.make_ndarray(e.tensor_proto)))
                for e in ea.Tensors(tag)
            ]

        print(f"  [WARN] Tag '{tag}' not found in {folder}")
        print(f"         Scalar tags : {scalar_tags}")
        print(f"         Tensor tags : {tensor_tags}")
        return []

    print("\nLoading TensorBoard logs...")
    val_steps   = load_scalars(val_dir,   val_tag)
    train_steps = load_scalars(train_dir, train_tag)

    if not val_steps:
        return None, None, train_steps, val_steps

    valid_val_steps = [(s, v) for s, v in val_steps if not np.isnan(v)]
    if not valid_val_steps:
        print("  [WARN] All val_loss values are NaN")
        return None, None, train_steps, val_steps

    best_epoch, best_val_loss = min(valid_val_steps, key=lambda x: x[1])

    print(f"  Best epoch   : {best_epoch}  (val_loss = {best_val_loss:.6f})")

    return best_epoch, best_val_loss, train_steps, val_steps


def smooth(values, weight=0.95):
    """Exponential moving average smoothing (TensorBoard-style)."""
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def plot_loss_curve(train_steps, val_steps, output_path):
    """Saves a training / validation loss curve as a PNG."""
    fig, ax = plt.subplots(figsize=(11, 5))

    def prepare(steps):
        """Deduplicate by step and drop NaNs."""
        seen = {}
        for s, v in steps:
            if not np.isnan(v):
                seen[s] = v
        xs = sorted(seen)
        return xs, [seen[x] for x in xs]

    if train_steps:
        t_x, t_y = prepare(train_steps)
        ax.plot(t_x, t_y, color="#90CAF9", linewidth=1.0, alpha=0.4, label="_nolegend_")
        ax.plot(t_x, smooth(t_y), color="#1565C0", linewidth=2.0, label="Train loss")

    if val_steps:
        v_x, v_y = prepare(val_steps)
        ax.plot(v_x, v_y, color="#EF9A9A", linewidth=1.0, alpha=0.4, label="_nolegend_")
        v_smooth = smooth(v_y)
        ax.plot(v_x, v_smooth, color="#C62828", linewidth=2.0, label="Val loss")

        # best on raw curve — consistent with reported value
        best_idx = int(np.argmin(v_y))
        best_x   = v_x[best_idx]
        best_y   = v_y[best_idx]
        ax.axvline(best_x, color="#C62828", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.scatter([best_x], [best_y], color="#C62828", zorder=5, s=60,
                   label=f"Best val loss {best_y:.3f} (epoch {best_x})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss",  fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    pdf_path = Path(output_path).with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Loss curve saved: {output_path} / {pdf_path}")


# ─────────────────────────────────────────────
#  Segmentation metrics
# ─────────────────────────────────────────────

def compute_metrics(pred_dir, gt_dir, num_classes, ignore_value=255, single_file=None):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    if single_file:
        files = [pred_dir / single_file]
    else:
        files = pred_dir.glob("*.tif")

    for pred_path in files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"Missing GT: {pred_path.name}")
            continue

        pred_ds = gdal.Open(str(pred_path))
        gt_ds   = gdal.Open(str(gt_path))

        pred = pred_ds.GetRasterBand(1).ReadAsArray()
        gt   = gt_ds.GetRasterBand(1).ReadAsArray()

        pred_ds = None
        gt_ds   = None

        valid_mask = gt != ignore_value
        pred = pred[valid_mask]
        gt   = gt[valid_mask]

        for true_class in range(num_classes):
            for pred_class in range(num_classes):
                confusion_matrix[true_class, pred_class] += np.sum(
                    (gt == true_class) & (pred == pred_class)
                )

    return confusion_matrix


def metrics_from_confusion_matrix(cm, class_names=None):
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    results = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        accuracy  = (tp + tn) / cm.sum() if cm.sum() > 0 else 0

        results[class_names[c]] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall,
            "f1": f1, "iou": iou, "accuracy": accuracy,
        }

    total_correct     = np.diag(cm).sum()
    total_pixels      = cm.sum()
    overall_accuracy  = total_correct / total_pixels if total_pixels > 0 else 0
    mean_f1           = np.mean([results[c]["f1"]     for c in class_names])
    mean_iou          = np.mean([results[c]["iou"]    for c in class_names])
    balanced_accuracy = np.mean([results[c]["recall"] for c in class_names])

    results["overall"] = {
        "accuracy":          overall_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mean_f1":           mean_f1,
        "mean_iou":          mean_iou,
    }

    return results


def print_metrics(results, class_names, best_epoch=None, best_val_loss=None):
    if any(v is not None for v in [best_epoch, best_val_loss]):
        print("\n── Training summary ─────────────────────────────────────────")
        if best_epoch    is not None: print(f"  Best epoch     : {best_epoch}")
        if best_val_loss is not None: print(f"  Best val loss  : {best_val_loss:.6f}")

    print(f"\n{'Class':<15} {'F1':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 55)
    for c in class_names:
        r = results[c]
        print(f"{c:<15} {r['f1']:>8.3f} {r['iou']:>8.3f} {r['precision']:>10.3f} {r['recall']:>8.3f}")
    print("-" * 55)
    r = results["overall"]
    print(f"{'Overall':<15} {r['mean_f1']:>8.3f} {r['mean_iou']:>8.3f} {'OA:':>6}{r['accuracy']:.3f}  {'BOA:':>5}{r['balanced_accuracy']:.3f}")


def metrics_to_latex(results, class_names, caption="", label="",
                     best_epoch=None, best_val_loss=None):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\hline")
    lines.append(r"Class & F1 & IoU & OA & BOA \\")
    lines.append(r"\hline")

    for c in class_names:
        r = results[c]
        lines.append(f"{c} & {r['f1']:.3f} & {r['iou']:.3f} & -- & -- \\\\")

    lines.append(r"\hline")
    r = results["overall"]
    lines.append(
        f"Overall & {r['mean_f1']:.3f} & {r['mean_iou']:.3f} & "
        f"{r['accuracy']:.3f} & {r['balanced_accuracy']:.3f} \\\\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    footnote_parts = []
    if best_epoch    is not None: footnote_parts.append(f"best epoch: {best_epoch}")
    if best_val_loss is not None: footnote_parts.append(f"best val loss: {best_val_loss:.6f}")
    if footnote_parts:
        lines.append(f"\\vspace{{2pt}}")
        lines.append(f"\\footnotesize{{Training info --- {', '.join(footnote_parts)}}}")

    if caption: lines.append(f"\\caption{{{caption}}}")
    if label:   lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def confusion_matrix_to_latex(cm, class_names, output_path, normalize=True, caption="", label=""):
    if normalize:
        cm_plot  = cm.astype(np.float64)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot  = np.divide(cm_plot, row_sums, where=row_sums != 0)
        fmt = lambda x: f"{x:.2f}"
    else:
        cm_plot = cm.astype(np.float64)
        fmt = lambda x: str(int(x))

    n          = len(class_names)
    cell_size  = 1.5
    row_counts = cm.sum(axis=1)

    lines = []
    lines.append(r"\begin{figure}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\definecolor{cellhigh}{RGB}{33,102,172}")
    lines.append(r"\definecolor{celllow}{RGB}{214,232,246}")
    lines.append(r"\definecolor{cellzero}{RGB}{247,251,255}")

    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            x   = j * cell_size
            y   = (n - 1 - i) * cell_size

            if val > 0.5:
                color = f"cellhigh!{min(max(int(val*100), 40), 75)}"
            elif val > 0.05:
                color = f"celllow!{max(int(val*200), 20)}"
            else:
                color = "cellzero"

            lines.append(f"\\fill[{color}] ({x},{y}) rectangle ({x+cell_size},{y+cell_size});")
            lines.append(
                f"\\node[black, font=\\normalsize\\bfseries] at ({x+cell_size/2},{y+cell_size/2}) {{{fmt(val)}}};"
            )

    lines.append(f"\\draw[black!50, line width=0.5pt] (0,0) rectangle ({n*cell_size},{n*cell_size});")
    for k in range(1, n):
        lines.append(f"\\draw[black!20, line width=0.3pt] ({k*cell_size},0) -- ({k*cell_size},{n*cell_size});")
        lines.append(f"\\draw[black!20, line width=0.3pt] (0,{k*cell_size}) -- ({n*cell_size},{k*cell_size});")

    for j, name in enumerate(class_names):
        x = j * cell_size + cell_size / 2
        lines.append(f"\\node[font=\\small] at ({x},{-0.35}) {{{name}}};")

    for i, name in enumerate(class_names):
        y = (n - 1 - i) * cell_size + cell_size / 2
        lines.append(f"\\node[anchor=east, font=\\small] at (-0.2,{y+0.18}) {{{name}}};")
        lines.append(f"\\node[anchor=east, font=\\footnotesize, black!60] at (-0.2,{y-0.18}) {{[{row_counts[i]}]}};")

    lines.append(f"\\node[font=\\small] at ({n*cell_size/2},{-0.9}) {{Predicted label}};")
    lines.append(f"\\node[rotate=90, font=\\small] at (-2.2,{n*cell_size/2}) {{True label}};")
    lines.append(f"\\node[font=\\normalsize\\bfseries] at ({n*cell_size/2},{n*cell_size+0.45}) {{Confusion matrix}};")

    lines.append(r"\end{tikzpicture}")
    if caption: lines.append(f"\\caption{{{caption}}}")
    if label:   lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{figure}")

    Path(output_path).write_text("\n".join(lines))
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate segmentation model predictions.")
    parser.add_argument("--pred_dir",  required=True, help="Directory with predicted .tif masks")
    parser.add_argument("--gt_dir",    required=True, help="Directory with ground truth .tif masks")
    parser.add_argument("--output_dir", help="Output directory (default: pred_dir)", default=None)
    parser.add_argument("--log_dir",    help="TensorBoard logs base directory", default=None)
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes (default: 3)")
    parser.add_argument("--class_names", nargs="+", default=["clear", "cloud", "shadow"],
                        help="Class names in order (default: clear cloud shadow)")
    parser.add_argument("--ignore_value", type=int, default=255,
                        help="Ignore value in masks (default: 255)")
    parser.add_argument("--single_file", help="Evaluate a single file only", default=None)
    parser.add_argument("--cm_caption",    default=None, help="Confusion matrix caption (default: none)")
    parser.add_argument("--cm_label",      default="fig:cm",      help="Confusion matrix label")
    parser.add_argument("--table_caption", default=None, help="Metrics table caption (default: none)")
    parser.add_argument("--table_label",   default="tab:metrics", help="Metrics table label")
    args = parser.parse_args()

    pred_dir   = Path(args.pred_dir)
    gt_dir     = Path(args.gt_dir)
    output_dir = Path(args.output_dir) if args.output_dir else pred_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse TensorBoard logs
    best_epoch = best_val_loss = train_steps = val_steps = None
    if args.log_dir:
        best_epoch, best_val_loss, train_steps, val_steps = parse_tensorboard_logs(
            args.log_dir, val_tag="epoch_loss", train_tag="epoch_loss"
        )
        plot_loss_curve(train_steps, val_steps, output_dir / "loss_curve.png")

    # 2. Segmentation metrics
    cm      = compute_metrics(pred_dir, gt_dir, args.num_classes, args.ignore_value,
                              single_file=args.single_file)
    results = metrics_from_confusion_matrix(cm, args.class_names)

    print_metrics(results, args.class_names, best_epoch, best_val_loss)

    # 3. LaTeX outputs
    latex_table = metrics_to_latex(
        results, args.class_names,
        caption=args.table_caption, label=args.table_label,
        best_epoch=best_epoch, best_val_loss=best_val_loss,
    )
    Path(output_dir / "metrics_table.tex").write_text(latex_table)
    print(f"\nLaTeX table saved: {output_dir / 'metrics_table.tex'}")

    confusion_matrix_to_latex(
        cm, args.class_names,
        output_dir / "confusion_matrix.tex",
        normalize=True,
        caption=args.cm_caption,
        label=args.cm_label,
    )