# Manual

This document describes installation, project structure, preprocessing pipeline, and evaluation tools for transfer learning for cloud semantic segmentation on satellite imagery.

---

# 1. Project Structure

```txt
src/
├── preprocessing/
├── evaluation/
├── notebooks/

config/
├── s2_config.json
├── venus_config.json
```

- `preprocessing/` – preprocessing pipeline for dataset creation
- `evaluation/` – evaluation metrics and visualization tools
- `notebooks/` – manual inspection and dataset annotation review
- `config/` – dataset configuration files

Two configurations are used:
- `s2_config.json` – Sentinel-2 processing configuration
- `venus_config.json` – VENµS processing configuration

# 2. Configuration

All parameters and paths are defined in JSON config file.

The project supports two dataset configurations:

- Sentinel-2 (`s2`)
- VENµS (`venus`)

These configurations define dataset-specific preprocessing and normalization settings.

---

## 2.1 Config structure

- **paths**
  - input/output dataset directories
  - AOI file
  - training/test directories
  - manual review CSV

- **band_extraction**
  - input spectral bands mapping
  - label band selection
  - data types
  - nodata handling
  - class remapping

- **dataset**
  - patch size (`tensor_shape`)
  - train/val split (`val_set_pct`)
  - padding mode
  - ignore label (`mask_ignore_value`)
  - optional class filtering

- **normalization**
  - `nodata_value`
  - predefined stats:
    - Sentinel-2 (`s2`)
    - VENµS (`venus`)
  - mean/std file paths + training source

---

# 3. Requirements

Main dependencies:

- GDAL  
- Rasterio  
- Fiona  
- Shapely  
- NumPy  
- Matplotlib  
- TensorFlow  

External dependency

- cnn-lib

https://github.com/ctu-geoforall-lab/cnn-lib

Library used during preprocessing for dataset structure generation

---

# 4. Preprocessing

Pipeline for preprocessing raw satellite data into training dataset supported by cnn-lib

Steps:

1. Raster selection (AOI filtering)
2. Band extraction
3. Manual quality filtering
4. Dataset generation
5. Normalization

---

## 4.1 Full pipeline

```bash
python src/preprocessing/run_pipeline.py --config config/s2_config.json
```

### Arguments

- `--config` (required)
- `--steps {select, extract, review, generate, normalize}`  
  Run selected pipeline stages
- `--test`  
  Run pipeline to create test set, otherwise train and val set are created
- `--no-augment`  
  Disable data augmentation during dataset generation
- `--stats {s2, venus}`  
  Select normalization statistics
- `--dynamic`  
  Enable dynamic (per-tile) normalization

---

## 4.2 select_rasters.py

Selects rasters intersecting AOI.

```bash
python src/preprocessing/select_rasters.py --config config/s2_config.json
```

### Arguments

- `--config` (required)
- `--project-root` (optional override)

---

## 4.3 extract_bands.py

Extracts spectral bands and creates image-label pairs.

```bash
python src/preprocessing/extract_bands.py --config config/s2_config.json
```

### Arguments

- `--config` (required)

Uses from config:
- `band_extraction.image_bands`
- `band_extraction.label_bands`
- `class_remapping`
- nodata handling

---

## 4.4 filter_by_review.py

Filters dataset using manually created CSV from notebook review.

```bash
python src/preprocessing/filter_by_review.py --config config/s2_config.json
```

### Arguments

- `--config` (required)

Input:
- `review_csv`

---

## 4.5 generate_dataset.py

Creates train/val dataset structure.

```bash
python src/preprocessing/generate_dataset.py --config config/s2_config.json
```

### Arguments

- `--config` (required)
- `--test`  
  Use test split instead of training set
- `--no-augment`  
  Disable augmentation

---

## 4.6 normalize.py

Applies dataset normalization.

```bash
python src/preprocessing/normalize.py --config config/s2_config.json --stats s2
```

### Arguments

- `--config` (required)
- `--test`  
  Normalize only test/validation split
- `--stats {s2, venus}`  
  Use predefined normalization statistics
- `--dynamic`  
  Per-image dynamic normalization

---

## 4.7 Output usage

The output dataset is compatible with cnn-lib training pipeline.

Generated structure (`train/`, `val/`) is directly used as input for cnn-lib training utilities.

---

# 5. Evaluation

Tools for quantitative and qualitative evaluation.

---

## 5.1 compute_metrics.py

Computes segmentation metrics and generates reports.

```bash
python src/evaluation/compute_metrics.py \
    --pred_dir results/pred \
    --gt_dir data/val_masks
```

### Arguments

- `--pred_dir` (required)
- `--gt_dir` (required)
- `--output_dir`
- `--log_dir`
- `--num_classes`
- `--class_names`
- `--ignore_value` (default 255)
- `--merge_shadow_to_clear`
- `--single_file`
- `--cm_caption`
- `--cm_label`
- `--table_caption`
- `--table_label`

### Outputs

- per-class metrics (F1, IoU, precision, recall)
- overall accuracy & balanced accuracy
- confusion matrix (LaTeX TikZ)
- metrics table (LaTeX table)
- optional training curves

---

## 5.2 visualize_tile.py

Visualizes prediction vs ground truth.

```bash
python src/evaluation/visualize_tile.py \
    --pred_mask results/tile.tif \
    --data_base_dir data
```

### Arguments

- `--pred_mask` (file or directory for batch mode)
- `--data_base_dir`
- `--batch`
- `--gamma`
- `--title`
- `--output`

### Output

- PNG/PDF visualization:
  - RGB image
  - ground truth
  - prediction

---

# 6. Notebooks

Used for manual dataset inspection.

Purpose:
- annotation quality control
- CSV generation for review filtering

---
# Note

The pipeline is designed specifically for the dataset used in this thesis and is not intended as a general-purpose preprocessing tool.
General dataset generation and training-related functionality is provided by cnn-lib.
