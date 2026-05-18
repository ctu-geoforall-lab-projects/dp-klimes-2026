import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from osgeo import gdal


def compute_normalization_stats(train_images_dir, nodata_value=-1):
    first_file = next(train_images_dir.glob("*.tif"))
    ds = gdal.Open(str(first_file))
    n_bands = ds.RasterCount
    ds = None

    sum_ = np.zeros(n_bands, dtype=np.float64)
    sum_sq = np.zeros(n_bands, dtype=np.float64)
    n_pixels = np.zeros(n_bands, dtype=np.int64)

    for tif in train_images_dir.glob("*.tif"):
        ds = gdal.Open(str(tif))
        for band in range(1, n_bands + 1):
            rb = ds.GetRasterBand(band)
            data = rb.ReadAsArray().astype(np.float64)
            gdal_nodata = rb.GetNoDataValue()

            mask = data != nodata_value
            if gdal_nodata is not None:
                mask &= data != gdal_nodata

            valid = data[mask]
            sum_[band - 1] += valid.sum()
            sum_sq[band - 1] += (valid ** 2).sum()
            n_pixels[band - 1] += valid.size
        ds = None

    mean = sum_ / n_pixels
    std = np.sqrt(sum_sq / n_pixels - mean ** 2)
    return mean, std


def save_normalization_stats(mean_path, std_path, mean, std):
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"mean: {mean}")
    print(f"std: {std}")


def load_normalization_stats(mean_path, std_path):
    mean = np.load(mean_path)
    std = np.load(std_path)
    return mean, std


def apply_normalization(images_dir, masks_dir, out_images_dir, out_masks_dir, mean, std):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")

    for img_path in images_dir.glob("*.tif"):
        ds = gdal.Open(str(img_path))
        data = ds.ReadAsArray().astype(np.float32)

        for b in range(data.shape[0]):
            data[b] = (data[b] - mean[b]) / std[b]

        out = driver.Create(
            str(out_images_dir / img_path.name),
            ds.RasterXSize, ds.RasterYSize,
            ds.RasterCount, gdal.GDT_Float32,
            ["COMPRESS=LZW"]
        )
        out.SetProjection(ds.GetProjection())
        out.SetGeoTransform(ds.GetGeoTransform())

        for b in range(ds.RasterCount):
            out.GetRasterBand(b + 1).WriteArray(data[b])

        out.FlushCache()
        out = None
        ds = None

        shutil.copy2(masks_dir / img_path.name, out_masks_dir / img_path.name)


def apply_dynamic_zscore(images_dir, masks_dir, out_images_dir, out_masks_dir, nodata_value=-1):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")

    for img_path in images_dir.glob("*.tif"):
        ds = gdal.Open(str(img_path))
        data = ds.ReadAsArray().astype(np.float32)

        for b in range(data.shape[0]):
            band = data[b]
            mask = band != nodata_value
            valid = band[mask]
            mean = valid.mean()
            std = valid.std()
            data[b] = np.where(mask, (band - mean) / (std + 1e-8), 0)

        out = driver.Create(
            str(out_images_dir / img_path.name),
            ds.RasterXSize, ds.RasterYSize,
            ds.RasterCount, gdal.GDT_Float32,
            ["COMPRESS=LZW"]
        )
        out.SetProjection(ds.GetProjection())
        out.SetGeoTransform(ds.GetGeoTransform())

        for b in range(ds.RasterCount):
            out.GetRasterBand(b + 1).WriteArray(data[b])

        out.FlushCache()
        out = None
        ds = None

        shutil.copy2(masks_dir / img_path.name, out_masks_dir / img_path.name)



def normalize_dataset(data_dir, dst_dir, mean, std, subsets=("train", "val")):
    for subset in subsets:
        apply_normalization(
            images_dir=data_dir / f"{subset}_images",
            masks_dir=data_dir / f"{subset}_masks",
            out_images_dir=dst_dir / f"{subset}_images",
            out_masks_dir=dst_dir / f"{subset}_masks",
            mean=mean,
            std=std,
        )
        print(f"normalize.py: {subset} normalized")
    print(f"normalize.py: Fixed z-score finished")


def normalize_dataset_dynamic(data_dir, dst_dir, subsets=("train", "val"), nodata_value=-1):
    for subset in subsets:
        apply_dynamic_zscore(
            images_dir=data_dir / f"{subset}_images",
            masks_dir=data_dir / f"{subset}_masks",
            out_images_dir=dst_dir / f"{subset}_images",
            out_masks_dir=dst_dir / f"{subset}_masks",
            nodata_value=nodata_value,
        )
        print(f"normalize.py: {subset} normalized")
    print(f"normalize.py: Dynamic z-score finished")


def main():
    parser = argparse.ArgumentParser(description="Normalize dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--stats", required=True, choices=["venus", "s2"])
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    project_root = Path(cfg["paths"]["project_root"]).resolve()
    nodata_value = cfg["normalization"]["nodata_value"]
    stats_cfg = cfg["normalization"]["stats"][args.stats]

    # compute stats
    source_dir = project_root / stats_cfg["source"]
    mean, std = compute_normalization_stats(source_dir, nodata_value=nodata_value)

    # input and output dirs
    if args.test:
        data_dir = project_root / cfg["paths"]["test_dir"]
        subsets = ("val",)
    else:
        data_dir = project_root / cfg["paths"]["train_dir"]
        subsets = ("train", "val")

    if args.dynamic:
        dst_dir = data_dir / f"norm_dynamic"
        normalize_dataset_dynamic(data_dir, dst_dir, subsets=subsets, nodata_value=nodata_value)
    else:
        source_dir = project_root / stats_cfg["source"]
        mean, std = compute_normalization_stats(source_dir, nodata_value=nodata_value)
        mean_path = project_root / stats_cfg["mean"]
        std_path = project_root / stats_cfg["std"]
        save_normalization_stats(mean_path, std_path, mean, std)
        dst_dir = data_dir / f"norm_{args.stats}_fix"
        normalize_dataset(data_dir, dst_dir, mean, std, subsets=subsets)


if __name__ == "__main__":
    main()