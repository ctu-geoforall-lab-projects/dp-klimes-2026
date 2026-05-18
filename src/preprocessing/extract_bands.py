import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from osgeo import gdal


@dataclass
class BandExtractionConfig:
    bands: dict
    output_dtype: int = gdal.GDT_UInt16
    nodata_in: float = None
    nodata_out: float = None
    class_remapping: dict = field(default_factory=dict)
    propagate_nodata_to_label: bool = False


def apply_remapping(data, remapping):
    if not remapping:
        return data
    result = data.copy()
    for src, dst in remapping.items():
        result[data == src] = dst
    return result


def extract_bands(raster_paths, output_folder, image_config, label_config=None):
    output_folder.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")

    for tif in raster_paths:
        ds = gdal.Open(str(tif))
        is_venus_label = tif.stem.endswith("_label")
        is_venus_image = tif.stem.endswith("_image")
        is_s2 = not is_venus_label and not is_venus_image

        if is_s2:
            configs_to_write = [(image_config, f"{tif.stem}_image.tif")]
            if label_config:
                configs_to_write.append((label_config, f"{tif.stem}_label.tif"))
        elif is_venus_label:
            configs_to_write = [(label_config, tif.name)]
        else:
            configs_to_write = [(image_config, tif.name)]

        for config, out_name in configs_to_write:
            out = driver.Create(
                str(output_folder / out_name),
                ds.RasterXSize, ds.RasterYSize,
                len(config.bands), config.output_dtype, ["COMPRESS=LZW"]
            )
            out.SetProjection(ds.GetProjection())
            out.SetGeoTransform(ds.GetGeoTransform())

            nodata_mask = None
            if is_venus_label and image_config.propagate_nodata_to_label:
                image_path = tif.with_name(tif.name.replace("_label", "_image"))
                ds_image = gdal.Open(str(image_path))
                nodata_mask = ds_image.GetRasterBand(1).ReadAsArray() == image_config.nodata_in
                ds_image = None

            for idx, (band_name, band_number) in enumerate(config.bands.items(), start=1):
                data = ds.GetRasterBand(band_number).ReadAsArray()
                if config.nodata_in is not None and config.nodata_out is not None:
                    data[data == config.nodata_in] = config.nodata_out
                data = apply_remapping(data, config.class_remapping)
                if nodata_mask is not None:
                    data[nodata_mask] = 255
                out_band = out.GetRasterBand(idx)
                out_band.WriteArray(data)
                out_band.SetDescription(band_name)

            out.FlushCache()
            out = None
        ds = None
    print('extract_bands.py: Extraction finished')


def config_from_dict(cfg):
    image_bands = cfg["band_extraction"]["image_bands"]
    label_bands = cfg["band_extraction"]["label_bands"]
    remapping = {int(k): v for k, v in cfg["band_extraction"].get("class_remapping", {}).items()}

    dtype_map = {
        "int16": gdal.GDT_Int16,
        "uint16": gdal.GDT_UInt16,
        "float32": gdal.GDT_Float32,
    }

    image_config = BandExtractionConfig(
        bands=image_bands,
        output_dtype=dtype_map[cfg["band_extraction"].get("image_dtype", "int16")],
        nodata_in=cfg["band_extraction"].get("nodata_in"),
        nodata_out=cfg["band_extraction"].get("nodata_out"),
        propagate_nodata_to_label=cfg["band_extraction"].get("propagate_nodata_to_label", False),
    )
    label_config = BandExtractionConfig(
        bands=label_bands,
        output_dtype=dtype_map[cfg["band_extraction"].get("label_dtype", "int16")],
        class_remapping=remapping,
    )
    return image_config, label_config


def main():
    parser = argparse.ArgumentParser(description="Extract bands from rasters")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    project_root = Path(cfg["paths"]["project_root"]).resolve()
    training_dataset = project_root / cfg["paths"]["training_dataset"]
    output_folder = training_dataset / "extracted"
    raster_list = training_dataset / "rasters_in_aoi.txt"

    raster_paths = [Path(p) for p in raster_list.read_text().splitlines() if p.strip()]
    image_config, label_config = config_from_dict(cfg)

    extract_bands(raster_paths, output_folder, image_config, label_config)


if __name__ == "__main__":
    main()