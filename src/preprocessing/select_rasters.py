import json
import argparse
from pathlib import Path

import fiona
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box


def load_aoi_polygons(aoi_file):
    with fiona.open(aoi_file) as aoi:
        aoi_crs = aoi.crs_wkt
        aoi_polygons = [shape(polygon["geometry"]) for polygon in aoi]
    return aoi_polygons, aoi_crs


def select_intersecting_rasters(data_folder, aoi_file, list_file):
    aoi_polygons, aoi_crs = load_aoi_polygons(aoi_file)

    selected = []
    aoi_crs_check = {}

    for raster_path in data_folder.rglob("*.tif"):
        with rasterio.open(raster_path) as raster:
            raster_bounds = raster.bounds
            raster_crs = raster.crs

        raster_box = box(*raster_bounds)

        if raster_crs not in aoi_crs_check:
            aoi_boxes = []
            for aoi_polygon in aoi_polygons:
                bounds_in_raster_crs = transform_bounds(
                    aoi_crs, raster_crs, *aoi_polygon.bounds
                )
                aoi_boxes.append(box(*bounds_in_raster_crs))
            aoi_crs_check[raster_crs] = aoi_boxes

        if any(raster_box.intersects(aoi_bbox) for aoi_bbox in aoi_crs_check[raster_crs]):
            selected.append(raster_path)

    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("\n".join(str(p) for p in selected), encoding="utf-8")
    print(f"select_rasters.py: Selected {len(selected)} rasters")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Select rasters intersecting AOI")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--project-root", default=None, help="Project root path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    project_root = Path(cfg["paths"]["project_root"]).resolve()
    data_folder = project_root / cfg["paths"]["raw_data"]
    aoi_file = project_root / cfg["paths"]["aoi_file"]
    list_file = project_root / cfg["paths"]["training_dataset"] / "rasters_in_aoi.txt"

    select_intersecting_rasters(data_folder, aoi_file, list_file)


if __name__ == "__main__":
    main()