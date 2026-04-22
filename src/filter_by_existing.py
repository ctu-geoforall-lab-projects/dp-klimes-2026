import argparse
from pathlib import Path


def filter_by_existing(source_dir, reference_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_names = {f.stem for f in reference_dir.rglob("*.tif")}

    moved = 0
    for tif in source_dir.rglob("*.tif"):
        if tif.stem in reference_names:
            tif.rename(output_dir / tif.name)
            moved += 1
    print(f"Moved: {moved} files")


def main():
    parser = argparse.ArgumentParser(description="Filter rasters by presence in reference folder")
    parser.add_argument("--source", required=True, help="Source folder")
    parser.add_argument("--reference", required=True, help="Reference folder")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    filter_by_existing(Path(args.source), Path(args.reference), Path(args.output))


if __name__ == "__main__":
    main()