import argparse
from pathlib import Path


def get_base_stem(stem):
    """Remove rotation suffixes like _rot90, _rot180, _rot270."""
    for suffix in ("_rot90", "_rot180", "_rot270"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def filter_by_existing(source_dir, reference_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_names = {f.stem for f in reference_dir.rglob("*.tif")}

    moved = 0
    deleted = 0
    for tif in source_dir.rglob("*.tif"):
        base = get_base_stem(tif.stem)
        if base in reference_names:
            if base == tif.stem:
                # originál — přesuň
                tif.rename(output_dir / tif.name)
                moved += 1
            else:
                # rotace originálu který je v reference — smaž
                tif.unlink()
                deleted += 1

    print(f"Moved: {moved}, Deleted: {deleted} files")


def main():
    parser = argparse.ArgumentParser(description="Filter rasters by presence in reference folder")
    parser.add_argument("--source", required=True, help="Source folder")
    parser.add_argument("--reference", required=True, help="Reference folder")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    filter_by_existing(Path(args.source), Path(args.reference), Path(args.output))


if __name__ == "__main__":
    main()