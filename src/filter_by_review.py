import csv
import shutil
import json
import argparse
from pathlib import Path


def filter_by_review(extracted_dir, output_dir, review_csv):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(review_csv) as f:
        reader = csv.DictReader(f)
        good = {row["file"] for row in reader if row["status"] == "g"}

    moved = 0
    for img_path in extracted_dir.glob("*_image.tif"):
        if img_path.stem in good:
            shutil.move(str(img_path), output_dir / img_path.name)
            label_path = img_path.with_name(img_path.name.replace("_image", "_label"))
            if label_path.exists():
                shutil.move(str(label_path), output_dir / label_path.name)
            moved += 1

    print(f"filter_by_review.py: Moved {moved} good quality files")


def main():
    parser = argparse.ArgumentParser(description="Filter extracted rasters by review CSV")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    project_root = Path(cfg["paths"]["project_root"]).resolve()
    extracted_dir = project_root / cfg["paths"]["training_dataset"] / "extracted"
    output_dir = project_root / cfg["paths"]["training_dataset"] / "reviewed"
    review_csv = project_root / cfg["paths"]["review_csv"]

    filter_by_review(extracted_dir, output_dir, review_csv)


if __name__ == "__main__":
    main()