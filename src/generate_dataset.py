import sys
import json
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cnn-lib"))

from cnn_lib.data_preparation import generate_dataset_structure


def generate_dataset(dataset_folder, output_folder, augment=True, is_test=False, cfg=None):
    generate_dataset_structure(
        data_dir=dataset_folder,
        augment=augment,
        tensor_shape=tuple(cfg["dataset"]["tensor_shape"]),
        val_set_pct=1 if is_test else cfg["dataset"]["val_set_pct"],
        padding_mode=cfg["dataset"]["padding_mode"],
        mask_ignore_value=cfg["dataset"]["mask_ignore_value"],
        input_regex='*.tif'
    )

    if dataset_folder == output_folder:
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    for subset in ("train_images", "train_masks", "val_images", "val_masks"):
        src = dataset_folder / subset
        dst = output_folder / subset
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"Moved: {subset} -> {dst}")
    print("generate_dataset.py: Dataset generated")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset structure")
    parser.add_argument("--config", required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    project_root = Path(cfg["paths"]["project_root"]).resolve()
    augment = not args.test and not args.no_augment

    if args.test:
        dataset_folder = project_root / cfg["paths"]["test_dir"]
        output_folder = dataset_folder
    else:
        reviewed_dir = project_root / cfg["paths"]["training_dataset"] / "reviewed"
        extracted_dir = project_root / cfg["paths"]["training_dataset"] / "extracted"
        dataset_folder = reviewed_dir if reviewed_dir.exists() else extracted_dir
        output_folder = project_root / cfg["paths"]["train_dir"]

    generate_dataset(dataset_folder, output_folder, augment=augment, is_test=args.test, cfg=cfg)


if __name__ == "__main__":
    main()