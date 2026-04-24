import sys
import json
import argparse
import subprocess
from pathlib import Path


steps = ["select", "extract", "review", "generate", "normalize"]



def run_step(script, args):
    cmd = [sys.executable, str(script)] + args
    print(f"\n{'='*50}")
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    print('='*50)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", nargs="+", choices=steps, default=steps)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--stats", choices=["venus", "s2"], default=None)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    src = Path(__file__).parent
    base_args = ["--config", args.config]

    if "select" in args.steps:
        run_step(src / "select_rasters.py", base_args)

    if "extract" in args.steps:
        run_step(src / "extract_bands.py", base_args)

    if "review" in args.steps:
        run_step(src / "filter_by_review.py", base_args)

    if "generate" in args.steps:
        generate_args = base_args.copy()
        if args.test:
            generate_args.append("--test")
        if args.no_augment:
            generate_args.append("--no-augment")
        run_step(src / "generate_dataset.py", generate_args)

    if "normalize" in args.steps:
        normalize_args = base_args.copy()
        if args.test:
            normalize_args.append("--test")
        if args.dynamic:
            normalize_args.append("--dynamic")
        stats = args.stats or ("venus" if "venus" in args.config else "s2")
        normalize_args += ["--stats", stats]
        run_step(src / "normalize.py", normalize_args)
    print("run_pipeline.py: Pipeline finished")

if __name__ == "__main__":
    main()