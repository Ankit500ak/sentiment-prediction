"""
Helper to create or install requirements for the project.
Run this from the project root where `hhe.py` lives.

Usage:
    python requirements.py --write   # write requirements.txt (will overwrite)
    python requirements.py --install # install requirements via pip

This script keeps a canonical list of deps (can be edited) and writes them to requirements.txt
or calls pip to install them into the active Python environment.
"""
import argparse
import subprocess
from pathlib import Path

DEFAULT_REQS = [
    "numpy",
    "pandas",
    "matplotlib",
    "tensorflow",
    "keras",
    "scikit-learn",
]

REQ_FILE = Path(__file__).parent / "requirements.txt"


def write_requirements(reqs=DEFAULT_REQS, path=REQ_FILE):
    path.write_text("\n".join(reqs) + "\n")
    print(f"Wrote {path} with {len(reqs)} packages")


def install_requirements(path=REQ_FILE):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run --write first or create the file.")
    cmd = ["python", "-m", "pip", "install", "-r", str(path)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help="Write requirements.txt")
    p.add_argument("--install", action="store_true", help="Install requirements from requirements.txt")
    args = p.parse_args()

    if args.write:
        write_requirements()
    elif args.install:
        install_requirements()
    else:
        print("No action provided. Use --write or --install")
