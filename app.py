"""Launcher script for the sentiment app.

Behavior:
- Optionally install requirements from requirements.txt
- Train the model by running hhe.py if sentiment_model.h5 is missing (or if --train is passed)
- Start the Flask UI (flask_app.py)

Usage examples:
  python app.py            # train if missing, then start server
  python app.py --no-install # don't run pip install
  python app.py --train     # force running hhe.py even if model exists
  python app.py --port 5000 # set port for Flask server

This script tries to run commands using the current Python interpreter. On Windows,
if you want to use the venv, activate it before running this script.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REQUIREMENTS = ROOT / 'requirements.txt'
MODEL = ROOT / 'sentiment_model.h5'

parser = argparse.ArgumentParser()
parser.add_argument('--no-install', action='store_true', help='skip pip install')
parser.add_argument('--train', action='store_true', help='force training (run hhe.py)')
parser.add_argument('--port', type=int, default=5000, help='port for Flask app')
parser.add_argument('--no-venv', action='store_true', help='do not create/use a .venv; run in the current interpreter')
parser.add_argument('--create-venv', action='store_true', help='create a .venv if missing')
parser.add_argument('--venv-path', type=str, default='.venv', help='path to virtualenv folder (default: .venv)')
args = parser.parse_args()

PY = sys.executable

VENV_PATH = Path(args.venv_path)

# Decide which Python executable to use for subsequent steps
def find_venv_python(venv_path: Path):
    win_py = venv_path / 'Scripts' / 'python.exe'
    posix_py = venv_path / 'bin' / 'python'
    if win_py.exists():
        return str(win_py)
    if posix_py.exists():
        return str(posix_py)
    return None

use_python = PY
if not args.no_venv:
    # create venv if requested or if missing
    if args.create_venv or (not VENV_PATH.exists() and args.create_venv):
        print('Creating virtual environment at', VENV_PATH)
        subprocess.check_call([PY, '-m', 'venv', str(VENV_PATH)])

    # if venv exists, prefer using it
    if VENV_PATH.exists():
        vpy = find_venv_python(VENV_PATH)
        if vpy:
            use_python = vpy
        else:
            print('Warning: .venv exists but no python executable found inside; using current interpreter')
else:
    print('Running without creating/using a .venv (using current interpreter)')

if not args.no_install and REQUIREMENTS.exists():
    print('Installing requirements from', REQUIREMENTS, 'using', use_python)
    subprocess.check_call([use_python, '-m', 'pip', 'install', '-r', str(REQUIREMENTS)])

# Train if model missing or if --train requested
if args.train or not MODEL.exists():
    print('Training model (running hhe.py) using', use_python, '. This may take a while...')
    subprocess.check_call([use_python, str(ROOT / 'hhe.py')])
else:
    print('Model exists, skipping training.')

# Start the Flask app
env = dict(**dict())
print('Starting Flask app (flask_app.py) on port', args.port)
# Start Flask app with chosen python
subprocess.check_call([use_python, str(ROOT / 'flask_app.py')])
