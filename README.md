# Modulator Auto-Design

LLM-driven design loop for electro-optic modulators using Tidy3D and PhotonForge.

## Setup (Windows, Python 3.10)

1. Create the venv:
py -3.10 -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
2. Install standard dependencies:
pip install -r requirements.txt

3. Install PhotonForge from the project's wheel:
pip install "wheels\photonforge-1.4.0-cp310-cp310-win_amd64.whl[live_viewer]"

   The wheel comes from Flexcompute's CI builds. To get a newer version,
   download the Windows artifact for Python 3.10 from the relevant build:
   https://github.com/flexcompute/compute/actions

4. Verify:
python -c "import tidy3d, photonforge; print(tidy3d.version, photonforge.version)"

## Reproducing the exact environment

The exact pinned environment is in `requirements-lock.txt`. To recreate:
pip install -r requirements-lock.txt
pip install "wheels\photonforge-1.4.0-cp310-cp310-win_amd64.whl[live_viewer]"