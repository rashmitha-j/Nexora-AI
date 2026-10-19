#!/usr/bin/env sh
# Installs the MetaGPT worker into the active Python. Run from the repository root.
# Usage: PYTHON=path/to/python sh worker/install.sh   (defaults to "python")
set -e
PY="${PYTHON:-python}"
"$PY" -m pip install -r worker/requirements.txt
"$PY" -m pip install --no-deps "metagpt @ git+https://github.com/FoundationAgents/MetaGPT.git@11cdf466d042aece04fc6cfd13b28e1a70341b1f"
