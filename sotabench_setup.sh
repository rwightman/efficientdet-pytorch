#!/usr/bin/env bash
source /workspace/venv/bin/activate

pip install -r requirements-sotabench.txt

apt-get git
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
