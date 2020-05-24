#!/usr/bin/env bash
source /workspace/venv/bin/activate

pip install -r requirements-sotabench.txt

apt-get git
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./

echo "Extracting dataset and annotations..."
cd ./.data/vision/coco
python -c 'import zipfile; zipfile.ZipFile("annotations_trainval2017.zip").extractall()'
python -c 'import zipfile; zipfile.ZipFile("val2017.zip").extractall()'
cd -
