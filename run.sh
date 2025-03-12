#!/bin/bash

set -e

export KAGGLE_USERNAME=$1
export KAGGLE_KEY=$2

pip install -U kaggle numpy umap-learn
kaggle datasets download -d orxaicom/daily-arxiv-embeddings
unzip daily-arxiv-embeddings.zip
rm -rf ./output
mkdir ./output
m4 app_m4.js > ./app.js
sed -i '1,2d' ./app.js
rm app_m4.js
cp ./index.html ./style.css ./app.js output
python umap_data.py
