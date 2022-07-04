#!/usr/bin/env bash

mkdir -p data
cd data
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"
unzip animately_data.zip
rm animately_data.zip
cd ..
mv data/animately_data/sample_video.mp4 .
mkdir -p $HOME/.torch/models/
mv data/animately_data/yolov3.weights $HOME/.torch/models/
