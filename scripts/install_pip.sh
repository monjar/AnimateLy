#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv animately-env
echo "Activating virtual environment"

source $PWD/animately-env/bin/activate

$PWD/animately-env/bin/pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
$PWD/animately-env/bin/pip install git+https://github.com/giacaglia/pytube.git --upgrade
$PWD/animately-env/bin/pip install -r requirements.txt
