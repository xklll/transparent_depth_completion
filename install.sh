#!/bin/bash

conda create -n xkl python==3.8 -y
sudo apt install screen -y
screen -S xkl
conda activate xkl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch -y


pip install -r requirements.txt
# tensorboard --logdir=/opt/data/private/xkl/Net/tb_log/default