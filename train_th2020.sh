#!/bin/bash
cd src
python train.py --gpus 0 mot --data_cfg ./lib/cfg/TH2020_split1.json
cd ..