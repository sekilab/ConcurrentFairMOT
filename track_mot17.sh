#!/bin/bash
cd src
python track.py mot --val_mot17 True --load_model ../models/fairmot_dla34.pth --gpus 0
cd ..