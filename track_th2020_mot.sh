#!/bin/bash
cd src
python track.py mot --load_model ../models/fairmot_th2020_dla34.pth --gpus 0 --val_seq_file TH2020_seqs_4p.json --cmc_on --cmc_type aff --mot_frame_skip 1
cd ..