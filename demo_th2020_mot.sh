#!/bin/bash
cd src
python demo.py mot --input-root ../videos --demo_seq_file TH2020_demo_seqs.json --load_model ../models/fairmot_th2020_dla34.pth --gpus -1
cd ..