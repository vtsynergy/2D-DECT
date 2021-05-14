#!/bin/sh


module load gcc cuda Anaconda3 jdk
source activate powerai16_ibm
#source activate pytf_cc
python train_main2.py

