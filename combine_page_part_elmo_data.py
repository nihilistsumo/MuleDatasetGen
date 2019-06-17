#!/usr/bin/python3

import math, json, os, sys, argparse
import numpy as np

#######################################
#
# Combine pagewise elmo data generated
# by elmo_embedding_generator.py
#
#######################################

indir = sys.argv[1]
outfile = sys.argv[2]

comb_dict = dict()
for f in os.listdir(indir):
    print(f)
    curr_dict = np.load(indir+"/"+f, allow_pickle=True)[()]
    comb_dict.update(curr_dict)

np.save(outfile, comb_dict)