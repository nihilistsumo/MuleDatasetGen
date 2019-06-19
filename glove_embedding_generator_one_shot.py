#!/usr/bin/python3

import math, json, os, sys, argparse, csv
import numpy as np
import pandas as pd
import spacy
import logging
from pathos.multiprocessing import ThreadPool

#######################################
#
# Single swipe through para list
#
#######################################

def construct_para_embedding(para):
    para_tokens = tokenized_para[()][para]
    print(para+": "+str(len(para_tokens))+" tokens")
    para_vec = []
    for t in para_tokens:
        if t in glove.index:
            para_vec.append(glove.loc[t].as_matrix())
    embed_vecs[para] = para_vec

parser = argparse.ArgumentParser(description='Create Glove paragraph embedding from a list of paragraphs.')
parser.add_argument("-g", "--glove_file", required=True, help="Pretrained Glove file")
parser.add_argument("-pl", "--para_list", required=True, help="List of paragraphs")
parser.add_argument("-pt", "--tokenized_para", required=True, help="Tokenized para file")
parser.add_argument("-o", "--out", required=True, help="Output file")
parser.add_argument("-pn", "--no_processes", type=int, required=True, help="No of parallel processes")
args = vars(parser.parse_args())
glove_file = args["glove_file"]
para_list = args["para_list"]
tokenized_para_file = args["tokenized_para"]
out_file = args["out"]
pno = args["no_processes"]

glove = pd.read_table(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
with open(para_list, 'r') as pl:
    paras = json.load(pl)
tokenized_para = np.load(tokenized_para_file)

print("Data loaded")
print(str(len(paras))+" total paras")
embed_vecs = dict()
with ThreadPool(pno) as pool:
    pool.map(construct_para_embedding, paras)

np.save(out_file, embed_vecs)