#!/usr/bin/python3

import random, json, sys
import numpy as np

def get_para_obj(paraid, jsonl_file_ptr, jsonl_lookup):
    para_span = jsonl_lookup[paraid]
    jsonl_file_ptr.seek(para_span[0])
    para_obj = json.loads(jsonl_file_ptr.read(para_span[1] - para_span[0]))
    return para_obj

def get_jsonl_lookup(jsonl_lookup_file):
    lookup = dict()
    with open(jsonl_lookup_file, 'r') as jl:
        for l in jl:
            lookup[l.split(" ")[0]] = (int(l.split(" ")[1]), int(l.split(" ")[2]))
    return lookup

def main():
    jsonl_file = sys.argv[1]
    jsonl_lookup_file = sys.argv[2]
    paraid = ""
    jsonl_lookup = get_jsonl_lookup(jsonl_lookup_file)
    jsonl = open(jsonl_file, 'r')
    para_obj = get_para_obj(paraid, jsonl, jsonl_lookup)