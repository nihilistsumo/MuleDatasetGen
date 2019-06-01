#!/usr/bin/python3

import random, json, sys
import numpy as np

def get_lookup_data(elmo_data):
    lookup_data = dict()
    for p in elmo_data[()].keys():
        print(p)
        page_data = elmo_data[()][p]
        si = 0
        for pi in range(len(page_data['paraids'])):
            paraid = page_data['paraids'][pi]
            para_sent_vecs = []
            for i in range(page_data['para_sent_count'][pi]):
                pvec = page_data['sent_vecs'][si]
                para_sent_vecs.append(pvec)
                si += 1
            lookup_data[paraid] = para_sent_vecs
    return lookup_data

def main():
    elmo_data_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-nodup-elmo-vec-data/by1test_merged_elmo_data_squeezed.npy"
    out_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-nodup-elmo-vec-data/by1test_merged_elmo_squeezed_para_vec_lookup"
    elmo_data = np.load(elmo_data_file)
    lookup = get_lookup_data(elmo_data)
    np.save(out_file, lookup)

if __name__ == '__main__':
    main()