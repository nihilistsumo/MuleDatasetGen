#!/usr/bin/python3

import random, json, sys

def get_clean_paras(all_para_list, paratext_dict, minlen):
    clean_paras = []
    para_word_count = []
    for p in all_para_list:
        ptext = paratext_dict[p]
        para_word_count.append((len(paratext_dict[p].split(" ")), p))
    para_word_count.sort()
    for ptup in para_word_count:
        if ptup[0] > minlen-1:
            clean_paras.append(ptup[1])
    return clean_paras

def main():
    all_paralist = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1test-nodup.json.data/by1test-para-set.json"
    all_paratexts_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-nodup.para.texts.json"
    min_para_word_count = 10
    output_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned-lengthwise-sorted-paralist.json"

    with open(all_paratexts_file, 'r') as pt:
        paratexts = json.load(pt)
    with open(all_paralist, 'r') as pl:
        paras = json.load(pl)
        sorted_clean_paras = get_clean_paras(paras, paratexts, min_para_word_count)
    with open(output_file, 'w') as out:
        json.dump(sorted_clean_paras, out)

if __name__ == '__main__':
    main()