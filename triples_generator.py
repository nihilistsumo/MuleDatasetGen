#!/usr/bin/python3

import random, json, sys
import numpy as np

def generate_triples(page_paras, page_para_labels_file):
    with open(page_para_labels_file, 'r') as pl:
        page_para_labels = json.load(pl)
    triples = []
    for page in page_paras.keys():
        paras_in_page = page_paras[page]
        para_label_dict_in_page = page_para_labels[page]
        max_label = max(para_label_dict_in_page.values())
        for l in range(-1, max_label+1):
            if list(para_label_dict_in_page.values()).count(l) > 1:
                similar = []
                dissimilar = []
                for p in para_label_dict_in_page.keys():
                    if para_label_dict_in_page[p] == l:
                        similar.append(p)
                    else:
                        dissimilar.append(p)
                for i in range(len(similar)-1):
                    for j in range(i+1, len(similar)):
                        for k in range(len(dissimilar)):
                            triples.append(page+" "+str(paras_in_page.index(similar[i]))+" "+str(paras_in_page.index(similar[j]))+" "+str(paras_in_page.index(dissimilar[k])))
    return triples

def main():
    page_paras_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned.page.paras.json"
    page_para_labels_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned.page.para.labels.json"
    output_file_json = "/home/sumanta/Documents/Mule-data/input_data_v2/test.triples.json"
    output_file_np = "/home/sumanta/Documents/Mule-data/input_data_v2/test.triples"
    with open(page_paras_file, 'r') as pp:
        page_paras = json.load(pp)
    triples = generate_triples(page_paras, page_para_labels_file)
    with open(output_file_json, 'w') as outj:
        json.dump(triples, outj)
    triples_dict = dict()
    for t in triples:
        page = t.split(" ")[0]
        p1 = page_paras[page][int(t.split(" ")[1])]
        p2 = page_paras[page][int(t.split(" ")[2])]
        p3 = page_paras[page][int(t.split(" ")[3])]
        if page not in triples_dict.keys():
            triples_dict[page] = [[p1, p2, p3]]
        else:
            triples_dict[page].append((p1, p2, p3))
    np.save(output_file_np, triples_dict)

if __name__ == '__main__':
    main()