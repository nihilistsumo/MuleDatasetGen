#!/usr/bin/python3

import random, json, sys

def get_page_paras(art_qrels_file, clean_paras_file):
    with open(clean_paras_file, 'r') as cp:
        clean_paras = json.load(cp)
    page_paras = dict()
    with open(art_qrels_file, 'r') as art:
        for l in art:
            page = l.split(" ")[0]
            para = l.split(" ")[2]
            if para in clean_paras:
                if page not in page_paras.keys():
                    page_paras[page] = [para]
                else:
                    page_paras[page].append(para)
    return page_paras

def main():
    art_qrels_file = "/home/sumanta/Documents/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels"
    clean_paras_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned-lengthwise-sorted-paralist.json"
    output_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned.page.paras.json"
    page_paras = get_page_paras(art_qrels_file, clean_paras_file)
    with open(output_file, 'w') as out:
        json.dump(page_paras, out)

if __name__ == '__main__':
    main()