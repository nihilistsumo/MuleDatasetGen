#!/usr/bin/python3

import random, json, sys

def get_page_para_labels(page_paras_file, page_topics_file, topqrels):
    with open(page_paras_file, 'r') as pp:
        page_paras = json.load(pp)
    with open(page_topics_file, 'r') as pt:
        page_topics = json.load(pt)
    page_para_labels = dict()
    with open(topqrels, 'r') as qrels:
        for l in qrels:
            section = l.split(" ")[0]
            para = l.split(" ")[2]
            if "/" in section:
                page = section.split("/")[0]
                section = section.split("/")[1]
            else:
                page = section
            if para in page_paras[page]:
                if page not in page_para_labels.keys():
                    page_para_labels[page] = dict()
                topics_in_page = page_topics[page]
                if section not in topics_in_page:
                    page_para_labels[page][para] = -1
                else:
                    page_para_labels[page][para] = topics_in_page.index(section)
    return page_para_labels

def main():
    page_paras_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned.page.paras.json"
    page_topics_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-nodup.page.tops.json"
    topqrels_file = "/home/sumanta/Documents/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels"
    output_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned.page.para.labels.json"
    page_para_labels = get_page_para_labels(page_paras_file, page_topics_file, topqrels_file)
    with open(output_file, 'w') as out:
        json.dump(page_para_labels, out)

if __name__ == '__main__':
    main()