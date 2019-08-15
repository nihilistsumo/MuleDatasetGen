import random, json, sys, argparse
import numpy as np

def generate_triples(page_para_dict, pagewise_hier_qrels, top_qrels_reversed):
    triples_data = dict()
    for page in page_para_dict.keys():
        paras_in_page = page_para_dict[page]
        hier_qrels_for_page = pagewise_hier_qrels[page]
        triples_data_in_page = []
        for hier in hier_qrels_for_page.keys():
            simparas = [p for p in hier_qrels_for_page[hier] if p in paras_in_page]
            if len(simparas) > 1:
                for i in range(len(simparas)-1):
                    for j in range(i+1, len(simparas)):
                        p1 = simparas[i]
                        p2 = simparas[j]
                        p3 = random.sample([p for p in paras_in_page if p not in simparas], 1)[0]
                        while top_qrels_reversed[p3] == top_qrels_reversed[p1]:
                            p3 = random.sample([p for p in paras_in_page if p not in simparas], 1)[0]
                        triples = [p1, p2, p3]
                        random.shuffle(triples)
                        triples.append(p3)
                        triples_data_in_page.append(triples)
        if len(triples_data_in_page) > 0:
            triples_data[page] = triples_data_in_page
            print(page)
    return triples_data

def get_pagewise_hier_qrels(hier_qrels_file):
    pagewise_hq = dict()
    with open(hier_qrels_file, 'r') as hq:
        for l in hq:
            hier_sec = l.split(' ')[0]
            page = hier_sec.split('/')[0]
            para = l.split(' ')[2]
            if page not in pagewise_hq.keys():
                pagewise_hq[page] = {hier_sec:[para]}
            else:
                if hier_sec in pagewise_hq[page].keys():
                    pagewise_hq[page][hier_sec].append(para)
                else:
                    pagewise_hq[page][hier_sec] = [para]
    return pagewise_hq

def get_reversed_top_qrels(top_qrels_file):
    top_qrels_reverse = dict()
    with open(top_qrels_file, 'r') as tq:
        for l in tq:
            top_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]
    return top_qrels_reverse

def main():
    parser = argparse.ArgumentParser(description="Generate pagewise discriminative triples")
    parser.add_argument('-p', '--parapairs', help='Path to parapairs json')
    parser.add_argument('-tq', '--top_qrels', help='Path to top-level qrels')
    parser.add_argument('-hq', '--hier_qrels', help='Path to hierarchical level qrels')
    parser.add_argument('-o', '--out', help='Path to output file')
    args = vars(parser.parse_args())
    parapairs_file = args['parapairs']
    top_qrels_file = args['top_qrels']
    hier_qrels_file = args['hier_qrels']
    outfile = args['out']

    with open(parapairs_file, 'r') as pp:
        parapairs = json.load(pp)
    page_paras_from_parapairs = dict()
    for page in parapairs.keys():
        pairs = parapairs[page]['parapairs']
        paras = set()
        for pair in pairs:
            paras.add(pair.split('_')[0])
            paras.add(pair.split('_')[1])
        page_paras_from_parapairs[page] = list(paras)
    top_qrels_rev = get_reversed_top_qrels(top_qrels_file)
    hier_qrels = get_pagewise_hier_qrels(hier_qrels_file)
    triples_data = generate_triples(page_paras_from_parapairs, hier_qrels, top_qrels_rev)
    with open(outfile, 'w') as out:
        json.dump(triples_data, out)

if __name__ == '__main__':
    main()