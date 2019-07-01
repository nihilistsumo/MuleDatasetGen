import argparse
import json
import spacy

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from pathos.threading import ThreadPool


def preprocess_para(paratext, nlp):
    paratext = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')
    paratext = ' '.join(paratext.split())
    doc = nlp(paratext)
    tokenized_sentences = []
    for s in doc.sents:
        # tokenized_sentences.append([t.text for t in nlp(s.string)])
        tokenized_sentences.append(s.string.split())
    return tokenized_sentences

def preprocessed_paratext(paratext_dict):
    nlp = spacy.load("en_core_web_sm")
    preproc_paratext_dict = dict()
    for para in paratext_dict.keys():
        preproc_paratext_dict[para] = preprocess_para(paratext_dict[para], nlp)
    return preproc_paratext_dict

def get_elmo_embeddings(paralist):
    paralist_index_dict = dict()
    start_index = 0
    for para in paralist:
        sent_count = len(preproc_paratext_dict[para])
        paralist_index_dict[para] = (start_index, start_index + sent_count)
        start_index += sent_count
    sentences = []
    for para in paralist:
        sentences = sentences + preproc_paratext_dict[para]
    elmo = ElmoEmbedder()
    embed_vecs = elmo.embed_sentences(sentences, 10)

    for para in paralist_index_dict.keys():
        para_embed_vecs = []
        for i in range(paralist_index_dict[para][0], paralist_index_dict[para][1]):
            para_embed_vecs.append(next(embed_vecs))
        para_embed_dict[para] = para_embed_vecs
    print("{} paras embedded".format(len(paralist)))

def get_mean_emb_vec(emb_vec, para):
    return np.mean(np.array([np.mean(np.mean(emb_vec[para][i], axis=0), axis=0) for i in range(len(emb_vec[para]))]), axis=0)

parser = argparse.ArgumentParser(description="Generate ELMo embeddings for paragraphs")
parser.add_argument("-pp", "--page_paras", required=True, help="Path to page-paras file")
parser.add_argument("-pt", "--para_text", required=True, help="Path to para-text dict file")
parser.add_argument("-tn", "--thread_count", type=int, required=True, help="No of threads in Thread pool")
parser.add_argument("-o", "--out", required=True, help="Path to output file")
args = vars(parser.parse_args())
page_paras_file = args["page_paras"]
para_text_file = args["para_text"]
thread_count = args["thread_count"]
outfile = args["out"]
with open(para_text_file, 'r') as pt:
    paratext = json.load(pt)
with open(page_paras_file, 'r') as pp:
    page_paras = json.load(pp)
preproc_paratext_dict = preprocessed_paratext(paratext)
para_embed_dict = dict()
print("Data loaded")

paras_in_page = []
for page in page_paras.keys():
    paras_in_page.append(page_paras[page])
with ThreadPool(nodes=thread_count) as pool:
    pool.map(get_elmo_embeddings, paras_in_page)

np.save(outfile, para_embed_dict)