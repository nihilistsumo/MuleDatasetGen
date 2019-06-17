#!/usr/bin/python3

import math, json, os, sys, argparse
import numpy as np
from scipy.spatial import distance
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
import spacy
from spacy.lang.en import English
from spacy import displacy
import logging

#######################################
#
# Page split wise calc
#
#######################################

def preprocess_text(paratext):
    text = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')  # get rid of problem chars
    text = ' '.join(text.split())
    return text

def get_elmo_embed_paras_in_page(page, page_paras, para_text_dict, nlp, embed, embed_style="def"):
    paraids = []
    for p in page_paras[page]:
        paraids.append(p)
    para_sentences = []
    para_sent_keys = []
    for para in paraids:
        paratext = str(para_text_dict[para])
        text = preprocess_text(paratext)
        doc = nlp(text)
        sent_count = 0
        for i in doc.sents:
            if len(i) > 1:
                para_sentences.append(i.string.strip())
                para_sent_keys.append(para+"_"+str(sent_count))
                sent_count += 1
    embed_dict = embed(para_sentences, signature="default", as_dict=True)
    if embed_style == 'concat':
        wemb = embed_dict["word_emb"]
        lstm1 = embed_dict["lstm_outputs1"]
        lstm2 = embed_dict["lstm_outputs2"]
        embeddings = tf.concat([wemb, lstm1, lstm2], axis=2)
    else:
        embeddings = embed_dict["default"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sentence_vecs = sess.run(embeddings)
    assert len(para_sent_keys) == sentence_vecs.shape[0]
    para_embeddings = dict()
    for i in range(len(para_sent_keys)):
        para_embeddings[para_sent_keys[i]] = sentence_vecs[i]
    return para_embeddings

def get_embeddings(pages, page_paras, para_text_dict, emb_style):
    logging.getLogger('tensorflow').disabled = True
    embed_data = dict()
    nlp = spacy.load('en_core_web_md')
    url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(url)
    for page in page_paras.keys():
        if page not in pages:
            continue
        print(page)
        para_embeddings_in_page = get_elmo_embed_paras_in_page(page, page_paras, para_text_dict, nlp, embed, emb_style)
        embed_data[page] = para_embeddings_in_page
    return embed_data

def main():
    # by1train_nodup_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.para.texts.json"
    # page_paras_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.paras.json"
    # page_para_labels_json = "/home/sumanta/Documents/Dugtrio-data/AttnetionWindowData/by1train-nodup.json.data/by1-train-nodup.page.para.labels.json"
    parser = argparse.ArgumentParser(description='Create ELMo paragraph embedding from a list of paragraphs.')
    parser.add_argument("-pp", "--page_parts", required=True, help="Parts of pages/ File containing list of pages")
    parser.add_argument("-pt", "--para_text", required=True, help="Para text file")
    parser.add_argument("-pgp", "--page_paras", required=True, help="Page paras file")
    parser.add_argument("-o", "--out", required=True, help="Output file")
    parser.add_argument("-es", "--embed_style", required=True, help="Style of embedding (def/concat)")
    parser.add_argument("-tfc", "--tf_cache_dir", help="Directory to tf cache")
    args = vars(parser.parse_args())
    pages_part_file = args["page_parts"]
    para_text_json = args["para_text"]
    page_paras_json = args["page_paras"]
    elmo_out_file = args["out"]
    emb_style = args["embed_style"]
    if args["tf_cache_dir"] != None:
        tf_cache_dir_path = args["tf_cache_dir"]
        os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir_path

    pages = []
    with open(pages_part_file, 'r') as p_part:
        for l in p_part:
            pages.append(l.rstrip("\r\n"))
    with open(para_text_json, 'r') as by:
        para_text_dict = json.load(by)
    with open(page_paras_json, 'r') as pp:
        page_paras = json.load(pp)

    embed_data = np.array(get_embeddings(pages, page_paras, para_text_dict, emb_style))
    np.save(elmo_out_file, embed_data)
    
    print("Done")

if __name__ == '__main__':
    main()