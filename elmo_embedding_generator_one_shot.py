#!/usr/bin/python3

import math, json, os, sys, argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import spacy
import logging
from pathos.multiprocessing import Pool

#######################################
#
# Single swipe through para list
#
#######################################

def construct_para_embedding(para):
    def preprocess_text(paratext):
        text = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')  # get rid of problem chars
        text = ' '.join(text.split())
        return text
    paratext = str(para_text_dict[para])
    text = preprocess_text(paratext)
    doc = nlp(text)
    sentences = []
    for i in doc.sents:
        if len(i) > 1:
            sentences.append(i.string.strip())
    print(para + ": {}".format(len(sentences)) + " sentences")
    embed_dict = embed(sentences, signature="default", as_dict=True)
    if emb_style == 'concat':
        wemb = embed_dict["word_emb"]
        lstm1 = embed_dict["lstm_outputs1"]
        lstm2 = embed_dict["lstm_outputs2"]
        para_embedding = tf.concat([wemb, lstm1, lstm2], axis=2)
    else:
        para_embedding = embed_dict["default"]
    embed_vecs[para] = para_embedding

def get_elmo_embed_paras(paras):

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    return embedding_dict


parser = argparse.ArgumentParser(description='Create ELMo paragraph embedding from a list of paragraphs.')
parser.add_argument("-pl", "--para_list", required=True, help="List of paragraphs")
parser.add_argument("-pt", "--para_text", required=True, help="Para text file")
parser.add_argument("-o", "--out", required=True, help="Output file")
parser.add_argument("-es", "--embed_style", required=True, help="Style of embedding (def/concat)")
parser.add_argument("-tfc", "--tf_cache_dir", help="Directory to tf cache")
parser.add_argument("-pn", "--no_processes", type=int, required=True, help="No of parallel processes")
args = vars(parser.parse_args())
para_list = args["para_list"]
para_text_file = args["para_text"]
elmo_out_file = args["out"]
emb_style = args["embed_style"]
pno = args["no_processes"]
if args["tf_cache_dir"] != None:
    tf_cache_dir_path = args["tf_cache_dir"]
    os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir_path
with open(para_list, 'r') as pl:
    paras = json.load(pl)
with open(para_text_file, 'r') as pt:
    para_text_dict = json.load(pt)

print("Data loaded")
logging.getLogger('tensorflow').disabled = True
nlp = spacy.load('en_core_web_md')
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)
print("Tensorflow-hub loaded")

print(str(len(paras))+" total paras")
embed_vecs = dict()

with Pool(pno) as pool:
    pool.map(construct_para_embedding, paras)

print("Starting tensorflow session...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    embedding_dict = sess.run(embed_vecs)
print("Done")
np.save(elmo_out_file, embed_vecs)