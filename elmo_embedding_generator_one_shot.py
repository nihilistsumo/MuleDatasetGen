#!/usr/bin/python3

import math, json, os, sys, argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import spacy
import logging

#######################################
#
# Single swipe through para list
#
#######################################

def preprocess_text(paratext):
    text = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')  # get rid of problem chars
    text = ' '.join(text.split())
    return text

def get_elmo_embed_paras(paras, para_text_dict, nlp, embed, embed_style="def"):
    print(str(len(paras))+" total paras")
    paraid_embed_index = dict()
    # sentence for padding
    pad_sentence = " "
    para_sentences = [pad_sentence]
    para_sent_index = 1
    for para in paras:
        paratext = str(para_text_dict[para])
        text = preprocess_text(paratext)
        doc = nlp(text)
        for i in doc.sents:
            if len(i) > 1:
                para_sentences.append(i.string.strip())
                if para in paraid_embed_index.keys():
                    paraid_embed_index[para].append(para_sent_index)
                else:
                    paraid_embed_index[para] = [para_sent_index]
                para_sent_index += 1
    print(str(len(para_sentences)-1) + " total sentences")
    embed_dict = embed(para_sentences, signature="default", as_dict=True)
    if embed_dict == 'concat':
        wemb = embed_dict["word_emb"]
        lstm1 = embed_dict["lstm_outputs1"]
        lstm2 = embed_dict["lstm_outputs2"]
        embeddings = tf.concat([wemb, lstm1, lstm2], axis=2)
    else:
        embeddings = embed_dict["default"]

    print("Starting tensorflow session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        embed_vecs = sess.run(embeddings)
    return embed_vecs, paraid_embed_index

def main():
    parser = argparse.ArgumentParser(description='Create ELMo paragraph embedding from a list of paragraphs.')
    parser.add_argument("-pl", "--para_list", required=True, help="List of paragraphs")
    parser.add_argument("-pt", "--para_text", required=True, help="Para text file")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("-es", "--embed_style", required=True, help="Style of embedding (def/concat)")
    parser.add_argument("-tfc", "--tf_cache_dir", help="Directory to tf cache")
    args = vars(parser.parse_args())
    para_list = args["para_list"]
    para_text_file = args["para_text"]
    outdir = args["outdir"]
    emb_style = args["embed_style"]
    if args["tf_cache_dir"] != None:
        tf_cache_dir_path = args["tf_cache_dir"]
        os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir_path
    with open(para_list, 'r') as pl:
        paras = json.load(pl)
    with open(para_text_file, 'r') as pt:
        para_text_dict = json.load(pt)

    logging.getLogger('tensorflow').disabled = True
    nlp = spacy.load('en_core_web_md')
    url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(url)

    embed_vecs, paraid_embed_index = get_elmo_embed_paras(paras, para_text_dict, nlp, embed, emb_style)
    print("Done")
    np.save(outdir+"/embeddings_vecs", embed_vecs)
    np.save(outdir+"/paraid_embed_index", paraid_embed_index)

if __name__ == '__main__':
    main()