#!/usr/bin/python3

import random, json, sys
import numpy as np

MAX_SENT_COUNT = 10

def get_sequence_vec_parapair(parapair, elmo_vec_lookup, sent_seq_len):
    p1 = parapair.split("_")[0]
    p2 = parapair.split("_")[1]
    p1_sent_seqs = np.array(elmo_vec_lookup[()][p1]).flatten()
    p2_sent_seqs = np.array(elmo_vec_lookup[()][p2]).flatten()
    if len(p1_sent_seqs) <= sent_seq_len:
        p1_sent_seqs = np.concatenate((p1_sent_seqs, np.zeros(sent_seq_len - len(p1_sent_seqs))))
    else:
        p1_sent_seqs = p1_sent_seqs[:sent_seq_len]
    if len(p2_sent_seqs) <= sent_seq_len:
        p2_sent_seqs = np.concatenate((p2_sent_seqs, np.zeros(sent_seq_len - len(p2_sent_seqs))))
    else:
        p2_sent_seqs = p2_sent_seqs[:sent_seq_len]
    return np.concatenate((p1_sent_seqs, p2_sent_seqs))

def create_train_data(parapair_data, elmo_lookup, neg_diff_page_pairs):
    elmo_vec_len = len(elmo_lookup[()][list(elmo_lookup[()].keys())[0]][0])
    print("ELMo vector length: {}".format(elmo_vec_len))
    train_labels = np.array(parapair_data['labels'])
    train_labels = train_labels.reshape((train_labels.size, 1))
    train_pairs = np.array(parapair_data['parapairs'])
    train_pairs = train_pairs.reshape((train_pairs.size, 1))
    train_sequences = []
    print("No. of parapairs: {}".format(len(parapair_data['parapairs'])))
    for i in range(len(parapair_data['parapairs'])):
        pp = parapair_data['parapairs'][i]
        seq_vec = np.concatenate(get_sequence_vec_parapair(pp, elmo_lookup, MAX_SENT_COUNT * elmo_vec_len), np.array([train_labels[i], train_pairs[i]]))
        train_sequences.append(seq_vec)
        if i % 1000 == 0:
            print(".")
    train_sequences = np.array(train_sequences)
    train_dat = np.hstack((train_sequences, train_labels, train_pairs))
    return train_dat

def main():
    # train_parapair_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/train-cleaned-parapairs/by1-train-cleaned-foodpages.parapairs.json"
    # train_elmo_lookup_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1train-nodup-elmo-vec-data/by1train_merged_elmo_squeezed_para_vec_lookup.npy"
    # test_parapair_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned-foodpages.parapairs.json"
    # test_elmo_lookup_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-nodup-elmo-vec-data/by1test_merged_elmo_squeezed_para_vec_lookup.npy"
    # out_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/by1_elmo_seq_data"
    train_parapair_file = sys.argv[1]
    train_elmo_lookup_file = sys.argv[2]
    test_parapair_file = sys.argv[3]
    test_elmo_lookup_file = sys.argv[4]
    outdir = sys.argv[5]
    train_elmo_lookup = np.load(train_elmo_lookup_file, allow_pickle=True)
    with open(train_parapair_file, 'r') as ppd:
        train_parapair_data = json.load(ppd)
    test_elmo_lookup = np.load(test_elmo_lookup_file, allow_pickle=True)
    with open(test_parapair_file, 'r') as ppdt:
        test_parapair_data = json.load(ppdt)
    elmo_vec_len = len(train_elmo_lookup[()][list(train_elmo_lookup[()].keys())[0]][0])
    print("ELMo vector length: {}".format(elmo_vec_len))
    train_labels = np.array(train_parapair_data['labels'])
    train_labels = train_labels.reshape((train_labels.size, 1))
    train_sequences = []
    for pp in train_parapair_data['parapairs']:
        train_sequences.append(get_sequence_vec_parapair(pp, train_elmo_lookup, MAX_SENT_COUNT * elmo_vec_len))
    train_sequences = np.array(train_sequences)
    test_labels = np.array(test_parapair_data['labels'])
    test_labels = test_labels.reshape((test_labels.size, 1))
    test_sequences = []
    for pp in test_parapair_data['parapairs']:
        test_sequences.append(get_sequence_vec_parapair(pp, test_elmo_lookup, MAX_SENT_COUNT * elmo_vec_len))
    test_sequences = np.array(test_sequences)
    print("train seq shape {}".format(train_sequences.shape)+" train labels shape {}".format(train_labels.shape))
    train_dat = np.hstack((train_sequences, train_labels))
    test_dat = np.hstack((test_sequences, test_labels))
    np.save(outdir+"/train", train_dat)
    np.save(outdir+"/test", test_dat)

if __name__ == '__main__':
    main()