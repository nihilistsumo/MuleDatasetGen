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
    out_file = sys.argv[5]
    train_elmo_lookup = np.load(train_elmo_lookup_file, allow_pickle=True)
    with open(train_parapair_file, 'r') as ppd:
        train_parapair_data = json.load(ppd)
    test_elmo_lookup = np.load(test_elmo_lookup_file, allow_pickle=True)
    with open(test_parapair_file, 'r') as ppdt:
        test_parapair_data = json.load(ppdt)
    elmo_vec_len = len(train_elmo_lookup[()][list(train_elmo_lookup[()].keys())[0]][0])
    print("ELMo vector length: {}".format(elmo_vec_len))
    train_labels = train_parapair_data['labels']
    train_sequences = []
    for pp in train_parapair_data['parapairs']:
        train_sequences.append(get_sequence_vec_parapair(pp, train_elmo_lookup, MAX_SENT_COUNT * elmo_vec_len))
    train_sequences = np.array(train_sequences)
    test_labels = test_parapair_data['labels']
    test_sequences = []
    for pp in test_parapair_data['parapairs']:
        test_sequences.append(get_sequence_vec_parapair(pp, test_elmo_lookup, MAX_SENT_COUNT * elmo_vec_len))
    test_sequences = np.array(test_sequences)
    sequence_data = dict()
    sequence_data['Xtrain'] = train_sequences
    sequence_data['ytrain'] = train_labels
    sequence_data['Xtest'] = test_sequences
    sequence_data['ytest'] = test_labels
    np.save(out_file, sequence_data)

if __name__ == '__main__':
    main()