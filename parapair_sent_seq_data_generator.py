#!/usr/bin/python3

import random, json, sys, math
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

def create_test_data(parapair_data, elmo_lookup):
    elmo_vec_len = len(elmo_lookup[()][list(elmo_lookup[()].keys())[0]][0])
    print("ELMo vector length: {}".format(elmo_vec_len))
    test_labels = parapair_data['labels']
    test_pairs = parapair_data['parapairs']
    test_pos_pairs = [p for p in test_pairs if test_labels[test_pairs.index(p)] == 1]
    neg_pairs = [p for p in test_pairs if test_labels[test_pairs.index(p)] == 0]
    assert (len(test_pos_pairs) <= len(neg_pairs))

    test_neg_pairs = random.sample(neg_pairs, len(test_pos_pairs))
    test_select_pairs = test_pos_pairs + test_neg_pairs
    random.shuffle(test_select_pairs)

    test_sequences = []
    print("No. of available parapairs from parapair data: {}".format(len(parapair_data['parapairs'])))
    print("Test set\n========")
    print("No of +ve samples: {}".format(len(test_pos_pairs)) +
          "\nNo of -ve samples same page: {}".format(len(test_neg_pairs)))
    print("No of selected parapairs: {}".format(len(test_select_pairs)))
    for i in range(len(test_select_pairs)):
        pp = test_select_pairs[i]
        if pp in test_pos_pairs:
            label = 1
        else:
            label = 0
        seq_vec = get_sequence_vec_parapair(pp, elmo_lookup, MAX_SENT_COUNT * elmo_vec_len)
        seq_vec = np.append(seq_vec, label)
        test_sequences.append(seq_vec)
        if i % 1000 == 0:
            print(".")
    test_sequences = np.array(test_sequences)
    return test_sequences, test_select_pairs

def create_train_data(parapair_data, elmo_lookup, neg_diff_page_pairs):
    elmo_vec_len = len(elmo_lookup[()][list(elmo_lookup[()].keys())[0]][0])
    print("ELMo vector length: {}".format(elmo_vec_len))
    train_labels = parapair_data['labels']
    train_pairs = parapair_data['parapairs']
    pos_pairs = [p for p in train_pairs if train_labels[train_pairs.index(p)] == 1]
    neg_pairs = [p for p in train_pairs if train_labels[train_pairs.index(p)] == 0]
    assert(len(pos_pairs) <= len(neg_pairs))
    num_tr_pos = math.floor(len(pos_pairs) * 0.8)
    num_tr_neg = num_tr_pos - len(neg_diff_page_pairs)
    num_v_pos_neg = len(pos_pairs) - num_tr_pos

    train_pos_pairs = random.sample(pos_pairs, num_tr_pos)
    train_neg_pairs = random.sample(neg_pairs, num_tr_neg)
    validation_pos_pairs = [p for p in pos_pairs if p not in train_pos_pairs]
    validation_neg_pairs = random.sample([p for p in neg_pairs if p not in train_neg_pairs], num_v_pos_neg)
    train_select_pairs = train_pos_pairs + train_neg_pairs + neg_diff_page_pairs
    validation_select_pairs = validation_pos_pairs + validation_neg_pairs
    random.shuffle(train_select_pairs)
    random.shuffle(validation_select_pairs)

    train_sequences = []
    print("No. of available parapairs from parapair data: {}".format(len(parapair_data['parapairs'])))
    print("No. of negative parapairs from different pages: {}".format(len(neg_diff_page_pairs)))
    print("Train set\n=========")
    print("No of +ve samples: {}".format(num_tr_pos) +
          "\nNo of -ve samples same page: {}".format(num_tr_neg) +
          "\nNo of -ve samples diff page: {}".format(len(neg_diff_page_pairs)))
    print("No of selected parapairs: {}".format(len(train_select_pairs)))
    for i in range(len(train_select_pairs)):
        pp = train_select_pairs[i]
        if pp in pos_pairs:
            label = 1
        else:
            label = 0
        seq_vec = get_sequence_vec_parapair(pp, elmo_lookup, MAX_SENT_COUNT * elmo_vec_len)
        seq_vec = np.append(seq_vec, label)
        train_sequences.append(seq_vec)
        if i % 1000 == 0:
            print(".")
    train_sequences = np.array(train_sequences)

    validation_sequences = []
    print("Validation set\n==============")
    print("No of +ve samples: {}".format(num_v_pos_neg) +
          "\nNo of -ve samples same page: {}".format(num_v_pos_neg))
    print("No of selected parapairs: {}".format(len(validation_select_pairs)))
    for i in range(len(validation_select_pairs)):
        pp = validation_select_pairs[i]
        if pp in pos_pairs:
            label = 1
        else:
            label = 0
        seq_vec = get_sequence_vec_parapair(pp, elmo_lookup, MAX_SENT_COUNT * elmo_vec_len)
        seq_vec = np.append(seq_vec, label)
        validation_sequences.append(seq_vec)
        if i % 1000 == 0:
            print(".")
    validation_sequences = np.array(validation_sequences)
    return train_sequences, train_select_pairs, validation_sequences, validation_select_pairs

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