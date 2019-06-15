#!/usr/bin/python3

import random, json, sys, math, argparse
import numpy as np
import parapir_generator

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
    test_labels = []
    test_pairs = []
    for p in parapair_data.keys():
        test_pairs = test_pairs + parapair_data[p]['parapairs']
        test_labels = test_labels + parapair_data[p]['labels']
    test_pos_pairs = [p for p in test_pairs if test_labels[test_pairs.index(p)] == 1]
    test_neg_pairs = [p for p in test_pairs if test_labels[test_pairs.index(p)] == 0]
    pair_shuffle_random(test_pos_pairs)
    pair_shuffle_random(test_neg_pairs)
    assert (len(test_pos_pairs) <= len(test_neg_pairs))

    test_neg_pairs = random.sample(test_neg_pairs, len(test_pos_pairs))
    test_select_pairs = test_pos_pairs + test_neg_pairs
    random.shuffle(test_select_pairs)

    test_sequences = []
    # print("No. of available parapairs from parapair data: {}".format(len(parapair_data['parapairs'])))
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

    for p in random.sample(range(len(test_select_pairs)), 10):
        p1 = test_select_pairs[p].split("_")[0]
        p2 = test_select_pairs[p].split("_")[1]
        p1vec = test_sequences[p][:25600]
        p2vec = test_sequences[p][25600:51200]
        p1vec_elmo = np.array(elmo_lookup[()][p1]).flatten()
        p2vec_elmo = np.array(elmo_lookup[()][p2]).flatten()
        min_p1_len = min(len(p1vec), len(p1vec_elmo))
        min_p2_len = min(len(p2vec), len(p2vec_elmo))
        assert np.allclose(p1vec[:min_p1_len], p1vec_elmo[:min_p1_len]) and np.allclose(p2vec[:min_p2_len], p2vec_elmo[:min_p2_len])

        # assert np.allclose(p1vec[:p1vec_elmo.size], p1vec_elmo) and np.allclose(p2vec[:p2vec_elmo.size], p2vec_elmo)

    return test_sequences, test_select_pairs

def pair_shuffle_random(parapair_list):
    for i in random.sample(range(len(parapair_list)), len(parapair_list)//2):
        p1 = parapair_list[i].split("_")[0]
        p2 = parapair_list[i].split("_")[1]
        parapair_list[i] = p2+"_"+p1

def create_train_data(parapair_data, elmo_lookup, page_paras, train_val_split = 0.5, num_neg_diff_page_count=0):
    TRAIN_PAGE_COUNT = math.floor(len(parapair_data.keys()) * train_val_split)
    VALIDATION_PAGE_COUNT = len(parapair_data.keys()) - TRAIN_PAGE_COUNT

    elmo_vec_len = len(elmo_lookup[()][list(elmo_lookup[()].keys())[0]][0])
    print("ELMo vector length: {}".format(elmo_vec_len))
    train_pages = random.sample(parapair_data.keys(), TRAIN_PAGE_COUNT)
    validation_pages = [p for p in parapair_data.keys() if p not in train_pages]
    if num_neg_diff_page_count > 0:
        print("Adding {} negative samples from different pages".format(num_neg_diff_page_count))
        neg_diff_page_pairs = \
            parapir_generator.get_random_neg_parapairs_different_page(page_paras, train_pages, num_neg_diff_page_count)
    train_labels = []
    train_pairs = []
    for p in train_pages:
        train_pairs = train_pairs + parapair_data[p]['parapairs']
        train_labels = train_labels + parapair_data[p]['labels']
    train_pos_pairs = [p for p in train_pairs if train_labels[train_pairs.index(p)] == 1]
    train_neg_pairs = [p for p in train_pairs if train_labels[train_pairs.index(p)] == 0]
    pair_shuffle_random(train_pos_pairs)
    pair_shuffle_random(train_neg_pairs)
    pair_shuffle_random(neg_diff_page_pairs)

    v_labels = []
    v_pairs = []
    for p in validation_pages:
        v_pairs = v_pairs + parapair_data[p]['parapairs']
        v_labels = v_labels + parapair_data[p]['labels']
    v_pos_pairs = [p for p in v_pairs if v_labels[v_pairs.index(p)] == 1]
    v_neg_pairs = [p for p in v_pairs if v_labels[v_pairs.index(p)] == 0]
    pair_shuffle_random(v_pos_pairs)
    pair_shuffle_random(v_neg_pairs)
    assert len(v_pos_pairs) <= len(v_neg_pairs)
    num_tr_pos = len(train_pos_pairs)
    num_tr_neg = num_tr_pos - len(neg_diff_page_pairs)
    num_v_pos_neg = len(v_pos_pairs)

    # train_pos_pairs = random.sample(train_pos_pairs, num_tr_pos)
    train_neg_pairs = random.sample(train_neg_pairs, num_tr_neg)
    # v_pos_pairs = [p for p in train_pos_pairs if p not in train_pos_pairs]
    v_neg_pairs = random.sample(v_neg_pairs, num_v_pos_neg)
    train_select_pairs = train_pos_pairs + train_neg_pairs + neg_diff_page_pairs
    validation_select_pairs = v_pos_pairs + v_neg_pairs
    random.shuffle(train_select_pairs)
    random.shuffle(validation_select_pairs)

    train_sequences = []
    # print("No. of available parapairs from parapair data: {}".format(len([p for p parapair_data['parapairs'])))
    print("No. of negative parapairs from different pages: {}".format(len(neg_diff_page_pairs)))
    print("Train set\n=========")
    print("No of +ve samples: {}".format(num_tr_pos) +
          "\nNo of -ve samples same page: {}".format(num_tr_neg) +
          "\nNo of -ve samples diff page: {}".format(len(neg_diff_page_pairs)))
    print("No of selected parapairs: {}".format(len(train_select_pairs)))
    for i in range(len(train_select_pairs)):
        pp = train_select_pairs[i]
        if pp in train_pos_pairs:
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
        if pp in v_pos_pairs:
            label = 1
        else:
            label = 0
        seq_vec = get_sequence_vec_parapair(pp, elmo_lookup, MAX_SENT_COUNT * elmo_vec_len)
        seq_vec = np.append(seq_vec, label)
        validation_sequences.append(seq_vec)
        if i % 1000 == 0:
            print(".")
    validation_sequences = np.array(validation_sequences)

    for p in random.sample(range(len(train_select_pairs)), 10):
        p1 = train_select_pairs[p].split("_")[0]
        p2 = train_select_pairs[p].split("_")[1]
        p1vec = train_sequences[p][:25600]
        p2vec = train_sequences[p][25600:51200]
        p1vec_elmo = np.array(elmo_lookup[()][p1]).flatten()
        p2vec_elmo = np.array(elmo_lookup[()][p2]).flatten()
        min_p1_len = min(len(p1vec), len(p1vec_elmo))
        min_p2_len = min(len(p2vec), len(p2vec_elmo))
        assert np.allclose(p1vec[:min_p1_len], p1vec_elmo[:min_p1_len]) and np.allclose(p2vec[:min_p2_len], p2vec_elmo[:min_p2_len])

    for p in random.sample(range(len(validation_select_pairs)), 10):
        p1 = validation_select_pairs[p].split("_")[0]
        p2 = validation_select_pairs[p].split("_")[1]
        p1vec = validation_sequences[p][:25600]
        p2vec = validation_sequences[p][25600:51200]
        p1vec_elmo = np.array(elmo_lookup[()][p1]).flatten()
        p2vec_elmo = np.array(elmo_lookup[()][p2]).flatten()
        min_p1_len = min(len(p1vec), len(p1vec_elmo))
        min_p2_len = min(len(p2vec), len(p2vec_elmo))
        assert np.allclose(p1vec[:min_p1_len], p1vec_elmo[:min_p1_len]) and np.allclose(p2vec[:min_p2_len], p2vec_elmo[:min_p2_len])

    return train_sequences, train_select_pairs, validation_sequences, validation_select_pairs

def main():
    # train_parapair_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/train-cleaned-parapairs/by1-train-cleaned-foodpages.parapairs.json"
    # train_elmo_lookup_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1train-nodup-elmo-vec-data/by1train_merged_elmo_squeezed_para_vec_lookup.npy"
    # test_parapair_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned-foodpages.parapairs.json"
    # test_elmo_lookup_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-nodup-elmo-vec-data/by1test_merged_elmo_squeezed_para_vec_lookup.npy"
    # out_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/by1_elmo_seq_data"
    parser = argparse.ArgumentParser(description='Create dataset suitable for direct training of MaLSTM.')
    parser.add_argument("-trp", "--train-pair", required=True, help="Path to train parapair json")
    parser.add_argument("-tp", "--test-pair", required=True, help="Path to test parapair json")
    parser.add_argument("-tre", "--train-elmo", required=True, help="Path to train ELMo lookup np file")
    parser.add_argument("-te", "--test-elmo", required=True, help="Path to test ELMo lookup np file")
    parser.add_argument("-pp", "--page-paras", required=True, help="Path to train page paras file")
    parser.add_argument("-n", "--neg-diff", type=int, required=True, help="No of negative parapair samples from different pages")
    parser.add_argument("-tv", "--train_val_split", type=float, required=True, help="Fraction of train/validation split")
    parser.add_argument("-o", "--out", required=True, help="Path to output directory")
    args = vars(parser.parse_args())
    train_parapair_file = args["train_pair"]
    train_elmo_lookup_file = args["train_elmo"]
    test_parapair_file = args["test_pair"]
    test_elmo_lookup_file = args["test_elmo"]
    page_paras_file = args["page_paras"]
    neg_pairs = args["neg_diff"]
    split_frac = args["train_val_split"]
    output_dir = args["out"]
    train_elmo_lookup = np.load(train_elmo_lookup_file, allow_pickle=True)
    with open(train_parapair_file, 'r') as ppd:
        train_parapair_data = json.load(ppd)
    test_elmo_lookup = np.load(test_elmo_lookup_file, allow_pickle=True)
    with open(test_parapair_file, 'r') as ppdt:
        test_parapair_data = json.load(ppdt)
    with open(page_paras_file, 'r') as n:
        page_paras = json.load(n)

    train_sequences, train_select_pairs, validation_sequences, validation_select_pairs = \
        create_train_data(train_parapair_data, train_elmo_lookup, page_paras, split_frac, neg_pairs)
    test_sequences, test_select_pairs = create_test_data(test_parapair_data, test_elmo_lookup)
    
    # seq_data = dict()
    # seq_data['train_data'] = train_sequences
    # seq_data['train_parapairs'] = train_select_pairs
    # seq_data['val_data'] = validation_sequences
    # seq_data['val_parapairs'] = validation_select_pairs
    # seq_data['test_data'] = test_sequences
    # seq_data['test_parapairs'] = test_select_pairs

    print("Going to save train data, size: {} GB".format(train_sequences.nbytes/(1024 * 1024 * 1024)))
    np.save(output_dir + "/train_data", train_sequences)
    print("Going to save vaidation data, size: {} GB".format(validation_sequences.nbytes / (1024 * 1024 * 1024)))
    np.save(output_dir + "/val_data", validation_sequences)
    print("Going to save test data, size: {} GB".format(test_sequences.nbytes / (1024 * 1024 * 1024)))
    np.save(output_dir + "/test_data", test_sequences)
    np.save(output_dir + "/train_parapair_list", train_select_pairs)
    np.save(output_dir + "/val_parapair_list", validation_select_pairs)
    np.save(output_dir + "/test_parapair_list", test_select_pairs)


if __name__ == '__main__':
    main()