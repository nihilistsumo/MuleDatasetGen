import numpy as np
import json, argparse, random

def get_hier_pos_pairs(pos_pairs, hier_qrels_rev):
    hier_pos_pairs = [p for p in pos_pairs if hier_qrels_rev[p.split("_")[0]] == hier_qrels_rev[p.split("_")[1]]]
    return hier_pos_pairs

def convert_to_balanced_discrim(parapair_dict, hier_qrels_reversed):
    bal_discrim_parapair_dict = dict()
    for page in parapair_dict.keys():
        parapairs = parapair_dict[page]['parapairs']
        labels = parapair_dict[page]['labels']
        pos_pairs = []
        neg_pairs = []
        for i in range(len(parapairs)):
            if labels[i] == 1:
                pos_pairs.append(parapairs[i])
            else:
                neg_pairs.append(parapairs[i])
        hier_pos_pairs = get_hier_pos_pairs(pos_pairs, hier_qrels_reversed)
        bal_neg_pairs = random.sample(neg_pairs, len(hier_pos_pairs))
        bal_discrim_pairs = hier_pos_pairs + bal_neg_pairs
        bal_discrim_labels = [1] * len(hier_pos_pairs) + [0] * len(bal_neg_pairs)
        bal_discrim_parapair_dict[page] = {'parapairs':bal_discrim_pairs, 'labels':bal_discrim_labels}
        print(page+" done")
    return bal_discrim_parapair_dict

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Dense-Siamese network for paragraph similarity task")
    parser.add_argument("-p", "--parapair", required=True, help="Path to parapair file")
    parser.add_argument("-hq", "--hier_qrels", required=True, help="Path to hierarchical qrels")
    parser.add_argument("-o", "--out", required=True, help="Path to save discriminative balanced parapairs")

    args = vars(parser.parse_args())
    pp_file = args["parapair"]
    hier_qrels_file = args["hier_qrels"]
    out_file = args["out"]

    with open(pp_file, 'r') as trpp:
        parapair = json.load(trpp)

    hier_qrels_reverse = dict()
    with open(hier_qrels_file, 'r') as hq:
        for l in hq:
            hier_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]

    bal_discrim_pair_dict = convert_to_balanced_discrim(parapair, hier_qrels_reverse)

    with open(out_file, 'w') as out:
        json.dump(bal_discrim_pair_dict, out)

if __name__ == '__main__':
    main()