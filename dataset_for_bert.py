import argparse, json, random, math

import numpy as np

def get_pairdata(train_parapair_dict, test_parapair_dict, train_val_ratio, train_pair_file, val_pair_file, test_pair_file):
    train_pages = train_parapair_dict.keys()
    val_pages = random.sample(train_pages, math.floor(len(train_pages) * train_val_ratio))
    train_pages = train_pages - val_pages
    test_pages = test_parapair_dict.keys()

    with open(train_pair_file, 'w') as trp:
        for page in train_pages:
            pairs = train_parapair_dict[page]["parapairs"]
            labels = train_parapair_dict[page]["labels"]
            for i in range(len(labels)):
                p1 = pairs[i].split("_")[0]
                p2 = pairs[i].split("_")[1]
                trp.write(str(labels[i])+"\t"+p1+"\t"+p2+"\n")

    with open(val_pair_file, 'w') as vp:
        for page in val_pages:
            pairs = train_parapair_dict[page]["parapairs"]
            labels = train_parapair_dict[page]["labels"]
            for i in range(len(labels)):
                p1 = pairs[i].split("_")[0]
                p2 = pairs[i].split("_")[1]
                vp.write(str(labels[i])+"\t"+p1+"\t"+p2+"\n")

    with open(test_pair_file, 'w') as tp:
        for page in test_pages:
            pairs = test_parapair_dict[page]["parapairs"]
            labels = test_parapair_dict[page]["labels"]
            for i in range(len(labels)):
                p1 = pairs[i].split("_")[0]
                p2 = pairs[i].split("_")[1]
                tp.write(str(labels[i])+"\t"+p1+"\t"+p2+"\n")

def produce_pair_data():
    train_parapair_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/train-cleaned-parapairs/by1-train-cleaned.parapairs.json"
    test_parapair_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned.parapairs.json"
    train_val_rat = 0.8
    outdir = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/pair_data_for_bert"
    with open(train_parapair_file, 'r') as trp:
        train_parapair = json.load(trp)
    with open(test_parapair_file, 'r') as tp:
        test_parapair = json.load(tp)
    get_pairdata(train_parapair, test_parapair, train_val_rat, outdir+"/train_pair", outdir+"/val_pair", outdir+"/test_pair")

def produce_bert_input_file(pair_file, paratext_dict, outfile):
    pair_lines = [line.rstrip('\n') for line in open(pair_file)]
    random.shuffle(pair_lines)
    print("{} lines to process".format(len(pair_lines)))
    count = 0
    with open(outfile, 'w') as out:
        for l in pair_lines:
            label = l.split("\t")[0]
            p1 = l.split("\t")[1]
            p2 = l.split("\t")[2]
            p1text = paratext_dict[p1]
            p2text = paratext_dict[p2]
            if random.random() > 0.5:
                out.write(label + "\t" + p1 + "\t" + p2 + "\t" + p1text + "\t" + p2text + "\n")
            else:
                out.write(label + "\t" + p2 + "\t" + p1 + "\t" + p2text + "\t" + p1text + "\n")
            count += 1
            if count % 10000:
                print(".")

def main():
    parser = argparse.ArgumentParser(description="Generate parapair file suitable for BERT")
    parser.add_argument("-p", "--pair", required=True, help="Path to pair file")
    parser.add_argument("-pt", "--paratext", required=True, help="Path to paratext file")
    parser.add_argument("-o", "--out", required=True, help="Path to output file")
    args = vars(parser.parse_args())
    pair_file = args["pair"]
    paratext_file = args["paratext"]
    outfile = args["out"]

    with open(paratext_file, 'r') as pt:
        paratext = json.load(pt)
    produce_bert_input_file(pair_file, paratext, outfile)

if __name__ == '__main__':
    main()