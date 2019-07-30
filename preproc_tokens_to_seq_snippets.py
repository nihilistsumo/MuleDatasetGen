import numpy as np
import json, argparse

# Following code converts preproc tokens file to its corresponding seq file
# Instead of token we now have either the index + 1 of the token in the vocab list
# or 0 if the token is absent in vocab

def convert_to_seq(tokens_dict, vocab):
    # stem_ts = np.load("/path/to/preproc_token_file")
    # with open("/path/to/vocab_list_file", 'r') as v:
        # vocab = json.load(v)
    stem_ts_seq = dict()
    for p in tokens_dict[()].keys():
        stem_ts_seq[p] = [vocab.index(t)+1 if t in vocab else 0 for t in tokens_dict[()][p]]
    return stem_ts_seq

def produce_vocab_dict(preproc_token_dicts):
    vocab_dict = dict()
    for d in preproc_token_dicts:
        for p in d[()].keys():
            for t in d[()][p]:
                if t in vocab_dict.keys():
                    vocab_dict[t] += 1
                else:
                    vocab_dict[t] = 1
    return vocab_dict

def main():
    parser = argparse.ArgumentParser(description="Script to manipulate preproc token files")
    parser.add_argument("-a", "--action", required=True, choices=["vocab", "seq"], default="seq", help="Create vocab or convert seq")
    parser.add_argument("-tl", "--token_file_list", nargs="+", required=True, help="List of preproc token file paths")
    parser.add_argument("-voc", "--vocab_list", help="Path to vocab list file")
    parser.add_argument("-o", "--out", required=True, help="Path to output file")
    args = vars(parser.parse_args())
    action = args["action"]
    token_file_list = args["token_file_list"]
    vocab_file = args["vocab_list"]
    outfile = args["out"]

    if action == "vocab":
        token_files = []
        for f in token_file_list:
            token_files.append(np.load(f))
        vocab_freq_dict = produce_vocab_dict(token_files)
        vocab_sorted = sorted(vocab_freq_dict, key=vocab_freq_dict.get, reverse=True)
        with open(outfile, 'w') as o:
            json.dump(vocab_sorted, o)
    elif action == "seq":
        preproc_tokens = np.load(token_file_list[0])
        with open(vocab_file, 'r') as v:
            vocab = json.load(v)
        preproc_tokens_seq = convert_to_seq(preproc_tokens, vocab)
        np.save(outfile, preproc_tokens_seq)
    else:
        print("Wrong choice! Choose between [vocab, seq]")


if __name__ == '__main__':
    main()