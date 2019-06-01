#!/usr/bin/python3

import random, json, sys
import numpy as np

def generate_parapairs(page_paras, page_para_labels_file, pages=[]):
    if len(pages) == 0:
        pages = list(page_paras.keys())
    parapairs = []
    labels = []
    with open(page_para_labels_file, 'r') as pl:
        page_para_labels = json.load(pl)
    for page in pages:
        print(page)
        paras_in_page = page_paras[page]
        for i in range(len(paras_in_page)-1):
            for j in range(i+1, len(paras_in_page)):
                pi = paras_in_page[i]
                pj = paras_in_page[j]
                parapairs.append(pi+"_"+pj)
                if page_para_labels[page][pi] == page_para_labels[page][pj]:
                    labels.append(1)
                else:
                    labels.append(0)
    assert len(parapairs) == len(labels)
    return parapairs, labels

def main():
    page_paras_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned.page.paras.json"
    page_para_labels_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1test-cleaned.json.data/by1-test-cleaned.page.para.labels.json"
    output_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/by1-test-cleaned-foodpages.parapairs.json"

    train_food_pages = ["enwiki:Carbohydrate",
"enwiki:Chocolate",
"enwiki:Flavor",
"enwiki:Sugar",
"enwiki:Smoothie",
"enwiki:Blueberry",
"enwiki:Espresso",
"enwiki:Junk%20food",
"enwiki:Candy%20making",
"enwiki:Yogurt",
"enwiki:Halva",
"enwiki:Food%20security",
"enwiki:Egg%20white",
"enwiki:Wedding%20cake",
"enwiki:Condensed%20milk",
"enwiki:Coffee",
"enwiki:Chocolate%20chip"]
    test_food_pages = ["enwiki:Mexican%20cuisine",
"enwiki:Olive%20oil",
"enwiki:Superfood",
"enwiki:Coffee%20preparation",
"enwiki:Aztec%20cuisine",
"enwiki:Quinoa",
"enwiki:Cocoa%20bean",
"enwiki:Chili%20pepper",
"enwiki:Taste",
"enwiki:Health%20effects%20of%20chocolate",
"enwiki:Avocado",
"enwiki:Bagel",
"enwiki:Instant%20coffee",
"enwiki:Christmas%20pudding",
"enwiki:Hot%20chocolate",
"enwiki:Fudge",
"enwiki:Cocoa%20butter"]

    with open(page_paras_file, 'r') as pp:
        page_paras = json.load(pp)
    parapairs, labels = generate_parapairs(page_paras, page_para_labels_file, test_food_pages)
    output_dict = {'parapairs':parapairs, 'labels':labels}
    with open(output_file, 'w') as out:
        json.dump(output_dict, out)

if __name__ == '__main__':
    main()