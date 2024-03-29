#!/usr/bin/python3

import random, json, sys
import numpy as np

def get_random_neg_parapairs_different_page(page_paras, pagelist, count):
    neg_parapairs = []
    while(len(neg_parapairs) < count):
        pages = random.sample(pagelist, 2)
        para1 = str(random.sample(page_paras[pages[0]], 1)[0])
        para2 = str(random.sample(page_paras[pages[1]], 1)[0])
        pp1 = para1+"_"+para2
        pp2 = para2+"_"+para1
        if pp1 not in neg_parapairs and pp2 not in neg_parapairs:
            neg_parapairs.append(para1+"_"+para2)
    return neg_parapairs

def generate_parapairs(page_paras, page_para_labels_file, pages=[]):
    if len(pages) == 0:
        pages = list(page_paras.keys())
    page_parapairs = dict()
    with open(page_para_labels_file, 'r') as pl:
        page_para_labels = json.load(pl)
    for page in pages:
        print(page)
        paras_in_page = page_paras[page]
        parapairs = []
        labels = []
        for i in range(len(paras_in_page)-1):
            for j in range(i+1, len(paras_in_page)):
                pi = paras_in_page[i]
                pj = paras_in_page[j]
                parapairs.append(pi+"_"+pj)
                if page_para_labels[page][pi] == page_para_labels[page][pj]:
                    labels.append(1)
                else:
                    labels.append(0)
        page_parapairs[page] = {'parapairs':parapairs, 'labels':labels}
        assert len(parapairs) == len(labels)
    return page_parapairs

def main():
    page_paras_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1train-cleaned.json.data/by1-train-cleaned.page.paras.json"
    page_para_labels_file = "/home/sumanta/Documents/Mule-data/input_data_v2/by1train-cleaned.json.data/by1-train-cleaned.page.para.labels.json"
    output_file = "/home/sumanta/Documents/Mule-data/input_data_v2/pairs/by1-train-cleaned-tiny-foodpages.parapairs.json"

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
    tiny_train_food_pages = ["enwiki:Sugar", "enwiki:Smoothie", "enwiki:Blueberry", "enwiki:Espresso"]
    tiny_test_food_pages = ["enwiki:Avocado", "enwiki:Bagel"]

    with open(page_paras_file, 'r') as pp:
        page_paras = json.load(pp)
    page_parapairs = generate_parapairs(page_paras, page_para_labels_file, tiny_train_food_pages)
    with open(output_file, 'w') as out:
        json.dump(page_parapairs, out)

if __name__ == '__main__':
    main()