import json, argparse, math, random
import trec_car_y3_conversion
from trec_car_y3_conversion.y3_data import OutlineReader, Page, Paragraph, ParagraphOrigin
import trec_car
from trec_car import read_data
import allennlp
from allennlp.commands.elmo import ElmoEmbedder

def get_outline_text_from_cbor(cbor_filepath):
    outline_text_dict = dict()
    with open(cbor_filepath, 'rb') as cb:
        for page in read_data.iter_annotations(cb):
            print("Page: "+page.page_name)
            sections = []
            for sec in page.outline():
                print(sec.heading)
                sections.append(sec.heading)
            print("=====================================")
            outline_text_dict[page] = sections
    return outline_text_dict

def get_page_sectionwise_elmo_vecs(page_outline_dict):
    outline_elmo_dict = dict()
    elmo = ElmoEmbedder()
    for page in page_outline_dict.keys():
        tokenized_headings = [page.split()]
        sections = page_outline_dict[page]
        for s in sections:
            tokenized_headings.append(s.split())
        embed_vecs = elmo.embed_sentences(tokenized_headings)
        outline_elmo_dict[page] = embed_vecs
    return outline_elmo_dict