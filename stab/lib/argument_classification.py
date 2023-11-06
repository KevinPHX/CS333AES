import stanza
import pandas as pd
import re
import numpy as np
from statsmodels.discrete.discrete_model import Poisson
from itertools import groupby
from operator import itemgetter
import sys
from gensim import models

w = models.KeyedVectors.load_word2vec_format(
    './lib/GoogleNews-vectors-negative300.bin', binary=True)

corenlp_dir = '../corenlp'
import os
os.environ["CORENLP_HOME"] = corenlp_dir

from stanza.server import CoreNLPClient
 
class ArgumentClassification():
    def __init__(self, data):
        self.data = data
        # self.components = self.process_data(data)
    def process_data(self, data):
        self.components = []
        component = []
        preceding_tokens = []
        following_tokens = []
        encountered_b = False
        paragraph = None
        sentence = None
        component_stats = {}
        for each in data:
            if each['IOB'] == 'Arg-B' or each['IOB'] == 'Arg-I':
                if not paragraph and not sentence:
                    paragraph = each['paragraph']
                    sentence = each['sentence']

                encountered_b = True
                component.append(each['token'])
            else:
                if not encountered_b:
                    preceding_tokens.append(each['token'])
                else:
                    if not each['token'] == '.':
                        following_tokens.append(each['token'])
                    else:
                        component_stats = self.paragraph_stats(data, each['essay'], paragraph, sentence)
                        fields = {
                            "essay":each['essay'],
                            "component":component,
                            "preceding_tokens":preceding_tokens,
                            "num_preceding":len(preceding_tokens),
                            "following_tokens":following_tokens,
                            "num_following":len(following_tokens),
                            "paragraph":paragraph,
                            "paragraph_size":component_stats['paragraph_size'],
                            "first/last": 1 if component_stats['max_sentence'] == sentence or component_stats['min_sentence'] == sentence else 0,
                            "intro/conc": 1 if each['docPosition'] == 'Introduction' or each['docPosition'] == 'Conclusion' else 0,
                            "sentence":sentence,
                            "sentence_size":component_stats['sentence_size'],
                            "ratio":len(component)/component_stats['sentence_size']
                        }
                        if len(component) > 0:
                            self.components.append(fields)
                        component = []
                        preceding_tokens = []
                        following_tokens = []
                        encountered_b = False
                        paragraph = None
                        sentence = None                

    def paragraph_stats(self, data, essay, paragraph, sentence):
        grouped = groupby(data, itemgetter('essay','paragraph'))
        max_index = 0
        min_index = sys.maxsize
        paragraph_size = 0
        sentence_size = 0
        for group in grouped:
            if group[0][0] == essay and group[0][1] == paragraph:
                for e in group[1]:
                    if e['sentence'] == sentence:
                        sentence_size+=1
                    if e['sentence'] > max_index:
                        max_index = e['sentence']
                    if e['sentence'] < min_index:
                        min_index = e['sentence']
                    paragraph_size += 1
        return {"max_sentence":max_index, "min_sentence":min_index, "paragraph_size":paragraph_size, "sentence_size":sentence_size}


    def embed_component(self):
        ret = 0
        for component in self.components:
            for word in component['preceding_included']:
                ret += w[word]
