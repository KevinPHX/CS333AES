import stanza
import pandas as pd
import re
import numpy as np
from statsmodels.discrete.discrete_model import Poisson
from itertools import groupby
from operator import itemgetter
import sys
from gensim import models
from indicators import FORWARD_INDICATORS, FIRST_PERSON_INDICATORS, THESIS_INDICATORS, BACKWARDS_INDICATORS, REBUTTAL_INDICATORS
from pos import pos
w = models.KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin', binary=True)

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
        pos_dist = pos.copy()
        for each in data:
            if each['IOB'] == 'Arg-B' or each['IOB'] == 'Arg-I':
                if not encountered_b:
                    paragraph = each['paragraph']
                    sentence = each['sentence']
                    encountered_b = True
                pos_dist[each['pos']]+=1
                component.append(each['token'])
            else:
                if not encountered_b:
                    preceding_tokens.append(each['token'])
                else:
                    if not each['token'] == '.':
                        following_tokens.append(each['token'])
                    else:
                        component_stats = self.paragraph_stats(data, each['essay'], paragraph, sentence)
                        preceding_tokens = [x for x in " ".join(preceding_tokens).split('.')[-1].split(' ') if x != '']
                        fields = {
                            "essay":each['essay'],
                            "component":component,
                            "preceding_tokens":preceding_tokens,
                            "num_preceding":len(preceding_tokens),
                            "following_tokens":following_tokens,
                            "num_following":len(following_tokens),
                            "type_indicators": self.type_indicators(preceding_tokens, component),
                            "first_person_indicators": self.first_person_indicators(preceding_tokens, component),
                            "paragraph":paragraph,
                            "paragraph_size":component_stats['paragraph_size'],
                            "first/last": 1 if component_stats['max_sentence'] == sentence or component_stats['min_sentence'] == sentence else 0,
                            "intro/conc": 1 if each['docPosition'] == 'Introduction' or each['docPosition'] == 'Conclusion' else 0,
                            "sentence":sentence,
                            "sentence_size":component_stats['sentence_size'],
                            "ratio":len(component)/component_stats['sentence_size'],
                            **pos_dist,
                            **self.embed_component(component+preceding_tokens)
                        }
                        if len(component) > 0:
                            self.components.append(fields)
                        component = []
                        preceding_tokens = []
                        following_tokens = []
                        encountered_b = False
                        paragraph = None
                        sentence = None  
                        pos_dist = pos.copy()              

    def paragraph_stats(self, data, essay, paragraph, sentence):
        grouped = groupby(data, itemgetter('essay','paragraph'))
        max_index = 0
        min_index = sys.maxsize
        paragraph_size = 0
        sentence_size = 0
        covering_sentence = []
        paragraph_text = []
        for group in grouped:
            if group[0][0] == essay and group[0][1] == paragraph:
                for e in group[1]:
                    paragraph_text.append(e['token'])
                    if e['sentence'] == sentence:
                        sentence_size+=1
                        covering_sentence.append(e['token'])
                    if e['sentence'] > max_index:
                        max_index = e['sentence']
                    if e['sentence'] < min_index:
                        min_index = e['sentence']
                    paragraph_size += 1
        return {"max_sentence":max_index, "min_sentence":min_index, "paragraph_size":paragraph_size, "sentence_size":sentence_size, "covering_sentence":covering_sentence, 'paragraph_text':paragraph_text}


    def embed_component(self, component):
        ret = 0
        
        for word in component:
            ret += w[word]
        ret_dict = {}
        for i, each in enumerate(ret):
            ret_dict[f"dim_{i}"] = each
        return ret_dict

    
    def type_indicators(self, preceding, component):
        prec = ' '.join(preceding)
        comp = ' '.join(component)
        for f in FORWARD_INDICATORS:
            if f in prec or f in comp:
                return 1
        for f in BACKWARDS_INDICATORS:
            if f in prec or f in comp:
                return 1
        for f in REBUTTAL_INDICATORS:
            if f in prec or f in comp:
                return 1
        for f in THESIS_INDICATORS:
            if f in prec or f in comp:
                return 1
        return 0
    def first_person_indicators(self, preceding, component):
        prec = ' '.join(preceding)
        comp = ' '.join(component)
        for f in FIRST_PERSON_INDICATORS:
            if f in prec or f in comp:
                return 1
            
        return 0

if __name__=='__main__':
    data = pd.read_csv('../indentification2.csv').to_dict('records')
    argclass = ArgumentClassification(data)
    argclass.process_data(data)
    print(argclass.components)