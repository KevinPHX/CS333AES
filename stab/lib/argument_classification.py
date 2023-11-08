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
    def __init__(self, data, client, text_file, ann_file=None):
        self.data = data
        self.client = client
        self.file = text_file
        self.ann_file = ann_file
        # self.components = self.process_data(data)
    def process_data(self):
        self.components = []
        component = []
        preceding_tokens = []
        following_tokens = []
        encountered_b = False
        paragraph = None
        sentence = None
        start_index = None
        component_stats = {}
        pos_dist = pos.copy()
        for each in self.data:
            if each['IOB'] == 'Arg-B' or each['IOB'] == 'Arg-I':
                if not encountered_b:
                    paragraph = each['paragraph']
                    sentence = each['sentence']
                    start_index = each['start']
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
                        following_tokens.append(each['token'])
                        component_stats = self.paragraph_stats(each['essay'], paragraph, sentence)
                        preceding_tokens = [x for x in " ".join(preceding_tokens).split('.')[-1].split(' ') if x != '']
                        text_info = self.read_file(each['essay'])
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
                            **self.annotate_sentence(sentence, text_info, component, each['essay'])
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

    def paragraph_stats(self, essay, paragraph, sentence):
        grouped = groupby(self.data, itemgetter('essay','paragraph'))
        max_index = 0
        min_index = sys.maxsize
        paragraph_size = 0
        sentence_size = 0
        # covering_sentence = []
        # paragraph_text = []
        for group in grouped:
            if group[0][0] == essay and group[0][1] == paragraph:
                for e in group[1]:
                    # paragraph_text.append(e['token'])
                    if e['sentence'] == sentence:
                        sentence_size+=1
                        # covering_sentence.append(e['token'])
                    if e['sentence'] > max_index:
                        max_index = e['sentence']
                    if e['sentence'] < min_index:
                        min_index = e['sentence']
                    paragraph_size += 1
        # return {"max_sentence":max_index, "min_sentence":min_index, "paragraph_size":paragraph_size, "sentence_size":sentence_size, "covering_sentence":covering_sentence, 'paragraph_text':paragraph_text}
        return {"max_sentence":max_index, "min_sentence":min_index, "paragraph_size":paragraph_size, "sentence_size":sentence_size}


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
    def read_file(self, file, paragraph, sentence):
        f = open(file,"r").read()
        paragraph_text = f.split('\n\n')[1].split('\n')[paragraph]
        sentence_text = paragraph_text.split('.')[sentence-1]
        intro = f.split('\n\n')[1].split('\n')[0]
        conclusion = f.split('\n\n')[1].split('\n')[-1]
        return {'paragraph':paragraph_text, 'sentence':sentence_text, 'introduction':intro, 'conclusion':conclusion}
    def annotate_sentence(self, sent_idx, paragraph, component, file):
        document = self.client.annotate(paragraph['paragraph'])
        sentence = None
        for i, sent in enumerate(document.sentence):
            if i+1 == sent_idx:
                # we want this sentence
                sentence = sent
                break
        depth = self.tree_depth(sentence.parseTree, 0)
        num_subclause = self.subclause(sentence.parseTree)
        tense = self.verb_tense(sentence.parseTree, component)
        nps = self.noun_phrases(sentence.parseTree, component)
        vps = self.verb_phrases(sentence.parseTree, component)        
        shared_p_values = self.shared_phrase([paragraph['intro'], paragraph['conclusion']], vps, nps)
        return {"depth":depth, "num_subclause":num_subclause, "tense":tense, **shared_p_values}


        
    def verb_tense(self, parse_tree, component):
        for each in component:
            path = []
            self.tree_path(parse_tree, each, path)
            if len(path) > 2:
                if path[-2] in ['VB', 'VBZ', 'VBP']:
                    return 0
                elif path[-2] in ['VBD', 'VBN']:
                     return 1
                elif path[-2] == 'VBG':
                    return 2
    def noun_phrases(self, parse_tree, component):
        ret = []
        noun_phrase = []
        in_np = False
        for each in component:
            path = []
            self.tree_path(parse_tree, each, path)
            if "NP" in path:
                noun_phrase.append(each)
                in_np = True
            else:
                if in_np:
                    ret.append(noun_phrase)
                    noun_phrase = []
                    in_np = False
        return ret
    
    def verb_phrases(self, parse_tree, component):
        ret = []
        verb_phrase = []
        in_vp = False
        for each in component:
            path = []
            self.tree_path(parse_tree, each, path)
            if "VP" in path:
                verb_phrase.append(each)
                in_vp = True
            else:
                if in_vp:
                    ret.append(verb_phrase)
                    verb_phrase = []
                    in_vp = False
        return ret
    
    def shared_phrase(self, paragraphs, verb_phrases, noun_phrases):
        np_count = 0
        vp_count = 0
        for paragraph in paragraphs:
            for phrase in verb_phrases:
                vp_count += paragraph.count(phrase.join(' '))
            for phrase in noun_phrases:
                np_count += paragraph.count(phrase.join(' '))
        
        return {"noun_phrases": np_count, "verb_phrases": vp_count}


    def indicators_context(self, paragraph, component):
        text = paragraph.replace(component.join(' '), '\n')
        preceding_text = text[0]
        following_text = text[1]
        ret = {
            "preceding_forward_context":0,
            "preceding_backward_context":0,
            "preceding_rebuttal_context":0,
            "preceding_thesis_context":0,
            "following_forward_context":0,
            "following_backward_context":0,
            "following_rebuttal_context":0,
            "following_thesis_context":0
        }
        for f in FORWARD_INDICATORS:
            if f in preceding_text:
                ret['preceding_forward_context']=1
            if f in following_text:
                ret['following_forward_context']=1
        for f in BACKWARDS_INDICATORS:
            if f in preceding_text:
                ret['preceding_backward_context']=1
            if f in following_text:
                ret['following_backward_context']=1
        for f in REBUTTAL_INDICATORS:
            if f in preceding_text:
                ret['preceding_rebuttal_context']=1
            if f in following_text:
                ret['following_rebuttal_context']=1
        for f in THESIS_INDICATORS:
            if f in preceding_text:
                ret['preceding_thesis_context']=1
            if f in following_text:
                ret['following_thesis_context']=1
        return ret
                

        # if len(parse_tree.child) == 0 and "VB" in parent and parse_tree.value in component:
        #     if parent in ['VB', 'VBZ', 'VBP']:
        #         return 0
        #     elif parent in ['VBD', 'VBN']:
        #         return 1
        #     else:
        #         return 2
        # if parse_tree.value == 'VP':
        #     return self.verb_tense(parse_tree.child[-1], parse_tree.value, component)
        # for child in parse_tree.child:
        #     return self.verb_tense(child, child.value, component)
        # return 0
    
    def tree_path(self, parse_tree, target, path):
        if parse_tree.value == target:        
            path.append(parse_tree.value)
            return True
        if len(parse_tree.child) == 0:
            return False
        for child in parse_tree.child:
            temp_check = self.tree_path(child, target, path)
            if temp_check:
                path.append(parse_tree.value)
                return True
        return False
            
    
    def subclause(self, parse_tree):
        tree_paths = []
        self.traverse(parse_tree, tree_paths)
        return tree_paths.count("S")
    def traverse(self, parse_tree, path):
        path.append(parse_tree.value)
        if len(parse_tree.child) == 0:
            return
        for child in parse_tree.child:
            self.traverse(child, path)
        return 
        
    def tree_depth(self, parse_tree, depth):
        if len(parse_tree.child) == 0:
            return depth
        max_depth = 0
        for child in parse_tree.child:
            temp_depth = self.tree_depth(parse_tree=child, depth=depth+1)
            if max_depth < temp_depth:
                max_depth = temp_depth
        return max_depth


    def read_data(self, ann_file): 
        components = []
        with open(ann_file,"r") as f: 
            for line in f.readlines(): 
                line = line.strip('\n')
                line = line.split("\t")
                if "T" in line[0]: 
                    components.append(line)
        return components
    def preprocess_components(self, components):
        augmented = []
        for component in components: 
            name = component[0]
            claim,start,end = component[1].split(' ')
            phrase = component[2]
            info = {"name": name, "claim": claim,"start":int(start),"end":int(end),"phrase": phrase}
            augmented.append(info)
        return augmented

if __name__=='__main__':
    data = pd.read_csv('../indentification2.csv').to_dict('records')
    argclass = ArgumentClassification(data)
    argclass.process_data()
    print(argclass.components)