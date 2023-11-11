import stanza
import pandas as pd
import re
import shutil

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import groupby
from operator import itemgetter
import sys
from gensim import models
from indicators import FORWARD_INDICATORS, FIRST_PERSON_INDICATORS, THESIS_INDICATORS, BACKWARDS_INDICATORS, REBUTTAL_INDICATORS
from pos import pos
import json
w = models.KeyedVectors.load_word2vec_format(
    '../models/GoogleNews-vectors-negative300.bin', binary=True)
import multiprocessing
corenlp_dir = '../corenlp'
import os
os.environ["CORENLP_HOME"] = corenlp_dir
pdtb_output_dir = '../../data/ArgumentAnnotatedEssays-2.0 2/data/brat-project-final/output'
from stanza.server import CoreNLPClient
import subprocess
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()


class ArgumentClassification():
    def __init__(self, data, client, probability=None, vectorizer=None, dependency_tuples=None):
        self.data = data
        self.client = client
        if probability:
            self.probability = probability
        if dependency_tuples:
            self.dependency_tuples = dependency_tuples
        else:
            self.dependency_tuples = {}
        
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = TfidfVectorizer()
    def process_data(self, train=None):
        self.components = []
        component = []
        preceding_tokens = []
        following_tokens = []
        encountered_b = False
        paragraph = None
        sentence = None
        start_index = None
        end_index = None
        component_stats = {}
        pos_dist = pos.copy()
        essays = set()
        

        if train:
            dependency_tuples_freq = {}
            for each in self.data:
                head = list(filter(lambda d: d['essay'] == each['essay'] and d['token'] == each['head'].split('-')[0], self.data))[0]
                dep_tuple = head["lemma"]+"-"+each['lemma']
                if dep_tuple in dependency_tuples_freq.keys():
                    dependency_tuples_freq[dep_tuple] += 1
                else:
                    dependency_tuples_freq[dep_tuple] = 1
            # print(sorted(dependency_tuples_freq, key=lambda k: dependency_tuples_freq[k],  reverse=True))
            tuples_2k = list(sorted(dependency_tuples_freq, key=lambda k: dependency_tuples_freq[k],  reverse=True))[:2000]
            for each in tuples_2k:
                self.dependency_tuples[each] = 0

        for essay in groupby(self.data, itemgetter('essay')):
            # print(essay)
            print(f"starting {essay[0]}")
            parsings = self.pdtb_parse(essay[0])
            # print(parsings)
            for index, each in enumerate(essay[1]):
                # print(each)
                if each['IOB'] == 'Arg-B' or each['IOB'] == 'Arg-I':
                    if not encountered_b:
                        paragraph = each['paragraph']
                        sentence = each['sentence']
                        start_index = each['start']
                        encountered_b = True
                    if each['pos'] in pos_dist.keys() :
                        pos_dist[each['pos']]+=1
                    end_index = each['start'] + len(each['token'])
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
                            text_info = self.read_file(each['essay'], paragraph, sentence, start_index, end_index)
                            fields = {
                                "essay":each['essay'],
                                "component":component,
                                "component_text":text_info['component_text'],
                                "start":start_index,
                                "end":end_index,
                                "preceding_tokens":preceding_tokens,
                                "num_preceding":len(preceding_tokens),
                                "following_tokens":following_tokens,
                                "num_following":len(following_tokens),
                                "type_indicators": self.type_indicators(preceding_tokens, text_info['component_text']),
                                "first_person_indicators": self.first_person_indicators(preceding_tokens, component),
                                "paragraph":paragraph,
                                "paragraph_size":component_stats['paragraph_size'],
                                "first/last": 1 if component_stats['max_sentence'] == sentence or component_stats['min_sentence'] == sentence else 0,
                                "intro/conc": 1 if each['docPosition'] == 'Introduction' or each['docPosition'] == 'Conclusion' else 0,
                                "sentence":sentence,
                                "sentence_size":component_stats['sentence_size'],
                                "ratio":len(component)/component_stats['sentence_size'],
                                # "probability":0,
                                "modal_present": 1 if pos_dist['MD'] > 0 else 0,
                                **pos_dist,
                                **self.annotate_sentence(sentence, text_info, component, each['essay']),
                                **self.indicators_context(text_info['paragraph'], text_info['component_text']),
                                **self.embed_component(component+preceding_tokens),
                                **self.component_pdtb(parsings, start_index, end_index),
                                "claim":None
                            }
                            essays.add(each['essay'])
                            if len(component) > 0:
                                self.components.append(fields)
                            component = []
                            preceding_tokens = []
                            following_tokens = []
                            encountered_b = False
                            paragraph = None
                            sentence = None  
                            start_index = None
                            end_index = None
                            pos_dist = pos.copy()                  
        labels = {}
        
        if train: 
            self.probability = []
            vectorization_data = []
            for essay in essays:
                labels[essay] = self.read_data(essay.replace('.txt', '.ann'))
            for component in self.components:
                list_labels = labels[component['essay']]
                vectorization_data.append(" ".join(component['preceding_tokens']) + ' ' + " ".join(component['component']))
                for label in list_labels:
                    if component['start'] == label['start']:
                        component['claim'] = label['claim']
                
                self.probability.append({'claim':component['claim'], 'preceding_tokens':component['preceding_tokens']})
            self.vectorizer.fit(vectorization_data)
    
            
        # with open('classification_probability.json', 'w') as f:
        #     json.dump(probability, f)
        # with open('classification_dependency.json', 'w') as f:
        #     json.dump(list(self.dependency_tuples), f)
        temp = []
        for component in self.components:
            vectorized_dep = self.vectorize(" ".join(component["preceding_tokens"]) + ' ' + " ".join(component["component"]))
            print(vectorized_dep)
            p=self.probability_calc(self.probability, component['preceding_tokens'])
            temp.append({**vectorized_dep, **p, **component})
        self.components = temp
        
        
    def vectorize(self, text):
        ret = {}
        vectorized  = self.vectorizer.transform([text])
        list_vectorized = vectorized.toarray()
        for i, each in enumerate(list_vectorized[0]):
            ret[f"dep_{i}"] = each
        return ret



    def probability_calc(self, probability, preceding):
        labels = ['MajorClaim','Claim','Premise']
        ret = {'p_MajorClaim':0, 'p_Claim':0, 'p_Premise':0}
        for c in labels:
            label_instance = [label for label in probability if label["claim"] == c]
            preceding_instance = [e for e in label_instance if e["preceding_tokens"] == preceding]
            if len(label_instance) > 0:
                ret["p_"+c] = len(preceding_instance)/len(label_instance)
            else:
                ret["p_"+c] = 0
        return ret

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
            try:
                ret += w[word]
            except:
                continue
        ret_dict = {}
        for i, each in enumerate(ret):
            ret_dict[f"dim_{i}"] = each
        return ret_dict

    
    def type_indicators(self, preceding, component):
        # TODO: Thiss also won't work
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
        # TODO FIX THIS
        prec = ' '.join(preceding)
        comp = ' '.join(component)
        for f in FIRST_PERSON_INDICATORS:
            if f in prec or f in comp:
                return 1
        return 0
    def read_file(self, file, paragraph, sentence, start, end):
        full_file = open(file,"r").read()
        # print(f.split('\n\n'))
        paragraphs = []

        with open(file,"r") as f: 
            for idx, line in enumerate(f.readlines()):
                if idx == 0: # skip prompt 
                    continue
                if idx == 1: # skip the additional newline after prompt 
                    continue
                paragraphs.append(line)
        paragraph_text = paragraphs[int(paragraph)]
        sentence_text = paragraph_text.split('.')[int(sentence)-1]
        intro = paragraphs[0]
        conclusion = paragraphs[-1]
        component_text = full_file[int(start):int(end)]
        return {'paragraph':paragraph_text, 'sentence':sentence_text, 'introduction':intro, 'conclusion':conclusion, 'component_text':component_text}
    def annotate_sentence(self, sent_idx, paragraph, component, file):
        document = self.client.annotate(paragraph['paragraph'])
        sentence = None
        # print(sent_idx)
        for i, sent in enumerate(document.sentence):
            if i == sent_idx:
                # we want this sentence
                sentence = sent
                break
        if sentence:
            depth = self.tree_depth(sentence.parseTree, 0)
            num_subclause = self.subclause(sentence.parseTree)
            tense = self.verb_tense(sentence.parseTree, component)
            nps = self.noun_phrases(sentence.parseTree, component)
            vps = self.verb_phrases(sentence.parseTree, component)        
            shared_p_values = self.shared_phrase([paragraph['introduction'], paragraph['conclusion']], vps, nps)
            dependency_info = self.dependency(sentence.token, sentence.basicDependencies)
        
        return {"depth":depth, "num_subclause":num_subclause, "tense":tense, **shared_p_values, **dependency_info}

    def dependency(self, tokens, dependency):
        ret = self.dependency_tuples.copy()
        for dep in dependency.ListFields()[1][1]:
            # self.dependency_tuples.add((tokens[dep['source']-1], tokens[dep['target']-1]))
            dep_tuple = tokens[dep.source-1].word+"-"+tokens[dep.target-1].word
            if dep_tuple in self.dependency_tuples.keys():
                ret[dep_tuple] += 1
        return ret
            


    

        
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
                vp_count += paragraph.count(' '.join(phrase))
            for phrase in noun_phrases:
                np_count += paragraph.count(' '.join(phrase))
        
        return {"noun_phrases": np_count, "verb_phrases": vp_count}


    def indicators_context(self, paragraph, component):
        # TODO Fix this
        text = paragraph.replace(component, '\n').split('\n')
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
        # print(components)
    #     return components
    # def preprocess_components(self, components):
        augmented = []
        for component in components: 
            name = component[0]
            claim,start,end = component[1].split(' ')
            phrase = component[2]
            info = {"name": name, "claim": claim,"start":int(start),"end":int(end),"phrase": phrase}
            augmented.append(info)
        return augmented
    
    def pdtb_parse(self, essay):
        cwd = os.getcwd() #current directory
        os.chdir('../models/pdtb-parser')
        subprocess.run(['mkdir', pdtb_output_dir])
        subprocess.run(['sudo','java', '-jar', 'parser.jar', f'../{essay}'])
        dir_list = os.listdir(pdtb_output_dir)
        pipe = [i for i in dir_list if '.pipe' in i][0]
        parsings = open(f'{pdtb_output_dir}/{pipe}', 'r').read().split('\n')
        list_parsings = [x.split('|') for x in parsings]
        print(dir_list)
        # shutil.rmtree(pdtb_output_dir) 
        os.chdir(cwd)
        return list_parsings
    def component_pdtb(self, parsings, start, end):
        print('start: ',start)
        print('end: ', end)
        ret = {"type": 0, "arg":0, "relation":0}
        key = {"Comparison":1, "EntRel":2, "Expansion":3, "NoRel":4, 'Temporal':5, 'Contingency':6}

        discourse = None
        for parse in parsings:
            if len(parse) > 34:
                arg1 = parse[22].split('..')
                arg2 = parse[32].split('..')
                print("Arg1: ", str(arg1))
                print("Arg2: ", str(arg2))
                if start <= int(arg1[0])  and end <= int(arg2[-1]):
                    # ret["arg"] = 1
                    discourse = parse
                    break
                # elif int(arg2[0]) >= start and int(arg2[-1]) <= end:
                #     ret["arg"] = 2
                #     discourse = parse
                #     break
        print(discourse)
        if discourse and discourse[0] in ["Explicit", "Implicit"]:
            arg1 = discourse[22].split('..')
            arg2 = discourse[32].split('..')
            prop1 = int(arg1[-1]) - start
            prop2 = end - int(arg2[0])
            if prop1 > prop2:
                ret['arg'] = 1
            else:
                ret['arg'] = 2
            if discourse[11] in key.keys():
                ret['type'] =key[discourse[11]]
            ret['relation'] = 1 if discourse[0] == "Explicit" else 2
        return ret





if __name__=='__main__':
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse'], 
        memory='4G', 
        endpoint='http://localhost:9005',
        be_quiet=True)
    client.start()
    data = pd.read_csv('test.csv').iloc[:408].to_dict('records')
    argclass = ArgumentClassification(data, client)
    argclass.process_data(True)
    # print(argclass.components[0])
    # with open("components.json", "w") as f:
    #     json.dump(argclass.components, f)
    pd.DataFrame(argclass.components).to_json("components.json", "records")

    client.stop()