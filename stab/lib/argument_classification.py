import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import groupby
from operator import itemgetter
import sys
import re
from gensim import models
from indicators import FORWARD_INDICATORS, FIRST_PERSON_INDICATORS, THESIS_INDICATORS, BACKWARDS_INDICATORS, REBUTTAL_INDICATORS
from pos import pos
import json
w = models.KeyedVectors.load_word2vec_format(
    '../models/GoogleNews-vectors-negative300.bin', binary=True)
corenlp_dir = '../corenlp'
import os
os.environ["CORENLP_HOME"] = corenlp_dir
pdtb_output_dir = '../../data/ArgumentAnnotatedEssays-2.0 2/data/brat-project-final/output'
from stanza.server import CoreNLPClient
import subprocess


class ArgumentClassification():
    def __init__(self, data, client, token_list, probability=None, vectorizer=None, dependency_tuples=None):
        self.data = data
        self.client = client
        self.token_list = token_list
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
        p_token = []
        component_lemma = []
        component_pos = []
        component_sent = []
        preceding_tokens = []
        preceding_lemmas = []
        following_tokens = []
        encountered_b = False
        paragraph = None
        sentence = None
        start_index = None
        end_index = None
        component_stats = {}
        pos_dist = pos.copy()
        essays = set()
        count = 1
        

        if train:
            dependency_tuples_freq = {}
            for each in self.data:  
                # head = list(filter(lambda d: d['essay'] == each['essay'] and d['token'] == each['head'].split('-')[0], self.data))
                # print(head)
                # if len(head)>0:
                dep_tuple = each["head_lemma"]+"-"+each['lemma']
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
                    component_sent.append(each['token_sentiment'])
                    component_lemma.append(each['lemma'])
                    component_pos.append(each['pos'])
                    p_token.append(self.token_list.count(each['token'])/len(self.token_list))
                else:
                    if not encountered_b:
                        preceding_tokens.append(each['token'])
                        preceding_lemmas.append(each['lemma'])
                    else:
                        if not each['token'] == '.':
                            following_tokens.append(each['token'])
                        else:
                            following_tokens.append(each['token'])
                            # print(each['essay'])
                            # print(paragraph)
                            # print(sentence)
                            
                            component_stats = self.paragraph_stats(each['essay'], paragraph, sentence)
                            preceding_tokens = [x for x in " ".join(preceding_tokens).split('.')[-1].split(' ') if x != '']
                            preceding_lemmas = preceding_lemmas[-len(preceding_tokens):]
                            text_info = self.read_file(each['essay'], paragraph, sentence, start_index, end_index)
                            # print('sentence_size',str(component_stats['sentence_size']))
                            fields = {
                                "essay":each['essay'],
                                "index":count,
                                "component":component,
                                "component_sent":component_sent,
                                "component_text":text_info['component_text'],
                                "component_lemmas":component_lemma,
                                "p_token":p_token, # FOR PMI!!!
                                "component_pos":component_pos,
                                "start":start_index,
                                "end":end_index,
                                "preceding_tokens":preceding_tokens,
                                "preceding_lemmas":preceding_lemmas, 
                                "num_preceding":len(preceding_tokens),
                                "following_tokens":following_tokens,
                                "num_following":len(following_tokens),
                                "type_indicators": self.type_indicators(preceding_tokens, text_info['component_text']),
                                "first_person_indicators": self.first_person_indicators(preceding_tokens, component),
                                "paragraph":paragraph,
                                "paragraph_size":component_stats['paragraph_size'],
                                # "first/last": 1 if component_stats['max_sentence'] == sentence or component_stats['min_sentence'] == sentence else 0,
                                "first/last":self.first_last(paragraph),
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
                                **self.type_indicators_text(" ".join(preceding_tokens), "preceding"),
                                **self.type_indicators_text(" ".join(following_tokens), "following"),
                                **self.type_indicators_text(text_info['component_text'], "component"),
                                "claim":None
                            }
                            essays.add(each['essay'])
                            if len(component) > 0:
                                self.components.append(fields)
                                count += 1
                            component = []
                            p_token = []
                            component_sent = []
                            component_lemma = []
                            component_pos = []
                            preceding_tokens = []
                            preceding_lemmas = []
                            following_tokens = []
                            encountered_b = False
                            paragraph = None
                            sentence = None  
                            start_index = None
                            end_index = None
                            pos_dist = pos.copy()
            encountered_b = False
            count = 1           
        labels = {}
        
        if train == "train": 
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
                
                self.probability.append({'claim':component['claim'], 'preceding_tokens':component['preceding_lemmas']})
            self.vectorizer.fit(vectorization_data)
        elif train=='eval':
            for essay in essays:
                    labels[essay] = self.read_data(essay.replace('.txt', '.ann'))
            for component in self.components:
                list_labels = labels[component['essay']]
                for label in list_labels:
                    if component['start'] == label['start']:
                        component['claim'] = label['claim']
    
        temp = []
        for component in self.components:
            vectorized_dep = self.vectorize(" ".join(component["preceding_tokens"]) + ' ' + " ".join(component["component"]))
            # print(vectorized_dep)
            p=self.probability_calc(self.probability, component['preceding_lemmas'])
            stats = self.component_stats(component['essay'], component['paragraph'], component['component'])
            temp.append({**vectorized_dep, **p, **stats, **component})
        self.components = temp
    
    def component_stats(self, essay, paragraph, component):
        grouped = groupby(self.components, itemgetter('essay', 'paragraph'))
        ret = {"num_component_components": 0, 'num_preceding_components': 0, 'num_following_components': 0}
        for group in grouped:
            if group[0][0] == essay and group[0][1] == paragraph:
                test = list(group[1])
                ret['num_component_components'] = len(test)
                for i, g in enumerate(test):
                    if g['component'] == component:
                        ret['num_preceding_components'] = i      
                ret['num_following_components'] = len(test) -  ret['num_preceding_components']-1
        return ret
     

    def first_last(self, paragraph):
        if len(self.components) == 0:
            return 1
        if self.components[-1]['paragraph'] < paragraph:
            self.components[-1]['first/last'] = 1
            return 1
        return 0
        
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
                    # print(e)
                    # paragraph_text.append(e['token'])
                    if e['sentence'] == sentence:
                        sentence_size+=1
                        # covering_sentence.append(e['token'])
                    if e['sentence'] > max_index:
                        max_index = e['sentence']
                    if e['sentence'] < min_index:
                        min_index = e['sentence']
                    paragraph_size += 1
        # print(sentence_size)
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
            ret_dict[f"dim_{i}"] = float(each)
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
    def type_indicators_text(self, text, part):
        # TODO: Thiss also won't work
        ret = {
            f"{part}_forward_indicators":0,
            f"{part}_backwards_indicators":0,
            f"{part}_rebuttal_indicators":0,
            f"{part}_thesis_indicators":0,
        }
        for f in FORWARD_INDICATORS:
            if f in text:
                ret[f"{part}_forward_indicators"] = 1
        for f in BACKWARDS_INDICATORS:
            if f in text:
                ret[f"{part}_backwards_indicators"] = 1
        for f in REBUTTAL_INDICATORS:
            if f in text:
                ret[f"{part}_rebuttal_indicators"] = 1
        for f in THESIS_INDICATORS:
            if f in text:
                ret[f"{part}_thesis_indicators"] = 1
        return ret
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
        print(paragraph_text.split('.'))
        print(int(sentence))
        sentence_text = re.split('[.?!]', paragraph_text)[int(sentence)]
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
            # print(sentence.sentiment)
            depth = self.tree_depth(sentence.parseTree, 0)
            num_subclause = self.subclause(sentence.parseTree)
            tense = self.verb_tense(sentence.parseTree, component)
            nps = self.noun_phrases(sentence.parseTree, component)
            vps = self.verb_phrases(sentence.parseTree, component)        
            shared_p_values = self.shared_phrase([paragraph['introduction'], paragraph['conclusion']], vps, nps)
            dependency_info = self.dependency(sentence.token, sentence.basicDependencies)
            sentiment_scores = self.sentiment_tree(sentence.annotatedParseTree)
            rules = self.production_rules(sentence.parseTree)
        return {"depth":depth, "num_subclause":num_subclause, "tense":tense, **shared_p_values, **dependency_info, **sentiment_scores, "production_rules":rules}
    def production_rules(self, tree):
        rules = []
        self.traverse_tree(tree, rules)
        return rules
    def traverse_tree(self, tree, rules):
        if len(tree.child[0].child) == 0:
            return
        children = [x.value for x in tree.child]
        rules.append((tree.value, tuple(children)))
        for child in tree.child:
            self.traverse_tree(child, rules)
    def sentiment_tree(self, ann_tree):
        ret = {
            "STRONG_NEGATIVE":0,
            "WEAK_NEGATIVE":0,
            "NEUTRAL":0,
            "WEAK_POSITIVE":0,
            "STRONG_POSITIVE":0,
        }
        self.tree_path_sentiment(ann_tree, ret)
        total = 0
        for key in ret.keys():
            total += ret[key]
        
        for key in ret.keys():
            ret[key] = ret[key]/total
        return ret
    def tree_path_sentiment(self, tree, count):
        legend = {
            0:"STRONG_NEGATIVE",
            1:"WEAK_NEGATIVE",
            2:"NEUTRAL",
            3:"WEAK_POSITIVE",
            4:"STRONG_POSITIVE"
        }
        count[legend[tree.sentiment]] += 1
        if len(tree.child) == 0:
            return
        for child in tree.child:
            self.tree_path_sentiment(child, count)
        



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
                if path[1] in ['VB', 'VBZ', 'VBP']:
                    return 0
                elif path[1] in ['VBD', 'VBN']:
                     return 1
                elif path[1] == 'VBG':
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
        essay_name = essay.split('/')[-1]
        # dir_list = os.listdir(pdtb_output_dir)
        pipe = f'{pdtb_output_dir}/{essay_name}.pipe'
        parsings = open(pipe, 'r').read().split('\n')
        list_parsings = [x.split('|') for x in parsings]
        # print(dir_list)
        # shutil.rmtree(pdtb_output_dir) 
        os.chdir(cwd)
        return list_parsings
    def component_pdtb(self, parsings, start, end):
        ret = {
            "Comparison_Arg1_Explicit":0,
            "Expansion_Arg1_Explicit":0,
            'Temporal_Arg1_Explicit':0,
            'Contingency_Arg1_Explicit':0,
            "Comparison_Arg2_Explicit":0,
            "Expansion_Arg2_Explicit":0,
            'Temporal_Arg2_Explicit':0,
            'Contingency_Arg2_Explicit':0,
            "Comparison_Arg1_Implicit":0,
            "Expansion_Arg1_Implicit":0,
            'Temporal_Arg1_Implicit':0,
            'Contingency_Arg1_Implicit':0,
            "Comparison_Arg2_Implicit":0,
            "Expansion_Arg2_Implicit":0,
            'Temporal_Arg2_Implicit':0,
            'Contingency_Arg2_Implicit':0
        }
        discourse = []
        for parse in parsings:
            if len(parse) > 34:
                arg1 = parse[22].split('..')
                arg2 = parse[32].split('..')
                # print("Arg1: ", str(arg1))
                # print("Arg2: ", str(arg2))
                if '' in arg1  or '' in arg2:
                    continue
                else:
                    if start >= int(arg1[0])  and end <= int(arg2[-1]):
                        # ret["arg"] = 1
                        discourse.append(parse)
                    # break
                # elif int(arg2[0]) >= start and int(arg2[-1]) <= end:
                #     ret["arg"] = 2
                #     discourse = parse
                #     break
        # print(discourse)
        for dis in discourse:
            temp = []
            if dis[0] in ["Explicit", "Implicit"]:
                arg1 = dis[22].split('..')
                arg2 = dis[32].split('..')
                prop1 = int(arg1[-1]) - start
                prop2 = end - int(arg2[0])
                temp.append(dis[11])
                if prop1 > prop2:
                    temp.append('Arg1')
                else:
                    temp.append('Arg2')
                
                temp.append(dis[0])
            # print(temp)

            key = "_".join(temp)
            if key in ret.keys():
                ret[key] += 1
            # print(key)
        return ret





if __name__=='__main__':
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse'], 
        memory='4G', 
        endpoint='http://localhost:9005',
        be_quiet=True)
    client.start()
    data = pd.read_csv('../outputs/train_old.csv')
    argclass = ArgumentClassification(data.iloc[:408].to_dict('records'), client, data.token.values.tolist())
    argclass.process_data(True)
    # print(argclass.components[0])
    with open("../outputs/components1.json", "w") as f:
        json.dump(argclass.components, f)
    # pd.DataFrame(argclass.components).to_json("components.json", "records")

    client.stop()