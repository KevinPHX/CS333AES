
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import groupby
from operator import itemgetter
from pos import pos
import sys
from indicators import FORWARD_INDICATORS, FIRST_PERSON_INDICATORS, THESIS_INDICATORS, BACKWARDS_INDICATORS, REBUTTAL_INDICATORS

class RelationIdentification():
    def __init__(self, data, train=True, vectorizer = None):
        self.data = data
        self.train = train
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = TfidfVectorizer()
    # def lexical_features(self, source, target):
    #     if self.train:



    def process_data(self):
        self.components = []
        component = []
        preceding_tokens = []
        following_tokens = []
        encountered_b = False
        paragraph = None
        sentence = None
        start_index = None
        end_index = None
        start_count = None
        end_count = None
        component_stats = {}
        pos_dist = pos.copy()
        essays = set()
        paragraph_components = {}
        for essay in groupby(self.data, itemgetter('essay')):
            # print(essay)
            
            paragraph_components[essay[0]] = {}
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
                        start_count = each['index']
                        encountered_b = True
                    if each['pos'] in pos_dist.keys() :
                        pos_dist[each['pos']]=1
                    end_index = each['start'] + len(each['token'])
                    end_count = each['index']
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
                                "num_tokens": len(component),
                                "start":start_index,
                                "end":end_index,
                                "start_i":start_count,
                                "end_i":end_count,
                                "preceding_tokens":preceding_tokens,
                                "num_preceding":len(preceding_tokens),
                                "following_tokens":following_tokens,
                                "num_following":len(following_tokens),
                                "first_person_indicators": self.first_person_indicators(preceding_tokens, component),
                                "paragraph":paragraph,
                                "paragraph_text":text_info['paragraph_text'],
                                "paragraph_size":component_stats['paragraph_size'],
                                "first/last": 1 if component_stats['max_sentence'] == sentence or component_stats['min_sentence'] == sentence else 0,
                                "intro/conc": 1 if each['docPosition'] == 'Introduction' or each['docPosition'] == 'Conclusion' else 0,
                                "sentence":sentence,
                                "sentence_size":component_stats['sentence_size'],
                                "ratio":len(component)/component_stats['sentence_size'],
                                "modal_present": 1 if pos_dist['MD'] > 0 else 0,
                                **pos_dist,
                                **self.annotate_sentence(sentence, text_info, component, each['essay']),
                                **self.type_indicators(text_info['component_text']),
                                "claim":None
                            }
                            essays.add(each['essay'])
                            
                            if len(component) > 0:
                                self.components.append(fields)
                                if paragraph in paragraph_components[essay[0]].keys():
                                    paragraph_components[essay[0]][paragraph] += 1
                                else:
                                    paragraph_components[essay[0]][paragraph] = 0
                            
                            component = []
                            preceding_tokens = []
                            following_tokens = []
                            encountered_b = False
                            paragraph = None
                            sentence = None  
                            start_index = None
                            end_index = None
                            start_count = None
                            end_count = None
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
            rules = self.production_rules(sentence.parseTree)

        return {"production_rules":rules}
        
    def production_rules(self, tree):
        rules = []
        self.traverse_tree(tree, rules)
        return rules
    def traverse_tree(self, tree, rules):
        if len(tree.child) == 0:
            return
        rules.append((tree.label, (x.label for x in tree.child)))
        for child in tree.child:
            self.traverse_tree(child, rules)
    def type_indicators(self, text):
        # TODO: Thiss also won't work
        ret = {
            "forward_indicators":0,
            "backwards_indicators":0,
            "rebuttal_indicators":0,
            "thesis_indicators":0,
        }
        for f in FORWARD_INDICATORS:
            if f in text:
                ret['forward_indicators'] = 1
        for f in BACKWARDS_INDICATORS:
            if f in text:
                ret['backwards_indicators'] = 1
        for f in REBUTTAL_INDICATORS:
            if f in text:
                ret['rebuttal_indicators'] = 1
        for f in THESIS_INDICATORS:
            if f in text:
                ret['thesis_indicators'] = 1
        return ret
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