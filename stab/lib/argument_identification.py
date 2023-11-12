import stanza
import pandas as pd
import re
import numpy as np
from statsmodels.discrete.discrete_model import Poisson
from multiprocessing import Pool    

#  per document list


class ArgumentIdentification():
    def __init__(self, client, text_file, ann_file=None):
        self.client = client
        self.file  = text_file
        self.ann_file = ann_file
    def __getstate__(self):
        state = self.__dict__.copy()
        print(state)
        del state['lock']
        return state
    def annotate_punc(self, token, data):
        # check if punc
        if not re.search(r'[A-Za-z0-9]',token.originalText) and len(data) > 0: # is punc
            # set precedes punc in last element to true
            data[-1]['precedesPunc'] = True
            return 1
        elif len(data) > 0:
            if data[-1]['isPunc']:
                return 2
        else:
            return 0

    def sent_pos(self, tokenIdx, sentLength):
        if tokenIdx == 1: 
            return "First"
        elif tokenIdx == sentLength: 
            return "Last"
        else: 
            return "Middle"

    def paragraph_pos(self, p_idx, paragraphLength):
        if p_idx == 0: 
            return "Introduction"
        elif p_idx == paragraphLength-1: 
            return "Conclusion"
        else: 
            return "Body"

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



    def iob(self, start, components):
        for c in components: 
            if c["start"] == start: # token that begins a component
                return "Arg-B"
            elif c["start"] < start < c["end"]: # token that is covered by an argument component 
                return "Arg-I"
        return "O"    



    def tree_depth(self, parse_tree, depth):
        if len(parse_tree.child) == 0:
            return depth
        max_depth = 0
        for child in parse_tree.child:
            temp_depth = self.tree_depth(parse_tree=child, depth=depth+1)
            if max_depth < temp_depth:
                max_depth = temp_depth
        return max_depth

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

    def LCA_path(self, parse_tree, token, preceding):
        preceding_path = []
        token_path = []
        if self.tree_path(parse_tree, token, token_path) and self.tree_path(parse_tree, preceding, preceding_path):
            preceding_path = preceding_path[::-1]
            token_path = token_path[::-1]
            for i in range(min(len(preceding_path), len(token_path))):
                if preceding_path[i] != token_path[i]:
                    return i, token_path[i-1]
            if preceding_path[i-1] == token_path[i-1]:
                return i, token_path[i-1]
        else:
            return 0, None


    # def following_LCA(self, parse_tree, token, following):
    #     following_path = []
    #     token_path = []
    #     if self.tree_path(parse_tree, token, token_path) and self.tree_path(parse_tree, following, following_path):
    #         following_path = following_path[::-1]
    #         token_path = token_path[::-1]
    #         for i in range(min(len(following_path), len(token_path))):
    #             if following_path[i] != token_path[i]:
    #                 return i, token_path[i-1]
    #         if following_path[i-1] == token_path[i-1]:
    #             return i, token_path[i-1]
    #         else:
    #             return 0, None
    #     else:
    #         return 0, None


    def LCA(self, parse_tree, token, preceding, following, depth):
        ret = []
        if preceding:
            lca_path_prec_path, lca_path_prec  = self.LCA_path(parse_tree, token, preceding)
            if (lca_path_prec_path > depth):
                print(lca_path_prec_path)
            ret.append((lca_path_prec_path/depth, lca_path_prec))
        if following:
            lca_path_fol_path, lca_path_fol = self.LCA_path(parse_tree, token, following)
            ret.append((lca_path_fol_path/depth, lca_path_fol))
        return ret

    def get_head(self, tree, index):
        # print(tree.ListFields())
        if len(tree.ListFields()) >=3:
            if index == tree.ListFields()[2][1][0]:
                return index
            for each in tree.ListFields()[1][1]:
                if each.target == index:
                    return each.source
        return 0
        
    def uppermost(self, parse_tree, head, token):
        if head == token:
            return 'ROOT', ['ROOT']
        head_path = []
        token_path = []
        if self.tree_path(parse_tree, token, token_path) and self.tree_path(parse_tree, head, head_path):
            head_path = head_path[::-1]
            token_path = token_path[::-1]
            for i in range(min(len(head_path), len(token_path))):
                if head_path[i] != token_path[i]:
                    return token_path[i], token_path[:i+1]
            if head_path[i-1] == token_path[i-1]:
                return token_path[i], token_path[:i+1]
            else:
                return 'S',['ROOT', 'S']
        else:
            return 'S', ['ROOT', 'S']

    def uppermost_child(self, parse_tree, true_path):
        if len(true_path) == 1 :
            if parse_tree.child:
                return parse_tree.child[0].value
        if parse_tree.value ==true_path[0]:
            for child in parse_tree.child:
                if child.value == true_path[1]:
                    return self.uppermost_child(child, true_path[1:])

    def uppermost_right(self, parse_tree, true_path):
        if len(true_path) == 1 :
            if parse_tree.child:
                return parse_tree.child[-1].value
        if parse_tree.value ==true_path[0]:
            for child in parse_tree.child:
                if child.value == true_path[1]:
                    return self.uppermost_right(child, true_path[1:])

    def get_heads(self,dep):
        heads = set()
        if len(dep.ListFields()) >=3:
            for each in dep.ListFields()[1][1]:
                heads.add(each.target)
        return list(heads)
    def get_right_sib_head(self, tree, dep, right_sib, heads):
        head = tree
        children = []
        target_head = None
        while head:
            children.extend(head.child)
            if head.value == right_sib:
                target_head = head
                break
            else:
                if len(children) > 0:
                    head = children.pop()
                else:
                    break
        children = []
        while target_head:
            children.extend(target_head.child)
            if target_head.value in heads.keys():
                return self.get_head(dep, heads[target_head.value])
            else:
                if len(children) > 0:
                    target_head = children.pop()
                else:
                    break
        return None


    def prob_train(self, train_set, target, n):
        # increment by 2 to get the targeted tag value (y)
        window_sz = np.arange(n)+2
        X= [[] for i in range(n)]
        Y = [[] for i in range(n)]
        ret = []
        for index, size in enumerate(window_sz):
            # preprocess data per dataset -> this is important when training across multiple docs
            for data in train_set:
                # creates sliding window of the desired size
                for window in np.lib.stride_tricks.sliding_window_view(data, size):
                    if len(window) == size:
                        x = window[:size-1][::-1].tolist()
                        # makes the target a binary of whether the y value is the target (Arg-B) or not - this is what we are ultimately trying to predict
                        y = (window[-1]==target).astype(int).tolist()
                        X[index].append(x)  
                        Y[index].append(y)
            ret.append([Y[index], X[index]])
            # if old_models:
            #     old_models[index] = old_models[index].
            # models.append(Poisson(Y[index], X[index]).fit())
        return ret
    def predict(self, models, X):
        probabilities = []
        # There should be n models, so we iterate across each model and get their respective probabilities
        for index, model in enumerate(models):
            if X[index]:
                probabilities.append(model.predict(X[:index+1]))
            else:
                probabilities.append(np.array([0]))
        # Find the max probability for any window size and return it
        # print(probabilities)
        index = np.argmax(probabilities)
        return probabilities[index][0]
    def process_file(self, file):
        print(f"starting essay {file}")
        paragraphs = []
        data =[]

        with open(file,"r") as f: 
            for idx, line in enumerate(f.readlines()):
                if idx == 0: # skip prompt 
                    prompt = line
                    continue
                if idx == 1: # skip the additional newline after prompt 
                    just_in_case = line
                    continue
                paragraphs.append(line)

        components = self.read_data(file.replace('.txt', '.ann'))

        adjust_sen = 0
        for k, para in enumerate(paragraphs):
            document = self.client.annotate(para)   
            for i, sent in enumerate(document.sentence):
                component_i = self.preprocess_components(components)
                for j, token in enumerate(sent.token):
                    start = len(prompt) + len(just_in_case) + adjust_sen + token.beginChar
                    punc = self.annotate_punc(token, data)
                    d = {
                        'essay':file,
                        'token':token.word,
                        'lemma':token.lemma,
                        'sentence':i,
                        'index':j+1,
                        'start':start,
                        'pos':token.pos,
                        'sentence_sentiment':sent.sentiment,
                        'token_sentiment':token.sentiment,
                        'paragraph':k,
                        'docPosition':self.paragraph_pos(k, len(paragraphs)),
                        'sentPosition':self.sent_pos(token.endIndex, len(sent.token)),
                        'isPunc':punc==1,
                        'followsPunc':punc==2,
                        'precedesPunc':False,
                        'followsLCA':None,
                        'precedesLCA':None,
                        'followsLCAPath':None,
                        'precedesLCAPath':None,
                        'head':self.get_head(sent.basicDependencies, j+1),
                        'uppermost':None,
                        'uppermost_child':None,
                        'right_sibling':None,
                        'right_sibling_head':None,
                        'probability':None,
                        'IOB':self.iob(start, component_i)
                    }
                    data.append(d)
                sentence_data = data[-j-1:]
                depth = self.tree_depth(sent.parseTree, 0)
                lexical_heads = self.get_heads(sent.basicDependencies)
                lex_heads_dict = {}
                for lex in lexical_heads:
                    lex_heads_dict[sentence_data[lex-1]['token']] = lex
                for index, t in enumerate(sentence_data):
                    upper_path = self.uppermost(sent.parseTree, sentence_data[t['head']-1]['token'], t['token'])
                    t['uppermost'] = upper_path[0]
                    t['uppermost_child'] = self.uppermost_child(sent.parseTree, upper_path[1])
                    right_sib = self.uppermost_right(sent.parseTree, upper_path[1])
                    t['right_sibling'] = right_sib
                    sib_head = self.get_right_sib_head(sent.parseTree, sent.basicDependencies, right_sib, lex_heads_dict)
                    if sib_head:
                        t['right_sibling_head'] = sentence_data[sib_head-1]['token']
                    head = t['head']
                    t['head'] = '-'.join([sentence_data[t['head']-1]['token'], str(t['head'])])
                    t['head_lemma'] = sentence_data[head-1]['lemma']
                    if index == 0 and len(sentence_data)>1:
                        t["precedesLCAPath"] = -1
                        lca_output = self.LCA(sent.parseTree, sentence_data[index]['token'], None, sentence_data[index+1]['token'], depth)
                        t['followsLCAPath'] = lca_output[0][0]
                        t["followsLCA"] = lca_output[0][1]
                    elif index == len(sentence_data) - 1:
                        t["followsLCAPath"] = -1
                        lca_output = self.LCA(sent.parseTree, sentence_data[index]['token'], sentence_data[index-1]['token'], None, depth)
                        t['precedesLCAPath'] = lca_output[0][0]
                        t["precedesLCA"] = lca_output[0][1]
                    else:
                        lca_output = self.LCA(sent.parseTree, sentence_data[index]['token'], sentence_data[index-1]['token'], sentence_data[index+1]['token'], depth)
                        t['precedesLCAPath'] = lca_output[0][0]
                        t["precedesLCA"] = lca_output[0][1]
                        t['followsLCAPath'] = lca_output[1][0]
                        t["followsLCA"] = lca_output[1][1]
            adjust_sen += len(para)
        return data
    def run_annotated(self):
        ret = []
        for index, file in enumerate(self.file):
            ret.append(self.process_file(file))
        convert = {"Arg-B":0, "Arg-I":1, "O":2}
        self.models = []
        X = {1:[], 2:[], 3:[]}
        Y = {1:[], 2:[], 3:[]}
        for data in ret:
            train_set = [convert[x["IOB"]] for x in data]
            temp = self.prob_train([train_set], 0, 3)
            X[1].extend(temp[0][1])
            X[2].extend(temp[1][1])
            X[3].extend(temp[2][1])
            Y[1].extend(temp[0][0])
            Y[2].extend(temp[1][0])
            Y[3].extend(temp[2][0])
        self.models = [Poisson(Y[1], X[1]).fit(), Poisson(Y[2], X[2]).fit(), Poisson(Y[3], X[3]).fit()]

            
        for data in ret:
            for i, d in enumerate(data):
                if i == 0:
                    d['probability'] = 0.5
                elif i == 1:
                    d['probability'] = self.predict(self.models, [convert[data[i-1]['IOB']], None, None])
                elif i == 2:
                    d['probability'] = self.predict(self.models, [convert[data[i-1]['IOB']], convert[data[i-2]['IOB']], None])
                else:
                    d['probability'] = self.predict(self.models, [convert[data[i-1]['IOB']], convert[data[i-2]['IOB']], convert[data[i-3]['IOB']]])
        self.train_data = sum(ret, [])
        print(f"finishing essay {file}")
    def save_data(self, file_name):
        pd.DataFrame(self.train_data).to_csv(f"{file_name}.csv", index=False)
    def run_predict(self, prob_model, arg_model):
        assert(self.ann_file == None)
        ret = []
        for i, file in enumerate(self.file):
            paragraphs = []
            data =[]
            
            with open(file,"r") as f: 
                for idx, line in enumerate(f.readlines()):
                    if idx == 0: # skip prompt 
                        prompt = line
                        continue
                    if idx == 1: # skip the additional newline after prompt 
                        just_in_case = line
                        continue
                    paragraphs.append(line)

            adjust_sen = 0
            for k, para in enumerate(paragraphs):
                document = self.client.annotate(para)   
                for i, sent in enumerate(document.sentence):
                    for j, token in enumerate(sent.token):
                        start = len(prompt) + len(just_in_case) + adjust_sen + token.beginChar
                        punc = self.annotate_punc(token, data)
                        d = {
                            'essay':file,
                            'token':token.word,
                            'lemma':token.lemma,
                            'sentence':i,
                            'index':j+1,
                            'start':start,
                            'pos':token.pos,
                            'sentiment':sent.sentiment,
                            'paragraph':k,
                            'docPosition':self.paragraph_pos(k, len(paragraphs)),
                            'sentPosition':self.sent_pos(token.endIndex, len(sent.token)),
                            'isPunc':punc==1,
                            'followsPunc':punc==2,
                            'precedesPunc':False,
                            'followsLCA':None,
                            'precedesLCA':None,
                            'followsLCAPath':None,
                            'precedesLCAPath':None,
                            'head':self.get_head(sent.basicDependencies, j+1),
                            'uppermost':None,
                            'uppermost_child':None,
                            'right_sibling':None,
                            'right_sibling_head':None,
                            'probability':None,
                            # 'IOB':None
                        }
                        data.append(d)
                    sentence_data = data[-j-1:]
                    depth = self.tree_depth(sent.parseTree, 0)
                    # print(depth)
                    lexical_heads = self.get_heads(sent.basicDependencies)
                    lex_heads_dict = {}
                    for lex in lexical_heads:
                        lex_heads_dict[sentence_data[lex-1]['token']] = lex
                    for index, t in enumerate(sentence_data):
                        upper_path = self.uppermost(sent.parseTree, sentence_data[t['head']-1]['token'], t['token'])
                        t['uppermost'] = upper_path[0]
                        t['uppermost_child'] = self.uppermost_child(sent.parseTree, upper_path[1])
                        right_sib = self.uppermost_right(sent.parseTree, upper_path[1])
                        t['right_sibling'] = right_sib
                        sib_head = self.get_right_sib_head(sent.parseTree, sent.basicDependencies, right_sib, lex_heads_dict)
                        if sib_head:
                            t['right_sibling_head'] = sentence_data[sib_head-1]['token']
                        # t['head'] = '-'.join([sentence_data[t['head']-1]['token'], str(t['head'])])
                        if index == 0:
                            t["precedesLCAPath"] = -1
                            lca_output = self.LCA(sent.parseTree, sentence_data[index]['token'], None, sentence_data[index+1]['token'], depth)
                            t['followsLCAPath'] = lca_output[0][0]
                            t["followsLCA"] = lca_output[0][1]
                        elif index == len(sentence_data) - 1:
                            t["followsLCAPath"] = -1
                            lca_output = self.LCA(sent.parseTree, sentence_data[index]['token'], sentence_data[index-1]['token'], None, depth)
                            t['precedesLCAPath'] = lca_output[0][0]
                            t["precedesLCA"] = lca_output[0][1]
                        else:
                            lca_output = self.LCA(sent.parseTree, sentence_data[index]['token'], sentence_data[index-1]['token'], sentence_data[index+1]['token'], depth)
                            t['precedesLCAPath'] = lca_output[0][0]
                            t["precedesLCA"] = lca_output[0][1]
                            t['followsLCAPath'] = lca_output[1][0]
                            t["followsLCA"] = lca_output[1][1]
                adjust_sen += len(para)
            convert = {"Arg-B":0, "Arg-I":1, "O":2}
            for i, d in enumerate(data):
                if i == 0:
                    d['probability'] = 0.5
                elif i == 1:
                    d['probability'] = self.predict(prob_model, [convert[data[i-1]['IOB']], None, None])
                elif i == 2:
                    d['probability'] = self.predict(prob_model, [convert[data[i-1]['IOB']], convert[data[i-2]['IOB']], None])
                else:
                    d['probability'] = self.predict(prob_model, [convert[data[i-1]['IOB']], convert[data[i-2]['IOB']], convert[data[i-3]['IOB']]])
            ret.extend(data)
        for d in ret:
            d['IOB'] = arg_model.tag(d)
        return ret