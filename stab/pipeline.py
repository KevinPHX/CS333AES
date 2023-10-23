import stanza
import pandas as pd
import re
import numpy as np
from statsmodels.discrete.discrete_model import Poisson

corenlp_dir = './corenlp'
# stanza.install_corenlp(dir=corenlp_dir)

import os
os.environ["CORENLP_HOME"] = corenlp_dir

from stanza.server import CoreNLPClient


def annotate_punc(token, data):
    # check if punc
    if not re.search(r'[A-Za-z0-9]',token.originalText): # is punc
        # set precedes punc in last element to true
        data[-1]['precedesPunc'] = True
        return 1
    elif len(data) > 0:
        if data[-1]['isPunc']:
            return 2
    else:
        return 0

def sent_pos(tokenIdx, sentLength):
    if tokenIdx == 1: 
        return "First"
    elif tokenIdx == sentLength: 
        return "Last"
    else: 
        return "Middle"

def paragraph_pos(p_idx, paragraphLength):
    if p_idx == 0: 
        return "Introduction"
    elif p_idx == paragraphLength-1: 
        return "Conclusion"
    else: 
        return "Body"

def read_data(essay_ann_file): 
    components = []
    with open(essay_ann_file,"r") as f: 
        for line in f.readlines(): 
            line = line.strip('\n')
            line = line.split("\t")
            if "T" in line[0]: 
                components.append(line)
    return components
def preprocess_components(components):
    augmented = []
    for component in components: 
        name = component[0]
        claim,start,end = component[1].split(' ')
        phrase = component[2]
        info = {"name": name, "claim": claim,"start":int(start),"end":int(end),"phrase": phrase}
        augmented.append(info)
    return augmented

def iob(start, components):
    for c in components: 
        if c["start"] == start: # token that begins a component
            return "Arg-B"
        elif c["start"] < start < c["end"]: # token that is covered by an argument component 
            return "Arg-I"
    return "O"    


def tree_depth(parse_tree, depth):
    if len(parse_tree.child) == 0:
        return depth
    max_depth = 0
    for child in parse_tree.child:
        temp_depth = tree_depth(parse_tree=child, depth=depth+1)
        if max_depth < temp_depth:
            max_depth = temp_depth
    return max_depth

def is_LCA(parse_tree, target, path):
    
    # path.append(parse_tree.value)
    
    if parse_tree.value == target:
        
        # print(parse_tree.value)
        path.append(parse_tree.value)
        # print(f"Tree Value: {parse_tree.value} Target: {target}")
        return True
    if len(parse_tree.child) == 0:
        return False
    for child in parse_tree.child:
        temp_check = is_LCA(child, target, path)
        if temp_check:
            # print(parse_tree.value)
            path.append(parse_tree.value)
            return True
    return False

def preceding_LCA(parse_tree, token, preceding):
    preceding_path = []
    token_path = []
    if is_LCA(parse_tree, token, token_path) and is_LCA(parse_tree, preceding, preceding_path):
        # print("IS TRUE")
        preceding_path = preceding_path[::-1]
        token_path = token_path[::-1]
        # print(preceding_path)
        # print(token_path)
        for i in range(min(len(preceding_path), len(token_path))):
            if preceding_path[i] != token_path[i]:
                
                return i, token_path[i]
        if preceding_path[i-1] == token_path[i-1]:
            return i, token_path[i]
    else:
        return 0, None


def following_LCA(parse_tree, token, following):
    following_path = []
    token_path = []
    if is_LCA(parse_tree, token, token_path) and is_LCA(parse_tree, following, following_path):
        # print(token_path)
        following_path = following_path[::-1]
        token_path = token_path[::-1]
        for i in range(min(len(following_path), len(token_path))):
            if following_path[i] != token_path[i]:
                return i, token_path[i]
        if following_path[i] == token_path[i]:
            return i, token_path[i]
        else:
            return 0, None
    else:
        return 0, None


def LCA(parse_tree, token, preceding, following, depth):
    ret = []
    if preceding:
        lca_path_prec_path, lca_path_prec  = preceding_LCA(parse_tree, token, preceding)
        if (lca_path_prec_path > depth):
            print(lca_path_prec_path)
            # print(depth)
        ret.append((lca_path_prec_path/depth, lca_path_prec))
    if following:
        lca_path_fol_path, lca_path_fol = following_LCA(parse_tree, token, following)
        ret.append((lca_path_fol_path/depth, lca_path_fol))
    return ret



def train(train_set, target, n):
    # increment by 2 to get the targeted tag value (y)
    window_sz = np.arange(n)+2
    X= [[] for i in range(n)]
    Y = [[] for i in range(n)]
    models = []
    for index, size in enumerate(window_sz):
        # preprocess data per dataset -> this is important when training across multiple docs
        for data in train_set:
            # creates sliding window of the desired size
            for window in np.lib.stride_tricks.sliding_window_view(data, size):
                if len(window) == size:
                    x = window[:size-1].tolist()
                    # makes the target a binary of whether the y value is the target (Arg-B) or not - this is what we are ultimately trying to predict
                    y = (window[-1]==target).astype(int).tolist()
                    X[index].append(x)  
                    Y[index].append(y)
        models.append(Poisson(Y[index], X[index]).fit())
    return models
def predict(models, X):
    probabilities = []
    # There should be n models, so we iterate across each model and get their respective probabilities
    for index, model in enumerate(models):
        if X[index]:
            probabilities.append(model.predict(X[:index+1]))
        else:
            probabilities.append(np.array([0]))
    # Find the max probability for any window size and return it
    print(probabilities)
    index = np.argmax(probabilities)
    return probabilities[index][0]
    


if __name__ == '__main__':
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment'], 
        memory='4G', 
        endpoint='http://localhost:9002',
        be_quiet=True)
    client.start()
    essayDir = 'data/ArgumentAnnotatedEssays-2.0 2/data/brat-project-final'
    for file in sorted(os.listdir(essayDir)):
        if ".ann" in file: 
            essay_ann_file = f"{essayDir}/{file}"
        if ".txt" in file: 
            essay_txt_file = f"{essayDir}/{file}"
        if "001.txt" in file: break
    components = read_data(essay_ann_file)
    data = []
    full_text = open(essay_txt_file, 'r').read().split('\n\n')
    prompt = ""
    paragraphs = []
    with open(essay_txt_file,"r") as f: 
        for idx, line in enumerate(f.readlines()):
            if idx == 0: # skip prompt 
                prompt = line
                continue
            if idx == 1: # skip the additional newline after prompt 
                continue
            paragraphs.append(line)


    essay = "".join(paragraphs)
    adjust_sen = 0
    for k, para in enumerate(paragraphs):
        document = client.annotate(para)   
        # sentence_data = []
        for i, sent in enumerate(document.sentence):
            component_i = preprocess_components(components)
            for j, token in enumerate(sent.token):
                start = len(prompt) + 1 + adjust_sen + token.beginChar
                punc = annotate_punc(token, data)
                d = {
                    'token':token.word,
                    'lemma':token.lemma,
                    'sentence':i,
                    'index':j+1,
                    'start':start,
                    'pos':token.pos,
                    'sentiment':sent.sentiment,
                    'paragraph':k,
                    'docPosition':paragraph_pos(k, len(paragraphs)),
                    'sentPosition':sent_pos(token.endIndex, len(sent.token)),
                    'isPunc':punc==1,
                    'followsPunc':punc==2,
                    'precedesPunc':False,
                    'followsLCA':None,
                    'precedesLCA':None,
                    'followsLCAPath':None,
                    'precedesLCAPath':None,
                    'lexsyn':None,
                    'probability':None,
                    'IOB':iob(start, component_i)
                }
                data.append(d)
            sentence_data = data[-j-1:]
            depth = tree_depth(sent.parseTree, 0)
            # print(depth)
            for index, t in enumerate(sentence_data):
                if index == 0:
                    t["precedesLCAPath"] = -1
                    lca_output = LCA(sent.parseTree, sentence_data[index]['pos'], None, sentence_data[index+1]['pos'], depth)
                    t['followsLCAPath'] = lca_output[0][0]
                    t["followsLCA"] = lca_output[0][1]
                elif index == len(sentence_data) - 1:
                    t["followsLCAPath"] = -1
                    lca_output = LCA(sent.parseTree, sentence_data[index]['pos'], sentence_data[index-1]['pos'], None, depth)
                    t['precedesLCAPath'] = lca_output[0][0]
                    t["precedesLCA"] = lca_output[0][1]
                else:
                    lca_output = LCA(sent.parseTree, sentence_data[index]['pos'], sentence_data[index-1]['pos'], sentence_data[index+1]['pos'], depth)
                    t['precedesLCAPath'] = lca_output[0][0]
                    t["precedesLCA"] = lca_output[0][1]
                    t['followsLCAPath'] = lca_output[1][0]
                    t["followsLCA"] = lca_output[1][1]
        # data.extend(sentence_data)
        adjust_sen += len(para)
    convert = {"Arg-B":0, "Arg-I":1, "O":2}
    train_set = [convert[x["IOB"]] for x in data]
    models = train([train_set], 0, 3)
    for i, d in enumerate(data):
        if i == 0:
            d['probability'] = 0.5
        elif i == 1:
            d['probability'] = predict(models, [convert[d['IOB']], None, None])
        elif i == 2:
            d['probability'] = predict(models, [convert[d['IOB']], convert[data[i-1]['IOB']], None])
        else:
            d['probability'] = predict(models, [convert[d['IOB']], convert[data[i-1]['IOB']], convert[data[i-2]['IOB']]])
    pd.DataFrame(data).to_csv("test.csv", index=False)







    client.stop()
