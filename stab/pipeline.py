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

# Structural

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

# Labelling


def iob(start, components):
    for c in components: 
        if c["start"] == start: # token that begins a component
            return "Arg-B"
        elif c["start"] < start < c["end"]: # token that is covered by an argument component 
            return "Arg-I"
    return "O"    


# Syntactical

def tree_depth(parse_tree, depth):
    if len(parse_tree.child) == 0:
        return depth
    max_depth = 0
    for child in parse_tree.child:
        temp_depth = tree_depth(parse_tree=child, depth=depth+1)
        if max_depth < temp_depth:
            max_depth = temp_depth
    return max_depth

def tree_path(parse_tree, target, path):
    if parse_tree.value == target:        
        path.append(parse_tree.value)
        return True
    if len(parse_tree.child) == 0:
        return False
    for child in parse_tree.child:
        temp_check = tree_path(child, target, path)
        if temp_check:
            path.append(parse_tree.value)
            return True
    return False

def preceding_LCA(parse_tree, token, preceding):
    preceding_path = []
    token_path = []
    if tree_path(parse_tree, token, token_path) and tree_path(parse_tree, preceding, preceding_path):
        preceding_path = preceding_path[::-1]
        token_path = token_path[::-1]
        for i in range(min(len(preceding_path), len(token_path))):
            if preceding_path[i] != token_path[i]:
                
                return i, token_path[i-1]
        if preceding_path[i-1] == token_path[i-1]:
            return i, token_path[i-1]
    else:
        return 0, None


def following_LCA(parse_tree, token, following):
    following_path = []
    token_path = []
    if tree_path(parse_tree, token, token_path) and tree_path(parse_tree, following, following_path):
        following_path = following_path[::-1]
        token_path = token_path[::-1]
        for i in range(min(len(following_path), len(token_path))):
            if following_path[i] != token_path[i]:
                return i, token_path[i-1]
        if following_path[i-1] == token_path[i-1]:
            return i, token_path[i-1]
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
        ret.append((lca_path_prec_path/depth, lca_path_prec))
    if following:
        lca_path_fol_path, lca_path_fol = following_LCA(parse_tree, token, following)
        ret.append((lca_path_fol_path/depth, lca_path_fol))
    return ret

# LEXICO-SYNTACTIC
def get_head(tree, index):
    if index == tree.ListFields()[2][1][0]:
        return index
    for each in tree.ListFields()[1][1]:
        if each.target == index:
            return each.source
    return 0
def uppermost(parse_tree, head, token):
    if head == token:
        return 'ROOT', ['ROOT']
    head_path = []
    token_path = []
    if tree_path(parse_tree, token, token_path) and tree_path(parse_tree, head, head_path):
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

def uppermost_child(parse_tree, true_path):
    if len(true_path) == 1 :
        if parse_tree.child:
            return parse_tree.child[0].value
    if parse_tree.value ==true_path[0]:
        for child in parse_tree.child:
            if child.value == true_path[1]:
                return uppermost_child(child, true_path[1:])

def uppermost_right(parse_tree, true_path):
    if len(true_path) == 1 :
        if parse_tree.child:
            return parse_tree.child[-1].value
    if parse_tree.value ==true_path[0]:
        for child in parse_tree.child:
            if child.value == true_path[1]:
                return uppermost_right(child, true_path[1:])

def get_heads(dep):
    heads = set()
    for each in dep.ListFields()[1][1]:
        heads.add(each.target)
    return list(heads)
def get_right_sib_head(tree, dep, right_sib, heads):
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
            return get_head(dep, heads[target_head.value])
        else:
            if len(children) > 0:
                target_head = children.pop()
            else:
                break
    return None


#     if parse_tree.value == true_path[0]:        
#         path.append(parse_tree.value)
#     if len(true_path) == 0:
#         return False
#     for child in parse_tree.child:
#         temp_check = tree_path(child, true_path[1:], path)
#         if temp_check:
#             path.append(parse_tree.value)
#             return True
#     return False

# def uppermost_child():

#     return

# PROBABILITY

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
                    x = window[:size-1][::-1].tolist()
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
    # print(probabilities)
    index = np.argmax(probabilities)
    return probabilities[index][0]
    


if __name__ == '__main__':
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse'], 
        memory='4G', 
        endpoint='http://localhost:9005',
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
                    'head':get_head(sent.basicDependencies, j+1),
                    'uppermost':None,
                    'uppermost_child':None,
                    'right_sibling':None,
                    'right_sibling_head':None,
                    'probability':None,
                    'IOB':iob(start, component_i)
                }
                data.append(d)
            sentence_data = data[-j-1:]
            depth = tree_depth(sent.parseTree, 0)
            # print(depth)
            lexical_heads = get_heads(sent.basicDependencies)
            lex_heads_dict = {}
            for lex in lexical_heads:
                lex_heads_dict[sentence_data[lex-1]['token']] = lex
            for index, t in enumerate(sentence_data):
                upper_path = uppermost(sent.parseTree, sentence_data[t['head']-1]['token'], t['token'])
                t['uppermost'] = upper_path[0]
                t['uppermost_child'] = uppermost_child(sent.parseTree, upper_path[1])
                right_sib = uppermost_right(sent.parseTree, upper_path[1])
                # if right_sib != t['token'] and upper_path[0] != right_sib :
                t['right_sibling'] = right_sib
                sib_head = get_right_sib_head(sent.parseTree, sent.basicDependencies, right_sib, lex_heads_dict)
                if sib_head:
                    t['right_sibling_head'] = sentence_data[sib_head-1]['token']
                t['head'] = '-'.join([sentence_data[t['head']-1]['token'], str(t['head'])])
                if index == 0:
                    t["precedesLCAPath"] = -1
                    lca_output = LCA(sent.parseTree, sentence_data[index]['token'], None, sentence_data[index+1]['token'], depth)
                    t['followsLCAPath'] = lca_output[0][0]
                    t["followsLCA"] = lca_output[0][1]
                elif index == len(sentence_data) - 1:
                    t["followsLCAPath"] = -1
                    lca_output = LCA(sent.parseTree, sentence_data[index]['token'], sentence_data[index-1]['token'], None, depth)
                    t['precedesLCAPath'] = lca_output[0][0]
                    t["precedesLCA"] = lca_output[0][1]
                else:
                    lca_output = LCA(sent.parseTree, sentence_data[index]['token'], sentence_data[index-1]['token'], sentence_data[index+1]['token'], depth)
                    t['precedesLCAPath'] = lca_output[0][0]
                    t["precedesLCA"] = lca_output[0][1]
                    t['followsLCAPath'] = lca_output[1][0]
                    t["followsLCA"] = lca_output[1][1]
        adjust_sen += len(para)
    convert = {"Arg-B":0, "Arg-I":1, "O":2}
    train_set = [convert[x["IOB"]] for x in data]
    models = train([train_set], 0, 3)
    for i, d in enumerate(data):
        if i == 0:
            d['probability'] = 0.5
        elif i == 1:
            d['probability'] = predict(models, [convert[data[i-1]['IOB']], None, None])
        elif i == 2:
            d['probability'] = predict(models, [convert[data[i-1]['IOB']], convert[data[i-2]['IOB']], None])
        else:
            d['probability'] = predict(models, [convert[data[i-1]['IOB']], convert[data[i-2]['IOB']], convert[data[i-3]['IOB']]])
    pd.DataFrame(data).to_csv("identification.csv", index=False)







    client.stop()
