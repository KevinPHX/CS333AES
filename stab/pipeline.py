import stanza
import pandas as pd
import re
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
    # print(components)
    # print(start)
    for c in components: 
        if c["start"] == start: # token that begins a component
            return "Arg-B"
        elif c["start"] < start < c["end"]: # token that is covered by an argument component 
            return "Arg-I"
    return "O"    


if __name__ == '__main__':
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment'], 
        memory='4G', 
        endpoint='http://localhost:9002',
        be_quiet=True)
    client.start()
    # essayDir = "data/ArgumentAnnotatedEssays-2.0/data/brat-project-final"
    essayDir = 'data/ArgumentAnnotatedEssays-2.0 2/data/brat-project-final'
    for file in sorted(os.listdir(essayDir)):
        if ".ann" in file: 
            essay_ann_file = f"{essayDir}/{file}"
        if ".txt" in file: 
            essay_txt_file = f"{essayDir}/{file}"
        if "001.txt" in file: break
    components = read_data(essay_ann_file)
    print(components)
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
            if i == 0:
                f = open(f"test{i}.txt", "a")
                f.write(str(sent))
                f.close()
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
                    'LCA':(),
                    'followsLCA':(),
                    'precedesLCA':(),
                    'lexsyn':(),
                    'probability':(),
                    'IOB':iob(start, component_i)
                }
                data.append(d)
        adjust_sen += len(para)
    pd.DataFrame(data).to_csv("test.csv", index=False)







    client.stop()
