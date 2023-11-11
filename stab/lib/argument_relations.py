corenlp_dir = '../corenlp'
# stanza.install_corenlp(dir=corenlp_dir)
import os
os.environ["CORENLP_HOME"] = corenlp_dir
from stanza.server import CoreNLPClient
import pandas as pd 
from itertools import groupby
from operator import itemgetter
from pos import pos

class ArgumentRelationIdentification(): 
    def __init__(self, data, client): 
        self.data = data
        self.client = client
        self.pairwise = {} # key is the essay idx (1-indexed). 

    def get_components(self):   
        self.components = {}
        for essay in groupby(self.data, itemgetter('essay')):
            # essay number
            essay_idx = int(essay[0].split("essay")[1].split(".txt")[0])  
            # create list to store the components for this essay 
            self.components[essay_idx] = []
            # a flag to help us identify if we are currently within a component 
            found_argB = False
            # flag to help us identify if a component is first in the paragraph 
            first_component = True 
            prior_paragraph_idx = 0
            prior_component_idx = 0 
            for token in essay[1]: 
                if token["IOB"] != "O":
                    if token["IOB"] == "Arg-B": 
                        found_argB = True 
                        # initialize info dictionary for the current component 
                        component = {
                            "tokens": [], # the length of this list will be used for getting token stats 
                            "pos_dist":pos.copy(), # pos distribution 
                            "nouns": [], # for getting shared nouns between pairs 
                            "start_position": token["start"],
                            "production_rules":[],
                            "paragraph_idx":token["paragraph"], 
                            "sentence_idx":token["sentence"],
                            "position_in_doc": token["docPosition"],
                            "first_or_last_in_paragraph": False, 
                            "indicator_type": None, 
                            "indicator_context": None, 
                            "discourse_triples":None, 
                            "pmi": None, 
                        }
                        if component["paragraph_idx"] > prior_paragraph_idx: 
                            # reset flag if we are in a new paragraph 
                            first_component = True 
                            # we know that the prior component must have been last in the prior paragraph 
                            self.components[essay_idx][prior_component_idx-1]["first_or_last_in_paragraph"] = True
                            prior_paragraph_idx += 1  
                        # the component is first in the paragraph 
                        if first_component: 
                            component["first_or_last_in_paragraph"] = True 
                            first_component = False
                        prior_component_idx += 1 
                    # add token to list component tokens 
                    component["tokens"].append(token["token"])
                    # update the POS distribution of component 
                    if token['pos'] in pos:
                        component["pos_dist"][token['pos']] += 1
                    # if the token is a noun, add its lemma to the nouns list 
                    if "NN" in token["pos"]: 
                        component["nouns"].append(token["lemma"])
                elif token["IOB"] == "O" and found_argB: 
                    # reached the end of a component, so reset our flag 
                    found_argB = False
                    # add component dictionary to list of components for the current essay 
                    self.components[essay_idx].append(component)

            # print(f"essay {essay_idx}")
            # for c in self.components[essay_idx]: 
            #     print(c["paragraph_idx"],c["first_or_last_in_paragraph"])
    
    def pairwise_features(self,essay_idx): 
        # key will be (i,j) where i is the idx of a source and j is the idx of a target 
        self.pairwise[essay_idx] = {}
        components = self.components[essay_idx]
        for i, source in enumerate(components):
            for j, target in enumerate(components):
                # the source cannot equal the target 
                if i == j: continue
                # the components must be in the same paragraph! 
                if source["paragraph_idx"] != target["paragraph_idx"]: 
                    continue 

                pairwise_info = {
                    # initialize the part-of-speech distribution for the pair with that of the source  
                    "pos_dist": source["pos_dist"].copy(),
                    # number of tokens in both source and target 
                    "num_tokens": len(source["tokens"]) + len(target["tokens"]),
                    # if source and target are present in the same sentence
                    "same_sentence": 0, # default is false 
                    # if target present before source 
                    "target_before_source": 0, # default is false,
                    # if pair is present in intro or conclusion. They are both in the same paragraph
                    "intro_or_conc": 0, # default is false 
                }
                # update binary POS distribution with the POS distribution of the target  
                for pos_type, count in target["pos_dist"].items():
                    pairwise_info["pos_dist"][pos_type] += count 
                # source and target are present in the same sentence
                if source["sentence_idx"] == target["sentence_idx"]: 
                    pairwise_info["same_sentence"] = 1 # true 
                # target is present before source 
                if source["start_position"] > target["start_position"]: 
                    pairwise_info["target_before_source"] = 1 # true  
                # if pair is present in intro or conclusion. They are both in the same paragraph 
                if source["position_in_doc"] == "Introduction" or source["position_in_doc"] == "Conclusion": 
                    pairwise_info["intro_or_conc"] = 1 
                # if target and source are first or last component in paragraph 
                if source["first_or_last_in_paragraph"] and target["first_or_last_in_paragraph"]: 
                    pairwise_info["first_or_last"] = 1 

                # add to information dictionary 
                self.pairwise[essay_idx][(i,j)] = pairwise_info
        
        for pair,info in self.pairwise[essay_idx].items(): 
            print(pair, ": ", info)
            break
        
    def syntactic(self): 
        # production rules 
        return

if __name__=='__main__':
    # client = CoreNLPClient(
    #     annotators=['tokenize','ssplit', 'pos', 'lemma'], 
    #     memory='4G', 
    #     endpoint='http://localhost:9005',
    #     be_quiet=True)
    # client.start()
    data = pd.read_csv('sample.csv').to_dict('records')
    argrelation = ArgumentRelationIdentification(data, None)
    argrelation.get_components()
    for essay in argrelation.components: 
        argrelation.pairwise_features(essay)
    # client.stop()