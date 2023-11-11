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

    def get_components(self): 
        self.components = {}
        for essay in groupby(self.data, itemgetter('essay')):
            # essay number
            idx = int(essay[0].split("essay")[1].split(".txt")[0]) + 1  
            # create list to store the components for this essay 
            self.components[idx] = []
            # a flag to help us identify if we are currently within a component 
            found_argB = False
            for token in essay[1]: 
                if token["IOB"] != "O":
                    if token["IOB"] == "Arg-B": 
                        found_argB = True 
                        # initialize info dictionary for the current component 
                        component = {
                            "tokens": [], # the length of this list will be used for getting token stats 
                            "pos_dist":pos.copy(), # pos distribution 
                            "nouns": [], # for getting shared nouns between pairs 
                            "production_rules":[],
                            "paragraph_idx":token["paragraph"], 
                            "sentence_idx":token["sentence"],
                            "docPosition": token["docPosition"],
                            "sentPosition":token["sentPosition"],
                            "indicator_type": None, 
                            "indicator_context": None, 
                            "discourse_triples":None, 
                            "pmi": None, 
                        }
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
                    self.components[idx].append(component)
        
        print(f"essay {idx}")
        for c in self.components[idx]: 
            print(c)

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
    # client.stop()