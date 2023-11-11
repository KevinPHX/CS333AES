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
            current_components = []
            found_argB = False
            idx = essay[0]
            for token in essay[1]: 
                if token["IOB"] != "O":
                    if token["IOB"] == "Arg-B": 
                        found_argB = True 
                        # initialize info dictionary for the current component 
                        component = {
                            "tokens": [token["token"]], # the length of this list will be used for getting token stats 
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
                elif token["IOB"] == "O" and not found_argB: 
                    # reached the end of a component 
                    found_argB = False

        return 

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
    # argrelation.process_data(True)
    # client.stop()