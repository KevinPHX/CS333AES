import pandas as pd
from statsmodels.discrete.discrete_model import Poisson
import sys, os
path = os.path.abspath("../")
sys.path.append(path)
from lib.argument_identification import ArgumentIdentification
corenlp_dir = '../corenlp'
import pickle
# stanza.install_corenlp(dir=corenlp_dir)

import os
os.environ["CORENLP_HOME"] = corenlp_dir

from stanza.server import CoreNLPClient


if __name__ == '__main__':
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse'], 
        memory='4G', 
        endpoint='http://localhost:9005',
        be_quiet=True)
    client.start()
    text = open("../assets/train_text.txt", "r").read().split('\n')
    ann = open("../assets/train_ann.txt", "r").read().split('\n')
    
    identifier = ArgumentIdentification(client, text, ann)
    identifier.run_annotated()
    pd.DataFrame(identifier.train_data).to_csv("../outputs/train.csv")
    # save models
    with open('../models/identification_probability.pkl', 'wb') as f:
        pickle.dump(identifier.models, f)
    
    
    
    client.stop()