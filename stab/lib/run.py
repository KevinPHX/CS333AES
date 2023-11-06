import stanza
import pandas as pd
import re
import numpy as np
from statsmodels.discrete.discrete_model import Poisson
from argument_identification import ArgumentIdentification
corenlp_dir = '../corenlp'
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
    text = open("train_text.txt", "r").read().split('\n')
    ann = open("train_ann.txt", "r").read().split('\n')
    
    identifier = ArgumentIdentification(client, text, ann)
    identifier.run_annotated()
    pd.DataFrame(identifier.train_data).to_csv("ALL.csv")
    client.stop()