import pandas as pd
from statsmodels.discrete.discrete_model import Poisson
import sys, os
path = os.path.abspath("../lib")
sys.path.append(path)
from argument_identification import ArgumentIdentification
from argument_classification import ArgumentClassification
from stance_recognition import StanceRecognition
corenlp_dir = '../corenlp'
import pickle
import json

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
    train_text = open("../assets/train_text.txt", "r").read().split('\n')
    train_ann = open("../assets/train_ann.txt", "r").read().split('\n')
    # print("Argument Identification")
    # identifier = ArgumentIdentification(client, train_text, train_ann)
    # identifier.run_annotated()
    # data = pd.DataFrame(identifier.train_data)
    # data.to_csv("../outputs/train.csv")
    # # save models
    # with open('../models/identification_probability.pkl', 'wb') as f:
    #     pickle.dump(identifier.models, f)
    data = pd.read_csv("../outputs/train.csv")
    print("Argument Classification")
    argclass = ArgumentClassification(data.to_dict('records'), client, data.token.values.tolist())
    argclass.process_data(True)
    with open("../outputs/components.json", "w") as f:
        json.dump(argclass.components, f)
    with open("../models/probability.json", "w") as f:
        json.dump(argclass.probability, f)
    with open("../models/dependency.json", "w") as f:
        json.dump(argclass.dependency_tuples, f)
    


    
    client.stop()