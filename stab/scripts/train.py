import pandas as pd
from statsmodels.discrete.discrete_model import Poisson
import sys, os
path = os.path.abspath("../lib")
sys.path.append(path)
from argument_identification import ArgumentIdentification
from argument_classification import ArgumentClassification
from argument_relations import ArgumentRelationIdentification
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
    print("Argument Identification")
    identifier = ArgumentIdentification(client, train_text, train_ann)
    identifier.run_annotated()
    data = pd.DataFrame(identifier.train_data)
    data.to_csv("../outputs/identification.csv")
    # save models
    with open('../models/identification_probability.json', 'w') as f:
        json.dump(identifier.probability, f)
    
    data = pd.read_csv("../outputs/identification.csv")
    print("Argument Classification")
    argclass = ArgumentClassification(data.to_dict('records'), client, data.lemma.values.tolist())
    argclass.process_data('train')
    # with open("../outputs/components.json", "w") as f:
    #     json.dump(argclass.components, f)
    with open("../models/probability.json", "w") as f:
        json.dump(argclass.probability, f)
    with open("../models/dependency.json", "w") as f:
        json.dump(argclass.dependency_tuples, f)
    with open('../models/classification_vectorizer.pkl', 'wb') as f:
        pickle.dump(argclass.vectorizer, f)
    
    essays = {}  
    for txt in train_text:
        essays[txt.split('/')[-1]] = []
    for d in argclass.components:
        if d['essay'].split('/')[-1] in essays.keys():
            e = d['essay'].split('/')[-1]
            essays[e].append(d)
    for key in essays.keys():
        with open(f"../outputs/classification/{key}.json", "w") as f:
            json.dump(essays[key], f)
    
    print("Relation Identification")

    for essay_name in train_text: 
        # read component data for this essay 
        with open(f'../outputs/classification/{essay_name}.json') as file: 
            components = json.load(file)
        # relation information for each essay 
        relation_info_file = "../models/argument_relation_info.json"
        # relation probabilities 
        relation_prob_file = "../models/relation_probabilities.json"
        # lemma information for components of training data 
        lemma_file = "../models/training_data_lemmas.json"
        # run argument relation features extraction 
        argrelation = ArgumentRelationIdentification(essay_name, components,relation_prob_file,lemma_file,relation_info_file)
        with open(f"../outputs/relations/{essay_name}.json", "w") as file:
            json.dump(argrelation.pairwise, file)
    
    print("Stance Identification")
    for essay_file in train_text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/classification/{essay_name}.json') as file: 
            components = json.load(file)
        stance = StanceRecognition(components)
        stance.process_data()
        with open(f"../outputs/stance/{essay_name}.json", "w") as file:
            json.dump(stance.components, file)
        


    
    client.stop()