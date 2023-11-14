import pandas as pd
import numpy as np
import sys, os
path = os.path.abspath("../lib")
sys.path.append(path)
from argument_identification import ArgumentIdentification
from argument_classification import ArgumentClassification
from argument_relations import ArgumentRelationIdentification
from stance_recognition import StanceRecognition
from features import COMPONENT_FEATURES, RELATION_FEATURES, STANCE_FEATURES

corenlp_dir = '../corenlp'
import pickle
import json
import pycrfsuite
from itertools import chain
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix


# stanza.install_corenlp(dir=corenlp_dir)

import os
os.environ["CORENLP_HOME"] = corenlp_dir

from stanza.server import CoreNLPClient


if __name__ == '__main__':

    # Import ID models
    with open('../models/identification_probability.json', 'r') as f:
        identifier_prob = json.load(f)
    tagger = pycrfsuite.Tagger()
    tagger.open('../models/argument_identification.crfsuite')           
   

    # Import Classification Models
    with open("../models/dependency.json", "r") as f:
        dependency_tuples = json.load(f)
    with open('../models/classification_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open("../models/probability.json", "r") as f:
        probability = json.load(f)
    with open('../models/classification_model.pkl', 'rb') as f:
        classifier_model = pickle.load(f)
    # Import Relation Models
    with open('../models/relation_model.pkl', 'rb') as f:
        relation_model = pickle.load(f)

    # Import Stance Model
    with open('../models/stance_model.pkl', 'rb') as f:
        stance_model = pickle.load(f)





    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse'], 
        memory='4G', 
        endpoint='http://localhost:9005',
        be_quiet=True)
    client.start()
    text = open("../assets/test_text.txt", "r").read().split('\n')
    ann = open("../assets/test_ann.txt", "r").read().split('\n')
    print("Argument Identification")
    identifier = ArgumentIdentification(client, text, ann, identifier_prob)
    identifier.run_evaluate()
    test = pd.DataFrame(identifier.predicted_data)
    test.to_csv("../outputs/test/identification.csv")
        
    
    data = pd.read_csv("../outputs/test/identification.csv")
    print("Argument Classification")
    argclass = ArgumentClassification(data.to_dict('records'), client, data.token.values.tolist(), probability, vectorizer, dependency_tuples)
    argclass.process_data('eval')
    with open("../outputs/test/components.json", "w") as f:
        json.dump(argclass.components, f)
    with open("../outputs/test/components.json", "r") as f:
        components = json.load(f)
    
    
    essays = {}  
    for txt in text:
        essays[txt.split('/')[-1]] = []
    for d in components:
        if d['essay'].split('/')[-1] in essays.keys():
            e = d['essay'].split('/')[-1]
            essays[e].append(d)
    for key in essays.keys():
        with open(f"../outputs/test/classification/{key}.json", "w") as f:
            json.dump(essays[key], f)
    
    


        

    print("Relation Identification")

    for essay_name in text: 
        essay_name = essay_name.split("-final/")[1]
        # read component data for this essay 
        with open(f'../outputs/test/classification/{essay_name}.json') as file: 
            components = json.load(file)
        # relation information for each essay 
        relation_info_file = "../models/argument_relation_info_TEST_SET.json"
        # relation probabilities 
        relation_prob_file = "../models/relation_probabilities.json"
        # lemma information for components of training data 
        lemma_file = "../models/training_data_lemmas.json"
        # run argument relation features extraction 
        argrelation = ArgumentRelationIdentification(essay_name, components,relation_prob_file,lemma_file, relation_info_file)
        with open(f"../outputs/test/relations/{essay_name}.json", "w") as file:
            json.dump(argrelation.pairwise, file)
    
    print("Stance Identification")
    for essay_file in text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/test/classification/{essay_name}.json') as file: 
            components = json.load(file)
        stance = StanceRecognition(components)
        stance.process_data()
        with open(f"../outputs/test/stance/{essay_name}.json", "w") as file:
            json.dump(stance.components, file)
        


    
    # client.stop()