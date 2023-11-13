import pandas as pd
import sys, os
path = os.path.abspath("../lib")
sys.path.append(path)
from argument_identification import ArgumentIdentification
from argument_classification import ArgumentClassification
from argument_relations import ArgumentRelationIdentification, relations
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
    # print("Argument Identification")
    # identifier = ArgumentIdentification(client, text, ann, identifier_prob)
    # identifier.run_evaluate()
    # test = pd.DataFrame(identifier.predicted_data)
    # test.to_csv("../outputs/test/identification.csv")
    # test = pd.read_csv('../outputs/test/identification.csv')
    # sent_x = []
    # sent_y = []
    # for e in set(test.essay.values):
    #     for p in set(test[test.essay==e].paragraph.values):
    #         for s in set(test[(test.essay==e)&(test.paragraph==p)].sentence.values):
    #             temp_test = test[(test.essay == e) & (test.paragraph == p) & (test.sentence == s)]
    #             sent_x.append(temp_test.loc[:, ~temp_test.columns.isin(['Unnamed: 0','IOB', 'essay', 'head_lemma', 'token_sentiment', 'sentence_sentiment'])].to_dict("records"))
    #             sent_y.append(temp_test.IOB.values)
    
    # y_pred = []
    # for i, each in enumerate(sent_x):
    #     y_pred.append(tagger.tag(each))
    # print(y_pred[80])
    # print(sent_y[80])
    # for i,s in enumerate(y_pred):
    #     if 'Arg-B' in s:
    #         print(s)
    #         print(sent_y[i])
    # lb = LabelBinarizer()
    # y_true_combined = lb.fit_transform(list(chain.from_iterable(sent_y)))
    # y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    # tagset = set(lb.classes_)
    # tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    # class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    # report = classification_report(
    #         y_true_combined,
    #         y_pred_combined,
    #         labels = [class_indices[cls] for cls in tagset],
    #         target_names = tagset,
    # )
    # print(report)

    # save models
    
    
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
    
      
    X_class = []
    y_class = []
    legend = {"MajorClaim":0, "Claim":1, "Premise":2}
    for essay_file in text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/test/classification/{essay_name}.json') as file: 
            components = json.load(file)
        for c in components:
            x = []
            for key in COMPONENT_FEATURES:
                if c[key] == None:
                    x.append(0)
                else:
                    x.append(c[key])
            X_class.append(x[:-1])

            y_class.append(legend[x[-1]])
    classifier_model.predict(y_class)

    # print("Relation Identification")

    # arguments, relation_probabilities = relations(text)

    # with open("../models/arguments.json", "w") as f:
    #     json.dump(arguments, f)
    # with open("../models/relation_probabilities.json", "w") as f:
    #     json.dump(relation_probabilities, f)
        

    # for essay_file in text: 
    #     essay_name = essay_file.split("-final/")[1]
    #     # read component data for this essay 
    #     with open(f'../outputs/classification/{essay_name}.json') as file: 
    #         components = json.load(file)
    #     # access the actual relations data for this essay 
    #     argument = arguments[essay_name]
    #     # run argument relation features extraction 
    #     argrelation = ArgumentRelationIdentification(components,argument,relation_probabilities)
    #     with open(f"../outputs/relations/{essay_name}.json", "w") as file:
    #         json.dump(argrelation.pairwise, file)
    
    # print("Stance Identification")
    # for essay_file in text: 
    #     essay_name = essay_file.split("-final/")[1]
    #     with open(f'../outputs/classification/{essay_name}.json') as file: 
    #         components = json.load(file)
    #     stance = StanceRecognition(components)
    #     stance.process_data()
    #     with open(f"../outputs/stance/{essay_name}.json", "w") as file:
    #         json.dump(stance.components, file)
        


    
    client.stop()