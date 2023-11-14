import pandas as pd
import numpy as np
import sys, os
path = os.path.abspath("../lib")
sys.path.append(path)
import pickle
import json
import pycrfsuite
from itertools import chain
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from features import COMPONENT_FEATURES, RELATION_FEATURES, STANCE_FEATURES



if __name__=='__main__':


    with open('../models/identification_probability.json', 'r') as f:
        identifier_prob = json.load(f)
    tagger = pycrfsuite.Tagger()
    tagger.open('../models/argument_identification.crfsuite')          

    with open('../models/classification_model.pkl', 'rb') as f:
        classifier_model = pickle.load(f)
    # Import Relation Models
    with open('../models/relation_model.pkl', 'rb') as f:
        relation_model = pickle.load(f)

    
    # Import Stance Model
    with open('../models/stance_model.pkl', 'rb') as f:
        stance_model = pickle.load(f)

    text = open("../assets/test_text.txt", "r").read().split('\n')



    print('Argument Identification')


    test = pd.read_csv('../outputs/test/identification.csv')
    sent_x = []
    sent_y = []
    for e in set(test.essay.values):
        for p in set(test[test.essay==e].paragraph.values):
            for s in set(test[(test.essay==e)&(test.paragraph==p)].sentence.values):
                temp_test = test[(test.essay == e) & (test.paragraph == p) & (test.sentence == s)]
                sent_x.append(temp_test.loc[:, ~temp_test.columns.isin(['Unnamed: 0','IOB', 'essay', 'head_lemma', 'token_sentiment', 'sentence_sentiment'])].to_dict("records"))
                sent_y.append(temp_test.IOB.values)
    
    y_pred = []
    for i, each in enumerate(sent_x):
        y_pred.append(tagger.tag(each))

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(sent_y)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    report = classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
    )
    print(report)


    print('Argument Classification')


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
    y_pred = classifier_model.predict(X_class)
    # print(y_pred - y_class)
    
    report = classification_report(
            y_class,
            y_pred,
            target_names = legend.keys(),
    )
    print(report)
    print(confusion_matrix(y_class, y_pred))




    print('Argument Relation Identification')
    X_rel = []
    y_rel = []
    for essay_file in text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/test/relations/{essay_name}.json') as file: 
            relations = json.load(file)
    
       
       
        for r in relations.keys():
            x = []
           
            for key in RELATION_FEATURES:
                if key not in relations[r].keys():
                    x.append(0)
                else:
                    x.append(relations[r][key])
            X_rel.append(x[1:])
            y_rel.append(x[0])
    y_pred = relation_model.predict(X_rel)

    report = classification_report(
            y_rel,
            y_pred,
            target_names = ["No", "Yes"],
    )
    print(report)
    print(confusion_matrix(y_rel, y_pred))



    print("Stance Identification")
    X_stance = []
    y_stance = []
    stance_legend = {'Against':0,'For':1}
    for essay_file in text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/test/stance/{essay_name}.json') as file: 
            stance = json.load(file)

        for s in stance:
            x = [s[key] for key in STANCE_FEATURES]
            X_stance.append(x[:-1])
            y_stance.append(stance_legend[x[-1]])
    
    y_pred = stance_model.predict(X_stance)

    report = classification_report(
            y_stance,
            y_pred,
            target_names = stance_legend.keys(),
    )
    print(report)
    print(confusion_matrix(y_stance, y_pred))

  