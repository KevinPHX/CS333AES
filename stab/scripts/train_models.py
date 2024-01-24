import pickle
import pycrfsuite
import pandas as pd
from sklearn.svm import SVC
import json
import numpy as np
import sys, os
path = os.path.abspath("../lib")
sys.path.append(path)
from features import COMPONENT_FEATURES, RELATION_FEATURES, STANCE_FEATURES

if __name__== '__main__':
    train_text = open("../assets/train_text.txt", "r").read().split('\n')
    # print("Training CRF for Argument Identification")
    # test = pd.read_csv('../outputs/identification_bert.csv')
    # sent_x = []
    # sent_y = []
    # for e in set(test.essay.values):
    #     for p in set(test[test.essay==e].paragraph.values):
    #         for s in set(test[(test.essay==e)&(test.paragraph==p)].sentence.values):
    #             temp_test = test[(test.essay == e) & (test.paragraph == p) & (test.sentence == s)]
    #             # if not (count_o > max_len and count_i > max_len) :
    #             #     unique, counts = np.unique(temp_test.IOB.values, return_counts=True)
    #             #     un = dict(zip(unique, counts))
    #             #     print(un)
    #             #     if 'O' in un.keys():
    #             #         count_o += un['O']
    #             #     if 'Arg-I' in un.keys():
    #             #         count_i += un['Arg-I']

    #             sent_x.append(temp_test.loc[:, ~temp_test.columns.isin(['Unnamed: 0','IOB', 'essay', 'head_lemma', 'token_sentiment', 'sentence_sentiment'])].to_dict("records"))
    #             sent_y.append(temp_test.IOB.values)
    # trainer = pycrfsuite.Trainer(verbose=False)
    
    # for xseq, yseq in zip(sent_x, sent_y):
    #     trainer.append(xseq, yseq)
    # trainer.set_params({
    #     'c1': 1.0,   # coefficient for L1 penalty
    #     'c2': 1e-3,  # coefficient for L2 penalty
    #     'max_iterations': 50,  # stop earlier

    #     # include transitions that are possible, but not observed
    #     'feature.possible_transitions': True
    # })
    # trainer.train('../models/argument_identification_bert.crfsuite')
    # print(trainer.logparser.last_iteration)
    # print("Done!")
    print("Training SVM for Argument Classification")
    
    X_class = []
    y_class = []
    clf_class = SVC(gamma='scale')
    legend = {"MajorClaim":0, "Claim":1, "Premise":2}
    for essay_file in train_text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/classification_bert/{essay_name}.json') as file: 
            components = json.load(file)
        for c in components:
            x = []
            for key in COMPONENT_FEATURES:
                if c[key] == None:
                    x.append(0)
                else:
                    x.append(c[key])
            # if legend[x[-1]] == 0:
            #     continue
            # else:
            X_class.append(x[:-1])

            y_class.append(legend[x[-1]])
    clf_class.fit(X_class, y_class)
    with open('../models/classification_model_bert.pkl', 'wb') as f:
        pickle.dump(clf_class, f)

    print("Done!")

    print("Training SVM for Argument Relation Identification")
    X_rel = []
    y_rel = []
    clf_rel = SVC(gamma='auto')
    max_count_rel = 2522
    count = 0
    for essay_file in train_text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/relations_bert/{essay_name}.json') as file: 
            relations = json.load(file)
    
       
       
        for r in relations.keys():
            x = []
           
            for key in RELATION_FEATURES:
                if key not in relations[r].keys():
                    x.append(0)
                else:
                    x.append(relations[r][key])
            if x[0] == 0:
                if count < max_count_rel:
                    X_rel.append(x[1:])
                    y_rel.append(x[0])
                    count += 1
            else:
                X_rel.append(x[1:])
                y_rel.append(x[0])

    clf_rel.fit(X_rel, y_rel)
    with open('../models/relation_model_bert.pkl', 'wb') as f:
        pickle.dump(clf_rel, f)
    
    print("Done!")

    print("Training SVM for Stance Identification")
    X_stance = []
    y_stance = []
    clf_stance = SVC(gamma='auto')
    stance_legend = {'Against':0,'For':1}
    max_count = 226
    count = 0
    for essay_file in train_text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/stance_bert/{essay_name}.json') as file: 
            stance = json.load(file)
    
        for s in stance:
            x = [s[key] for key in STANCE_FEATURES]
            if x[-1] =='For':
                if count < max_count:
                    X_stance.append(x[:-1])
                    y_stance.append(stance_legend[x[-1]])
                    count += 1
            else:
                X_stance.append(x[:-1])
                y_stance.append(stance_legend[x[-1]])
           
    clf_stance.fit(X_stance, y_stance)
    with open('../models/stance_model_bert.pkl', 'wb') as f:
        pickle.dump(clf_stance, f)
    






