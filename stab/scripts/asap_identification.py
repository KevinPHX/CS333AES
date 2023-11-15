import pandas as pd
import sys, os
path = os.path.abspath("../CS333_Project/CS333AES/stab")
sys.path.append(path)

from argument_identification import ArgumentIdentification
corenlp_dir = '../corenlp'
import pickle
import json
import pycrfsuite


import os
os.environ["CORENLP_HOME"] = corenlp_dir

from stanza.server import CoreNLPClient

path = '/Users/amycweng/Downloads/CS333_Project/CS333AES/stab'
if __name__ == '__main__':

    # Import ID models
    with open(f'{path}/models/identification_probability.json', 'r') as f:
        identifier_prob = json.load(f)
    tagger = pycrfsuite.Tagger()
    tagger.open('/Users/amycweng/Downloads/CS333_Project/models/argument_identification.crfsuite')           

    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse'], 
        memory='4G', 
        endpoint='http://localhost:9005',
        be_quiet=True)
    client.start()
    sample = {}
    with open(f"/Users/amycweng/Downloads/CS333_Project/asap-aes/test_essays.json") as file: 
        text = json.load(file) 
    count = 0
    for id, info in text.items():
        count += 1  
        sample[id] = info["text"]
        if count == 200: break

    '''Extract features for the component identification step'''
    print("Argument Identification - Features")
    identifier = ArgumentIdentification(client, sample, None, identifier_prob)
    identifier.run_evaluate()
    test = pd.DataFrame(identifier.predicted_data)
    test.to_csv(f"{path}/outputs/asap_set2/identification_01.csv")
    client.stop()

    '''Predict components'''
    with open(f'{path}/models/identification_probability.pkl', 'rb') as f:
        identifier_prob = pickle.load(f)
    tagger = pycrfsuite.Tagger()
    tagger.open('/Users/amycweng/Downloads/CS333_Project/models/argument_identification.crfsuite')          

    print('Argument Identification - Predictions')
    test = pd.read_csv(f'{path}/outputs/asap_set2/identification.csv')
    sent_x = []
    sent_y = []
    for e in sorted(set(test.essay.values)):
        for p in set(test[test.essay==e].paragraph.values):
            for s in set(test[(test.essay==e)&(test.paragraph==p)].sentence.values):
                temp_test = test[(test.essay == e) & (test.paragraph == p) & (test.sentence == s)]
                sent_x.append(temp_test.loc[:, ~temp_test.columns.isin(['Unnamed: 0','IOB', 'head_lemma', 'token_sentiment', 'sentence_sentiment'])].to_dict("records"))
                sent_y.append(temp_test.essay.values)

    y_pred = {}
    for i, sentence in enumerate(sent_x):
        tags = tagger.tag(sentence)
        essay = sentence[0]["essay"]
        if essay not in y_pred: 
            y_pred[essay] = []
        y_pred[essay].extend(tags)

    CURRENT = ""
    pred_idx = 0 
    rows = []
    for idx, essay in enumerate(test.essay.values):
        if essay != CURRENT: 
            if CURRENT != "": 
                print(f"{CURRENT} done")
            CURRENT = essay
            pred_idx = 0  
        row_idx = test["Unnamed: 0"][idx]
        row_dict = {}
        row = test.loc[test["Unnamed: 0"] == row_idx]
        for field, item in row.to_dict().items():
            if field == "Unnamed: 0": continue 
            row_dict[field] = item[row_idx]
        row_dict["IOB"] = y_pred[essay][pred_idx]
        rows.append(row_dict)
    results = pd.DataFrame(rows)
    results.to_csv(f"{path}/outputs/asap_set2/identification_01_predictions.csv")
