import pandas as pd
import sys, os
path = os.path.abspath("../CS333_Project")
sys.path.append(path)
from features import COMPONENT_FEATURES

from argument_classification import ArgumentClassification

corenlp_dir = '../corenlp'
import pickle
import json
os.environ["CORENLP_HOME"] = corenlp_dir
from stanza.server import CoreNLPClient

def features(start,end):

    # Import Classification Models
    with open(f"{path}/CS333AES/stab/models/dependency.json", "r") as f:
        dependency_tuples = json.load(f)
    with open(f'{path}/CS333AES/stab/models/classification_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f"{path}/CS333AES/stab/models/probability.json", "r") as f:
        probability = json.load(f)

    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse'], 
        memory='4G', 
        endpoint='http://localhost:9005',
        be_quiet=True)
    client.start()
    
    # for the first batch, read in the first 200 essays of Set 2 
    sample = {}
    with open(f"/Users/amycweng/Downloads/CS333_Project/asap-aes/test_essays.json") as file: 
        text = json.load(file) 
    count = 0
    for name in text:
        count += 1  
        sample[name] = text[name]["text"]
        if count == 1: break
    
    data = pd.read_csv(f"{path}/CS333AES/stab/outputs/asap_set2/identification.csv")
    print("Argument Classification")
    argclass = ArgumentClassification(data.iloc[start:end].to_dict('records'), client, data.lemma.values.tolist(), probability, vectorizer, dependency_tuples)
    argclass.process_data('eval')
    
    client.stop()

    essays = {}  
    for d in argclass.components:
        e = d['essay']
        if e in essays.keys():
            essays[e].append(d)
        else: 
            essays[e] = [d]
    
    for key in essays.keys():
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification/{key}.json", "w") as f:
            json.dump(essays[key], f)


def predict():
    with open(f'{path}/models/classification_model.pkl', 'rb') as f:
        classifier_model = pickle.load(f)
    
    sample = {}
    with open(f"/Users/amycweng/Downloads/CS333_Project/asap-aes/test_essays.json") as file: 
        text = json.load(file) 
    count = 0
    for name in text:
        count += 1  
        sample[name] = text[name]["text"]
        if count == 200: break

    print('Argument Classification')


    X_class = []
    y_class = []
    legend = {"MajorClaim":0, "Claim":1, "Premise":2}
    for essay_file in None: 
        essay_name = essay_file.split("-final/")[1]
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification/{essay_name}.json") as file: 
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
    with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification_predictions.json","w") as file: 
        json.dump(y_pred.tolist(),file)

if __name__ == '__main__':
    # # essays 2978 to 3050
    # features(0,29659)
    # # essays 3051 to 3077
    # features(29660,41583)
    # # essays 3078 to 3129
    # features(41584,61753)
    # # essays 3130 to 3177 
    features(61754,81305)
    