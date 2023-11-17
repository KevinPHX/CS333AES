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

def features(start,end,id_file):

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
    data = pd.read_csv(id_file)
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
    first = int(argclass.components[0]['essay'].split(".txt")[0])
    last = int(argclass.components[-1]['essay'].split(".txt")[0])
    sample = {}
    with open(f"/Users/amycweng/Downloads/CS333_Project/asap-aes/test_essays.json") as file: 
        text = json.load(file) 
    count = 0
    for name in text:
        sample[name] = text[name]["text"]
        if count == 500: break

    for key in sample:
        if first <= int(key) <= last: 
            key = f"{key}.txt"
            print(f"printing {key}")
            with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification/{key}.json", "w") as f:
                if key in essays: 
                    json.dump(essays[key], f)
                else: 
                    json.dump([],f)


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
        if count == 400: break
    
    print('Argument Classification')


    X_class = []
    y_class = []
    legend = {"MajorClaim":0, "Claim":1, "Premise":2}
    l  = {v:k for k,v in legend.items()}
    for essay in sample: 
        essay_name = essay 
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification/{essay_name}.txt.json") as file: 
            components = json.load(file)
            print(f"processing {essay_name}")
        for c in components:
            x = []
            for key in COMPONENT_FEATURES:
                if key == "dep_6499":
                    x.append(0)
                elif c[key] == None:
                    x.append(0)
                else:
                    x.append(c[key])
        
            X_class.append(x[:-1])

            y_class.append(l[x[-1]])
    y_pred = classifier_model.predict(X_class)
    with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification_predictions.json","w") as file: 
        json.dump(y_pred.tolist(),file)

if __name__ == '__main__':
    # first batch of essays 
    # file = f"{path}/CS333AES/stab/outputs/asap_set2/identification_0.csv"
    # essays 2978 to 3050
    # features(0,29660,file)
    # # essays 3051 to 3077
    # features(29660,41584,file)
    # # essays 3078 to 3129
    # features(41584,61754,file)
    # # essays 3130 to 3177 
    # features(61754,-1,file)

    # next batch of essays 
    # file = f"{path}/CS333AES/stab/outputs/asap_set2/identification_01.csv"

    # essays 3178 to 3250
    # features(0,32069,file)
    # # essays 3251 to 3277
    # features(32069,45074,file)
    # # essays 3278 to 3329 
    # features(45074,69686,file)
    # # essays 3330 to 3372 
    # features(69686, 87493,file) 
    # # essays 3373 to 3377
    # features(87493, -1,file)

    # predictions for all 400 essays 
    predict()
    