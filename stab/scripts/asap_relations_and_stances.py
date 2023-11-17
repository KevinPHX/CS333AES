from argument_relations import ArgumentRelationIdentification
from stance_recognition import StanceRecognition
from features import RELATION_FEATURES, STANCE_FEATURES
import pickle
import json
import sys, os
path = os.path.abspath("../CS333_Project")
sys.path.append(path)

if __name__ == '__main__':
    sample = {}
    with open(f"/Users/amycweng/Downloads/CS333_Project/asap-aes/test_essays.json") as file: 
        text = json.load(file) 
    count = 0
    for name in text:
        count += 1  
        sample[name] = text[name]["text"]
        if count == 400: break

    # Import Relation Models
    with open(f'{path}/models/relation_model.pkl', 'rb') as f:
        relation_model = pickle.load(f)
    
    print("Relation Identification")

    for essay_name in sample: 
        essay_name = essay_name + ".txt"
        # read component data for this essay 
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification/{essay_name}.json") as file: 
            components = json.load(file)
        # relation probabilities 
        relation_prob_file = f"{path}/CS333AES/stab/models/relation_probabilities.json"
        # lemma information for components of training data 
        lemma_file = f"{path}/CS333AES/stab/models/training_data_lemmas.json"
        # run argument relation features extraction 
        argrelation = ArgumentRelationIdentification(essay_name, components,relation_prob_file,lemma_file, None)
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/relations/{essay_name}.json", "w") as file:
            json.dump(argrelation.pairwise, file)
    
    print('Argument Relation Identification')
    X_rel = []
    y_rel = []
    for essay_name in sample: 
        essay_name = essay_name + ".txt"
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/relations/{essay_name}.json") as file: 
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
    with open(f"{path}/CS333AES/stab/outputs/asap_set2/relation_predictions.json","w") as file: 
        json.dump(y_pred.tolist(),file)
    print(f"Done with relation identification")
   
    # Import Stance Model
    with open(f'{path}/models/stance_model.pkl', 'rb') as f:
        stance_model = pickle.load(f)
    
    # read in classification predictions 
    with open(f'{path}/CS333AES/stab/outputs/asap_set2/classification_predictions.json') as file: 
        c_predictions = json.load(file)
    label_to_type = {0: "MajorClaim", 1: "Claim", 2: "Premise"}
    c_idx = 0 # used to find prediction info 
    
    print("Stance Identification")
    for essay_name in sample: 
        essay_name = essay_name + ".txt"
        print(essay_name)
        # read in classification information 
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/classification/{essay_name}.json") as file: 
            components_info = json.load(file)
        for c in components_info: 
            c["claim"] = label_to_type[int(c_predictions[c_idx])]
            c_idx += 1
        stance = StanceRecognition(components_info)
        stance.process_data(stance_model, False)
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/stance/{essay_name}.json", "w") as file:
            json.dump(stance.components, file)

    
    print("Stance Identification")
    X_stance = []
    stance_legend = {'Against':0,'For':1}
    l = {v:k for k,v in stance_legend.items()}
    for essay_name in sample: 
        essay_name = essay_name + ".txt"
        print(essay_name)
        with open(f"{path}/CS333AES/stab/outputs/asap_set2/stance/{essay_name}.json") as file: 
            stance = json.load(file)

        for s in stance:
            x = []
            for key in STANCE_FEATURES: 
                if key not in s: 
                    x.append(0)
                else: 
                    x.append(s[key])
            X_stance.append(x[:-1])
    y_pred = stance_model.predict(X_stance)
    with open(f"{path}/CS333AES/stab/outputs/asap_set2/stance_predictions.json","w") as file: 
        json.dump(y_pred.tolist(),file)
  


