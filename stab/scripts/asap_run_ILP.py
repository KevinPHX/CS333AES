
from argumentILP_withJSON import ArgumentTrees
import json, os, sys
path = os.path.abspath("../CS333_Project/CS333AES/stab/outputs/asap_set2")
sys.path.append(path)

if __name__ == "__main__":    
    sample = {}
    with open(f"/Users/amycweng/Downloads/CS333_Project/asap-aes/test_essays.json") as file: 
        text = json.load(file) 
    count = 0
    for name in text:
        count += 1  
        sample[name] = text[name]["text"]
        if count == 200: break

    optimized_relations = {} # maps essay_name to another dictionary 
    overall_rates = {"TPR":[],"TNR":[],"FPR":[],"FNR":[],"accuracy":[],"precision":[]}
    label_to_type = {0: "MajorClaim", 1: "Claim", 2: "Premise"}
    c_idx, r_idx = 0, 0 # used to find prediction info  
    for essay in sample:
        essay_name = essay + ".txt"
        # read in relation information 
        with open(f'{path}/relations/{essay_name}.json') as file: 
            relation_info = json.load(file)
        # read in classification information 
        with open(f'{path}/classification/{essay_name}.json') as file: 
            components_info = json.load(file)
        # read in relation predictions 
        with open(f'{path}/relation_predictions.json') as file: 
            r_predictions = json.load(file)
        # read in classification predictions 
        with open(f'{path}/classification_predictions.json') as file: 
            c_predictions = json.load(file)
        for pair in relation_info:
            relation_info[pair]["is_predicted_relation"] = r_predictions[r_idx]
            r_idx += 1 
        for c in components_info: 
            c["predicted_type"] = label_to_type[int(c_predictions[c_idx])]
            c_idx += 1 

        # initialize class for data formatting & ILP evaluation 
        argument = ArgumentTrees()
        print(f"{essay_name}")
        argument.process_data(components_info, relation_info)
        argument.optimize()

        # add output to the optimized_relations dictionary 
        optimized_relations[essay_name] = {"relations": {}, "evaluation": None } 
        for source,target_list in argument.results_indices.items(): 
            for target in target_list: 
                optimized_relations[essay_name]["relations"][f"{source},{target}"] = True 

    with open(f"{path}/optimized_relations.json","w") as file: 
        json.dump(optimized_relations,file)
