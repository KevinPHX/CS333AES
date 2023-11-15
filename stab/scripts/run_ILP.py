
from argumentILP_withJSON import ArgumentTrees
import json, os, sys
path = os.path.abspath("../CS333_Project/CS333AES/stab")
sys.path.append(path)

if __name__ == "__main__":
    essay_files = []
    # open document that lists all the essays of the test set in the Argument Annotated essays dataset 
    with open(f"{path}/assets/test_text.txt","r") as file: 
        for line in file.readlines(): 
            essay_files.append(line.split("../data/")[1].strip("\n").replace(" 2/data/","/"))
    # annotated relations for each essay 
    with open(f"{path}/models/argument_relation_info_TEST_SET.json") as file: 
        ground_truth = json.load(file)
    
    optimized_relations = {} # maps essay_name to another dictionary 
    overall_rates = {"TPR":[],"TNR":[],"FPR":[],"FNR":[],"accuracy":[],"precision":[]}
    label_to_type = {0: "MajorClaim", 1: "Claim", 2: "Premise"}
    c_idx, r_idx = 0, 0 # used to find prediction info  
    for essay_file in essay_files:
        essay_name = essay_file.split("-final/")[1]
        essay_ann_file = essay_file.replace(".txt",".ann")
        # read in relation information 
        with open(f'{path}/outputs/test/relations/{essay_name}.json') as file: 
            relation_info = json.load(file)
        # read in classification information 
        with open(f'{path}/outputs/test/classification/{essay_name}.json') as file: 
            components_info = json.load(file)
        # read in relation predictions 
        with open(f'{path}/outputs/relation_predictions.json') as file: 
            r_predictions = json.load(file)
        # read in classification predictions 
        with open(f'{path}/outputs/classification_predictions.json') as file: 
            c_predictions = json.load(file)
        for pair in relation_info:
            relation_info[pair]["is_predicted_relation"] = r_predictions[r_idx]
            r_idx += 1 
        for c in components_info: 
            c["predicted_type"] = label_to_type[int(c_predictions[c_idx])]
            c_idx += 1 

        # initialize class for data formatting & ILP evaluation 
        argument = ArgumentTrees(ground_truth[essay_name])
        print(f"{essay_name}")
        argument.process_data(components_info, relation_info)
        argument.optimize()

        # add output to the optimized_relations dictionary 
        optimized_relations[essay_name] = {"relations": {}, "evaluation": None } 
        for source,target_list in argument.results_indices.items(): 
            for target in target_list: 
                optimized_relations[essay_name]["relations"][f"{source},{target}"] = True 

        argument.evaluate()
        optimized_relations[essay_name]["evaluation"] = argument.evaluations
        for type,rate in argument.evaluation_rates.items(): 
            overall_rates[type].append(rate)
    
    with open(f"{path}/outputs/test_set_optimized_relations.json","w") as file: 
        json.dump(optimized_relations,file)
    
    for type, rates in overall_rates.items(): 
        print(f"{type}: {sum(rates)/len(rates)}")  