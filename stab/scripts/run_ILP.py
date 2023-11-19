
from argumentILP_withJSON import ArgumentTrees
import json, os, sys
path = "/Users/amycweng/Downloads/CS333_Project/CS333AES/stab"
from sklearn.metrics import classification_report

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
    all_outgoing = []
    all_optimized = []
    revisions = {}
    revised_to_premise,revised_to_claim = 0,0
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

        names_to_idx = {v["name"]:k for k,v in argument.predicted_info.items()}
        for s,t_list in argument.outgoing_relations.items(): 
            for t in argument.outgoing_relations: 
                if s != t: 
                    s_p = argument.predicted_info[names_to_idx[s]]["paragraph"]
                    t_p = argument.predicted_info[names_to_idx[t]]["paragraph"]
                    if s_p != t_p:
                        continue # they must be in the same paragraph
                    if t in t_list: 
                        all_outgoing.append(1)
                    else: 
                        all_outgoing.append(0)
                    if t in argument.results_names[s]: 
                        all_optimized.append(1)
                    else: 
                        all_optimized.append(0)

        idx_to_names = {v:k for k,v in names_to_idx.items()}
        for c in components_info:
            index = c["index"] 
            name = idx_to_names[int(index)]
            if essay_name not in revisions: 
                revisions[essay_name] = {}
            revisions[essay_name][index] = "Premise" # default is premise. Only components without outgoing relations are claims 
            if len(argument.outgoing_relations[name]) == 0: 
                revisions[essay_name][index] = "Claim"
            optimized_type = revisions[essay_name][index]
            predicted_type = c["predicted_type"]
            if optimized_type != predicted_type:
                if predicted_type == "Premise": 
                    revised_to_premise += 1 
                elif predicted_type == "Premise" and optimized_type == "Claim": 
                    revised_to_claim += 1 
        
    print(f"Classification Report for Optimized Relations")
    report = classification_report(
        all_outgoing, # ground truth 
        all_optimized, # predictions 
    )
    print(report)
    print(f"ILP revised {revised_to_premise} claims to premises.")
    print(f"ILP revised {revised_to_claim} premises to claims.")

    true_components = []
    optimized_components = []
    label_to_type = {"MajorClaim": 1, "Claim": 1, "Premise":2}
    for essay, info in ground_truth.items(): 
        for c, type in info["true_type"].items(): 
            true_components.append(label_to_type[type])
    for essay_name, info in revisions.items(): 
        for idx, type in info.items(): 
            optimized_components.append(label_to_type[type])
    print(f"Classification Report for Optimized Components")
    report = classification_report(
        true_components, # ground truth 
        optimized_components, # predictions 
    )
    print(report)
    # with open(f"{path}/outputs/test_set_optimized_relations.json","w") as file: 
    #     json.dump(optimized_relations,file)
    