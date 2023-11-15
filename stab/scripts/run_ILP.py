
from CS333AES.stab.lib.argumentILP_withJSON import ArgumentTrees 
import json 

if __name__ == "__main__": 
    essay_files = []
    # open document that lists all the essays of the test set in the Argument Annotated essays dataset 
    with open(f"CS333AES/stab/assets/test_text.txt","r") as file: 
        for line in file.readlines(): 
            essay_files.append(line.split("../data/")[1].strip("\n").replace(" 2/data/","/"))

    optimized_relations = {} # maps essay_name to another dictionary 
    overall_rates = {"TPR":[],"TNR":[],"FPR":[],"FNR":[]}
    for essay_file in essay_files:
        essay_name = essay_file.split("-final/")[1]
        essay_ann_file = essay_file.replace(".txt",".ann")
        # read in relation information 
        with open(f'CS333AES/stab/outputs/test/relations/{essay_name}.json') as file: 
            relation_info = json.load(file)
        with open(f'CS333AES/stab/outputs/test/classification/{essay_name}.json') as file: 
            components_info = json.load(file)

        # initialize class for data formatting & ILP evaluation 
        argument = ArgumentTrees(essay_ann_file)
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
        for type,rate in argument.evaluation_rates: 
            overall_rates[type].append(rate)
    
    with open(f"CS333AES/stab/outputs/test_set_optimized_relations.json","w") as file: 
        json.dump(optimized_relations,file)
    