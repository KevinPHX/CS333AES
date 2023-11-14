from argumentILP import ArgumentTrees 
from collections import Counter
import json 

class ArgumentRelationPreprocessing():
    def __init__(self, all_components): 
        self.components = all_components 

    def relation_probabilities(self, essay_files,type): 
        # get the probability that a component is related to another component 
        # ratio of num of components with at least one incoming relation to all components 
        #   to the total number of components in the entire dataset 
        num_components_with_relation = {"outgoing": 0, "incoming":0}
        total_components = 0 
        self.arguments = {}
        argument_info = {}
        for essay_file in essay_files:
            essay_ann_file = essay_file.replace(".txt",".ann")
            # the init function of the class in the ILP code file is helpful here 
            argument = ArgumentTrees(essay_ann_file)
            # update total count of components 
            total_components += len(argument.outgoing_relations)
            # update count of components that have at least one outgoing edge 
            for out_neighbors in argument.outgoing_relations.values(): 
                if len(out_neighbors) > 0: 
                    num_components_with_relation["outgoing"] += 1 
            # update count of components that have at least one incoming edge 
            for in_neighbors in argument.incoming_relations.values(): 
                if len(in_neighbors) > 0: 
                    num_components_with_relation["incoming"] += 1 
            # add argument info to dictionary 
            self.arguments[essay_file.split("-final/")[1]] = {  
                                                                "idx_to_start": argument.idx, 
                                                                "incoming_relations": argument.incoming_relations,
                                                                "outgoing_relations": argument.outgoing_relations
                                                            }
        relation_probabilities = {"outgoing": num_components_with_relation["outgoing"] / total_components,
                                "incoming": num_components_with_relation["incoming"] / total_components }
        if type == "training": 
            with open(f"CS333AES/stab/models/relation_probabilities.json","w") as file: 
                json.dump(relation_probabilities, file)
            with open(f"CS333AES/stab/models/argument_relation_info.json","w") as file: 
                json.dump(self.arguments, file)
        else: 
            with open(f"CS333AES/stab/models/argument_relation_info_TEST_SET.json","w") as file: 
                json.dump(self.arguments, file)
        
    def get_all_lemmas(self): 
        # map components to their names in the ann files using their start positions 
        # number of lemmas that occur in argument components of training data 
        all_lemmas = 0 
        # all lemmas that occur in components with an incoming relation 
        lemmas_incoming = []
        # all lemmas that occur in components with an outgoing relation 
        lemmas_outgoing = []
        
        for essay_name, c_info in self.components.items(): 
            position_to_name = {v:k for k,v in self.arguments[essay_name].idx.items()}
            for c in c_info: 
                name = position_to_name[c["start"]]
                all_lemmas += len(c["component_lemmas"]) 
                if len(self.arguments[essay_name].incoming_relations[name]) > 0: 
                    lemmas_incoming.extend(c["component_lemmas"])
                if len(self.arguments[essay_name].outgoing_relations[name]) > 0: 
                    lemmas_outgoing.extend(c["component_lemmas"])
        
        with open("CS333AES/stab/models/training_data_lemmas.json","w") as file: 
            json.dump({"num_all_lemmas": all_lemmas, 
                    "lemmas_incoming": dict(Counter(lemmas_incoming)), 
                    "lemmas_outgoing": dict(Counter(lemmas_outgoing))}, file)

 # this is for the ArgumentAnnotatedEssays-2.0 dataset by Stab and Gurveych 
if __name__ == "__main__":
    # for training set  
    essay_files = []
    with open(f"CS333AES/stab/assets/train_text.txt","r") as file: 
        for line in file.readlines(): 
            essay_files.append(line.split("../data/")[1].strip("\n").replace(" 2/data/","/"))

    all_components = {}
    for essay_file in essay_files: 
        essay_name = essay_file.split("-final/")[1]
        # read component data for this essay 
        with open(f'CS333AES/stab/outputs/classification/{essay_name}.json') as file: 
            components = json.load(file)
        all_components[essay_name] = components
    
    training_data_relations = ArgumentRelationPreprocessing(all_components)
    training_data_relations.relation_probabilities(essay_files,"training")
    training_data_relations.get_all_lemmas()

    # for test set 
    essay_files = []
    with open(f"CS333AES/stab/assets/test_text.txt","r") as file: 
        for line in file.readlines(): 
            essay_files.append(line.split("../data/")[1].strip("\n").replace(" 2/data/","/"))

    all_components = {}
    for essay_file in essay_files: 
        essay_name = essay_file.split("-final/")[1]
        # read component data for this essay 
        with open(f'CS333AES/stab/outputs/test/classification/{essay_name}.json') as file: 
            components = json.load(file)
        all_components[essay_name] = components
    
    training_data_relations = ArgumentRelationPreprocessing(all_components)
    training_data_relations.relation_probabilities(essay_files,"test")

  