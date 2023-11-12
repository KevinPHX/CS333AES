from pos import pos
import json
from collections import defaultdict, Counter
from argumentILP import ArgumentTrees 
from math import log
DISCOURSE_RELATIONS = ["Comparison","Contingency","Expansion","Temporal"]
INDICATOR_TYPES = ["forward","backwards","thesis","rebuttal"]

class ArgumentRelationIdentification(): 
    def __init__(self, components, argument, relation_probabilities): 
        self.components = components
        self.relations = argument.outgoing_relations 
        self.position_to_name = {v:k for k,v in argument.idx.items()}
        self.idx_to_name = []
        # number of lemmas that occur in argument components 
        self.all_lemmas = 0 
        # all lemmas that occur in components with an incoming relation 
        self.lemmas_incoming = []
        # all lemmas that occur in components with an outgoing relation 
        self.lemmas_outgoing = []
        for c in self.components:
            # map components to their names in the ann files 
            name = self.position_to_name[c["start"]]
            self.idx_to_name.append(name)
            self.all_lemmas += len(c["component_lemmas"]) 
            if len(argument.incoming_relations[name]) > 0: 
                self.lemmas_incoming.extend(c["component_lemmas"])
            if len(argument.outgoing_relations[name]) > 0: 
                self.lemmas_outgoing.extend(c["component_lemmas"])

        self.p_outgoing = relation_probabilities["outgoing"]
        self.p_incoming = relation_probabilities["incoming"]
        self.get_pointwise_mutual_info()
        self.get_production_rules()
        self.pairwise_features()
    
    def get_pointwise_mutual_info(self): 
        # get PMI(t,d) for each token t and direction d 
        for c in self.components: 
            c["pmi_incoming"] = []
            c["pmi_outgoing"] = []
            for idx,prob in enumerate(c["p_token"]):
                lemma = c["component_lemmas"][idx]
                p_t_in = self.lemmas_incoming.count(lemma) / self.all_lemmas
                p_t_out = self.lemmas_outgoing.count(lemma) / self.all_lemmas
                if p_t_in != 0: 
                    c[f"pmi_incoming"].append( log( p_t_in / (prob * self.p_incoming)) )
                else: 
                    c[f"pmi_incoming"].append(0)
                if p_t_out != 0: 
                    c[f"pmi_outgoing"].append( log( p_t_out / (prob * self.p_outgoing)) )
                else: 
                    c[f"pmi_outgoing"].append(0)
            
        # for c in self.components: 
        #     print(c["pmi_incoming"],c["pmi_outgoing"])


    def get_production_rules(self): 
        production_rules = []
        for c in self.components: 
            for rule in c["production_rules"]: 
                production_rules.append(f"{rule}")
        self.production_rules_500_most_common = Counter(production_rules).most_common(n=500)
    
    def pairwise_features(self): 
        # key will be (i,j) where i is the idx of a source and j is the idx of a target 
        self.pairwise = {} 

        self.components_per_paragraph = defaultdict(list)
        idx_in_paragraph = 0 # variable to indicate the position of the component in a paragraph (e.g. idx 0 out of 4 components in the paragraph)
        for idx in range(len(self.components)): 
            p_idx = self.components[idx]["paragraph"]
            idx_in_paragraph += 1 
            if p_idx not in self.components_per_paragraph:
                idx_in_paragraph = 0  
            self.components[idx]["idx_in_paragraph"] = idx_in_paragraph
            self.components_per_paragraph[p_idx].append(idx)

        for i, source in enumerate(self.components):
            for j, target in enumerate(self.components):
                # the source cannot equal the target 
                if i == j: continue
                # the components must be in the same paragraph! 
                if source["paragraph"] != target["paragraph"]: 
                    continue 
                
                self.pairwise[f"{i},{j}"] = {
                    # there is actually a directed edge from source to target 
                    "is_a_relation": 0, # default is false 
                    # number of tokens in both source and target 
                    "num_tokens": len(source["component"]) + len(target["component"]),
                    # if source and target are present in the same sentence
                    "same_sentence": 0, # default is false 
                    # if target present before source 
                    "target_before_source": 0, # default is false,
                    # if pair is present in intro or conclusion. They are both in the same paragraph
                    "intro_or_conc": source["intro/conc"],
                    # number of components between source and target
                    "num_between": abs(source["idx_in_paragraph"]-target["idx_in_paragraph"])-1, 
                    # number of components in the covering paragraph 
                    "num_in_paragraph": len(self.components_per_paragraph[source["paragraph"]]),
                    # if target and source share at least one noun 
                    "share_noun": 0, # default is false 
                    # the number of nouns shared by target and source 
                    "num_shared_nouns":0, # default is none 
                    # source or target is first or last in paragraph 
                    "first_or_last": 0 # default is none 
                }
                
                if self.idx_to_name[j] in self.relations[self.idx_to_name[i]]: 
                    self.pairwise[f"{i},{j}"]["is_a_relation"] = 1 

                self.pairwise[f"{i},{j}"].update(self.get_indicator_info(source,target))
                
                # get binary POS distribution with the POS distribution of the target  
                for pos_type in pos.keys():
                    self.pairwise[f"{i},{j}"][pos_type] = source[pos_type] + target[pos_type]

                # source and target are present in the same sentence
                if source["sentence"] == target["sentence"]: 
                    self.pairwise[f"{i},{j}"]["same_sentence"] = 1 # true 
                
                # target is present before source 
                if source["start"] > target["start"]: 
                    self.pairwise[f"{i},{j}"]["target_before_source"] = 1 # true  
                
                # if target and source are first or last component in paragraph 
                if source["first/last"] or target["first/last"]: 
                    self.pairwise[f"{i},{j}"]["first_or_last"] = 1 
                
                # find shared nouns (both binary and number)
                shared_nouns = []
                for idx, lemma in enumerate(source["component_lemmas"]):
                    if "NN" in source["component_pos"][idx]: 
                        if lemma in target["component_lemmas"]:
                            shared_nouns.append(lemma)
                if len(shared_nouns) > 0: 
                    self.pairwise[f"{i},{j}"]["share_noun"] = 1
                    self.pairwise[f"{i},{j}"]["num_shared_nouns"] = len(shared_nouns)

                # count how many times a production rule is shared by source and target 
                self.pairwise[f"{i},{j}"].update(self.shared_production_rules(source,target))
                
                # get binary discourse triples of source and target 
                self.pairwise[f"{i},{j}"].update(self.get_discourse_triples(source,target))

                # get pmi features 
                self.pairwise[f"{i},{j}"].update(self.get_pmi_features(source,target))
        # get binary representation of the types of indicators that occur in and around 
        # components between source and target 
        self.get_indicators_between()

        # print for testing purposes 
        # for pair,info in self.pairwise.items(): 
        #     # if pair[0] > 1: break
        #     if info["is_a_relation"]: 
        #         print(f"{self.idx_to_name[pair[0]]} to {self.idx_to_name[pair[1]]}: {info}\n")
        #         break
        
    def get_indicator_info(self,source,target):
        info = {}
        for type in INDICATOR_TYPES: 
            component_key = f"component_{type}_indicators"
            if source[component_key] == 1 or target[component_key] == 1: 
                # this indicator type is present in source or target 
                info[component_key] = 1 
            else: 
                # this indicator type is not present in source or target 
                info[component_key] = 0 
            for context in ["preceding","following"]: 
                context_key = f"{context}_{type}_indicators"
                if source[context_key] == 1 or target[context_key] == 1:
                    # this indicator type is present in the context of either source or target 
                    info[f"context_{type}_indicators"] = 1    
                else: 
                    # this indicator type is not present in the context of either source or target 
                    info[f"context_{type}_indicators"] = 0   
        return info

    def get_indicators_between(self):
        for pair in self.pairwise.keys(): 
            s,t = int(pair.split(",")[0]), int(pair.split(",")[1])
            p_idx = self.components[s]["paragraph"]
            for type in INDICATOR_TYPES: 
                key = f"{type}_indicators"
                self.pairwise[pair][f"between_{key}"] = 0
            
            for c in self.components_per_paragraph[p_idx]: 
                # find a component that is between source and target 
                # check if any of the four types of indicators occur in this component or its context 
                if min(s,t) < c < max(s,t):
                    for type in INDICATOR_TYPES: 
                        for location in ["component","preceding","following"]: 
                            key = f"{type}_indicators"
                            if self.components[c][f"{location}_{key}"] == 1:     
                                self.pairwise[pair][f"between_{key}"] = 1         

    def shared_production_rules(self,source,target): 
        info = { rule: 0 for rule, freq in self.production_rules_500_most_common}
        for rule in source["production_rules"]: 
            if rule in target["production_rules"]: 
                info[f"{rule}"] += 1       
        return info      
    
    def get_discourse_triples(self,source,target): 
        info = {}
        for relation in DISCOURSE_RELATIONS: 
            for arg in ["Arg1","Arg2"]:
                for type in ["Explicit","Implicit"]: 
                    key = f"{relation}_{arg}_{type}"
                    info[key] = source[key] + target[key]
        return info
    
    def get_pmi_features(self,source,target): 
        info = {
            "presence_positive_associations":0, # default is false 
            "presence_negative_associations":0, # default is false 
        } 
        positive, negative = 0,0 
        total = len(source["component_lemmas"]) + len(target["component_lemmas"])
        for direction in ["incoming","outgoing"]: 
            for lemma_pmi in source[f"pmi_{direction}"]: 
                if lemma_pmi > 0: 
                    positive += 1 
                elif lemma_pmi < 0: 
                    negative += 1 
        info["ratio_positive_associations"] = positive / total
        info["ratio_negative_associations"] = negative / total 
        if positive > 0: 
            info["presence_positive_associations"] = 1 
        if negative > 0: 
            info["presence_negative_associations"] = 1 
        return info

def relations(essay_files): 
    # get the probability that a component is related to another component 
    # ratio of num of components with at least one incoming relation to all components 
    #   to the total number of components in the entire dataset 
    num_components_with_relation = {"outgoing": 0, "incoming":0}
    total_components = 0 
    arguments = {}
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
        # add argument structure to dictionary 
        arguments[essay_file.split("-final/")[1]] = argument
    relation_probabilities = {"outgoing": num_components_with_relation["outgoing"] / total_components,
                            "incoming": num_components_with_relation["incoming"] / total_components }
    return arguments, relation_probabilities 

if __name__=='__main__':
    essay_files = []
    with open(f"CS333AES/stab/assets/train_text.txt","r") as file: 
        for line in file.readlines(): 
            essay_files.append(line.split("../data/")[1].strip("\n").replace(" 2/data/","/"))
    arguments, relation_probabilities = relations(essay_files)
    for essay_file in essay_files: 
        essay_name = essay_file.split("-final/")[1]
        # read component data for this essay 
        with open(f'CS333AES/stab/outputs/classification/{essay_name}.json') as file: 
            components = json.load(file)
        # access the actual relations data for this essay 
        argument = arguments[essay_name]
        # run argument relation features extraction 
        argrelation = ArgumentRelationIdentification(components,argument,relation_probabilities)
        with open(f"CS333AES/stab/outputs/relations/{essay_name}.json", "w") as file:
            json.dump(argrelation.pairwise, file)

        



