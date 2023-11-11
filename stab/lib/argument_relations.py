from pos import pos
import json
from collections import defaultdict, Counter

class ArgumentRelationIdentification(): 
    def __init__(self, components): 
        self.components = components
        self.get_production_rules()
        self.pairwise_features()
    
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
                
                self.pairwise[(i,j)] = {
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
                self.pairwise[(i,j)].update(self.get_indicator_info(source,target))
                
                # get binary POS distribution with the POS distribution of the target  
                for pos_type in pos.keys():
                    self.pairwise[(i,j)][pos_type] = source[pos_type] + target[pos_type]

                # source and target are present in the same sentence
                if source["sentence"] == target["sentence"]: 
                    self.pairwise[(i,j)]["same_sentence"] = 1 # true 
                
                # target is present before source 
                if source["start"] > target["start"]: 
                    self.pairwise[(i,j)]["target_before_source"] = 1 # true  
                
                # if target and source are first or last component in paragraph 
                if source["first/last"] or target["first/last"]: 
                    self.pairwise[(i,j)]["first_or_last"] = 1 
                
                # find shared nouns (both binary and number)
                shared_nouns = []
                for idx, lemma in enumerate(source["component_lemmas"]):
                    if "NN" in source["component_pos"][idx]: 
                        if lemma in target["component_lemmas"]:
                            shared_nouns.append(lemma)
                if len(shared_nouns) > 0: 
                    self.pairwise[(i,j)]["share_noun"] = 1
                    self.pairwise[(i,j)]["num_shared_nouns"] = len(shared_nouns)

                # count how many times a production rule is shared by source and target 
                self.pairwise[(i,j)].update(self.shared_production_rules(source,target))
        
        self.get_indicators_between()
        for pair,info in self.pairwise.items(): 
            if pair[0] > 1: break
            print(pair, ": ", info, "\n")
    
    def get_indicator_info(self,source,target):
        indicator_types = ["forward","backwards","thesis","rebuttal"]
        info = {}
        for type in indicator_types: 
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
        indicator_types =  ["forward","backwards","thesis","rebuttal"]
        for pair in self.pairwise.keys(): 
            s,t = pair[0],pair[1]
            p_idx = self.components[s]["paragraph"]
            for type in indicator_types: 
                key = f"{type}_indicators"
                self.pairwise[pair][f"between_{key}"] = 0
            
            for c in self.components_per_paragraph[p_idx]: 
                # find a component that is between source and target 
                # check if any of the four types of indicators occur in this component or its context 
                if min(s,t) < c < max(s,t):
                    for type in indicator_types: 
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

if __name__=='__main__':
    with open('components.json') as file: 
        components = json.load(file)  
    argrelation = ArgumentRelationIdentification(components)



