import os
from collections import defaultdict 
from structural import * 

class Syntactic(): 
    def __init__(self): 
        self.lca_info = defaultdict(list)
        self.node_info = {}
        self.token_info = defaultdict(dict)
        self.has_modal = {}
        self.pos_distribution = {}

    def component_syntactic(self,component_tokens,component_info,annotations): 
        for component, token_indices in component_tokens.items(): 
            self.has_modal[component] = False 
            self.pos_distribution[component] = []
            s_idx = component_info[component]["sentIdx"]
            for t_idx in token_indices: 
                pos = annotations[s_idx][t_idx]["pos"]
                self.pos_distribution[component].append(pos) 
                if pos == "MD": 
                    self.has_modal[component] = True 
                    break

    def get_lca(self,syn_file): 
        ''' 
        Note that c_token means the constituent type of the current token.
        Likewise for the preceding and following tokens, i.e., c_preceding and c_following. 
        '''
        with open(syn_file,"r") as f: 
            for line in f.readlines(): 
                line = line.split("\t")
                sentIdx = line[0]
                label = line[2]
                position = label.split("-")[1]
                if sentIdx not in self.lca_info: 
                    self.lca_info[sentIdx] = []
                token_info = {
                    "token": line[1], "position": int(position), "c_token": line[3],
                    "preceding": line[4], "c_preceding": line[5], "lcaPath_preceding": line[6],
                    "following": line[7], "c_following": line[8], "lcaPath_following": line[9].strip("\n")
                }
                self.lca_info[sentIdx].append(token_info)

    def get_lex(self,lex_file): 
        ''' 
        Info for sentences are divided by newlines. 
        Formatting of info for each sentence: 
        1. For each token, <token_label>\t<label of uppermost node>
        2. (if applicable) <token_label>\t"child_of_uppermost"\t<label of child> 
        3. (if applicable)  <token_label>\t"right_sibling_of_child"\t<right_sibling_label>\t<its lexical head> 
        4. List with each entry being <node_label>\t<node_index>
        5. HashMap mapping <node_index>=<lexical_head>
        '''
        sentIdx = 0
        node_indices = {}
        lexical_heads = defaultdict(list)
        with open(lex_file,"r") as f: 
            for line in f.readlines(): 
                if line == "\n": # reached a new sentence 
                    self.node_info[sentIdx] = dict(lexical_heads) 
                    sentIdx += 1 
                    node_indices = defaultdict(list)
                    lexical_heads = defaultdict(list)
                    if sentIdx == 1: break # for sample printing purposes 
                    continue
                if line[0] == "[":
                    # reached info of type 4 
                    line = line.replace("]","").replace("[","").split(", ")
                    for entry in line: 
                        entry = entry.strip("\n").split("\t")
                        node_indices[entry[1]] = entry[0]
                elif line[0] == "{": 
                    # reached info of type 5 
                    line = line.replace("{","").replace("}","").split(", ")
                    for entry in line: 
                        entry = entry.split("=")
                        lexical_heads[entry[0]].append({"node": node_indices[entry[0]], 
                                                "head": entry[1] })
                else: 
                    line = line.strip("\n").split("\t")
                    token = line[0].split("-")[0]
                    position = int(line[0].split("-")[1])
                    if len(line) == 2: 
                        uppermost = line[1]
                        token_dict = { "token": token, "uppermost": uppermost}
                        self.token_info[sentIdx][position] = token_dict
                    else: # intermediary info 
                        type = line[1]
                        label = line[2]
                        self.token_info[sentIdx][position][type] = label
                        if type == "right_sibling_of_child": 
                            self.token_info[sentIdx][position]["right_sibling_type"] = line[3]


if __name__ == "__main__":
    essayDir = "/Users/amycweng/Downloads/CS333_Project/ArgumentAnnotatedEssays-2.0/brat-project-final"
    annDir = "/Users/amycweng/Downloads/CS333_Project/CS333AES/stab/preprocessing/src/main/resources/token_level"
    sentDir = "/Users/amycweng/Downloads/CS333_Project/CS333AES/stab/preprocessing/src/main/resources/sentence_sentiment"
    filename = 'essay001'
    essay_ann_file = f"{essayDir}/{filename}.ann"
    essay_txt_file = f"{essayDir}/{filename}.txt"
    token_file = f"{annDir}/{filename}.txt"
    sentence_file = f"{sentDir}/{filename}.txt"
    essay = Structural()
    syntactic = Syntactic()
    essay.read_data(essay_ann_file, essay_txt_file, token_file, sentence_file)
    essay.annotate_tokens()
    essay.component_stats()
    syntactic.component_syntactic(essay.component_tokens, essay.component_info, essay.annotations)
    print(syntactic.has_modal)
    print(syntactic.pos_distribution)