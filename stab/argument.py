import gurobipy as gp
from gurobipy import GRB
import csv 
from structural import Structural 
from collections import defaultdict

class Argument(): 
    def __init__(self, ann_file, txt_file, sent_file): 
        self.components = [] 
        self.incoming_relations = defaultdict(list)
        self.outgoing_relations = defaultdict(list)
        self.type = {}
        with open(ann_file,"r") as f: 
            for line in f.readlines(): 
                info = line.strip('\n').split("\t")
                name = info[0]
                if "T" in name: 
                    c = int(name.split("T")[1])-1
                    self.components.append(int(name.split("T")[1])-1)
                    self.type[c] = info[1].split(" ")[0]
                elif "R" in name:
                    info = info[1].split(' ')
                    arg1 = int(info[1].split(":")[1].split("T")[1])-1 
                    arg2 = int(info[2].split(":")[1].split("T")[1])-1
                    self.outgoing_relations[arg1].append(arg2)
                    self.incoming_relations[arg2].append(arg1)
        self.matrix = defaultdict(dict) 
        for c1,c2_list in self.outgoing_relations.items(): 
            for c2 in c2_list: 
                self.matrix[c1][c2] = 1 
        
        self.structure = Structural()
        self.structure.read_data(ann_file,txt_file,None,sent_file)
        self.components_per_paragraph = defaultdict(list)
        self.component_paragraph_idx = {}
        for p,c_list in self.structure.paragraph_info()[0].items(): 
            for c in c_list: 
                c = int(c.split("T")[1])-1
                self.components_per_paragraph[p].append(c)
                self.component_paragraph_idx[c] = p

    def get_weights(self): 
        # cs_i = (relin_i - relout_i + n - 1) / (rel + n - 1) 
        cs = {}
        n = len(self.matrix)
        for c in self.components: 
            relin = len(self.incoming_relations[c])
            relout = len(self.outgoing_relations[c])
            p_idx = self.component_paragraph_idx[c]
            rel = len(self.components_per_paragraph[p_idx]) # total number of relations predicted in the covering paragraph
            cs[c] =  (relin-relout+n-1) / (rel+n-1)
        
        # cr_ij = cs_j - cs_i 
        cr = defaultdict(dict)
        c = defaultdict(dict)  
        for s,c2_list in self.outgoing_relations.items(): 
            for t in c2_list: 
                cr[s][t] = cs[t] - cs[s]
                target_type = self.type[t]
                if target_type == "Claim" or target_type == "MajorClaim":
                    c[s][t] = 1  
        
        # w_ij = (1/2)*r_ij + (1/4)*cr_ij + (1/4)*type
        self.weights = [[0 for _ in range(len(self.components))] for _ in range(len(self.components))] # nxn matrix  
        for s,c2_list in self.outgoing_relations.items(): 
            for t in c2_list: 
                self.weights[s][t] = (1/2)*self.matrix[s][t] + (1/4)*cr[s][t] + (1/4)*c[s][t]
        for w_list in self.weights: print(w_list) 

    

essayDir = "/Users/amycweng/Downloads/CS333_Project/ArgumentAnnotatedEssays-2.0/brat-project-final"
annDir = "/Users/amycweng/Downloads/CS333_Project/CS333AES/stab/preprocessing/src/main/resources/token_level"
sentDir = "/Users/amycweng/Downloads/CS333_Project/CS333AES/stab/preprocessing/src/main/resources/sentence_sentiment"

filename = 'essay001'
essay_ann_file = f"{essayDir}/{filename}.ann"
essay_txt_file = f"{essayDir}/{filename}.txt"
sentence_file = f"{sentDir}/{filename}.txt"

argument = Argument(essay_ann_file, essay_txt_file, sentence_file)
argument.get_weights()