import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import json 

class ArgumentTrees(): 
    def __init__(self, ann_file): 
        self.incoming_relations = {}
        self.outgoing_relations = {}
        self.type = {}
        self.idx = {}
        self.claim_types = ["Claim","MajorClaim"]
        with open(ann_file,"r") as f: 
            for line in f.readlines(): 
                info = line.strip('\n').split("\t")
                name = info[0]
                if "T" in name: 
                    self.idx[name] = int(info[1].split(" ")[1])
                    self.type[name] = info[1].split(" ")[0]
                    self.outgoing_relations[name] = []
                    self.incoming_relations[name] = []
                elif "R" in name:
                    info = info[1].split(' ')
                    arg1 = info[1].split(":")[1]
                    arg2 = info[2].split(":")[1]
                    self.outgoing_relations[arg1].append(arg2)
                    self.incoming_relations[arg2].append(arg1)
        self.start_to_name = {position: name for name,position in self.idx.items()}

    def read_predicted_data(self,components_info,relation_info): 
        relations = []
        for pair, info in relation_info.items(): 
            if info["is_a_relation"]: 
                relations.append(pair)

        self.predicted_info = {}
        self.predicted_per_paragraph = {}
        self.position_to_name = {}
        for k, info in enumerate(components_info): 
            self.predicted_info[k+1] = {
                "type": info["claim"], 
                "start": info["start"],
                "paragraph": info["paragraph"],
                "name": self.start_to_name[info["start"]]
            } 
            if info["paragraph"] not in self.predicted_per_paragraph: 
                self.predicted_per_paragraph[info["paragraph"]] = []
            self.predicted_per_paragraph[info["paragraph"]].append(k+1)

        self.predicted_outgoing = {k: [] for k in self.predicted_info}
        self.predicted_incoming = {k: [] for k in self.predicted_info}
        for pair in relations: 
            source, target = int(pair.split(",")[0]), int(pair.split(",")[1])
            self.predicted_outgoing[source].append(target)
            self.predicted_incoming[target].append(source)   

    def get_weights(self,components,out_neighbors,in_neighbors): 
        n = len(components)
        rel = 0 # total number of relations predicted WITHIN the paragraph
        # r[i][j]=1 means that there is a predicted relation from source i to target j 
        r = defaultdict(dict)
        for i in components:
            for j in components:  
                if j in out_neighbors[i]:
                    r[i][j] = 1 
                    rel += 1 
                else: 
                    r[i][j] = 0  
        # cs_i = (relin_i - relout_i + n - 1) / (rel + n - 1) 
        cs = {}
        for c in components: 
            relin = len(in_neighbors[c])
            relout = len(out_neighbors[c])
            cs[c] =  (relin-relout+n-1) / (rel+n-1)

        # cr_ij = cs_j - cs_i 
        cr = defaultdict(dict)
        c = defaultdict(dict)
        for i in components: 
            for j in components:
                cr[i][j] = cs[j] - cs[i]
                target_type = self.predicted_info[j]["type"]
                if target_type in self.claim_types: 
                    c[i][j] = 1 # target is a claim or majorclaim 
                else: 
                    c[i][j] = 0   # target is a premise 
        # w_ij = (1/2)*r_ij + (1/4)*cr_ij + (1/4)*type where r is the relation matrix and type is the dictionary c 
        w = defaultdict(dict) # nxn matrix  
        for i in components: 
            for j in components: 
                w[i][j] = (1/2)*r[i][j] + (1/4)*cr[i][j] + (1/4)*c[i][j]
        
        # for i, info in w.items(): 
        #     for j, weight in info.items(): 
        #         print(f"{i},{j}: {weight}")
        
        return w 

    def optimize(self): 
        self.results_indices = defaultdict(list)
        self.results_names = defaultdict(list) # paragraph idx to optimized relations
        for p, components in self.predicted_per_paragraph.items(): 
            if len(components) > 1: 
                out_neighbors = defaultdict(dict) 
                in_neighbors = defaultdict(dict) 
                for c in self.predicted_outgoing.keys(): 
                    for target in self.predicted_outgoing[c]:  
                        out_neighbors[c][target] = True  
                    for source in self.predicted_incoming[c]:  
                        in_neighbors[c][source] = True  
                w = self.get_weights(components,out_neighbors,in_neighbors)

                self.solve_paragraph(p,components,w)
            
    def solve_paragraph(self,p,components,w): 
        # silence Gurobi output 
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        env.start()
        # create model 
        model = gp.Model(f"argument_in_paragraph",env=env)
        
        # Create variables
        pairs = []
        for i in components: 
            for j in components: 
                # i is the source, j the target 
                pairs.append((i,j))
        # x_ij = 1 means that a directed edge exists from i to j. 0 otherwise  
        x = model.addVars(pairs, name="x",vtype=GRB.BINARY)
        # b_ij = 1 means that a directed path exists from i to j. 0 otherwise 
        # each b_ij is a binary auxiliary variable 
        b = model.addVars(pairs, name="b",vtype=GRB.BINARY)
        
        # set objective function 
        model.setObjective(sum(w[i][j]*x[(i,j)] for i,j in pairs), GRB.MAXIMIZE)
        
        # set constraints 
        for i in components: 
            
            # make sure each component has at most one outgoing edge 
            model.addConstr(sum(x[(i,j)] for j in components) <= 1, 'at_most_one_outgoing')
            
            # make sure that the source and target components are not identical 
            model.addConstr(x[(i,i)] == 0, 'no_self_loops') 
       
        # ensures each paragraph contains at least one root node (a node without an outgoing relation)
        model.addConstr(sum(x[(i,j)] for i,j in pairs) <= (len(components)-1), 'has_root')  
        
        for i in components: 
           
            # no directed paths starting and ending with the same node 
            model.addConstr(b[(i,i)]==0, 'avoid_cycles')
            
            for j in components: 
                # if a relation exists between i and j (i.e., x_ij is 1), then b_ij is also 1 
                model.addConstr(x[(i,j)] - b[(i,j)] <= 0, 'direct_relation')
        
        for i in components: 
            for j in components: 
                for k in components:
                    # if there is a directed path from i to j and from j to k, then there should be a relation from i to k 
                    model.addConstr((b[(i,k)] - b[(i,j)] - b[(j,k)] >= -1), 'transitive')
        # now solve for the optimal values of x 
        model.optimize()
        
        # write output to another file 
        if model.status == GRB.OPTIMAL:
            num_relations = 0
            # f = open(f'argument_tree_optimal.txt','a+')
            # f.write(f"\nResult for Paragraph {p} (0-indexed):\n")
            for i in components: 
                for j in components:
                    num_relations += int(x[(i,j)].X)
                    decision = int(x[(i,j)].X)
                    if decision == 1: 
                        source = self.predicted_info[i]["name"]
                        target = self.predicted_info[j]["name"]
                        # enforce the constraint within the annotation guidelines that no claims should be linked to each other  
                        source_type = self.predicted_info[i]["type"]
                        target_type = self.predicted_info[j]["type"]
                        if source_type in self.claim_types and target_type in self.claim_types:
                            continue 
                        # only link premises to claims 
                        self.results_names[source].append(target) 
                        self.results_indices[i].append(j)
            # f.write(f'Number of Relations: {num_relations}\n')
            # f.close()
    
    
    def evaluate(self):
        true_pos, true_neg, false_pos, false_neg = 0,0,0,0
        for i,j_list in self.results_names.items():
            if i in self.outgoing_relations: 
                for j in j_list: 
                    if j in self.outgoing_relations[i]: 
                        true_pos += 1 
                    else: 
                        false_pos += 1 
                        print(f"False Positive ({i},{j})")
            else: 
                true_neg += 1 
        for i,j_list in self.outgoing_relations.items(): 
            if i in self.outgoing_relations:
                for j in j_list: 
                    if j not in self.results_names[i]: 
                        false_neg += 1
                        print(f"False Negative ({i},{j})")  
        if false_neg == 0 and false_pos == 0: 
            return True 
        # else: 
        #     print(f"True positive rate: {true_pos/(true_pos+false_neg)}") 
        #     print(f"True negative rate: {true_neg/(true_neg+false_pos)}") 
        #     print(f"False negative rate: {false_pos/(true_neg+false_pos)}") 
        #     print(f"False positive rate: {false_neg/(true_pos+false_neg)}") 
        return False


if __name__ == "__main__": 
    essay_files = []
    with open(f"CS333AES/stab/assets/train_text.txt","r") as file: 
        for line in file.readlines(): 
            essay_files.append(line.split("../data/")[1].strip("\n").replace(" 2/data/","/"))

    arguments = {}
    for essay_file in essay_files:
        essay_name = essay_file.split("-final/")[1]
        essay_ann_file = essay_file.replace(".txt",".ann")
        # read in relation information 
        with open(f'CS333AES/stab/outputs/relations/{essay_name}.json') as file: 
            relation_info = json.load(file)
        with open(f'CS333AES/stab/outputs/classification/{essay_name}.json') as file: 
            components_info = json.load(file)

        # initialize class for data formatting & ILP evaluation 
        argument = ArgumentTrees(essay_ann_file)
        print(f"{essay_name}")
        argument.read_predicted_data(components_info, relation_info)
        argument.optimize()

        # prepare to write output out to the relations json file 
        for i,j_list in argument.predicted_outgoing.items(): 
            for j in j_list: 
                if i == j: continue 
                # initialize field to know whether the ILP makes this pair a relation 
                relation_info[f"{i},{j}"]["is_optimized_relation"] = 0 
        for source,target_list in argument.results_indices.items(): 
            for target in target_list: 
                relation_info[f"{source},{target}"]["is_optimized_relation"] = 1 

        argument.evaluate()
        with open(f"CS333AES/stab/outputs/relations/{essay_name}.TESTING.json","w") as file: 
            json.dump(relation_info,file)
        
        break