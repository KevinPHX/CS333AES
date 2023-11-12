import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

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

    def get_components_per_paragraph(self,txt_file): 
        self.components_per_paragraph = defaultdict(list)
        paragraphs = []
        self.paragraph_idx = []
        with open(txt_file,"r") as f: 
            for idx, line in enumerate(f.readlines()):
                if idx == 0: # skip prompt 
                    self.prompt = line
                    continue
                if idx == 1: # skip the additional newline after prompt 
                    continue
                paragraphs.append(line)
        self.essay = "".join(paragraphs)
        start = len(self.prompt)  
        for paragraph in paragraphs:
            self.paragraph_idx.append(start)
            start += len(paragraph) 
        for c,component_start in self.idx.items():
            for p_idx, paragraph_start in enumerate(self.paragraph_idx):
                if p_idx + 1 <= len(self.paragraph_idx)-1: 
                    if paragraph_start <= component_start < self.paragraph_idx[p_idx+1]: 
                        self.components_per_paragraph[p_idx].append(c)
                        break
                elif p_idx == len(self.paragraph_idx)-1: # final paragraph
                    if paragraph_start <= component_start: 
                        self.components_per_paragraph[p_idx].append(c) 
                

    def get_weights(self,components,out_neighbors,in_neighbors,types): 
        n = len(components)
        rel = 0 # total number of relations predicted WITHIN the paragraph
        # r[i][j] means that there is a predicted relation from source i to target j 
        r = [[0 for _ in range(n)] for _ in range(n)] # nxn matrix 
        for i in components: 
            for j in out_neighbors[i]: 
                r[i][j] = 1 
                rel += 1  

        # cs_i = (relin_i - relout_i + n - 1) / (rel + n - 1) 
        cs = {}
        for c in components: 
            relin = len(in_neighbors[c])
            relout = len(out_neighbors[c])
            cs[c] =  (relin-relout+n-1) / (rel+n-1)

        # cr_ij = cs_j - cs_i 
        cr = [[0 for _ in range(n)] for _ in range(n)] # nxn matrix 
        c = [[0 for _ in range(n)] for _ in range(n)] # nxn matrix  
        for i in components: 
            for j in components:
                if i==j: continue 
                cr[i][j] = cs[j] - cs[i]
                target_type = types[j]
                if target_type in self.claim_types: 
                    c[i][j] = 1  
        # w_ij = (1/2)*r_ij + (1/4)*cr_ij + (1/4)*type where r is the relation matrix and type is the dictionary c 
        w = [[0 for _ in range(n)] for _ in range(n)] # nxn matrix  
        for i in components: 
            for j in components: 
                w[i][j] = (1/2)*r[i][j] + (1/4)*cr[i][j] + (1/4)*c[i][j]
        return w 

    def solve(self,txt_file): 
        self.get_components_per_paragraph(txt_file)
        self.results = defaultdict(list) # paragraph idx to optimized relations
        for p,components in self.components_per_paragraph.items():
            if len(components) > 1:
                idx_to_name = {idx:name for idx,name in enumerate(components)} 
                name_to_idx = {name:idx for idx,name in idx_to_name.items()}
                out_neighbors = defaultdict(dict) 
                in_neighbors = defaultdict(dict) 
                types = {}
                for idx, name in idx_to_name.items():
                    for target in self.outgoing_relations[name]:  
                        out_neighbors[idx][name_to_idx[target]] = True 
                    for source in self.incoming_relations[name]: 
                        in_neighbors[idx][name_to_idx[source]] = True
                    types[idx] = self.type[name]
                
                # TEST w/ ESSAY 001 PARAGRAPH 2 TO SEE IF THE PROGRAM PREVENTS CYCLES 
                # out_neighbors[0][1] = True 
                # out_neighbors[0][2] = True 
                
                # get weights 
                w = self.get_weights(list(idx_to_name.keys()),out_neighbors,in_neighbors,types)
                # print(f"weights")
                # for w_list in w: print(w_list)
                
                self.solve_paragraph(p,idx_to_name,w)

    def solve_paragraph(self,p,idx_to_name,w): 
        components = list(idx_to_name.keys())

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
                        # f.write(f"Relation from {idx_to_name[i]} to {idx_to_name[j]}\n")
                        source = idx_to_name[i]
                        target = idx_to_name[j]

                        # enforce the constraint within the annotation guidelines that no claims should be linked to each other  
                        if self.type[source] in self.claim_types and self.type[target] in self.claim_types:
                            continue 
                        # only link premises to claims 
                        self.results[source].append(target) 
            # f.write(f'Number of Relations: {num_relations}\n')
            # f.close()

    def evaluate(self):
        true_pos, true_neg, false_pos, false_neg = 0,0,0,0
        for i,j_list in self.results.items():
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
                    if j not in self.results[i]: 
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
    essayDir = "/Users/amycweng/Downloads/CS333_Project/ArgumentAnnotatedEssays-2.0/brat-project-final"
    NUM_ESSAYS = 402 #402 
    for num in range(NUM_ESSAYS):
        if num+1 < 10: filename = f'essay00{num+1}'
        elif num+ 1 < 100: filename = f'essay0{num+1}'
        else: filename = f'essay{num+1}'

        essay_ann_file = f"{essayDir}/{filename}.ann"
        essay_txt_file = f"{essayDir}/{filename}.txt"
        argument = ArgumentTrees(essay_ann_file)
        argument.solve(essay_txt_file)
        print(f"{filename}")
        argument.evaluate()