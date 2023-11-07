import gurobipy as gp
from gurobipy import GRB
from structural import Structural 
from collections import defaultdict
import csv 

class ArgumentTrees(): 
    def __init__(self, ann_file, txt_file, sent_file): 
        self.incoming_relations = defaultdict(list)
        self.outgoing_relations = defaultdict(list)
        self.type = {}
        with open(ann_file,"r") as f: 
            for line in f.readlines(): 
                info = line.strip('\n').split("\t")
                name = info[0]
                if "T" in name: 
                    self.type[name] = info[1].split(" ")[0]
                elif "R" in name:
                    info = info[1].split(' ')
                    arg1 = info[1].split(":")[1]
                    arg2 = info[2].split(":")[1]
                    self.outgoing_relations[arg1].append(arg2)
                    self.incoming_relations[arg2].append(arg1)

        self.structure = Structural()
        self.structure.read_data(ann_file,txt_file,None,sent_file)
        self.components_per_paragraph = defaultdict(list)
        for p,c_list in self.structure.paragraph_info()[0].items(): 
            for c in c_list: 
                self.components_per_paragraph[p].append(c)

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
                if target_type == "Claim" or target_type == "MajorClaim":
                    c[i][j] = 1  
        # w_ij = (1/2)*r_ij + (1/4)*cr_ij + (1/4)*type where r is the relation matrix and type is the dictionary c 
        w = [[0 for _ in range(n)] for _ in range(n)] # nxn matrix  
        for i in components: 
            for j in components: 
                w[i][j] = (1/2)*r[i][j] + (1/4)*cr[i][j] + (1/4)*c[i][j]
        return w 

    def solve(self): 
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
                # get weights 
                w = self.get_weights(list(idx_to_name.keys()),out_neighbors,in_neighbors,types)
                # print(f"weights")
                # for w_list in w: print(w_list)
                
                self.solve_paragraph(p,list(idx_to_name.keys()),w)
                break

    def solve_paragraph(self,p,components,w): 
        # create model 
        model = gp.Model(f"argument_in_paragraph")
        # Create variables
        pairs = []
        for i in components: 
            for j in components: 
                # i is the source, j the target 
                pairs.append((i,j))
        # x_ij = 1 means that a directed edge exists from i to j. 0 otherwise  
        x = model.addVars(pairs, name="x")
        # b_ij = 1 means that a directed path exists from i to j. 0 otherwise 
        # each b_ij is an auxiliary variable 
        b = model.addVars(pairs, name="b")
        # set objective function 
        model.setObjective(sum(w[i][j]*x[(i,j)] for i,j in pairs), GRB.MAXIMIZE)
        for i in components: 
            # make sure each component has at most one outgoing edge 
            model.addConstr(sum(x[(i,j)] for j in components) <= 1, 'at_most_one_outgoing')
            # make sure that the source and target components are not identical 
            model.addConstr(x[(i,i)] == 0, 'no_self_loops') 
            # enforce integer constraint 
            for j in components: 
                model.addConstr(x[(i,j)] <= 1, 'binary_decision') 
        # ensures each paragraph contains at least one root node (a node without an outgoing relation)
        model.addConstr(sum(x[(i,j)] for i,j in pairs) <= (len(components)-1), 'has_root')  
        for i in components: 
            # no directed paths starting and ending with the same node 
            model.addConstr(b[(i,i)]==0, 'avoid_cycles')
            for j in components: 
                # if a relation exists between i and j (i.e., x_ij is 1), then b_ij is also 1 
                model.addConstr(x[(i,j)]-b[(i,j)] <= 0, 'direct_relation')
                # b_ij is either 0 or 1 
                model.addConstr(b[(i,j)] <= 1, 'binary_auxiliary')
        # if there is a path from i to j and from j to k, then there is a path from i to k 
        for k in components: 
            for i in components: 
                for j in components: 
                    model.addConstr(b[(i,k)] - b[(i,j)] - b[(j,k)] <= -1, 'path_exists')
        # now solve for the optimal values of x 
        model.optimize()
        if model.status == GRB.OPTIMAL:
            num_relations = 0
            # write solution out to another csv file 
            # outfile = open(f'argument_tree_optimal.csv','w+')
            # w = csv.writer(outfile)
            print("\nResult:")
            for i in components: 
                results = []
                for j in components:
                    num_relations += int(x[(i,j)].X)
                    results.append(int(x[(i,j)].X))
                print(results)
                # w.writerow(results)
            # outfile.close()
            print(f'\nNumber of Relations in Paragraph {p}: {num_relations}')
        print('----------------------------------\n')

essayDir = "/Users/amycweng/Downloads/CS333_Project/ArgumentAnnotatedEssays-2.0/brat-project-final"
annDir = "/Users/amycweng/Downloads/CS333_Project/CS333AES/stab/preprocessing/src/main/resources/token_level"
sentDir = "/Users/amycweng/Downloads/CS333_Project/CS333AES/stab/preprocessing/src/main/resources/sentence_sentiment"

filename = 'essay001'
essay_ann_file = f"{essayDir}/{filename}.ann"
essay_txt_file = f"{essayDir}/{filename}.txt"
sentence_file = f"{sentDir}/{filename}.txt"

argument = ArgumentTrees(essay_ann_file, essay_txt_file, sentence_file)
argument.solve()