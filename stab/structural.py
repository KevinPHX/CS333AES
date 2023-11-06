import re,csv

class Structural():

    def __init__(self): 
        self.paragraphs = [] 
        self.components = []
        self.component_tokens = {}
        self.annotations = {}
            # maps sentence index to token information (represented as a dictionary per token)
        self.sentences = []
        self.prompt = ""
        self.essay = ""
        self.paragraph_idx = [] # starting position of the first character of each paragraph (1-indexing)
        self.sentence_idx = [] # tuple with (start position of sentence, paragraph the sentence belongs to)
        self.token_info = {}
        self.component_info = {}
        self.pairwise_tokens = {} 
        self.pairwise_components = {}

    ''' 
    Read in relevant data: 
        - Raw essay text 
        - Annotated argument graphs  
        - Token-level annotations (idx of covering sentence, idx of token within sentence, token, lemma, POS, sentiment)
        - Segmented Sentences
    Call necessary preprocessing functions 
    '''
    def read_data(self,essay_ann_file, essay_text_file, token_file, sentence_file): 
        with open(essay_ann_file,"r") as f: 
            for line in f.readlines(): 
                line = line.strip('\n')
                line = line.split("\t")
                if "T" in line[0]: 
                    self.components.append(line)
        
        with open(essay_text_file,"r") as f: 
            for idx, line in enumerate(f.readlines()):
                if idx == 0: # skip prompt 
                    self.prompt = line
                    continue
                if idx == 1: # skip the additional newline after prompt 
                    continue
                self.paragraphs.append(line)

        self.essay = "".join(self.paragraphs)
        with open(token_file,"r") as f: 
            for line in f.readlines():
                info =  line.replace("[","").replace("]","").strip("\n")
                info = info.split(', ')
                sentIdx = int(info[0])
                tokenIdx = int(info[1])
                info_dict = {"token": info[2], "sentence": sentIdx,"index": tokenIdx,"lemma": info[3], "pos": info[4],"sentiment": info[5]}
                if sentIdx not in self.annotations: 
                    self.annotations[sentIdx] = {}
                self.annotations[sentIdx][tokenIdx] = info_dict

        with open(sentence_file,"r") as f: 
            for line in f.readlines():
                s = line.split("SENTENCE:\t\t")[1].strip("]\n") 
                self.sentences.append(s)

        self.paragraph_and_sentence_positions()
        self.preprocess_components()


    def paragraph_and_sentence_positions(self):
        prompt_length = len(self.prompt) + 1 # plus 1 for the extra newline in the essay TXT files 
        
        start = prompt_length  
        for paragraph in self.paragraphs:
            self.paragraph_idx.append(start)
            start += len(paragraph) 

        for sentence in self.sentences: 
            start = prompt_length + self.essay.index(sentence)
            for idx in range(len(self.paragraphs)): 
                if idx + 1 < len(self.paragraphs) -1: 
                    if self.paragraph_idx[idx] <= start < self.paragraph_idx[idx+1]: 
                        p_idx = idx 
                        break
                else: # final paragraph
                    if self.paragraph_idx[idx] <= start: 
                        p_idx = idx 
            self.sentence_idx.append((start,p_idx))


    def preprocess_components(self):
        augmented = {k:[] for k in range(len(self.sentence_idx))}
        for component in self.components: 
            name = component[0]
            type,start,end = component[1].split(' ')
            phrase = component[2]
            for idx in range(len(self.sentence_idx)): 
                sent_start = self.sentence_idx[idx][0]
                if idx + 1 < len(self.sentence_idx) -1: 
                    if sent_start <= int(start) < self.sentence_idx[idx+1][0]: 
                        s_idx = idx 
                        break
                else: 
                    if sent_start <= int(start): 
                        s_idx = idx
            info = {"name": name, "type": type,"start":int(start),"end":int(end),"sentence":s_idx,"phrase": phrase}
            augmented[s_idx].append(info)
        self.components = augmented 

    def annotate_tokens(self):
        for sentIdx, token_dict in self.annotations.items(): 
            start,p_idx = self.sentence_idx[sentIdx]
            for tokenIdx, info in token_dict.items(): 
                token_dict[tokenIdx]["start"] = start
                token_dict[tokenIdx]["paragraph"] = p_idx
                
                if p_idx == 0: 
                    token_dict[tokenIdx]["docPosition"] = "Introduction"
                elif p_idx == len(self.paragraphs)-1: 
                    token_dict[tokenIdx]["docPosition"] = "Conclusion"
                else: 
                    token_dict[tokenIdx]["docPosition"] = "Body"

                if tokenIdx == 1: 
                    token_dict[tokenIdx]["sentPosition"] = "First"
                elif tokenIdx == len(token_dict): 
                    token_dict[tokenIdx]["sentPosition"] = "Last"
                else: 
                    token_dict[tokenIdx]["sentPosition"] = "Middle"

                components = self.components[sentIdx]
                # initialize to default --> "O" meaning non-argumentative 
                token_dict[tokenIdx]["IOB"] = "O"
                for c in components: 
                    if c["start"] == start: # token that begins a component
                        token_dict[tokenIdx]["IOB"] = "Arg-B"
                        self.component_tokens[c["name"]] = [tokenIdx]
                        break
                    elif c["start"] < start < c["end"]: # token that is covered by an argument component 
                        token_dict[tokenIdx]["IOB"] = "Arg-I"
                        self.component_tokens[c["name"]].append(tokenIdx)
                        break

                token = info['token']
                if not re.search(r'[A-Za-z0-9]',token): # is punctuation
                    token_dict[tokenIdx]["isPunc"] = True
                    start += len(token)
                else: # is not punctuation  
                    token_dict[tokenIdx]["isPunc"] = False
                    start += len(token) + 1 # for white space  
                
        self.annotate_punctuation()

    def annotate_punctuation(self): 
        for sentIdx, token_dict in self.annotations.items(): 
            for tokenIdx, info in token_dict.items():
                # initialize to default 
                token_dict[tokenIdx]["followsPunc"] = False
                token_dict[tokenIdx]["precedesPunc"] = False
                # if token is not a punctuation mark 
                if not info["isPunc"]:
                    if tokenIdx == 1 and sentIdx > 0: 
                         token_dict[tokenIdx]["followsPunc"] = True
                    if tokenIdx > 1: 
                        if token_dict[tokenIdx-1]["isPunc"]: 
                            token_dict[tokenIdx]["followsPunc"] = True
                    if tokenIdx < len(token_dict): 
                        if token_dict[tokenIdx+1]["isPunc"]: 
                            token_dict[tokenIdx]["precedesPunc"] = True 
    
    def write_data(self,file_path): 
        with open(file_path, 'w+') as f:
            columns = ['token', 'sentence', 'index', 'lemma', 'pos', 
                       'sentiment', 'start', 'paragraph', 'docPosition',
                       'sentPosition', 'IOB', 'isPunc', 'followsPunc', 'precedesPunc']
            w = csv.DictWriter(f, columns)
            w.writeheader()
            for token_dict in self.annotations.values(): 
                for info in token_dict.values(): 
                    w.writerow(info)

    ''' 
    Helper function 
    '''
    def paragraph_info(self): 
        paragraph_components = {p_id: [] for p_id in range(len(self.paragraph_idx))} # p_id to components 
        paragraph_sentences = {p_id: [] for p_id in range(len(self.paragraph_idx))} # p_id to sentence indices 
        for sentIdx, components in self.components.items():  
            p_id = int(self.sentence_idx[sentIdx][1])
            paragraph_sentences[p_id].append(sentIdx)
            for component in components: 
                paragraph_components[p_id].append(component["name"])
        return paragraph_components,paragraph_sentences 
    ''' 
    Token Statistics for Each Component 
        For component classification:  
        - (1) Number of tokens in component
        - (2) Number of tokens in covering sentence 
        - (3) Number of tokens in covering paragraph 
        - (4) Number of tokens preceding component in sentence 
        - (5) Number of tokens succeeding component in sentence

        For component stance recognition:   
        - (2), (4), (5)
        - (6) Ratio of number of component to sentence tokens 
    '''
    def token_stats(self): 
        paragraph_tokens = {p_id: 0 for p_id in range(len(self.paragraph_idx))} # p_id to num_tokens 
        for sentIdx in range(len(self.sentence_idx)):
            p_id = self.sentence_idx[sentIdx][1]
            paragraph_tokens[p_id] += len(self.annotations[sentIdx])

        for sentIdx, components in self.components.items(): 
            p_id = self.sentence_idx[sentIdx][1]
            for component in components: 
                within,preceding,following = 0,0,0
                start = component["start"]
                name = component["name"]
                self.token_info[name] = {}
                self.token_info[name]["sentence"] = len(self.annotations[sentIdx])
                self.token_info[name]["paragraph"] = paragraph_tokens[p_id]
                for t_dict in self.annotations[sentIdx].values(): 
                    if t_dict["IOB"] == "O" and t_dict["start"] < start: 
                        preceding += 1 
                    elif t_dict["IOB"] == "O": 
                        following += 1 
                    else: 
                        within += 1 
                self.token_info[name]["within"] = within 
                self.token_info[name]["preceding"] = preceding
                self.token_info[name]["following"] = following                 

    '''
    Component Statistics
    - (1) If first or last in paragraph 
    - (2) Present in intro or conclusion
    - (3) Relative position in paragraph 
    - (4) Number of preceding and following components in paragraph 
    '''
    def component_stats(self): 
        paragraph_components,paragraph_sentences = self.paragraph_info()
        
        for sentIdx, components in self.components.items(): 
            p_id = self.sentence_idx[sentIdx][1]
            first_sent = paragraph_sentences[p_id][0]
            last_sent = paragraph_sentences[p_id][len(paragraph_sentences[p_id])-1]
            first_sent_start = self.annotations[first_sent][1]["start"] # first token 
            first_sent_end = self.annotations[first_sent][len(self.annotations[first_sent])]["start"] # last token 
            last_sent_start = self.annotations[last_sent][1]["start"]
            last_sent_end = self.annotations[last_sent][len(self.annotations[last_sent])]["start"]

            for component in components: 
                start = component["start"]
                name = component["name"]
                self.component_info[name] = {"sentIdx": sentIdx, "first/last":False, "intro/conc":False, 
                                        "num_paragraph": len(paragraph_components[p_id]),
                                        "num_preceding": 0, "num_following": 0}
                if p_id == 0 or p_id == len(paragraph_components)-1: 
                    # covering paragraph is either the intro or conclusion 
                    self.component_info[name]["intro/conc"] = True
                if first_sent_start <= start <= first_sent_end or last_sent_start <= start <= last_sent_end: 
                    # covering paragraph is either part of the first or last sentence in paragraph
                    self.component_info[name]["first/last"] = True
                for sentence in paragraph_sentences[p_id]:
                    for component in self.components[sentence]:
                        if component["start"] <  start: 
                            self.component_info[name]["num_preceding"] += 1 
                        elif component["start"] == start: 
                            continue
                        else: 
                            self.component_info[name]["num_following"] += 1 
    '''
    Structural Features for Relation Identification
    - (5) Number of components between source and target
    - (6) Number of components in covering paragraph
    - (7) If source and target are present in the same sentence
    - (8) If target present before source 
    - (9) If source and target are first or last component in paragraph 
    - (10) If source and target present in introduction or conclusion
    - Number of tokens in source 
    - Number of tokens in target 
    '''
    def pairs(self): 
        self.pairwise_tokens = {p_id: {} for p_id in range(len(self.paragraph_idx))}
        self.pairwise_components = {p_id: {} for p_id in range(len(self.paragraph_idx))}
        paragraph_components = self.paragraph_info()[0]
        
        for sentIdx, components in self.components.items(): 
            p_id = self.sentence_idx[sentIdx][1]
            for c1 in components:
                c1name = c1["name"]
                self.pairwise_tokens[p_id][c1name] = {}
                self.pairwise_components[p_id][c1name] = {}
                for c2name in paragraph_components[p_id]:
                    # c1 is source, c2 is target  
                    if c1name == c2name: continue 
                    self.pairwise_tokens[p_id][c1name][c2name] = (self.token_info[c1name]["within"], self.token_info[c2name]["within"])
                    
                    if c2name in self.pairwise_components[p_id]: # already encountered this pair before 
                        if c1name in self.pairwise_components[p_id][c2name]: 
                            self.pairwise_components[p_id][c1name][c2name] = self.pairwise_components[p_id][c2name][c1name].copy()
                    
                    # novel pair 
                    c2sentIdx = self.component_info[c2name]["sentIdx"]
                    self.pairwise_components[p_id][c1name][c2name] = {"first/last": False, "num_between": None,
                                                                      "num_paragraph": self.component_info[c1name]["num_paragraph"],
                                                                      "intro/conc": self.component_info[c1name]["intro/conc"],
                                                                      "targetBeforeSource": False, "sameSentence":sentIdx == c2sentIdx}
                    if self.component_info[c1name]["first/last"] and self.component_info[c2name]["first/last"]: 
                        self.pairwise_components[p_id][c1name][c2name]["first/last"] = True
                    c1idx,c2idx = self.component_info[c1name]["num_preceding"],self.component_info[c2name]["num_preceding"]
                    if c1idx < c2idx: 
                        self.pairwise_components[p_id][c1name][c2name]["num_between"] = c2idx - c1idx - 1
                    else: 
                        self.pairwise_components[p_id][c1name][c2name]["num_between"] = c1idx - c2idx - 1
                        self.pairwise_components[p_id][c1name][c2name]["targetBeforeSource"] = True


