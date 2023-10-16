import os,re,csv

class Structural():

    def __init__(self): 
        self.paragraphs = [] 
        self.components = []
        self.annotations = {}
            # maps sentence index to token information (represented as a dictionary per token)
        self.sentences = []
        self.prompt = ""
        self.essay = ""
        self.paragraph_idx = [] # starting position of the first character of each paragraph (1-indexing)
        self.sentence_idx = [] # tuple with (start position of sentence, paragraph the sentence belongs to)

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
            claim,start,end = component[1].split(' ')
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
            info = {"name": name, "claim": claim,"start":int(start),"end":int(end),"sentence":s_idx,"phrase": phrase}
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
                        break
                    elif c["start"] < start < c["end"]: # token that is covered by an argument component 
                        token_dict[tokenIdx]["IOB"] = "Arg-I"
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


if __name__ == "__main__": 
    essayDir = "ArgumentAnnotatedEssays-2.0/brat-project-final"
    annDir = "CS333AES/stab/preprocessing/src/main/resources/token_level"
    sentDir = "CS333AES/stab/preprocessing/src/main/resources/sentence_sentiment"

    for file in sorted(os.listdir(essayDir)):
        if ".ann" in file: 
            essay_ann_file = f"{essayDir}/{file}"
        if ".txt" in file: 
            essay_txt_file = f"{essayDir}/{file}"
            token_file = f"{annDir}/{file}"
            sentence_file = f"{sentDir}/{file}"
        # testing only with essay001.txt 
        if "001.txt" in file: break
        
    essay001 = Structural()
    essay001.read_data(essay_ann_file, essay_txt_file, token_file, sentence_file)
    essay001.annotate_tokens()
    outputdir = "CS333AES/stab/token_annotations" 
    essay001.write_data(f"{outputdir}/essay001.csv")