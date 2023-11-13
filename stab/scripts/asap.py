import pandas as pd 
if __name__ ==  "__main__": 
    all_asap = pd.read_csv("asap-aes/training_set_rel3.tsv",sep="\t",encoding='latin')
    essays = {}
    for idx, essay_set in enumerate(all_asap["essay_set"]): 
        if essay_set == 2: 
            essay_id = all_asap["essay_id"][idx]
            # domain1 is scored based on ideas & content, organization, style & voice on a 1-6 scale 
            essays[essay_id] = {
                                "text": all_asap["essay"][idx],
                                "domain1_score": all_asap["domain1_score"]
            }
        if essay_set > 2: 
            break