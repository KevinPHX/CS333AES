path = '/Users/amycweng/Downloads/CS333_Project'
import pandas as pd 
import re, json
if __name__ ==  "__main__": 
    all_asap = pd.read_csv(f"{path}/asap-aes/training_set_rel3.tsv",sep="\t",encoding='latin')
    essays = {}
    for idx, essay_set in enumerate(all_asap["essay_set"]): 
        if essay_set == 2: 
            essay_id = all_asap["essay_id"][idx]
            # domain1 is scored based on ideas & content, organization, style & voice on a 1-6 scale 
            essays[int(essay_id)] = {
                                "text": all_asap["essay"][idx],
                                "domain1_score": int(all_asap["domain1_score"][idx])
            }
            essays[essay_id]["text"] = re.sub(r"\s{2}"," ",essays[essay_id]["text"])
            essays[essay_id]["text"] = re.sub(r"\s{2,}","\n",essays[essay_id]["text"])
            with open(f"{path}/asap_set2/{essay_id}.txt","w") as file:
                file.write(essays[essay_id]["text"])
        if essay_set > 2: 
            break
    with open(f"{path}/asap-aes/test_essays.json","w") as file: 
        json.dump(essays, file)