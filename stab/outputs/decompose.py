import json




if __name__ == '__main__':
    with open("components.json", "r") as f:
        data = json.load(f)
    train_text = open("../assets/train_text.txt", "r").read().split('\n')
    essays = {}  
    for txt in train_text:
        essays[txt.split('/')[-1]] = []
    for d in data:
        if d['essay'].split('/')[-1] in essays.keys():
            e = d['essay'].split('/')[-1]
            essays[e].append(d)
    for key in essays.keys():
        with open(f"./classification/{key}.json", "w") as f:
            json.dump(essays[key], f)

    

      
