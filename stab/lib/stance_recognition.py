from itertools import groupby
from operator import itemgetter
import json
import re
from features import STANCE_FEATURES
'''
Number of components in paragraph
number of preceding and following components in paragraph
Relative position of the argument component in paragraph
'''

class StanceRecognition():
    def __init__(self, components):
        self.all_components = components
        self.stances = {}
    
    def process_data(self, model=None, train=True):
        self.components = []
        for component in self.all_components:
            if component['claim'] =='Claim':
                if train:
                    stance = self.get_labels(component['essay'], component['index'])
                else:
                    assert(model)
                    stance = model.predict([component[key] for key in STANCE_FEATURES])
                if stance:
                    self.components.append({**component, 'stance':stance})
        


    def get_labels(self, essay, index):
        if essay not in self.stances.keys():
            self.stances[essay] = {}
            file = essay.replace(".txt", ".ann")
            f = open(file, 'r').read()
            lines = f.split('\n')
            for line in lines:
                # print(line)
                if len(line) > 0 and 'A' == line[0]:
                    values = re.split('\t| ', line)
                    self.stances[essay][values[2]] = values[3]
        if f'T{index}' in self.stances[essay].keys():
            return self.stances[essay][f'T{index}']
        return None
    
            
if __name__=='__main__':
    with open("components.json", "r") as f:
        data = json.load(f)
    stance = StanceRecognition(data)
    stance.process()
    print(stance.components)
