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
        self.component_map = {}
    
    def process_data(self, model=None, train=True):
        self.components = []
        for component in self.all_components:
            if component['claim'] =='Claim':
                if train:
                    stance = self.get_labels(component['essay'], component['start'], component['end'])
                else:
                    assert(model)
                    stance = model.predict([component[key] for key in STANCE_FEATURES])
                if stance:
                    clues = self.subjectivity_clues(component)
                    self.components.append({**component, **clues, 'stance':stance})
    def subjectivity_clues(self, component):
        ret = {
            'num_positive':0,
            'num_negative':0,
            'num_neutral':0,
            'positive_negative':0
        }

        for sentiment in component['component_sentiment']:
            if sentiment == 'Positive':
                ret['num_positive'] +=1
            elif sentiment == 'Neutral':
                ret['num_neutral'] +=1
            elif sentiment == 'Negative':
                ret['num_negative'] +=1

        ret['positive_negative'] = ret['num_positive'] - ret['num_negative']
        return ret

    def get_labels(self, essay, start, end):
        if essay not in self.stances.keys():
            self.stances[essay] = {}
            self.component_map[essay] = {}
            file = essay.replace(".txt", ".ann")
            f = open(file, 'r').read()
            lines = f.split('\n')

            for line in lines:
                # print(line)
                if len(line) > 0 and 'A' == line[0]:
                    values = re.split('\t| ', line)
                    self.stances[essay][values[2]] = values[3]
            
                
            for line in lines:
                values = re.split('\t| ', line)
                for index in self.stances[essay].keys():
                    print(values)
                    if values[0] == index:
                        self.component_map[essay][values[2]+'-'+values[3]] = index
        print(self.component_map[essay])
        for key in self.component_map[essay].keys():
            val = key.split('-')
            if start >= int(val[0]) and end <= int(val[1]):
                return self.stances[essay][self.component_map[essay][key]]
        return None
    
            
if __name__=='__main__':
    with open("../outputs/classification/essay001.txt.json", "r") as f:
        data = json.load(f)
    # print(data)
    stance = StanceRecognition(data)
    stance.process_data()
    print(stance.components)
