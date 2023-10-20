import stanza
import pandas as pd

corenlp_dir = './corenlp'
# stanza.install_corenlp(dir=corenlp_dir)

import os
os.environ["CORENLP_HOME"] = corenlp_dir

from stanza.server import CoreNLPClient



if __name__ == '__main__':
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment'], 
        memory='4G', 
        endpoint='http://localhost:9001',
        be_quiet=True)
    client.start()
    text = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."
    document = client.annotate(text)
    text_file = open("output.txt", "w")
    to_write = ''    
    for i, sent in enumerate(document.sentence):
        to_write += str(sent) + '\n'
    text_file.write(to_write)
    text_file.close()

   
    data = []
    for i, sent in enumerate(document.sentence):
        for j, token in enumerate(sent.token):
            d = {
                'token':token.word,
                'lemma':token.lemma,
                'sentence':i,
                'index':j,
                'start':token.beginChar,
                'end':token.endChar,
                'pos':token.pos,
                'LCA':(),
                'followsLCA':(),
                'precedesLCA':(),
                'lexsyn':(),
                'probability':(),
                'sentiment':token.sentiment,
                'paragraph':(),
                'docPosition':len(data),
                'sentPosition':(),
                'isPunc':(),
                'followsPunc':(),
                'precedesPunc':(),
            }
            data.append(d)
    pd.DataFrame(data).to_csv("test.csv", index=False)







    client.stop()
