# import gensim

from gensim import models

w = models.KeyedVectors.load_word2vec_format(
    './lib/GoogleNews-vectors-negative300.bin', binary=True)

if __name__ == '__main__':
    print(w['Here'])