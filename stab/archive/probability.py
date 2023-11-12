from statsmodels.discrete.discrete_model import Poisson
import numpy as np
import pandas as pd

class Probability:
    def __init__(self, n, train_set, target, probability_func = Poisson):
        self.probability_func = probability_func
        self.n = n
        self.target=target
        self.train_set = train_set
        self.train()
    # Stab needs at least 3 different models since they compare among a range of the last three tokens
    # We create 3 models for different token ingestions
    def train(self):
        # increment by 2 to get the targeted tag value (y)
        window_sz = np.arange(self.n)+2
        X= [[] for i in range(self.n)]
        Y = [[] for i in range(self.n)]
        self.models = []
        for index, size in enumerate(window_sz):
            # preprocess data per dataset -> this is important when training across multiple docs
            for data in self.train_set:
                # creates sliding window of the desired size
                for window in np.lib.stride_tricks.sliding_window_view(data, size):
                    if len(window) == size:
                        x = window[:size-1].tolist()
                        # makes the target a binary of whether the y value is the target (Arg-B) or not - this is what we are ultimately trying to predict
                        y = (window[-1]==self.target).astype(int).tolist()
                        X[index].append(x)  
                        Y[index].append(y)
            self.models.append(self.probability_func(Y[index], X[index]).fit())
    
    def predict(self, X):
        assert(len(X) == self.n)
        probabilities = []
        # There should be n models, so we iterate across each model and get their respective probabilities
        for index, model in enumerate(self.models):
            probabilities.append(model.predict(X[:index+1]))
        # Find the max probability for any window size and return it
        index = np.argmax(probabilities)
        return probabilities[index]


if __name__ == "__main__":
    df = pd.read_csv('./token_annotations/essay001.csv')
    df['IOB_cat'] = df.IOB.astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    # With this categorization, Arg-B = 0, Arg-I = 1, O = 2
    prob = Probability(3, [df.IOB_cat.values], 0)
    # order starts were index 0 is previous token
    test_x = [1, 0, 1]
    print(prob.predict(test_x))
