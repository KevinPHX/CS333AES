import json
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from sklearn.preprocessing import OneHotEncoder, normalize
import keras.backend as K
from keras import regularizers
enc = OneHotEncoder(sparse=False)
from sentence_transformers import SentenceTransformer

embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def encode(s:str):
    return embed.encode(s)

import tensorflow as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

if __name__=='__main__':
    train_encode = []
    train_true = []
    test_encode = []
    test_true = []
    max_count_rel = 1199
    max_count_rel_test = 304
    count = 0
    count_i = 0
    count_test = 0
    count_i_test = 0
    legend = {"MajorClaim":0, "Claim":0, "Premise":1}
    # enc.fit([])
    train_text = open("../assets/train_text.txt", "r").read().split('\n')
    test_text = open("../assets/test_text.txt", "r").read().split('\n')
    for essay_file in train_text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/classification/{essay_name}.json') as file: 
            components = json.load(file)
        for c in components:
            train_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            train_true.append(legend[c['claim']])
            # if legend[c['claim']] == 1:
            #     if count < max_count_rel:
            #         train_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            #         train_true.append(legend[c['claim']])
            #         count += 1
            # elif legend[c['claim']] == 2:
            #     if count_i < max_count_rel:
            #         train_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            #         train_true.append(legend[c['claim']])
            #         count_i += 1
            # else:
            #     train_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            #     train_true.append(legend[c['claim']])
                

    
    # with open("train_bert.json", "w") as f:
    #     json.dump(str({"data":train_encode, "label":train_true}), f)
    for essay_file in test_text: 
        essay_name = essay_file.split("-final/")[1]
        with open(f'../outputs/test/classification/{essay_name}.json') as file: 
            components = json.load(file)
        for c in components:
            test_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            test_true.append(legend[c['claim']])
            # if legend[c['claim']] == 1:
            #     if count_test < max_count_rel_test:
            #         test_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            #         test_true.append(legend[c['claim']])
            #         count_test += 1
            # elif legend[c['claim']] == 2:
            #     if count_i_test < max_count_rel_test:
            #         test_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            #         test_true.append(legend[c['claim']])
            #         count_i_test += 1
            # else:
            #     test_encode.append(encode(' '.join(c['preceding_tokens'])+' '+c['component_text']+' '+' '.join(c['following_tokens'])))
            #     test_true.append(legend[c['claim']])
    # with open("test_bert.json", "w") as f:
    #     json.dump(str({"data":test_encode, "label":test_true}), f)
    train_encode = normalize(train_encode)
    test_encode = normalize(test_encode)
    
    nn_train=enc.fit_transform(np.array(train_true).reshape(-1, 1))
    nn_test=enc.transform(np.array(test_true).reshape(-1, 1))
    X = np.concatenate([train_encode, test_encode])
    y = np.concatenate([nn_train, nn_test])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    # X_train, X_test, y_train, y_test = train_encode, test_encode, nn_train, nn_test

    clf_class = SVC(gamma='auto', C=1)
    clf_class.fit(train_encode, train_true)
    
    model = Sequential()
    model.add(Input(shape=(384,)))
    # model.add(Dense(200, activation='relu'))
    
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(200, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    optimizer = keras.optimizers.Adam(lr=0.001)
    # categorical_crossentropy
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1])
    # with open('classification_model_bert.pkl', 'wb') as f:
    #     pickle.dump(clf_class, f)
    # test_pred = clf_class.predict(train_encode)
    # report = classification_report(
    #         train_true,
    #         test_pred,
    #         # target_names = [0, 1],
    # )
    # print(report)
    # print(confusion_matrix(train_true, test_pred))

    
    # test_pred = clf_class.predict(test_encode)
    # report = classification_report(
    #         test_true,
    #         test_pred,
    #         # target_names = [0, 1],
    # )
    # print(report)
    # print(confusion_matrix(test_true, test_pred))

    
    model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_test), np.array(y_test)), epochs=15)
    nn_pred = model.predict(np.array(X_test))

    y_pred=np.argmax(nn_pred, axis=1)
    y_test_r=np.argmax(y_test, axis=1)
    report = classification_report(
            y_test_r,
            y_pred,
            # target_names = [0, 1],
    )
    print(report)
    print(confusion_matrix(y_test_r, y_pred))