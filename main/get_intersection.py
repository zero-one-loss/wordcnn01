import pickle
import sys
sys.path.append('..')
import pandas as pd
import numpy as np


def map_(label):

    return 'positive' if label == 1 else 'negative'


def prediction(sentence, model):

    ts = sentence.split()
    print(f'This review is {map_(model.text_pred(ts).argmax(dim=1).item())}.')


with open('checkpoints/mr_wordcnn01_bp_01_fs_2.pkl', 'rb') as f:
    model_01 = pickle.load(f)

with open('checkpoints/mr_wordcnn01_bp_relu_2.pkl', 'rb') as f:
    model_bp = pickle.load(f)


def evaluation(text):

    print('review: ', text)
    print('model_bp: ')
    prediction(text, model_bp)
    print('model_01: ')
    prediction(text, model_01)
    print()


if __name__ == '__main__':

    df = pd.read_csv('../data/mr/test_token.csv')[:1000]

    yp_bp = []
    yp_01 = []

    for text in df.text:
        yp_bp.append(model_bp.text_pred(text.split()).argmax(dim=1).item())
        yp_01.append(model_01.text_pred(text.split()).argmax(dim=1).item())

    yp_bp = np.array(yp_bp)
    yp_01 = np.array(yp_01)
    index = np.array(1000)

    both_correct = (yp_bp == df.label.values).astype(np.int8) + (yp_01 == df.label.values).astype(np.int8)
    both_correct = both_correct==2

    correct_index = np.nonzero(both_correct)[0]

    df.iloc[correct_index].to_csv('mr_samples.csv', index=False)
