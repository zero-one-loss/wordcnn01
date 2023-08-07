import torch
import os
import pickle
import torch.nn as nn
import sys
sys.path.append('..')
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class TextWrapper(object):
    def __init__(self, embedding_layer=None, model=None, stoi=None):
        self.emb_layer = embedding_layer
        self.scd_type = None
        self.pad = False
        self.unk = '<unk>'
        self.word2id = stoi
        self.max_length = 0
        self.model = model
        self.cuda = True if torch.cuda.is_available() else False
        if self.cuda:
            self.emb_layer.cuda()
            self.model.cuda()

    def text_pred(self, text, batch_size=None):

        x = self.convert_features(text)
        if len(x.size()) == 3:
            x.unsqueeze_(dim=1)
        else:
            x.unsqueeze_(dim=0)
        yp = self.model(x)

        return yp

    def word2vector(self, text):
        x = [self.word2id.get(word, 0) for word in text]
        # ids = x
        # weights = torch.from_numpy(np.array(self.tfidf.transform([" ".join(text)]).todense()))[:, ids].float().T
        if self.pad:
            if len(x) < self.max_length:
                x += [self.word2id['<pad>']] * (self.max_length - len(x))
        x = torch.LongTensor(x)
        if self.cuda:
            x = x.cuda()
            # weights = weights.cuda()
        x_vectors = self.emb_layer(x.unsqueeze(dim=0))
        # x_tfidf = self.tfidf.transform(text)
        if self.pad:
            return x_vectors.reshape((1, -1))
        return x_vectors

    def convert_features(self, text):
        if isinstance(text[0], list):
            x = torch.cat([self.word2vector(samples) for samples in text], dim=0)
        else:
            x = self.word2vector(text)

        return x