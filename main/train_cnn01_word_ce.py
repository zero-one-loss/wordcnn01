import torch
import os
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
sys.path.append('..')

from tools import args, load_data, ModelArgs, BalancedBatchSampler
from core.lossfunction import ZeroOneLoss, BCELoss, CrossEntropyLoss
from core.cnn01 import *
from core.text_wrap import TextWrapper
from core.train_cnn01_word_ce import train_single_cnn01
import pickle
import pandas as pd
# Args assignment
scd_args = ModelArgs()

scd_args.nrows = args.nrows
scd_args.local_iter = args.localit
scd_args.num_iters = args.iters
scd_args.interval = args.interval
scd_args.rounds = 1
scd_args.w_inc1 = args.w_inc1
scd_args.updated_fc_features = args.updated_fc_features
scd_args.updated_conv_features = args.updated_conv_features
scd_args.n_jobs = 1
scd_args.num_gpus = 1
scd_args.adv_train = False
scd_args.eps = args.eps
scd_args.w_inc2 = args.w_inc2
scd_args.hidden_nodes = 20
scd_args.evaluation = True
scd_args.verbose = True
scd_args.b_ratio = 0.2
scd_args.cuda = True if torch.cuda.is_available() else False
scd_args.seed = args.seed
scd_args.source = args.source
scd_args.save = True
scd_args.resume = True if args.resume else False
scd_args.criterion = CrossEntropyLoss
scd_args.structure = arch[args.version]
scd_args.dataset = args.dataset
scd_args.num_classes = args.n_classes
scd_args.gpu = 0
scd_args.fp16 = True if args.fp16 and scd_args.cuda else False
scd_args.act = args.act
scd_args.updated_fc_nodes = args.updated_fc_nodes
scd_args.updated_conv_nodes = args.updated_conv_nodes
scd_args.width = args.width
scd_args.normal_noise = True
scd_args.verbose = True
scd_args.normalize = bool(args.normalize)
scd_args.batch_size = 256
scd_args.sigmoid = False
scd_args.softmax = True
scd_args.percentile = bool(args.percentile)
scd_args.fail_count = args.fail_count
scd_args.loss = args.loss
scd_args.diversity = False
scd_args.fc_diversity = bool(args.fc_diversity)
scd_args.conv_diversity = False
scd_args.updated_conv_features_diversity = 16
scd_args.diversity_train_stop_iters = 3000
scd_args.init = args.init
scd_args.target = args.target
scd_args.logs = {}
scd_args.no_bias = args.no_bias
scd_args.record = False
scd_args.scale = args.scale
scd_args.save_path = os.path.join('checkpoints', 'pt')
scd_args.adaptive_loss_epoch = args.epoch
scd_args.updated_fc_ratio = args.updated_fc_ratio
scd_args.updated_conv_ratio = args.updated_conv_ratio
scd_args.divmean = args.divmean
scd_args.cnn = args.cnn
scd_args.verbose_iter = args.verbose_iter
scd_args.freeze_layer = args.freeze_layer
scd_args.temp_save_per_iter = args.temp_save_per_iter
scd_args.lr_decay_iter = args.lr_decay_iter
scd_args.batch_increase_iter = args.batch_increase_iter
scd_args.aug = args.aug
scd_args.balanced_sampling = args.balanced_sampling
np.random.seed(scd_args.seed)
scd_args.embedding_path = 'glove.6B.200d.txt'
scd_args.drop_p = args.dropout

data_set = [f'../data/{scd_args.dataset}/all_token.csv',
            f'../data/{scd_args.dataset}/train_token.csv',
            f'../data/{scd_args.dataset}/test_token.csv', ]

best_model, embedding, val_acc, stoi = train_single_cnn01(
    scd_args, None, None, data_set)

model = TextWrapper(embedding_layer=embedding, model=best_model, stoi=stoi)

# text = pd.read_csv(data_set[2])['text'].values.tolist()
# text = ['"well', 'thats', 'nice.', 'too', 'bad', 'i', 'cant', 'eat', 'it']
# print(model.text_pred([text]*6))

with open(f'checkpoints/{args.target}.pkl', 'wb') as f:
    pickle.dump(model, f)