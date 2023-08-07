import argparse
import os
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
import sys
import pickle
sys.path.append('..')
from core.text_wrap import TextWrapper
import readline
import tensorflow.compat.v1 as tf

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig


# from attack_classification import args

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def attack(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    # first check the prediction of the original text
    # print(text_ls)
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<unk>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[
                    0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def random_attack(text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
                  sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                  synonym_num=50, batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # randomly get perturbed words
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[
                    0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data",
                    type=str,
                    required=True,
                    help="Which dataset to attack.")
parser.add_argument("--dataset_path",
                    type=str,
                    # required=True,
                    help="Which dataset to attack.")
parser.add_argument("--nclasses",
                    type=int,
                    default=2,
                    help="How many classes for classification.")
parser.add_argument("--vote",
                    type=int,
                    default=None,
                    help="How many classes for classification.")
parser.add_argument("--target_model",
                    type=str,
                    required=True,
                    # choices=['wordLSTM', 'bert', 'wordCNN'],
                    help="Target models for text classification: fasttext, charcnn, word level lstm "
                         "For NLI: InferSent, ESIM, bert-base-uncased")
parser.add_argument("--target_model_path",
                    type=str,
                    required=True,
                    help="pre-trained target model path")
parser.add_argument("--word_embeddings_path",
                    type=str,
                    default='glove.6B.200d.txt',
                    help="path to the word embeddings for the target model")
parser.add_argument("--counter_fitting_embeddings_path",
                    type=str,
                    # required=True,
                    default='counter-fitted-vectors.txt',
                    help="path to the counter-fitting embeddings we used to find synonyms")
parser.add_argument("--counter_fitting_cos_sim_path",
                    type=str,
                    default='cos_sim_counter_fitting.npy',
                    help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
parser.add_argument("--USE_cache_path",
                    type=str,
                    # required=True,
                    default='checkpoints',
                    help="Path to the USE encoder cache.")
parser.add_argument("--output_dir",
                    type=str,
                    default='adv_results',
                    help="The output directory where the attack results will be written.")

## Model hyperparameters
parser.add_argument("--sim_score_window",
                    default=15,
                    type=int,
                    help="Text length or token number to compute the semantic similarity score")
parser.add_argument("--import_score_threshold",
                    default=-1.,
                    type=float,
                    help="Required mininum importance score.")
parser.add_argument("--sim_score_threshold",
                    default=0.7,
                    type=float,
                    help="Required minimum semantic similarity score.")
parser.add_argument("--synonym_num",
                    default=50,
                    type=int,
                    help="Number of synonyms to extract")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Batch size to get prediction")
parser.add_argument("--data_size",
                    default=1000,
                    type=int,
                    help="Data size to create adversaries")
parser.add_argument("--perturb_ratio",
                    default=0.,
                    type=float,
                    help="Whether use random perturbation for ablation study")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="max sequence length for BERT target model")
parser.add_argument("--soft",
                    default=0,
                    type=int,
                    help="max sequence length for BERT target model")
parser.add_argument("--dropout", type=float, default=0.3)
args = parser.parse_args()





np.random.seed(2018)
torch.manual_seed(2018)

if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
else:
    os.makedirs(args.output_dir, exist_ok=True)


# construct the model
print("Building Model...")
if args.target_model == 'wordLSTM':
    model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
    checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint)
elif 'cnn' in args.target_model:
    model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True,
                  kernel=[4,], dropout=args.dropout).cuda()
    checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint)
    model.eval()
elif args.target_model == 'bert':
    model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
else:
    # model = TextWrapper(embedding_path=args.word_embeddings_path, scd_path=args.target_model_path, vote=args.vote,
    #                     soft=args.soft)
    with open(args.target_model_path, 'rb') as f:
        model = pickle.load(f)
predictor = model.text_pred
print("Model built!")

# prepare synonym extractor
# build dictionary via the embedding file
idx2word = {}
word2idx = {}

print("Building vocab...")
with open(args.counter_fitting_embeddings_path, 'r', encoding='utf8') as ifile:
    for line in ifile:
        word = line.split()[0]
        if word not in idx2word:
            idx2word[len(idx2word)] = word
            word2idx[word] = len(idx2word) - 1

print("Building cos sim matrix...")
if args.counter_fitting_cos_sim_path:
    # load pre-computed cosine similarity matrix if provided
    print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
    cos_sim = np.load(args.counter_fitting_cos_sim_path)
else:
    # calculate the cosine similarity matrix
    print('Start computing the cosine similarity matrix!')
    embeddings = []
    with open(args.counter_fitting_embeddings_path, 'r', encoding='utf8') as ifile:
        for line in ifile:
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(embedding)
    embeddings = np.array(embeddings).astype(np.float32)
    product = np.dot(embeddings, embeddings.T)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    cos_sim = product / np.dot(norm, norm.T)
    np.save('cos_sim_counter_fitting.npy', cos_sim)
print("Cos sim import finished!")

# build the semantic similarity module
use = USE(args.USE_cache_path)

# start attacking
orig_failures = 0.
adv_failures = 0.
changed_rates = []
nums_queries = []
orig_texts = []
adv_texts = []
true_labels = []
new_labels = []
# log_file =

stop_words_set = criteria.get_stopwords()

# get data to attack
# texts, labels = dataloader.read_corpus('data/%s' % args.data, clean=True)
# texts, labels = dataloader.read_corpus(os.path.join(f'../data/{args.data}',
#                                                          'test_token.csv'),
#                                             clean=False, MR=False, shuffle=False)

while True:
    print('Type "retry" without quotes if you want to retry typing')
    print('Type your review, press Enter when typing is done')
    ts = input()
    if ts == 'retry':
        continue
    texts = [ts.split()]

    while True:
        print('Type the correct label (0 or 1)')
        lb = input()
        if lb == 'retry':
            continue
        try:
            labels = [int(lb)]
            break
        except:
            del lb
            print('Please type integer')
            continue


    pred = model.text_pred(texts[0]).argmax(dim=1).item()
    if pred != labels[0]:
        print(f'Models prediction: {pred} is not correct, attack based '
              f'on the current prediction will inverse the prediction')
    labels = [pred]
    data = list(zip(texts, labels))
    data = data[:min(args.data_size, len(data))]
    data_size = len(data)# choose how many samples for adversary
    print('data_size: ', data_size)
    print("Data import finished!")

    print('Start attacking!')
    for idx, (text, true_label) in enumerate(data):
        if idx % 20 == 0:
            print('{} samples out of {} have been finished!'.format(idx, data_size))
        if args.perturb_ratio > 0.:
            new_text, num_changed, orig_label, \
            new_label, num_queries = random_attack(text, true_label, predictor, args.perturb_ratio, stop_words_set,
                                                   word2idx, idx2word, cos_sim, sim_predictor=use,
                                                   sim_score_threshold=args.sim_score_threshold,
                                                   import_score_threshold=args.import_score_threshold,
                                                   sim_score_window=args.sim_score_window,
                                                   synonym_num=args.synonym_num,
                                                   batch_size=args.batch_size)
        else:
            new_text, num_changed, orig_label, \
            new_label, num_queries = attack(text, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, cos_sim, sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / len(text)

        if true_label == orig_label and true_label != new_label:
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)

            print('Attack successfully. Adversary example:')
            print(new_text)

        else:
            print('Attack failed')

# message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
#           'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(args.target_model,
#                                                                        (1 - orig_failures / data_size) * 100,
#                                                                        (1 - adv_failures / data_size) * 100,
#                                                                        np.mean(changed_rates) * 100,
#                                                                        np.mean(nums_queries))
# print(message)
# with open(os.path.join(args.output_dir, '%s_results_log' % args.data), 'a') as f:
#     f.write(message)
#
# with open(os.path.join(args.output_dir, '%s_%s_adversaries.txt' % (args.target_model, args.data)), 'w',
#           encoding='utf8') as ofile:
#     for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
#         ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))
