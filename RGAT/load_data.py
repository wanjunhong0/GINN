import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import random


class Data(object):
    def __init__(self, path, seed):
        random.seed(seed)
        # read file
        train = pd.read_csv(path + '/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        valid = pd.read_csv(path + '/valid.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        test = pd.read_csv(path + '/test.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        # get full entity and relation set
        data = pd.concat([train, test, valid], axis=0)
        entity = pd.concat([data['head'], data['tail']], axis=0)
        self.n_entity = entity.nunique()
        self.n_relation = data['relation'].nunique()
        # map entities and relations to index
        entity_map = dict(zip(entity.unique(), range(entity.nunique())))
        relation_map = dict(zip(data['relation'].unique(), range(data['relation'].nunique())))

        train['head'] = train['head'].map(entity_map)
        train['tail'] = train['tail'].map(entity_map)
        train['relation'] = train['relation'].map(relation_map)

        valid['head'] = valid['head'].map(entity_map)
        valid['tail'] = valid['tail'].map(entity_map)
        valid['relation'] = valid['relation'].map(relation_map)

        test['head'] = test['head'].map(entity_map)
        test['tail'] = test['tail'].map(entity_map)
        test['relation'] = test['relation'].map(relation_map)
        # graph
        self.graph = torch.LongTensor(train.values)

        # train, negative sampling
        train_neg = neg_sampling(train, self.n_entity, self.n_relation, 'head')
        train['label'] = 1
        train_neg['label'] = 0
        train = pd.concat([train, train_neg], axis=0)
        self.train = torch.LongTensor(train[['head', 'relation', 'tail']].values)
        self.label_train = torch.FloatTensor(train['label'].values)
        # valid, test
        self.valid = torch.LongTensor(valid.values)
        self.test = torch.LongTensor(test.values)


def neg_sampling(dataset, n_entity, n_relation, mode):
    head_relation = dataset.groupby('relation')['head'].apply(set)
    tail_relation = dataset.groupby('relation')['tail'].apply(set)
    relation_entity_neg = dict()
    for i in range(n_relation):
        relation_entity_neg[i] = list(set(range(n_entity)) - (head_relation[i] | tail_relation[i]))

    head_neg, tail_neg = [], []
    for i in dataset['relation']:
        sample = random.choices(relation_entity_neg[i], k=2)
        head_neg.append(sample[0])
        tail_neg.append(sample[1])
    neg = dataset.copy()
    if mode == 'head':
        neg['head'] = head_neg
    if mode == 'tail':
        neg['tail'] = tail_neg
    if mode == 'all':
        neg['head'] = head_neg
        neg['tail'] = tail_neg

    return neg
