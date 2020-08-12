import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import random


class Data(object):
    def __init__(self, path):
        # read file
        self.train = pd.read_csv(path + '/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        self.val = pd.read_csv(path + '/valid.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        self.test = pd.read_csv(path + '/test.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        # get full entity and relation set
        data = pd.concat([self.train, self.test, self.val], axis=0)
        entity = pd.concat([data['head'], data['tail']], axis=0)
        self.n_entity = entity.nunique()
        self.n_relation = data['relation'].nunique()
        # map entities and relations to index
        entity_map = dict(zip(entity.unique(), range(entity.nunique())))
        relation_map = dict(zip(data['relation'].unique(), range(data['relation'].nunique())))

        self.train['head'] = self.train['head'].map(entity_map)
        self.train['tail'] = self.train['tail'].map(entity_map)
        self.train['relation'] = self.train['relation'].map(relation_map)

        self.val['head'] = self.val['head'].map(entity_map)
        self.val['tail'] = self.val['tail'].map(entity_map)
        self.val['relation'] = self.val['relation'].map(relation_map)

        self.test['head'] = self.test['head'].map(entity_map)
        self.test['tail'] = self.test['tail'].map(entity_map)
        self.test['relation'] = self.test['relation'].map(relation_map)


    def neg_sampling(self, dataset, predict):
        head_relation = dataset.groupby('relation')['head'].apply(set)
        tail_relation = dataset.groupby('relation')['tail'].apply(set)
        relation_entity_neg = dict()
        for i in range(self.n_relation):
            relation_entity_neg[i] = list(set(range(self.n_entity)) - (head_relation[i] | tail_relation[i]))

        head_neg, tail_neg = [], []
        for i in dataset['relation']:
            sample = random.choices(relation_entity_neg[i], k=2)
            head_neg.append(sample[0])
            tail_neg.append(sample[1])
        neg = dataset.copy()
        if predict == 'head':
            neg['head'] = head_neg
        if predict == 'tail':
            neg['tail'] = tail_neg

        return torch.LongTensor(neg.values)
