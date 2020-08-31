import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import random
from sklearn.preprocessing import MultiLabelBinarizer


class Data(object):
    def __init__(self, path):
        # read file
        train = pd.read_csv(path + '/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        val = pd.read_csv(path + '/valid.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        test = pd.read_csv(path + '/test.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],
                            keep_default_na=False, encoding='utf-8')
        # get full entity and relation set
        data = pd.concat([train, test, val], axis=0)
        entity = pd.concat([data['head'], data['tail']], axis=0)
        self.n_entity = entity.nunique()
        self.n_relation = data['relation'].nunique()
        # map entities and relations to index
        entity_map = dict(zip(entity.unique(), range(entity.nunique())))
        relation_map = dict(zip(data['relation'].unique(), range(data['relation'].nunique())))

        train['head'] = train['head'].map(entity_map)
        train['tail'] = train['tail'].map(entity_map)
        train['relation'] = train['relation'].map(relation_map)

        val['head'] = val['head'].map(entity_map)
        val['tail'] = val['tail'].map(entity_map)
        val['relation'] = val['relation'].map(relation_map)

        test['head'] = test['head'].map(entity_map)
        test['tail'] = test['tail'].map(entity_map)
        test['relation'] = test['relation'].map(relation_map)

        self.triple_train = torch.LongTensor(train.values)
        self.triple_val = torch.LongTensor(val.values)
        self.triple_test = torch.LongTensor(test.values)
        # prepare (h, r): [t...]
        self.train, self.label_train = self.prepare_input(train)
        self.val, self.label_val = self.prepare_input(val)
        self.test, self.label_test = self.prepare_input(test)
        

    def prepare_input(self, data):
        data = data.groupby(by=['head', 'relation'], as_index=False).agg(list)
        h_r = torch.LongTensor(data[['head', 'relation']].values)
        mlb = MultiLabelBinarizer(classes=range(self.n_entity))
        label = torch.FloatTensor(mlb.fit_transform(data['tail'].values))
        return h_r, label




















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
