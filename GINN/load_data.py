import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer


class Data(object):
    """Data loading
    """
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
        self.entity_map = dict(zip(entity.unique(), range(entity.nunique())))
        self.relation_map = dict(zip(data['relation'].unique(), range(data['relation'].nunique())))

        train = self.map_index(train)
        val = self.map_index(val)
        test = self.map_index(test)

        self.triple_train = torch.LongTensor(train.values)
        self.triple_val = torch.LongTensor(val.values)
        self.triple_test = torch.LongTensor(test.values)
        # prepare (h, r): [t...]
        self.train, self.label_train = self.prepare_input(train)
        self.val, self.label_val = self.prepare_input(val)
        self.test, self.label_test = self.prepare_input(test)
        # prepare filter and index
        self.filter_val = self.prepare_filter(self.val, self.label_val)
        self.filter_test = self.prepare_filter(self.test, self.label_test)
        # index for output(groupby [h, r]) to each triple
        self.index_val = [(self.val == i).all(1).nonzero(as_tuple=False).item() for i in self.triple_val[:, :2]]
        self.index_test = [(self.test == i).all(1).nonzero(as_tuple=False).item() for i in self.triple_test[:, :2]]
        
    def map_index(self, dataset):
        """Map entities and relations to index

        Args:
            dataset (pandas DataFrame): dataset to map

        Returns:
            (pandas DataFrame): dataset after mapping
        """
        dataset['head'] = dataset['head'].map(self.entity_map)
        dataset['tail'] = dataset['tail'].map(self.entity_map)
        dataset['relation'] = dataset['relation'].map(self.relation_map)
        return dataset

    def prepare_input(self, dataset):
        """Prepare model input for multiple BCE loss

        Args:
            dataset (pandas DataFrame): triple to process

        Returns:
            h_r (torch tensor): unique [h, r] after groupby
            label (torch tensor): onehot t(dim = n_entity) for each [h, r] 
        """
        dataset = dataset.groupby(by=['head', 'relation'], as_index=False).agg(list)
        h_r = torch.LongTensor(dataset[['head', 'relation']].values)
        mlb = MultiLabelBinarizer(classes=range(self.n_entity))
        label = torch.FloatTensor(mlb.fit_transform(dataset['tail'].values))
        return h_r, label
        
    def prepare_filter(self, h_r, label):
        """Prepare filter to filter out results in train

        Args:
            h_r (torch tensor): unique [h, r] after groupby
            label (torch tensor): onehot t(dim = n_entity) for each [h, r]

        Returns:
            (torch tensor): matrix only contain 0 and 1, and 0 for the results in train
        """
        filter_ = []
        for i in range(h_r.shape[0]):
            try:
                filter_.append(self.label_train[(self.train == h_r[i]).all(1).nonzero(as_tuple=False).item()])
            except ValueError:
                filter_.append(label[i])
        filter_ = torch.stack(filter_, dim=0)
        filter_ = torch.ones_like(filter_) - filter_ + torch.mul(label, filter_)
        return filter_
