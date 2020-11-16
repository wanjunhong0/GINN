import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer


class Data(object):
    """Data loading
    """
    def __init__(self, path):
        # read file
        train, val, test = [pd.read_csv(path + i, sep='\t', header=None,
                            names=['head', 'relation', 'tail'], keep_default_na=False)
                            for i in ['/train.txt', '/valid.txt', '/test.txt']]
        # get full entity and relation set
        data = pd.concat([train, test, val], axis=0)
        entity = pd.concat([data['head'], data['tail']], axis=0)
        self.n_entity = entity.nunique()
        self.n_relation = data['relation'].nunique()
        # map entities and relations to index
        self.entity_map = dict(zip(entity.unique(), range(self.n_entity)))
        self.relation_map = dict(zip(data['relation'].unique(), range(self.n_relation)))
        train, val, test, data = [self.map_index(i) for i in [train, val, test, data]]
        self.triple_train, self.triple_val, self.triple_test = [torch.LongTensor(i.values) for i in [train, val, test]]
        # prepare (h, r): [t...]
        [self.train, self.label_train], [self.val, self.label_val], [self.test, self.label_test], [self.data, self.label_data] = \
            [self.prepare_input(i) for i in [train, val, test, data]]
        # prepare filter and index
        self.filter_val = self.prepare_filter(self.val, self.label_val)
        self.filter_test = self.prepare_filter(self.test, self.label_test)
        # index for output(groupby [h, r]) to each triple
        self.index_val = [torch.where((self.val.T == self.triple_val[:, :2].unsqueeze(-1)).all(1))[1], self.triple_val[:, 2]]
        self.index_test = [torch.where((self.test.T == self.triple_test[:, :2].unsqueeze(-1)).all(1))[1], self.triple_test[:, 2]]

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

        filter = self.label_data[torch.where((self.data.T == h_r.unsqueeze(-1)).all(1))[1]]
        filter = torch.ones_like(filter) - filter + label
        return filter
