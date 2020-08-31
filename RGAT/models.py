import torch
import torch.nn.functional as F
from RGAT.layers import ConvAttentionLayer


class RGAT(torch.nn.Module):
    def __init__(self, n_entity, n_relation, dim, dropout, n_head, n_channel, kernel_size):
        """
        Args:
            n_entity (int): the number of entities
            n_relation (int): the number of relations
            dim (int): the dimension of hidden layer
            dropout (float): dropout rate
            n_head (int): the number of attention head
        """
        super(RGAT, self).__init__()
        self.dropout = dropout
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_channel = n_channel
        self.kernel_size = kernel_size

        self.entity_embeddings = torch.nn.Embedding(self.n_entity, self.dim)
        self.relation_embeddings = torch.nn.Embedding(self.n_relation, self.dim)
        torch.nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embeddings.weight.data)

        # multi-head graph attention
        self.attentions = [ConvAttentionLayer(self.entity_embeddings, self.relation_embeddings, self.dim, self.dropout, 
                                              self.n_channel, self.kernel_size) for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, triple, data):
        """update all entities' embeddings using attention mechanism and calculate 0-1 score for each triple 

        Args:
            triple (torch tensor): [h, r, t]

        Returns:
            (torch tensor): 0-1 score for each triple 
        """
        x = torch.mean(torch.stack([att(triple) for att in self.attentions]), dim=0)
        x = F.dropout(x, self.dropout, training=self.training)

        h = x[data[:, 0]]
        # h = self.entity_embeddings(data[:, 0])
        r = self.relation_embeddings(data[:, 1])
        h = F.dropout(h, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        
        score = torch.mm(torch.mul(h, r), self.entity_embeddings.weight.transpose(1,0))

        return torch.sigmoid(score)
