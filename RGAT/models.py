import torch
import torch.nn.functional as F
from RGAT.layers import ConvAttentionLayer


class RGAT(torch.nn.Module):
    def __init__(self, n_entity, n_relation, dim, dropout, n_head):
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

        self.entity_embeddings = torch.nn.Embedding(self.n_entity, self.dim)
        self.relation_embeddings = torch.nn.Embedding(self.n_relation, self.dim)
        torch.nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        torch.nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        # multi-head graph attention
        self.attentions = [ConvAttentionLayer(self.entity_embeddings, self.relation_embeddings, self.dim) for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.fc = torch.nn.Linear(dim*3, 1, bias=False)

    def forward(self, edge_list, triple):
        """update all entities' embeddings using attention mechanism and calculate 0-1 score for each triple 

        Args:
            triple (torch tensor): [h, r, t]

        Returns:
            (torch tensor): 0-1 score for each triple 
        """
        x = torch.mean(torch.stack([att(edge_list) for att in self.attentions]), dim=0)
        # x = F.dropout(x, self.dropout, training=self.training)

        h = x[triple[:, 0]]
        r = self.relation_embeddings(triple[:, 1])
        t = x[triple[:, 2]]
        # score = self.fc(torch.cat([h, r, t], dim=1))
        score = torch.sum(torch.mul(h, t), dim=1)

        return torch.sigmoid(score)
