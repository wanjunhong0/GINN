import torch
import torch.nn.functional as F
from GINN.layers import ConvAttentionLayer, GraphAttentionLayer
from GINN.score_function import ConvE, DistMult


class GINN(torch.nn.Module):
    def __init__(self, n_entity, n_relation, dim, dropout, n_head, n_channel, kernel_size, attention, score_func, reshape_size):
        """
        Args:
            n_entity (int): the number of entities
            n_relation (int): the number of relations
            dim (int): the dimension of hidden layer
            dropout (float): dropout rate
            n_head (int): the number of attention head
        """
        super(GINN, self).__init__()

        self.dropout = dropout
        self.entity_embed = torch.nn.Embedding(n_entity, dim)
        self.relation_embed = torch.nn.Embedding(n_relation, dim)
        self.attention = attention
        torch.nn.init.xavier_normal_(self.entity_embed.weight)
        torch.nn.init.xavier_normal_(self.relation_embed.weight)

        # multi-head graph attention
        if self.attention == 'GINN':
            self.attentions = [ConvAttentionLayer(self.relation_embed, dim, dim, n_channel, kernel_size) for _ in range(n_head)]
            self.out_attention = ConvAttentionLayer(self.relation_embed, dim * n_head, dim, n_channel, kernel_size)
        if self.attention == 'GAT':
            self.attentions = [GraphAttentionLayer(dim, dim) for _ in range(n_head)]
            self.out_attention = GraphAttentionLayer(dim * n_head, dim)
        if self.attention in ['GINN', 'GAT']:
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)
        
        # scoring function
        if score_func == 'ConvE':
            self.score_function = ConvE(dim, reshape_size, n_channel, kernel_size, dropout)
        if score_func == 'DistMult':
            self.score_function = DistMult(dropout)

    def forward(self, triple_hop1, triple_hop2, data):
        """update all entities' embeddings using attention mechanism and calculate 0-1 score for each triple 

        Args:
            triple (torch tensor): [h, r, t]

        Returns:
            (torch tensor): 0-1 score for each triple 
        """
        if self.attention == 'None':
            h = self.entity_embed(data[:, 0])
        else:
            x = torch.cat([att(self.entity_embed.weight, triple_hop2) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.out_attention(x, triple_hop1)
            x = F.dropout(x, self.dropout, training=self.training)
            h = x[data[:, 0]]

        r = self.relation_embed(data[:, 1])
        score = self.score_function(h, r, self.entity_embed)

        return torch.sigmoid(score)
