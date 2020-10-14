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
        torch.nn.init.xavier_normal_(self.entity_embed.weight)
        torch.nn.init.xavier_normal_(self.relation_embed.weight)

        # multi-head graph attention
        if attention == 'GINN':
            self.attentions = [ConvAttentionLayer(self.relation_embed, dim, dim, n_channel, kernel_size) for _ in range(n_head)]
            self.out_attention = ConvAttentionLayer(self.relation_embed, dim * n_head, dim, n_channel, kernel_size)
        if attention == 'GAT':
            self.attentions = [GraphAttentionLayer(dim, dim) for _ in range(n_head)]
            self.out_attention = GraphAttentionLayer(dim * n_head, dim)

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        # scoring function
        if score_func == 'ConvE':
            self.score_function = ConvE(dim, reshape_size, n_channel, kernel_size)
        if score_func == 'DistMult':
            self.score_function = DistMult()

    def forward(self, triple, data):
        """update all entities' embeddings using attention mechanism and calculate 0-1 score for each triple 

        Args:
            triple (torch tensor): [h, r, t]

        Returns:
            (torch tensor): 0-1 score for each triple 
        """
        x = F.dropout(self.entity_embed.weight, self.dropout, training=self.training)
        x = torch.cat([att(x, triple) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_attention(x, triple)

        h = x[data[:, 0]]
        r = self.relation_embed(data[:, 1])
        h = F.dropout(h, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        score = self.score_function(h, r, self.entity_embed)

        return torch.sigmoid(score)
