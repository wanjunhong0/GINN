import torch
import torch.nn.functional as F


class ConvAttentionLayer(torch.nn.Module):

    def __init__(self, entity_embeddings, relation_embeddings, dim, dropout, n_channel, kernel_size):
        """
        Args:
            entity_embeddings (torch embedding): the embedding of all the entities
            relation_embeddings (torch embedding): the embedding of all the relations
            dim (int): the dimension of hidden layer
            out_channels (int): the number of convolution channels
            kernel_size (int): the size of kernel
        """
        super(ConvAttentionLayer, self).__init__()

        self.dim = dim
        self.n_channel = n_channel
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.n_entity = entity_embeddings.weight.shape[0]
        self.n_relation = relation_embeddings.weight.shape[0]

        self.conv1_bn = torch.nn.BatchNorm2d(1)
        self.conv_layer = torch.nn.Conv2d(1, self.n_channel, (self.kernel_size, self.kernel_size))
        self.conv2_bn = torch.nn.BatchNorm2d(self.n_channel)
        self.fc_layer = torch.nn.Linear(self.n_channel * (self.dim - self.kernel_size + 1) * (3 - self.kernel_size + 1), 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc_layer.weight.data)
        torch.nn.init.xavier_uniform_(self.conv_layer.weight.data)


    def energy_function(self, h, r, t):
        """Calculate the attention coefficients for each triple [h, r, t]

        Args:
            h (torch tensor): The head of triple [h, r, t]
            r (torch tensor): The relation of triple [h, r, t]
            t (torch tensor): The tail of triple [h, r, t]

        Returns:
            (torch tensor): The attention coefficients of each triple [h, r, t]
        """
        h = h.unsqueeze(1)
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)
        conv_input = F.dropout(conv_input, self.dropout, training=self.training)
        # To make tensor of size 3, where second dim is for input channels
        conv_input = conv_input.transpose(1, 2).unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = F.relu(out_conv)
        out_conv = out_conv.view(conv_input.shape[0], -1)
        score = self.fc_layer(out_conv).view(-1)

        return score

    def forward(self, data):
        """Calculate the new embeddings of the entities according to attention coefficients

        Args:
            data (torch tensor): The matrix of the index of triple [h, r, t]

        Returns:
            (torch tensor): The new updated embeddings of the entities
        """
        h = data[:, 0]
        r = data[:, 1]
        t = data[:, 2]

        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        e = self.energy_function(h_emb, r_emb, t_emb)

        data_index = data.transpose(0, 1)
        data_index = torch.stack((data_index[0, :], data_index[2, :]))
        attention = torch.sparse.FloatTensor(data_index, e, torch.Size([self.n_entity, self.n_entity]))
        attention = attention.add(torch.eye(self.n_entity).to_sparse())
        attention = torch.sparse.softmax(attention, dim=1)
    
        return torch.sparse.mm(attention, self.entity_embeddings.weight)
