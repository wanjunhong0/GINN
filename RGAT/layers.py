import torch
import torch.nn.functional as F


class ConvAttentionLayer(torch.nn.Module):

    def __init__(self, entity_embeddings, relation_embeddings, hidden_size, out_channels=32, kernel_size1=2, kernel_size2=2):
        """
        Args:
            entity_embeddings (torch embedding): the embedding of all the entities
            relation_embeddings (torch embedding): the embedding of all the relations
            hidden_size (int): the dimension of hidden layer
            out_channels (int, optional): the number of convolution channels. Defaults to 32.
            kernel_size1 (int, optional): the size of kernel. Defaults to 2.
            kernel_size2 (int, optional): the size of kernel. Defaults to 2.
        """
        super(ConvAttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.n_entity = entity_embeddings.weight.shape[0]
        self.n_relation = relation_embeddings.weight.shape[0]

        self.conv1_bn = torch.nn.BatchNorm2d(1)
        self.conv_layer = torch.nn.Conv2d(1, self.out_channels, (self.kernel_size1, self.kernel_size2))  # kernel_size1 x kernel_size2
        self.conv2_bn = torch.nn.BatchNorm2d(self.out_channels)
        self.fc_layer = torch.nn.Linear(self.out_channels * (self.hidden_size - self.kernel_size1 + 1) * (3 - self.kernel_size2 + 1), 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc_layer.weight.data)
        torch.nn.init.xavier_uniform_(self.conv_layer.weight.data)


    def _calc(self, h, r, t):
        """Calculate the attention coefficients for each triple [h, r, t]

        Args:
            h (torch tensor): The head of triple [h, r, t]
            r (torch tensor): The relation of triple [h, r, t]
            t (torch tensor): The tail of triple [h, r, t]

        Returns:
            (torch tensor): The attention coefficients of each triple [h, r, t]
        """
        h = h.unsqueeze(1)  # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
        n_dim = conv_input.shape[1]  # 感知的维度
        conv_input = conv_input.transpose(1, 2)  # bs x dim x 3
        # To make tensor of size 3, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)  # bs x 1 x dim x 3
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)  # bs x out_channels x (dim-kernel_size1+1) x (3-kernel_size2+1)
        out_conv = self.conv2_bn(out_conv)
        out_conv = F.relu(out_conv)
        # dim = hidden_size
        out_conv = out_conv.view(-1, self.out_channels * (self.hidden_size - self.kernel_size1 + 1) * (n_dim - self.kernel_size2 + 1))
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
        score = self._calc(h_emb, r_emb, t_emb)

        data_index = data.transpose(0, 1)
        data_index = torch.stack((data_index[0, :], data_index[2, :]))
        att_matrix = torch.sparse.FloatTensor(data_index, score, torch.Size([self.n_entity, self.n_entity]))
        att_matrix = att_matrix.add(torch.eye(self.n_entity).to_sparse())
        att_matrix = torch.sparse.softmax(att_matrix, dim=1)
    
        return torch.sparse.mm(att_matrix, self.entity_embeddings.weight)
