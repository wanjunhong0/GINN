import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ConvAttentionLayer(torch.nn.Module):

    def __init__(self, relation_embed, in_dim, out_dim, n_channel, kernel_size):
        """
        Args:
            relation_embed (torch embedding): the embedding of all the relations
            in_dim (int): input dimension
            out_dim (int): output dimension
            n_channel (int): the number of convolution channels
            kernel_size (int): the size of kernel
        """
        super(ConvAttentionLayer, self).__init__()

        self.relation_embed = relation_embed
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.conv = torch.nn.Conv2d(1, n_channel, (kernel_size, kernel_size))
        self.bn2 = torch.nn.BatchNorm2d(n_channel)
        self.fc = torch.nn.Linear(n_channel * (out_dim - kernel_size + 1) * (3 - kernel_size + 1), 1, bias=False)
        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))

        torch.nn.init.xavier_uniform_(self.W)

    def energy_function(self, h, r, t):
        """Calculate the attention coefficients for each triple [h, r, t]

        Args:
            h (torch tensor): The head representation of triple [h, r, t]
            r (torch tensor): The relation representation of triple [h, r, t]
            t (torch tensor): The tail of representation triple [h, r, t]

        Returns:
            (torch tensor): The attention coefficients of each triple [h, r, t]
        """        
        h = h.view(-1, 1, h.shape[1], 1)
        r = r.view(-1, 1, r.shape[1], 1)
        t = t.view(-1, 1, t.shape[1], 1)
        # to make tensor of size 3, where second dim is for input channels
        x = torch.cat([h, r, t], dim=3)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x).view(-1)
        return x

    def forward(self, input, triple):
        """Calculate the new embeddings of the entities according to attention coefficients

        Args:
            input (torch Tensor): The entity embedding of last layer    
            triple (torch tensor): The matrix of the index of triple [h, r, t]

        Returns:
            (torch tensor): The new updated embeddings of the entities
        """
        N = input.shape[0]
        input_ = torch.mm(input, self.W)
        h = input_[triple[:, 0]]
        r = self.relation_embed(triple[:, 1])
        t = input_[triple[:, 2]]
        e = F.leaky_relu(self.energy_function(h, r, t))
        e = torch.sparse.FloatTensor(triple[:, [0, 2]].T, e, torch.Size([N, N]))
        attention = torch.sparse.softmax(e, dim=1)
        output = torch.sparse.mm(attention, input_) + input_

        return F.elu(output)


class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
        """
        super(GraphAttentionLayer, self).__init__()

        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.a = Parameter(torch.FloatTensor(out_dim * 2, 1))
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.a)

    def forward(self, input, triple):
        """
        Args:
            input (torch Tensor): The entity embedding of last layer
            triple (torch tensor): The matrix of the index of triple [h, r, t]

        Returns:
            (torch tensor): The new updated embeddings of the entities
        """
        N = input.shape[0]
        input_ = torch.mm(input, self.W)
        a_input = torch.cat([input_[triple[:, 0], :], input_[triple[:, 2], :]], dim=1)
        e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=0.2).view(-1)
        e = torch.sparse.FloatTensor(triple[:, [0, 2]].T, e, torch.Size([N, N]))
        attention = torch.sparse.softmax(e, dim=1)
        output = torch.sparse.mm(attention, input_) + input_

        return F.elu(output)
