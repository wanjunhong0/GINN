import torch
import torch.nn.functional as F


class DistMult(torch.nn.Module):
    """
    DistMult scoring function as https://arxiv.org/abs/1412.6575
    Change to 1-N scoring as https://github.com/TimDettmers/ConvE
    """
    def __init__(self, dropout):
        super(DistMult, self).__init__()

        self.dropout = dropout

    def forward(self, h, r, entity_embed):
        """
        Args:
            h (torch tensor): The head representation of triple [h, r, t]
            r (torch tensor): The relation representation of triple [h, r, t]
            entity_embed (torch embedding): the embedding of all the entites

        Returns:
            (torch tensor): 1-N score of each [h, r]
        """
        h = F.dropout(h, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        score = torch.mm(torch.mul(h, r), entity_embed.weight.T)

        return score


class ConvE(torch.nn.Module):
    """
    ConvE scoring function as https://arxiv.org/abs/1707.01476
    """
    def __init__(self, dim, reshape_size, n_channel, kernel_size, dropout):
        """
        Args:
            dim (int): dimension
            reshape_size (int): reshape size for input
            n_channel (int): the number of convolution channels
            kernel_size (int): the size of kernel
        """
        super(ConvE, self).__init__()

        assert dim % reshape_size == 0, "Invaid reshpe size"
        self.reshape_size1 = reshape_size
        self.reshape_size2 = int(dim / reshape_size)
        self.dropout = dropout

        self.conv = torch.nn.Conv2d(1, n_channel, (kernel_size, kernel_size))
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn2 = torch.nn.BatchNorm2d(n_channel)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.fc = torch.nn.Linear(n_channel * (self.reshape_size1 * 2 - kernel_size + 1) * (self.reshape_size2 - kernel_size + 1), dim)

    def forward(self, h, r, entity_embed):
        """
        Args:
            h (torch tensor): The head representation of triple [h, r, t]
            r (torch tensor): The relation representation of triple [h, r, t]
            entity_embed (torch embedding): the embedding of all the entites

        Returns:
            (torch tensor): 1-N score of each [h, r]
        """
        h = h.view(-1, 1, self.reshape_size1, self.reshape_size2)
        r = r.view(-1, 1, self.reshape_size1, self.reshape_size2)

        x = torch.cat([h, r], 2)
        x = self.bn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.bn3(x)
        x = F.relu(x)
        score = torch.mm(x, entity_embed.weight.T)

        return score