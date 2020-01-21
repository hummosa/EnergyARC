import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, normal_
import torch.nn.functional as F


class RelationNetworks(nn.Module):
    def __init__(
        self,
        conv_out=64,
        embed_size=32,
        mlp_hidden=256,
        n_vocab = 10,
        embed_dim = 3,
        latents_dim = 64,
        classes=1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d( 3      , conv_out, [3, 3], 1, 0, bias=False),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(),
            nn.Conv2d(conv_out, conv_out, [3, 3], 2, 0, bias=False),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(),
            nn.Conv2d(conv_out, conv_out, [3, 3], 2, 0, bias=False),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(),
            nn.Conv2d(conv_out, conv_out, [3, 3], 1, 0, bias=False),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(),
        )

        self.embed = nn.Embedding(n_vocab, embed_dim)

        self.n_concat = conv_out * 2 + latents_dim + 2 * 2 # the 2*2 is for coordinates

        self.g = nn.Sequential(
            nn.Linear(self.n_concat, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden/2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden/2, classes),
        )

        self.conv_out = conv_out
        self.latents_dim = latents_dim
        self.mlp_hidden = mlp_hidden

        coords = torch.linspace(-4, 4, 8)
        x = coords.unsqueeze(0).repeat(8, 1)
        y = coords.unsqueeze(1).repeat(1, 8)
        coords = torch.stack([x, y]).unsqueeze(0)
        self.register_buffer('coords', coords)

    def forward(self, image, latents):
        conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        w = latents[0]
        w_dim = w.size()[-1] 

        w_tile = w.expand(batch_size, n_pair * n_pair, w_dim)

        conv = torch.cat([conv, self.coords.expand(batch_size, 2, conv_h, conv_w)], 1)
        n_channel += 2
        conv_tr = conv.view(batch_size, n_channel, -1).permute(0, 2, 1)
        conv1 = conv_tr.unsqueeze(1).expand(batch_size, n_pair, n_pair, n_channel)
        conv2 = conv_tr.unsqueeze(2).expand(batch_size, n_pair, n_pair, n_channel)
        conv1 = conv1.contiguous().view(-1, n_pair * n_pair, n_channel)
        conv2 = conv2.contiguous().view(-1, n_pair * n_pair, n_channel)

        concat_vec = torch.cat([conv1, conv2, w_tile], 2).view(-1, self.n_concat)
        g = self.g(concat_vec)
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden).sum(1).squeeze()

        f = self.f(g)

        return f
