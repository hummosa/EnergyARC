import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, normal_
import torch.nn.functional as F
from utils import *
import wandb

class RelationNetworks(nn.Module):
    ''' Code modified from https://github.com/rosinality/relation-networks-pytorch '''
    def __init__(
        self,
        channels_out=64,
        embed_size=32,
        mlp_hidden=64,
        n_vocab = 10,
        embed_dim = 3,
        latents_dim = 64,
        classes=1,
        use_wandb = False,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d( 3      , channels_out, [3, 3], 1, 0, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, [3, 3], 1, 0, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, [3, 3], 2, 0, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, [3, 3], 2, 0, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )

        self.embed = nn.Embedding(n_vocab, embed_dim)

        self.n_concat = channels_out * 2 + latents_dim + 2 * 2 # the 2*2 is for coordinates

        self.g = nn.Sequential(
            nn.Linear(self.n_concat, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            # nn.Linear(mlp_hidden, mlp_hidden),
            # nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f_cat_dim = mlp_hidden * 2 + latents_dim * 4

        self.f = nn.Sequential(
            nn.Linear(self.f_cat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, int(mlp_hidden/2)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(int(mlp_hidden/2), classes),
        )

        self.channels_out = channels_out
        self.latents_dim = latents_dim
        self.mlp_hidden = mlp_hidden
        self.use_wandb = use_wandb

        coords_dim = 5
        coords = torch.linspace(-int(coords_dim/2), int(coords_dim/2), coords_dim)
        x = coords.unsqueeze(0).repeat(coords_dim, 1)
        y = coords.unsqueeze(1).repeat(1, coords_dim)
        coords = torch.stack([x, y]).unsqueeze(0)
        self.register_buffer('coords', coords)

    def rn_embed(self, image, latents):
        conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        w, a = latents
        w_dim = w.size()[-1] 

        a = softmax_this(a)
        conv = torch.mul(conv, a.unsqueeze(2).unsqueeze(2))

        w_tile = w.unsqueeze(1).expand(batch_size, n_pair * n_pair, w_dim)

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

        return g

    def forward(self, input, latents):
        x0, x1 = input

        wx, wa, a0, a1 = latents.permute([1,0,2])

        g0 = self.rn_embed(x0, (wx, a0))
        g1 = self.rn_embed(x1, (wa, a1))
        
        g = torch.cat([g0, g1, latents.view([x0.size()[0], -1])], dim=1)
        
        f = self.f(g)

        # stats(f, 'f before')
        # f = torch.sigmoid(f)
        # stats(f, 'f after ')
        
        # ef = torch.exp(f)
        # print('check for inf values in f: finite? {}'.format(torch.isfinite(f).any()))
        # print('check for inf values in g1: finite? {}'.format(torch.isfinite(g1).any()))
        # print('check for inf values in g0: finite? {}'.format(torch.isfinite(g0).any()))

        # if self.use_wandb and torch.isfinite(f).any(): 
        #     wandb.log ({
        #     'g0': wandb.Histogram (g0.detach().cpu().numpy()),
        #     'g1': wandb.Histogram (g1.detach().cpu().numpy()),
        #     'f' : wandb.Histogram (f.detach().cpu().numpy() ),
        #     # 'exp_f': ef.detach().cpu().numpy(),
        #     }, commit=False)
        return f

class Ereason():
    def __init__(self,        channels_out=64, latents_dim = 64, use_wandb = False):
        self.rn = RelationNetworks(channels_out=64, latents_dim = 64, use_wandb = False)
    def forward(self, input, latents):
        return self.rn(self, input, latents)


def softmax_this(input, beta = 1.):
    z = torch.sum( torch.exp(beta*input) )
    input = torch.exp(beta*input) / (z + + 1e-9 )
    return (input)