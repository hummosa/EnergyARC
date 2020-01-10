import torch

from torch import nn
from torch.nn import functional as F
from torch.nn import utils


class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module

class conv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_out = 128
        # self.conv1 = spectral_norm(nn.Conv2d(1, self.channels_out, 3, padding=1), std=1)
        self.conv1 = nn.Conv2d(2 , self.channels_out, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(self.channels_out, self.channels_out, padding=0, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(self.channels_out, self.channels_out, padding=0, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        
        
    def forward(self, inputs, a):
        out = self.conv1(inputs)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.conv3(out)
        out = F.leaky_relu(out, negative_slope=0.2)
        
        
        # a0 = torch.randn(size = [ self.batch_size, self.out_channels ] ) #noise the shape of a0 = torch.randn(size = [c0.shape[0], c0.shape[1] ] )
        # a1 = torch.randn(size = [ self.batch_size, self.out_channels ] ) 

        a = softmax_this(a)
        out = torch.mul(out, a.unsqueeze(2).unsqueeze(2))
        
        return out

class e_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_out = 128
        self.lstm1 = nn.LSTM(self.channels_out, self.channels_out, batch_first = True)
        
    def forward(self, inputs, w):
        out = inputs.view([inputs.shape[0], self.channels_out, -1])
        out = torch.cat([w.view([out.shape[0], -1, 1]), out], dim=2)
        out = out.permute( [0, 2, 1])
        output_lstm, hidden = self.lstm1(out)
        # hidden is a tuple of (last output, last hidden state), last output has [batch, seq (=1), lstm dim]
        out = hidden[0].squeeze(1) #take the last hidden output as final out, remove the sequence dim

        return out

class h_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_out = 128
        self.lstm1 = nn.LSTM(self.channels_out, self.channels_out, batch_first = True)
        
    def forward(self, inputs, w):
        out = inputs.view([inputs.shape[0], self.channels_out, -1])
        out = torch.cat([w.view([out.shape[0], -1, 1]), out], dim=2)
        out = out.permute( [0, 2, 1])
        
        out, _ = self.lstm1(out)  # take all outputs for all conv locations

        return out

def softmax_this(input, beta = 1.):
	z = torch.sum( torch.exp(beta*input) ) 
	input = torch.exp(beta*input) / (z + + 1e-9 ) # added for numerical stability
	return (input)

class Ereason(nn.Module):
    def __init__(self, batch_size = 1, channels_out = 128):
        super().__init__()
        self.channels_out = channels_out
        self.batch_size = batch_size

        self.conv1 = conv3()
        self.conv2 = conv3()
        self.elstm1 = e_lstm()
        self.elstm2 = e_lstm()
        self.hlstm1 = h_lstm()
        self.hlstm2 = h_lstm()

        self.fc1 = nn.Linear(self.channels_out*5, 256)
        self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, inputs, latents):

        a0 = latents[:,0]
        a1 = latents[:,1]
        w = latents[:,2]

        c0 = self.conv1(inputs[0], a0)
        c1 = self.conv2(inputs[1], a1)

        # ## init w, if not provided
        # if w: 
        #    assert (w.shape == tensor.shape([out.shape[0], 1, self.channels_out])) 
        # else: # if w is not provided, initizlie from a random normal.
        #     w = torch.randn(size=[out.shape[0],1, self.channels_out])

        #get embeddings
        e0 = self.elstm1(c0, w)
        e1 = self.elstm2(c1, w)

        # ? why??? e0 has shape   [1, batch, channels_out ]
        e0.squeeze_()
        e1.squeeze_()
        
        # translation: TODO 
        # h0 = self.hlstm1( c0, w )
        # h1 = self.hlstm2( h0, w) #lstm with attention 
                

        out = torch.cat([e0, e1, latents.view([inputs[0].shape[0],-1])], dim=1)
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        out = (self.fc3(out)) #F.relu

        return out


