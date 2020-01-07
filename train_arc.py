
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import datasets, transforms, utils
torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm

from model_arc import Ereason
import dataset_arc
from utils import *

import wandb
wandb.init(project="earc")

def to_tensor_transforms(item_input):
    return (
        (0.1) * torch.tensor(item_input, dtype=torch.float32, device=device).unsqueeze_(0).unsqueeze_(1) 
    )

def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)
            yield next(loader_iter)


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

def train(model, alpha=1, step_size=10, sample_step=30, device=device):
    dataset = dataset_arc.ARCDataset('./ARC/data/', 'both', transform= dataset_arc.all_transforms, no_of_demos=5) #, transform=transforms.Normalize(0., 10.)
    # dataset = ARCDataset('./ARC/data/', 'both', transform= all_transforms) #, transform=transforms.Normalize(0., 10.)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1) # Error: Couldn't open shared event, when workers is 4, hmmm but also 1
    loader = tqdm(enumerate(sample_data(loader)))

    # buffer = SampleBuffer()
    # noise = torch.randn(128, 3, 32, 32, device=device)

    batch_size = 1
    w_dim = 128
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0.0, 0.999))
    #for t, batch in loader: # [i/o, demos, batch,   c, w, h]
    for t, task in loader: #batch:
        # here I am assuming that all examples in the task share a common w. I think w[0] is in common, but a (w[1], and w[2]) might not be in common necessarily
        w = torch.randn([3, w_dim], device=device)
        w.requires_grad = True

        requires_grad(parameters, False)
        # model.eval() #because lstm layer insists to  be in training mode to allow backprop through it. 
        model.train()

        task=  [(inp.to(device), out.to(device)) for (inp, out) in task]

        for k in tqdm(range(sample_step)):

            for i, pos_inp in enumerate(task):
                    # pos_inp = pos_inp.to(device)
                
                    noise = torch.randn_like(w, device=device)
                    noise.normal_(0, 0.005)
                    w.data.add_(noise.data)

                    w_out = model(pos_inp, w)
                    w_out.sum().backward()
                    w.grad.data.clamp_(-0.01, 0.01)

                    w.data.add_(-step_size, w.grad.data)

                    w.grad.detach_()
                    w.grad.zero_()

                    # w.data.clamp_(0, 1)

        w.detach_() 


        total_loss = 0.
        for i, pos_inp in enumerate(task):# the order here flips (for task for k, because each x1_ is propagated serparately for each input-output in training set)
            for k in tqdm(range(sample_step)): 
                
                # pos_inp = pos_inp.to(device)
                x0 = pos_inp[0]
                x1 = pos_inp[1]

                #init predicted answer x1_ to x0, alternatively start from noise.  torch.randn([batch_size, 2, 30, 30])
                x1_ = x0.clone()
                x1_.requires_grad = True

                noise = torch.randn_like(x1, device=device)
                noise.normal_(0, 0.005)
                x1_.data.add_(noise.data)

                x1_out = model((pos_inp[0], x1_), w)
                x1_out.sum().backward()
                x1_.grad.data.clamp_(-0.01, 0.01)

                x1_.data.add_(-step_size, x1_.grad.data)  #Step size according to RL reward!!! That might be it!

                # x1_.grad.detach_() # consider removing so I backpro through every step every step
                # save the pixel rewards and nudge each backprop step based on how it changes pixel reward?

                x1_.grad.zero_()



                requires_grad(parameters, True)
                model.train()

                model.zero_grad()

                kl_loss = model([x0, x1_], w.detach()) # grad through x1_ and the Langevin dynamics. 

                pos_out = model(pos_inp, w)
                #force gradient to go through w, so that params os model change to give this particular one more energy.
                neg_out = model([x0, x1_.detach()], w)
                
                #KL loss

                loss = alpha * (pos_out ** 2 + neg_out ** 2)
                loss = loss + (pos_out - neg_out) + kl_loss
                loss = loss.mean()
                total_loss += loss
            
        total_loss.backward()
        clip_grad(parameters, optimizer)
        optimizer.step()

        # buffer.push(w, neg_id)
        loader.set_description(f'loss: {total_loss.item():.5f}')
        total_loss = 0.
        wandb.log({
            'model_loss': total_loss
        })
        if t % 5 == 0:
            try:
                # utils.save_image(  #this is nice because it can display a whole batch of photos
                #     w.detach().to('cpu'),
                #     f'samples/{str(i).zfill(5)}.png',
                #     nrow=16,
                #     normalize=True,
                #     range=(0, 1),
                # )
                plt.figure(dpi=1200)
                plt.subplot(1, 2, 1)
                plt.imshow(x1.detach().to('cpu').squeeze().numpy()[0])
                plt.title('answer')
                plt.subplot(1, 2, 2)
                plt.imshow(x1_.detach().to('cpu').squeeze().numpy()[0])
                plt.title('prediction')
                plt.savefig(f'samples/{str(i).zfill(5)}.png')

            except:
                print('the shape of the image: {} \n info: {}'.format(w.shape, w))

if __name__ == '__main__':
    model = Ereason().to(device)
    wandb.watch(model)
    train(model)
