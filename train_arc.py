
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

wandb_use = False
if wandb_use: import wandb
if wandb_use: wandb.init(project="earc")

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


def im_log(x):
    return(  plt.imshow(x[0][0].to('cpu').numpy()) )

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




def train(model, alpha=0.1, step_size=1, sample_step=30, device=device):
    batch_size = 256
    w_dim = 128
    no_of_demos = 5
    
    dataset = dataset_arc.ARCDataset('./ARC/data/', 'both', transform= dataset_arc.all_transforms, no_of_demos=no_of_demos) #, transform=transforms.Normalize(0., 10.)
    # dataset = ARCDataset('./ARC/data/', 'both', transform= all_transforms) #, transform=transforms.Normalize(0., 10.)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) # Error: Couldn't open shared event, when workers is 4, hmmm but also 1
    loader = tqdm(enumerate(sample_data(loader)))

    # buffer = SampleBuffer()
    # noise = torch.randn(128, 3, 32, 32, device=device)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0.0, 0.999))

    # pixel_reward_demos = [0. for _ in range(no_of_demos)]
    # pixel_reward_tasks = [0. for _ in range(no of tasks)]

    for t, batch in loader: # [i/o, demos, batch,   c, w, h]
        w = torch.randn([len(batch[0]), 3, w_dim], device=device)
        w.requires_grad = True

        model.train()

        batch[0] = batch[0].to(device)
        batch[1] = batch[1].to(device)
        
        '''############# Getting W across all demos ################'''
        for k in tqdm(range(sample_step)):

            for i in range(no_of_demos):
                pos_inp = (batch[0][:,i], batch[1][:,i])
            
                noise = torch.randn_like(w, device=device)
                noise.normal_(0, 0.005)
                w.data.add_(noise.data)

                w_out = model(pos_inp, w)
                w_grad = torch.autograd.grad(w_out.sum(), w)

                #TODO consider an alpha of 0 for demos that are just zero pads
                w.data.add_(-step_size, w_grad[0]) # dunno why grad returns a tuple of len 1!
                #note: the differentiation path to w remains in the graph, and will take grad across it
        '''############# Solving for X1_ using W ####################'''
        if wandb_use: wandb.log( {'w': wandb.Histogram( w.detach().to('cpu').numpy() )}, commit=False)
        total_loss = 0.
        for i in range(no_of_demos): # the order here flips (for task for k, because each x1_ is propagated serparately 
                                                #for each input-output in training set)
            pos_inp = (batch[0][:,i], batch[1][:,i])
            x0 = pos_inp[0]
            x1 = pos_inp[1]
            
            #init predicted answer x1_ to x0, alternatively start from noise.  torch.randn([batch_size, 2, 30, 30])
            x1_ = x0.detach().clone()
            x1_.requires_grad = True
            
            demo_loss = 0.
            sample_step_tqdm = tqdm(range(sample_step))
            for k in sample_step_tqdm:

                noise = torch.randn_like(x1, device=device)
                noise.normal_(0, 0.005)
                x1_.data.add_(noise.data)

                x1_out = model((x0, x1_), w)
                x1_grad = torch.autograd.grad(x1_out.sum(), x1_)

                x1_.data.add_(-step_size, x1_grad[0])  # TODO Step size according to RL reward!!! That might be it!  save the pixel rewards and nudge each backprop step based on how it changes pixel reward?
                # accumlate loss every step
                demo_loss_local = torch.norm(x1_- x1)/ (torch.norm(x1) + torch.norm(x1_))
                sample_step_tqdm.set_description(f'demo_loss: {demo_loss_local.item():.5f}')
                demo_loss += demo_loss_local
            
            # log occasionally
            if i == 0 and t % 50:
                if wandb_use: wandb.log({ "examples":[wandb.Image( im_log(x0 ), caption='question'),
                                        wandb.Image( im_log(x1 ), caption='answer'),
                                        wandb.Image( im_log(x1_), caption='prediction')],
                            })
        
            #######################################################################
            #? Does this need to happen every k or every t??????????????????????
            # Every input pair i.  
            # OPTION 1: calculate loss and prop for every demo pair.
            # Option 2: accumulate losses and keep the graph, then optimize once at the end.
            #? consider making latent w for the task, and another one for the specific demo example
            #######################################################################

            # TODO check the model grad here. It should be hopefully zero. unless it was updated to 
            # model.zero_grad()
            # TODO to calculate the above partial derivatives. 

            # now, push up on the value of x0, x1, w(theta). So push up by adjusting theta to be closer to x0 x1 but also make the sampler of w(theta) more likely to produce these specific w

            # ! for each demo:
            # copy model and run through copy. The goal of this is not to create a local punch at x1_!
            #  but to change the slope towards E minima
            
            for from_param, to_param in zip(model.parameters(), model_back.parameters()):
                to_param.data.copy_(from_param)
            
            kl_loss = model_back([x0, x1_], w.detach()) # Grad Not: model, w   Grad x1_ and the Langevin dynamics. 

            # requires_grad(parameters, True)
            # model.train()

            pos_out = model([x0, x1], w)
            #force gradient to go through w, so that params os model change to give this particular one more energy.
            neg_out = model([x0, x1_.detach()], w)
            
            #KL loss
            
            loss = alpha * (pos_out ** 2 + neg_out ** 2)
            loss = loss + torch.abs(  (pos_out - neg_out)  ) + kl_loss 
            loss = loss.mean()

            total_loss += loss

        #! for all demos
        optimizer.zero_grad()  # ? hmmmmmm, am I erasing intermediate grads for w and x1_ ?
        total_loss.backward()
        clip_grad(parameters, optimizer)
        optimizer.step()

        x1_.detach_()
        w.detach_()
            
        # buffer.push(w, neg_id)
        loader.set_description(f'loss: {total_loss.item():.5f}')
        # if i == 0:# log only the first input deomo into wandb
        if wandb_use: wandb.log({
            'model_loss': total_loss,
            # "latents": wandb.Histogram(w[0].detach().to('cpu').numpy()),
            })
            # wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})
        # else:
            # wandb.log({
            #     'model_loss': total_loss,
            # })


        total_loss = 0.
        if t % 20 == 0:
            try:
                # utils.save_image(  #this is nice because it can display a whole batch of photos
                #     w.detach().to('cpu'),
                #     f'samples/{str(i).zfill(5)}.png',
                #     nrow=16,
                #     normalize=True,
                #     range=(0, 1),
                # )
                plt.figure(dpi=200, figsize=[10, 10])
                plt.subplot(2, 2, 1)
                plt.imshow(x0[0].detach().to('cpu').squeeze().numpy()[0])
                plt.title('question')
                
                plt.subplot(2, 2, 2)
                plt.imshow(x1[0].detach().to('cpu').squeeze().numpy()[0])
                plt.title('answer')

                plt.subplot(2, 2, 4)
                plt.imshow(x1_[0].detach().to('cpu').squeeze().numpy()[0])
                plt.title('prediction')
                plt.savefig(f'samples/{str(t).zfill(5)}.png')
                if wandb_use: wandb.log ( {'plots': plt}, commit=False)
            except:
                print('the shape of the image: {} \n info: {}'.format(w.shape, w))

    

if __name__ == '__main__':
    model = Ereason().to(device)
    model_copy = Ereason().to(device)
    if wandb_use: wandb.watch(model)
    train(model)
