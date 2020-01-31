
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using : ', device)

from torchvision import datasets, transforms, utils

from tqdm import tqdm

import dataset_arc
from utils import *

use_wandb = False
if use_wandb: import wandb
if use_wandb: wandb.init(project="earc")

from model_rn import RelationNetworks as Ereason

def sample_data(loader, one_batch= False, loop= True, max_loops = 1000000):
    loader_iter = iter(loader)
    loop_count = 0
    memorize_one_batch = next(loader_iter)
    while one_batch and loop_count<max_loops: #if one_batch is True, keep yielding the exact same batch, for model checking.
        loop_count += 1
        yield (memorize_one_batch)
        
    while loop and loop_count<max_loops:
        try:
            loop_count += 1
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)
            yield next(loader_iter)


most_recent_time = time.process_time()

no_of_demos = 5
batch_size = 16
one_batch_debug = True 
conv_channels_dim = 64
w_dim = 64
debug=True
torch.autograd.set_detect_anomaly(debug) # this slows down computations but allows better traces for backward functions. 
                                         # it also checks for NaN during the backward computation.
#%%

def train(model, loader, alpha=0.1, step_size=0.1, sample_step=10, device=device, most_recent_time=most_recent_time):
    # tracemalloc.start(5)
    # snapshots = []
        
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0.0, 0.999))
    w_init = torch.randn([512, 4, w_dim], device=device)

    for t, batch in loader: # [i/o, demos, batch,   c, w, h]
        # for each batch update the copy of the model, used for backward prop through the gradient
        for from_param, to_param in zip(model.parameters(), model_copy.parameters()):
            to_param.data.copy_(from_param)
        requires_grad(model_copy.parameters(), False)
        
        # w = nn.Parameter(torch.randn([len(batch[0]), 4, w_dim], device=device)) 
        w = nn.Parameter(w_init[:len(batch[0])]) 
        # w = torch.randn([len(batch[0]), 4, w_dim], device=device)
        w.requires_grad = True

        model.train()

        batch[0] = batch[0].to(device) # send input x0 to cuda
        batch[1] = batch[1].to(device) # send outputs x1 to cuda
        
        '''############# Getting W across all demos ################'''
        for k in tqdm(range(sample_step)):

            for i in range(1,no_of_demos): # leave the first pair for the training of the meta-loop
                pos_inp = (batch[0][:,i], batch[1][:,i])
            
                noise = torch.randn_like(w, device=device)
                noise.normal_(0, 0.005)
                w= w + (noise.data)

                w_out = model(pos_inp, w)
                w_grad = torch.autograd.grad(w_out.sum(), w,
                create_graph=True)[0] #Create_graph ensures that differentiation is added to the gradient graph for later derivatives calculation

                w = w + (-0.1 * step_size * w_grad) # ! make lr smaller
            # if debug: print('k*i: {} \t norm of w: {:.4e}  \t of grad w: {:.4e}'.format(k*i, torch.norm(w), torch.norm(w_grad[0])))
        
        '''############# Solving for X1_ using W ####################'''
        if use_wandb: wandb.log( {'w': wandb.Histogram( w.detach().to('cpu').numpy() ),
                                'E_out': wandb.Histogram( w_out.detach().to('cpu').numpy() ),
        }, commit=False)
        grad_viz_container = []; grad_norms = []
        total_loss = 0.

        for i in tqdm(range(1)):    # ! look only at the first pair
                                                # the order here flips (for task for k, because each x1_ is propagated serparately 
                                                #for each input-output in training set)
            x0, x1 = (batch[0][:,i], batch[1][:,i])
            
            # x1_ = x0.detach().clone()
            x1_ = nn.Parameter(torch.rand_like(x0.detach()))
            # x1_ = torch.rand_like(x0.detach())
            x1_.requires_grad = True
            
            demo_loss = 0.
            for k in range(sample_step):

                noise = torch.randn_like(x1, device=device)
                noise.normal_(0, 0.005)
                x1_ = x1_ + (noise.data)

                x1_out = model((x0, x1_), w) 
                x1_grad = torch.autograd.grad(
                    outputs= x1_out.sum(), 
                    inputs = x1_,
                    only_inputs=True,
                    create_graph=True,)[0] #grad returns a tuple of 1 :/
                    # retain_graph=True )
                
                x1_grad = torch.sign(x1_grad) # TODO taking the sign

                # if debug:
                    # grad_norms.append(torch.norm((x1_grad)))
                    # print('grad_norm for step {}: {} '.format(k, grad_norms[k]))
                    # grad_viz_container.append(x1_grad)
                
                # x1_= x1_ + (- 1000 * step_size* x1_grad)  # ! multiply by 100
                x1_= x1_ + (- step_size* x1_grad)  # ! multiply by 100
                # TODO Step size according to RL reward!!! That might be it!  save the pixel rewards and nudge each backprop step based on how it changes pixel reward?
                # accumlate loss every step
                # demo_loss_local = torch.norm(x1_.detach().cpu()- x1)/ (torch.norm(x1) + torch.norm(x1_.detach().cpu()))
                # demo_loss += demo_loss_local
            if debug:
                plt.subplot(1,3,1)
                im_log(x1); plt.title('x1')
                plt.subplot(1,3,2)
                im_log(x1_); plt.title('x1_')
                plt.subplot(1,3,3)
                im_log((x1_grad)); plt.title('gard')
                plt.savefig('samples/x1_out_and_grad.png')

                stats(x1_, 'x1_')
                stats((x1_grad), 'signed x1_grad')
            # ! for each demo:
            kl_loss = model_copy([x0, x1_], w.detach()) # Grad Not: model, w   Grad x1_ and the Langevin dynamics. 

            # requires_grad(parameters, True)
            pos_out = model([x0, x1], w)
            #force gradient to go through w, so that params os model change to give this particular one more energy.
            neg_out = model([x0, x1_.detach()], w)
            # norm_loss = alpha * (pos_out ** 2 + neg_out ** 2)
            # loss = (  (pos_out - neg_out)  ) + (kl_loss ) # + norm_loss
            loss = F.softplus(  (pos_out - neg_out + 1)  ) #+ F.softplus(kl_loss ) # + norm_loss
            print('pos_out: {}\t neg_out {} \t kl_loss {}\t loss: {}'.format(pos_out[0].item(), neg_out[0].item(), kl_loss[0].item(), loss[0].item()))
            total_loss += loss.mean()
        #! for all demos 
        # with torch.autograd.profiler.record_function("Outer optim"): # label the block    
        # import graphviz
        # import torchviz
        # dot = torchviz.make_dot(loss)
        # ff = dot.render('round-table2.gv', view=True) 

        optimizer.zero_grad()  
        total_loss.backward()
        # clip_grad(parameters, optimizer)
        optimizer.step()
    
        x1_.detach_()
        w.detach_()
        total_loss.detach_()
        
        # buffer.push(w, neg_id)
        loader.set_description(f'loss: {total_loss.item():.5f}')
        # if i == 0:# log only the first input deomo into wandb
        if use_wandb: wandb.log({
            'model_loss': total_loss,
            # "latents": wandb.Histogram(w[0].detach().to('cpu').numpy()),
            })
            # wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})
      

        # log occasionally
        # if t % 50 == 0:
        #     if use_wandb: wandb.log({ "examples":[wandb.Image( im_log(x0 ), caption='question'),
        #                             wandb.Image( im_log(x1 ), caption='answer'),
        #                             wandb.Image( im_log(x1_), caption='prediction')],
        #                 })
        total_loss = 0.
        if t % 50 == 0:
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
                if use_wandb: wandb.log ( {'plots':wandb.Image(plt)}, commit=False)
                plt.close()
            except:
                print('the shape of the image: {} \n info: {}'.format(w.shape, w))

    #     snapshot = tracemalloc.take_snapshot()
    #     snapshots.append(snapshot)
    # display_top(snapshots[0], limit=20)

#%%
import tracemalloc

if __name__ == '__main__':
    model = Ereason(channels_out = conv_channels_dim,use_wandb=use_wandb).to(device)
    model_copy = Ereason(channels_out = conv_channels_dim).to(device)
    if use_wandb: wandb.watch(model)
    
    dataset = dataset_arc.ARCDataset('./ARC/data/', 'both', transform= dataset_arc.all_transforms, no_of_demos=no_of_demos) #, transform=transforms.Normalize(0., 10.)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) # Error: Couldn't open shared event, when workers is 4, hmmm but also 1
    loader = tqdm(enumerate(sample_data(loader, max_loops = 1000, one_batch=one_batch_debug)))
    print('loaded data {}'.format(time.process_time()))
    most_recent_time = time.process_time()
    
else:
    model = Ereason(channels_out = conv_channels_dim, use_wandb=use_wandb).to(device)
    model_copy = Ereason(channels_out = conv_channels_dim).to(device)

# with torch.autograd.profiler.profile(use_cuda = torch.cuda.is_available(), enabled=debug,) as prof: # record_shapes=True
    # with torch.backends.cudnn.flags(enabled=False):
most_recent_time = time.process_time()
train(model, loader)
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
