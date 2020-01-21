
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import datasets, transforms, utils

from tqdm import tqdm

import dataset_arc
from utils import *

wandb_use = False
if wandb_use: import wandb
if wandb_use: wandb.init(project="earc")

from model_arc import Ereason


def to_tensor_transforms(item_input):
    return (
        (0.1) * torch.tensor(item_input, dtype=torch.float32, device=device).unsqueeze_(0).unsqueeze_(1) 
    )

def sample_data(loader, one_batch= False, loop= True):
    loader_iter = iter(loader)
    memorize_one_batch = next(loader_iter)
    while one_batch: #if one_batch is True, keep yielding the exact same batch, for model checking.
        yield (memorize_one_batch)
        
    while loop:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)
            yield next(loader_iter)


def im_log(x):
    return(  plt.imshow(x[0].detach().permute([1,2,0]).to('cpu').numpy()) )

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


import time
no_of_demos = 5
batch_size = 32
conv_channels_dim = 128
w_dim = 128
debug=True
torch.autograd.set_detect_anomaly(debug) # this slows down computations but allows better traces for backward functions. 
                                         # it also checks for NaN during the backward computation.
#%%

def train(model, loader, alpha=0.1, step_size=10, sample_step=7, device=device):
    
    # buffer = SampleBuffer()
    # noise = torch.randn(128, 3, 32, 32, device=device)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0.0, 0.999))

    # pixel_reward_demos = [0. for _ in range(no_of_demos)]
    # pixel_reward_tasks = [0. for _ in range(no of tasks)]

    for t, batch in loader: # [i/o, demos, batch,   c, w, h]
        # for each batch update the copy of the model, used for backward prop through the gradient
        for from_param, to_param in zip(model.parameters(), model_copy.parameters()):
            to_param.data.copy_(from_param)
        requires_grad(model_copy.parameters(), False)
        print('model copy requires grad? :', np.array([p.requires_grad for p in model_copy.parameters()]).any()
)
        #this is how it is done in dspn code.
        w = nn.Parameter(torch.randn([len(batch[0]), 3, w_dim], device=device))
        
        # w = torch.randn([len(batch[0]), 3, w_dim], device=device)
        w.requires_grad = True

        model.train()

        batch[0] = batch[0].to(device) # send input x0 to cuda
        batch[1] = batch[1].to(device) # send outputs x1 to cuda
        print('batch loaded into cuda after : ', time.clock())        

        '''############# Getting W across all demos ################'''
        for k in tqdm(range(sample_step)):

            for i in range(no_of_demos):
                pos_inp = (batch[0][:,i], batch[1][:,i])
            
                noise = torch.randn_like(w, device=device)
                noise.normal_(0, 0.005)
                w= w + (noise.data)

                w_out = model(pos_inp, w)
                w_grad = torch.autograd.grad(w_out.sum(), w,
                create_graph=True) #Create_graph ensures that differentiation is added to the gradient graph for later derivatives calculation

                w = w + (-step_size * w_grad[0]) # grad returns a tuple of len 1
                #note: the differentiation path to w remains in the graph, and will take grad across it. So that finding w, can be informed by 
                # later operations and how successfull it turned out to be. Another option would be to coevolve w and x1_ taking turns.
                # TODO this loop by itself is a resource hog, retaining the graph of the model and its gradient for 100 (no__demos*sample_step) times
        '''############# Solving for X1_ using W ####################'''
        if wandb_use: wandb.log( {'w': wandb.Histogram( w.detach().to('cpu').numpy() )}, commit=False)
        grad_viz_container = []; grad_norms = []
        total_loss = 0.

        for i in tqdm(range(no_of_demos)): # the order here flips (for task for k, because each x1_ is propagated serparately 
                                                #for each input-output in training set)
            pos_inp = (batch[0][:,i], batch[1][:,i])
            x0 = pos_inp[0]
            x1 = pos_inp[1]
            
            #init predicted answer x1_ to x0, alternatively start from noise.  torch.randn([batch_size, 2, 30, 30])
            # TODO three alternatives:
            # x1_ = x0.detach().clone()
            # x1_ = nn.Parameters(torch.rand_like(x0.detach()))
            x1_ = torch.rand_like(x0.detach())
            x1_.requires_grad = True
            
            demo_loss = 0.
            for k in range(sample_step):

                noise = torch.randn_like(x1, device=device)
                noise.normal_(0, 0.005)
                x1_ = x1_ + (noise.data)

                x1_out = model((x0, x1_), w) #igor does not detach w here.
                x1_grad = torch.autograd.grad(
                    outputs= x1_out.sum(), 
                    inputs = x1_,
                    only_inputs=True,
                    create_graph=True,)
                    # retain_graph=True )

                if debug:
                    grad_norms.append(torch.norm(x1_grad[0]))
                    print('grad_norm for step {}: {}'.format(k, grad_norms[k]))
                    grad_viz_container.append(x1_grad[0])

                x1_= x1_ + (-step_size* x1_grad[0])  # TODO Step size according to RL reward!!! That might be it!  save the pixel rewards and nudge each backprop step based on how it changes pixel reward?
                # accumlate loss every step
                demo_loss_local = torch.norm(x1_- x1)/ (torch.norm(x1) + torch.norm(x1_))
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
            kl_loss = model_copy([x0, x1_], w.detach()) # Grad Not: model, w   Grad x1_ and the Langevin dynamics. 

            # requires_grad(parameters, True)
            pos_out = model([x0, x1], w)
            #force gradient to go through w, so that params os model change to give this particular one more energy.
            neg_out = model([x0, x1_.detach()], w)
            
            # norm_loss = alpha * (pos_out ** 2 + neg_out ** 2)
            loss = F.softplus(  (pos_out - neg_out)  ) + F.softplus(kl_loss ) # + norm_loss
            loss = loss.mean()
            total_loss += loss

        #? should I run opt.step every demo instead, to free up memory. Or do I need to loop up to w?
        #? I  prob should. It changes the parameters of the model, but that should not affecting
        # ? learning subsequent demos I do not think
        # note: I'm copying the model every batch, not every demo
        # so the copy is outdated now after processing the first demo
        #! for all demos 
        #     optimizer.zero_grad() # TODO although this needs testing: # can safely zero .grad because autograd.grad fn does not accumlate in them
        #     loss.backward()
        #     optimizer.step()
        
        #     total_loss += loss.detach_() #though I think pytorch already does that.
        # x1_.detach_()
        # w.detach_()
        with torch.autograd.profiler.record_function("Outer optim"): # label the block    
            optimizer.zero_grad()  
            total_loss.backward()
            # clip_grad(parameters, optimizer)
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
                plt.close()
            except:
                print('the shape of the image: {} \n info: {}'.format(w.shape, w))

    

if __name__ == '__main__':
    model = Ereason(channels_out = conv_channels_dim,wandb_use=wandb_use).to(device)
    model_copy = Ereason(channels_out = conv_channels_dim).to(device)
    if wandb_use: wandb.watch(model)
    
    dataset = dataset_arc.ARCDataset('./ARC/data/', 'both', transform= dataset_arc.all_transforms, no_of_demos=no_of_demos) #, transform=transforms.Normalize(0., 10.)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) # Error: Couldn't open shared event, when workers is 4, hmmm but also 1
    loader = tqdm(enumerate(sample_data(loader)))
    print('loaded data {}'.format(time.clock()))
    
else:
    model = Ereason(channels_out = conv_channels_dim, wandb_use=wandb_use).to(device)
    model_copy = Ereason(channels_out = conv_channels_dim).to(device)

with torch.autograd.profiler.profile(use_cuda = True, enabled=debug, record_shapes=True) as prof:
    with torch.backends.cudnn.flags(enabled=False):
        train(model, loader)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))