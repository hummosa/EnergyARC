import numpy as np
import torch
import matplotlib.pyplot as plt

def dlist(li):
    """keeps going into a nested list printing len of each dim"""
    c = li
    keep_going = True
    while keep_going:
        try:
            print(len(c))
            c = c[0]
        except:
            keep_going=False

def stats(var, var_name=None):
  if type(var) == type([]):
    var = np.array(var)
  elif type(var) == type(np.array([])):
    pass #if already a numpy array, just keep going.
  else: #assume torch tensor
    var = var.detach().cpu().numpy()
  if var_name:
    print(var_name, ':')   
  print('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}, norm {:1.3e}'.format(var.mean(), var.var(), var.min(), var.max(), torch.norm(torch.from_numpy(var))))



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




### trace allocate memory

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))



# General notes:
            
            #######################################################################
            #? Does this need to happen every k or every t??????????????????????
            # Every input pair i.  
            # OPTION 1: calculate loss and prop for every demo pair.
            # Option 2: accumulate losses and keep the graph, then optimize once at the end.
            #? consider making latent w for the task, and another one for the specific demo example
            #######################################################################
        #? should I run opt.step every demo instead, to free up memory. Or do I need to loop up to w?
        #? I  prob should. It changes the parameters of the model, but that should not affecting
        # ? learning subsequent demos I do not think
        # note: I'm copying the model every batch, not every demo
        # so the copy is outdated now after processing the first demo
