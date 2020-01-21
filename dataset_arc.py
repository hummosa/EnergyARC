#%% Dataset
from __future__ import print_function, division

import numpy as np
import json
import matplotlib.pyplot as plt

training_data_path = './ARC/data/training'
import os

train = []
training_json = os.listdir(training_data_path)
for tj in training_json:
    with open(os.path.join(training_data_path,tj)) as f:
        data = json.load(f)
        train.append(data)
# %%
# data['test'][0]['input']
# %%
# 
# plt.imshow(data['train'][0]['output'])

# %%
items = []
for item in train:
    for train_item in item['train']:
        items.append(train_item['input'])
        items.append(train_item['output'])
    for train_item in item['test']:
        items.append(train_item['input'])
        items.append(train_item['output'])

items = np.array(items)
#%
sizes_x = [(len(xx)) for xx in items]
sizes_y = [(len(xx[0])) for xx in items]
#%


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ARCDataset(Dataset):
    """ARC Torch dataset."""

    def __init__(self,  root_dir, mode='training', transform=None, no_of_demos=12):
        """
        Args:
            root_dir (string): Directory with all the data.
            mode: 'training' to load training tasks
                  'evaluation' to load evalualtion tasks
               or 'both' to load all tasks available
            transform (callable, optional): Optional transform to be applied
                on a sample.
            no_of_demos: for efficiency task demo pairs will be made into batches. 
                This limit on the max no of demos per task improves efficiency and number of computations.
                but exludes tasks with high number of demos. 
        """
        self.root_dir = root_dir
        if mode == 'training':
            self.tasks = self.__load_split( self.root_dir + 'training')
        if mode == 'evaluation':
            self.tasks = self.__load_split( self.root_dir + 'evaluation')
        if mode == 'both':
            tasks_training   = self.__load_split(self.root_dir + 'training')
            tasks_evaluation = self.__load_split(self.root_dir + 'evaluation')
            self.tasks = (tasks_evaluation[0] + tasks_training[0], tasks_evaluation[1] + tasks_training[1])

        # Limit to tasks with a number of demos below cut off given by no_of_demos
        # convert tasks from a tuple of lists to a list of tuples
        # tasks = []
        # for i in range(len(self.tasks[0])):
        #     tasks.append((self.tasks[0][i],self.tasks[1][i]))
        # self.tasks = tasks

        # self.tasks = [task for task in self.tasks if len(task) <= no_of_demos]

        self.transform = transform
        

        ## Preprocess
        # for i, task in enumerate(self.tasks):
        #     self.tasks[i] = self.transform(task)

        self.inputs = self.tasks[0]
        self.outputs = self.tasks[1]

        for i, task in enumerate(self.inputs):
             self.inputs[i] = self.transform(task)
        for i, task in enumerate(self.outputs):
             self.outputs[i] = self.transform(task)

        self.inputs = [torch.stack(demos) for demos in self.inputs]
        self.inputs = torch.stack(self.inputs)
        self.outputs = [torch.stack(demos) for demos in self.outputs]
        self.outputs = torch.stack(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        sample = (self.inputs[idx], self.outputs[idx])
        return sample

    def __load_split(self, split_dir):
        tasks = []
        tasks_json = os.listdir(split_dir)
        for task_file in tasks_json:
            with open(os.path.join(split_dir,task_file)) as f:
                data = json.load(f)
                tasks.append(data)
        return self.__break_split(tasks)

    def __break_split(self, tasks):
        
        inputs = []
        outputs = []
        for item in tasks:
            demo_inputs = []
            demo_outputs = []
            for tasks_item in item['train']:

                demo_inputs.append( [ti for ti in tasks_item['input']] )
                demo_outputs.append( [ti for ti in tasks_item['output']] )

            test_inputs=[]
            test_outputs=[]
            for tasks_item in item['test']:
                test_inputs.append([ti for ti in tasks_item['input']] )
                test_outputs.append([ti for ti in tasks_item['output']])
            
            #demo_inputs are [input_no, h, w]
            inputs.append (demo_inputs + test_inputs )
            #inputs is [task_no, input_no, h, w]
            outputs.append(demo_outputs+ test_outputs)
            
        return (inputs,outputs)


# %%
class Preprocess(object):
    """ Converts task inputs and outputs to tensors, but leaves the outer list structure"""

    def __call__(self, sample):
        inputs = sample

        for i, inp in enumerate(inputs):
            inputs[i] = torch.tensor(inp, dtype=torch.float32) ## convert to Tensor
            # inputs[i] = inputs[i]/10.  # normalize
            inputs[i] = inputs[i].unsqueeze(0) #  channels dim

        return inputs

class Padded_with_mask(object):
    """ Pads all inputs and outputs to a 30x30 but adds another channel with a mask"""

    def __call__(self, sample, pad=30):

        inputs = sample
        for i, inp in enumerate(inputs):
            canvas = torch.ones([ 1, pad, pad]) * 10. 
            # pad_mask = torch.zeros([ 1, pad, pad])
            canvas[ 0, :inp.shape[1], :inp.shape[2]] = inp 
            # pad_mask[ 0, :inp.shape[1], :inp.shape[2]] = torch.ones_like(inp)
            # canvas = map(convert_to_rgb, canvas.view([-1])).view([3, pad,pad])
            canvas = list(map(convert_to_rgb, torch.unbind(canvas.view([-1]) )  ))
            canvas = torch.stack(canvas)
            inputs[i] = canvas.view([pad, pad, 3]).permute([2,0,1]) 

        return inputs

class Padded_no_of_demos(object):
    """ Pads tasks with demo pairs duplicates"""
    def __init__(self, no_of_demos):
        self.no_of_demos = no_of_demos
        super(self.__class__, self).__init__()

    def __call__(self, sample):
        inputs = sample
        no_of_demos = self.no_of_demos
        li = len(inputs)
        
        #padd up to no_of_demos
        if li < no_of_demos:
            for it in range(no_of_demos-li):
                inputs.append(inputs[it])
                # inputs.append(torch.zeros(size=[2, 30, 30]))
        
        # limit anything above no_of_demos
        inputs = inputs[:no_of_demos]

        # Transforms from tuplie of lists to list of tuples. Each input output demo example is a tuple. 
        return inputs
        # return ([(i, o) for i, o in zip(inputs, outputs)])


# preprocess = Preprocess()
# pad_with_mask= Padded_with_mask(30)
all_transforms = transforms.Compose([Preprocess(), Padded_with_mask(), Padded_no_of_demos(6)])


# arc = ARCDataset('./ARC/data/', 'both', transform=all_transforms)
# aa = arc[3]



def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)
            yield next(loader_iter)

#%%

# Load as 9 channels
# layernorm??
# drop out?
# 
colors = {  0: [0., 0., 0.],  #black
            1: [0., 0., 1.],  #blue
            2: [1., 0., 0.],  #red
            3: [0., 1., 0.],  #green
            4: [1., 1., 0.],  #yellow
            5: [.7, .6, .5], #?grey
            6: [1., 0., 1.],  #magenta
            7: [.5, .5, .1],  #?orange?
            8: [0., 1., 1.],  #cyan
            9: [.5, .1, .1],  #?maroon
            10:[1., 1., 1.], #mask
            }
def convert_to_rgb(c):
    return torch.tensor(colors[c.item()]) #add item() to get the data from a zero dim tensor.

import time
print('started dataset: {}'.format(time.clock()))
# dataset = ARCDataset('./ARC/data/', 'training', transform= all_transforms,no_of_demos=6) #, transform=transforms.Normalize(0., 10.)
# loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0) # Error: Couldn't open shared event, when workers is 4, hmmm but also 1
# from tqdm import tqdm
# loader = tqdm(enumerate(sample_data(loader)))

# loader_iter = iter(loader)
# # #%%
# enum, n = next(loader_iter)
# n[0].shape # n[0] is inputs and n[1] is outputs

# torch.Size([128, 6, 2, 30, 30])
#%%
def show_task(inputs, outputs, channels_first=True):
    fig = plt.figure(figsize=[8,17],facecolor=(0, 0, 0))
    no_of_demos = (inputs.shape[0])
    for i, (inp, out) in enumerate( zip(inputs, outputs)):
        subplot = plt.subplot(no_of_demos+1,2, 2*i+1)
        channels = np.argmin(inp.shape)
        if channels_first:
            inp = inp.permute([ 1, 2, 0])
            out = out.permute([ 1, 2, 0])
        plt.imshow(inp)
        plt.title('demo ' +str(i)+ ' q', color='w')
        plt.axis('off')
        subplot = plt.subplot(no_of_demos+1,2, 2*i+2)
        plt.imshow(out)
        plt.title('demo ' +str(i)+ ' ans', color='w')
        plt.axis('off')
    return fig

#%%
# show_task(n[0][2], n[1][2])   

# %%
