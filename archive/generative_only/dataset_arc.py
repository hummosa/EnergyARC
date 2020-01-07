from __future__ import print_function, division

#%%
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
data['test'][0]['input']
# %%
# 
plt.imshow(data['train'][0]['output'])

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


# %%
sizes_x ==sizes_y

# %%

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ARCDataset(Dataset):
    """ARC Torch dataset."""

    def __init__(self,  root_dir, mode='training', transform=None):
        """
        Args:
            root_dir (string): Directory with all the data.
            mode: 'training' to load training tasks
                  'evaluation' to load evalualtion tasks
               or 'both' to load all tasks available
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if mode == 'training':
            self.tasks = self.__load_split( self.root_dir + 'training')
        if mode == 'evaluation':
            self.tasks = self.__load_split( self.root_dir + 'evaluation')
        if mode == 'both':
            tasks_training   = self.__load_split(self.root_dir + 'training')
            tasks_evaluation = self.__load_split(self.root_dir + 'evaluation')
            self.tasks = tasks_evaluation + tasks_training

        self.transform = transform

    def __len__(self):
        return len(self.tasks)

    def __load_split(self, split_dir):
        tasks = []
        tasks_json = os.listdir(split_dir)
        for task_file in tasks_json:
            with open(os.path.join(split_dir,task_file)) as f:
                data = json.load(f)
                tasks.append(data)
        return tasks

        # demos = []
        # tests = []
        # for item in tasks:
        #     for tasks_item in item['train']:
        #         demos.append(tasks_item['input'])
        #         demos.append(tasks_item['output'])
        #     for tasks_item in item['test']:
        #         tests.append(tasks_item['input'])
        #         tests.append(tasks_item['output'])

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        sample = self.tasks[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

# %%
arc = ARCDataset('./ARC/data/', 'training')
aa = arc[3]

# %%
def show_task(task):
    plt.figure(figsize=[6,15],facecolor=(0, 0, 0))
    no_of_demos = len(task['train'])
    for i, demo in enumerate(task['train']):
        subplot = plt.subplot(no_of_demos+1,2, 2*i+1)
        plt.imshow(demo['input'])
        plt.title('demo ' +str(i)+ ' q', color='w')
        plt.axis('off')
        subplot = plt.subplot(no_of_demos+1,2, 2*i+2)
        plt.imshow(demo['output'])
        plt.title('demo ' +str(i)+ ' ans', color='w')
        plt.axis('off')
        
    subplot = plt.subplot(no_of_demos+1,2, 2*(i+1)+1)
    plt.imshow(task['test'][0]['input'])
    plt.axis('off')
    plt.title('test q', color='w')
    subplot = plt.subplot(no_of_demos+1,2, 2*(i+1)+2)
    plt.imshow(task['test'][0]['output'])
    plt.title('test ans', color='w')
    plt.axis('off')
        
show_task(arc[5])
# %%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        demos, test = sample['train'], sample['test']

        return {'train': torch.tensor(demos),
                'test': torch.tensor(test)}

# %%
class Pad(object):
    """Pads task arrays that are smaller than 3x3."""

    def __call__(self, sample):
        demos, test = sample['train'], sample['test']
        for d in demos:
            input = d['input']
            if len(input) < 3:
                other_dim = len(input[0])
                if isinstance(input, list):
                    d['input'].append( [0.] * other_dim)
            if len(input[0]) < 3:
                other_dim = len(input)
                if isinstance(input, list):
                    for input_row in d['input']:
                        input_row.append( [0.] )
            d['input'] = input

            if (len(input) < 3) or (len(input[0]) < 3): # if either dim is still under 3 pad again.
                self(sample)
            

        return {'train': demos,
                'test': test}
# %% Define simple model. 
# Conv layers. 2 of them
# then LSTM or ConvLSTM cc
# then deconv
