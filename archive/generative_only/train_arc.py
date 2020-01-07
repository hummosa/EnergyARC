import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm

from model_arc import IGEBM

import dataset_arc

device = 'cuda'

class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples, class_ids=None):
        samples = samples.detach().to('cpu')
        class_ids = class_ids.detach().to('cpu')

        for sample, class_id in zip(samples, class_ids):
            self.buffer.append((sample.detach(), class_id))

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples, device='cuda'):
        items = random.choices(self.buffer, k=n_samples)
        samples, class_ids = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.tensor(class_ids)
        samples = samples.to(device)
        class_ids = class_ids.to(device)

        return samples, class_ids


def sample_buffer(buffer, batch_size=128, p=0.95, device=device):
    if len(buffer) < 1:
        return (
            torch.rand(batch_size, 3, 32, 32, device=device),
            torch.randint(0, 10, (batch_size,), device=device),
        )

    n_replay = (np.random.rand(batch_size) < p).sum()

    replay_sample, replay_id = buffer.get(n_replay)
    random_sample = torch.rand(batch_size - n_replay, 3, 32, 32, device=device)
    random_id = torch.randint(0, 10, (batch_size - n_replay,), device=device)

    return (
        torch.cat([replay_sample, random_sample], 0),
        torch.cat([replay_id, random_id], 0),
    )

def to_tensor_transforms(item_input):
    return (
        (0.1) * torch.tensor(item_input, dtype=torch.float32, device=device).unsqueeze_(0).unsqueeze_(1) 
    )

def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            item = next(loader_iter)
            item_train = item['train']
            for it in item_train:
                item_train_input = it['input']
                yield(to_tensor_transforms(item_train_input))
                
                item_train_output = it['output']
                yield(to_tensor_transforms(item_train_output))

            item_test_input = item['test'][0]['input']
            yield(to_tensor_transforms(item_test_input))
            
            item_test_output = item['test'][0]['output']
            yield(to_tensor_transforms(item_test_output))
            
            
        except StopIteration:
            loader_iter = iter(loader)

            item = next(loader_iter)
            item_train = item['train']
            for it in item_train:
                item_train_input = it['input']
                yield(to_tensor_transforms(item_train_input))
                
                item_train_output = it['output']
                yield(to_tensor_transforms(item_train_output))

            item_test_input = item['test'][0]['input']
            yield(to_tensor_transforms(item_test_input))
            
            item_test_output = item['test'][0]['output']
            yield(to_tensor_transforms(item_test_output))
            
            # yield next(loader_iter)


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
    # dataset = datasets.CIFAR10('.', download=True, transform=transforms.ToTensor())
    # loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    # loader = tqdm(enumerate(sample_data(loader)))
    dataset = dataset_arc.ARCDataset('./ARC/data/', 'training')#, transform=transforms.Normalize(0., 10.)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1) # Error: Couldn't open shared event, when workers is 4
    loader = tqdm(enumerate(sample_data(loader)))

    # buffer = SampleBuffer()

    # noise = torch.randn(128, 3, 32, 32, device=device)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0.0, 0.999))

    for i, pos_img in loader:
        pos_img = pos_img.to(device)
        # neg_img = sample_buffer(buffer, pos_img.shape[0])
        neg_img = torch.randn_like(pos_img)
        neg_img.requires_grad = True

        requires_grad(parameters, False)
        # model.eval() #because lstm layer insists to  be in training mode to allow backprop through it. 
        model.train()

        for k in tqdm(range(sample_step)):
            # if noise.shape[0] != neg_img.shape[0]:
            #     noise = torch.randn(neg_img.shape[0], 3, 32, 32, device=device)

            # noise.normal_(0, 0.005)
            # neg_img.data.add_(noise.data)

            neg_out = model(neg_img)
            neg_out.sum().backward()
            neg_img.grad.data.clamp_(-0.01, 0.01)

            neg_img.data.add_(-step_size, neg_img.grad.data)

            neg_img.grad.detach_()
            neg_img.grad.zero_()

            neg_img.data.clamp_(0, 1)

        neg_img = neg_img.detach()

        requires_grad(parameters, True)
        model.train()

        model.zero_grad()

        pos_out = model(pos_img)
        neg_out = model(neg_img)

        loss = alpha * (pos_out ** 2 + neg_out ** 2)
        loss = loss + (pos_out - neg_out)
        loss = loss.mean()
        loss.backward()

        clip_grad(parameters, optimizer)

        optimizer.step()

        # buffer.push(neg_img, neg_id)
        loader.set_description(f'loss: {loss.item():.5f}')

        if i % 1000 == 0:
            try:
                # utils.save_image(  #this is nice because it can display a whole batch of photos
                #     neg_img.detach().to('cpu'),
                #     f'samples/{str(i).zfill(5)}.png',
                #     nrow=16,
                #     normalize=True,
                #     range=(0, 1),
                # )
                plt.close()
                plt.imshow(neg_img.detach().to('cpu').squeeze().numpy())
                plt.savefig(f'samples/{str(i+1).zfill(5)}.png')

            except:
                print('the shape of the image: {} \n info: {}'.format(neg_img.shape, neg_img))

if __name__ == '__main__':
    model = IGEBM(10).to(device)
    train(model)
