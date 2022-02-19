import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import pickle
import shutil
import os
import glob

def do_test(net, testloader, device="cuda", mode="eval"):
    if mode=="train":
        net.train()
    elif mode=="eval":
        net.eval()
    else:
        return None
    labels = np.array([])
    with torch.no_grad():
        for itr, data in enumerate(testloader):
            images = data[0].to(device)
            output = cuda2numpy(net(images))
            label = cuda2numpy(data[1])
            if len(label.shape)<2:
                label = label.reshape(-1, 1)
            if itr==0:
                outputs = output
                labels = label
            else:
                outputs = np.concatenate([outputs, output], axis=0)
                labels = np.concatenate([labels, label], axis=0)
    return labels, outputs

def do_test_VAE(net, testloader, device="cuda", mode="eval"):
    if mode=="train":
        net.train()
    elif mode=="eval":
        net.eval()
    else:
        return None
    losses = 0
    with torch.no_grad():
        for itr, data in enumerate(testloader):
            images = data[0].to(device)
            output, latent, loss = net(images, True)
            input = cuda2numpy(images)
            output = cuda2numpy(output)
            latent = cuda2numpy(latent)
            losses += cuda2numpy(loss)*output.shape[0]
            label = cuda2numpy(data[1])
            if len(label.shape)<2:
                label = label.reshape(-1, 1)
            if itr==0:
                inputs = input
                latents = latent
                outputs = output
                labels = label
            else:
                inputs = np.concatenate([inputs, input], axis=0)
                latents = np.concatenate([latents, latent], axis=0)
                outputs = np.concatenate([outputs, output], axis=0)
                labels = np.concatenate([labels, label], axis=0)
    return labels, inputs, outputs, losses/inputs.shape[0], latents

class ToPIL(object):
    """
    convert torch.Tensor into PIL image

    ------------
    Parameters
    ------------

    img : torch.Tensor, shape=(sample_num, channel, length, width)
        either cuda or cpu tensor
        
    ------------
    Returns
    ------------

    image_list : PIL image
        PIL image of the input tensor

    ------------

    """
    def __init__(self):
        pass

    def __call__(self, img):
        return image_from_output(torch.reshape(img, (1,img.shape[0],img.shape[1],img.shape[2])))[0]

    def __repr__(self):
        return self.__class__.__name__
    
class MinMax(object):
    def __init__(self, mean0=True):
        self.mean0 = mean0
        pass
    def __call__(self, img):
        return torch.Tensor(min_max(cuda2numpy(img), mean0=self.mean0))
    def __repr__(self):
        return self.__class__.__name__

def cuda2numpy(x):
    """
    convert cuda(cpu) tensor into ndarray

    ------------
    Parameters
    ------------

    x : torch.Tensor
        either cuda or cpu tensor

    ------------
    Returns
    ------------

    y : ndarray
        numpy version of the input tensor

    ------------

    """
    return x.detach().to("cpu").numpy()

def cuda2cpu(x):
    """
    convert cuda tensor into cpu tensor

    ------------
    Parameters
    ------------

    x : torch.Tensor
        cuda tensor

    ------------
    Returns
    ------------

    y : torch.Tensor
        cpu tensor

    ------------

    """
    return x.detach().to("cpu")

def image_from_output(output):
    """
    convert torch.Tensor into PIL image

    ------------
    Parameters
    ------------

    output : torch.Tensor, shape=(sample_num, channel, length, width)
        either cuda or cpu tensor
        
    ------------
    Returns
    ------------

    image_list : list
        list includes PIL images

    ------------

    """
    if len(output.shape)==3:
        output = output.unsqueeze(0)
        
    image_list = []
    output = cuda2numpy(output)
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list
