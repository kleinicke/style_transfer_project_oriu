# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

import copy

import sys
import argparse

import scipy.misc
import imageio

import os

import time
from flow_methods import warp_image, read_flow
"""
    example run this file with command
    python neural_style_tutorial.py --content test6 --style style3 --size 384 512
    this file applies the style transfer for the given parameter
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = [512 if torch.cuda.is_available() else 128]  # use small size if no gpu

parser = argparse.ArgumentParser(description='Style transfer Configuration')
parser.add_argument('--content', dest='content', default='dancing', type=str, help='content image name without .jpg')
parser.add_argument('--style', dest='style', default='Van_Gogh', type=str, help='style image name without .jpg')
parser.add_argument('--init', dest='init', action='store_true',default=False, help='uses init value')
parser.add_argument('--steps', dest='steps', default=300, type=int, help='number of steps')
parser.add_argument('--size', nargs='+', dest='size',  default=imsize, type=int, help='resulution of images hight width')
parser.add_argument('--video', dest='video', action='store_true', default=False, help='set if source is video')
parser.add_argument('--frame', dest='frame', default=1, type=int, help='frame number for video')
parser.add_argument('--flow', dest='flow', action='store_true', default=False, help='set if should init with flow')
parser.add_argument('--memory', dest='memory', action='store_true', default=False, help='set if should use short and long term memory')
parser.add_argument('--savesteps', dest='savesteps', action='store_true', default=False, help='set if should create pictures during training')

print(parser.parse_args())
parser: argparse.Namespace = parser.parse_args()

imsize = (parser.size)  # [0]
style_name = parser.style
content_name = parser.content
video = parser.video
init_image = parser.init
num_steps = parser.steps
num_frame = parser.frame
flow = parser.flow
memory = parser.memory
savesteps = parser.savesteps

print("Learns {} frame {:04d} for {} steps".format(content_name, num_frame, num_steps))

"""
The code below loads all the required images and warped frames.
"""

print("size: {}".format(imsize))
print(video)
if len(imsize) > 1:
    loader = transforms.Compose([
        transforms.Resize((imsize[0], imsize[1])),  # scale imported image to squared image
        transforms.ToTensor()])  # transform it into a torch tensor
    imsize_name = "{}_{}".format(imsize[0], imsize[1])
else:
    loader = transforms.Compose([
        transforms.Resize((imsize[0], imsize[0])),  # scale imported image to squared image
        transforms.ToTensor()])  # transform it into a torch tensor
    imsize_name = "{}".format(imsize[0])


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("images/{}.jpg".format(style_name))

if video:
    content_img = image_loader("videos/{}/frame_{:04d}.ppm".format(content_name, num_frame))
else:
    content_img = image_loader("images/{}.jpg".format(content_name))
if init_image:
    if flow:
        save_name = "videos/{}flow/initImage_{:04d}.png".format(content_name, num_frame)
    else:
        save_name = "output/video/{}/stylized-{}-{}-{}-v{:04d}.png".format(content_name, content_name, style_name, imsize_name, num_frame-1)
    input_img = image_loader(save_name)
else:
    #Doesn't init with random noise but with content image
    input_img = content_img.clone()

print(content_name,style_name)
print(style_img.size(),content_img.size())

if memory:
    save_name = "videos/{}flow/initImage_{:04d}.png".format(content_name, num_frame)
    temp_loss_image = image_loader(save_name)


assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"



unloader = transforms.ToPILImage()  # reconvert into PIL image


def temporal_loss(x, w, c):
    """
    x: image
    w: weights
    c: consistency mask
        per-pixel weighting of the loss element [0,1]^D.
        its 0 in disoccluded regions (as detected by forward-backward
        consistency) and at the motion boundaries, and 1 everywhere else
    Sum through all the single pixels of image D = W ×H ×C
    """
    D = (np.size(x))
    #print(x.shape, w.shape, c.shape, D)
    #print("c dimesions {} ones of {} fields".format(np.sum(c),np.size(c)))
    #D=100

    diff = x-w
    thesum = np.multiply(c, np.square(diff))
    #print(np.shape(thesum))
    #print(type(thesum))
    loss = np.sum(thesum)/D
    #print("The temporal loss is {}.".format(loss))
    return loss


def sum_shortterm_temporal_losses(num_frame, input_img, style_flow):
    """Computes 
    
    Arguments:
        num_frame {[type]} -- [description]
        input_img {[type]} -- [description]
        style_flow {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    #detach tensors so we can us numpy.
    input_img=input_img.detach().cpu().numpy()
    style_flow=style_flow.detach().cpu().numpy()

    prev_frame = num_frame - 1

    #loads the 2d c and transforms into the right 4d format as the other tensors
    c = np.load("videos/{}flow/consistency_{:04d}_{:04d}.npy".format(content_name, num_frame-1, num_frame))
    c = c[np.newaxis,:,:]
    c = np.repeat(c[:, :, :, np.newaxis], 3, axis=3)
    c = c.swapaxes(1,3)
    c = c.swapaxes(2,3)

    t_loss = temporal_loss(x=input_img, w=style_flow, c=c)
    return t_loss


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std



# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []
    temporal_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses



def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def saveimage(tensor,save_name="test"):
    """Transforms the pytorch tensor into an image

    Arguments:
        tensor {torch.float} -- the original loaded input_image, now the stylized image

    Keyword Arguments:
        save_name {str} -- name of the file  (default: {"test"})
    """

    if video:
        save_name="output/video/{}/{}-v{:04d}.png".format(content_name,save_name,num_frame)
    else:
        if os.path.exists('output/{}.png'.format(save_name)):
            i = 0
            while os.path.exists('output/{}-{:d}.png'.format(save_name, i)):
                i += 1
            save_name = 'output/{}-{:d}.png'.format(save_name, i)
        else:

            save_name = "output/{}.png".format(save_name)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)

    imageio.imwrite(save_name, image)
    print("saved at {}".format(save_name))

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000 , content_weight=1, temporal_weight=5000):
    """Run the style transfer."""
    print('Building the style transfer model..')
    time.sleep(2)
    init_time = time.time()
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    print("Init Time {:4f}s".format(time.time()-init_time))
    start_time = time.time()
    while run[0] <= num_steps:
        #step_time = time.time()

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            temporal_score = 0
            if memory:
                temporal_score = sum_shortterm_temporal_losses(num_frame ,input_img, temp_loss_image)
                temporal_score *= temporal_weight

            loss = style_score + content_score + temporal_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} Temporal Loss: {:4f} Optimization Time: {:4f}s'.format(
                    style_score.item(), content_score.item(),temporal_score,time.time()-start_time))
                print()
            return style_score + content_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


#run god function
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img,num_steps=num_steps)

##saves unique name
save_name="stylized-{}-{}-{}".format(content_name,style_name,imsize_name)

saveimage(output, save_name)