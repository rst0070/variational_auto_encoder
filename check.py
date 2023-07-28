from torchvision.io import read_image
import torch
import numpy as np
from model.mlp import BernoulliMLP
from PIL import Image
from matplotlib.image import imsave



img1 = read_image('/home/rst/dataset/celeba/img_align_celeba/000005.jpg')
img2 = read_image('/home/rst/dataset/celeba/img_align_celeba/000012.jpg')
img = torch.stack((img1/ 255., img2 / 255.))

model = BernoulliMLP('cpu')
state_dict = torch.load('/home/rst/workspace/vae/result/BernoulliMLP.pt')
model.load_state_dict(state_dict)

img, _, _ = model(img)

img1 = img[0].view(3, 218, 178).permute(1, 2, 0).detach().numpy()
img2 = img[1].view(3, 218, 178).permute(1, 2, 0).detach().numpy()

imsave('/home/rst/workspace/vae/result/img1.png', img1)
imsave('/home/rst/workspace/vae/result/img2.png', img2)