from torchvision.io import read_image
import torch
import numpy as np
from model.mlp import BernoulliMLP
from model.unet import UNet
from model.resnet import ResNet34Bernoulli
from PIL import Image
from matplotlib.image import imsave



img1 = read_image('/home/rst/dataset/celeba/img_align_celeba/000001.jpg')
# img1 = torch.ones((3, 218, 178)) * 255
# img1[:, 0 : 109, 0 : 89] = img1[:, 0 : 109, 0 : 89] * 0
#img1 = img1[:, 45 : 45+128, 25 : 25+128]

img2 = read_image('/home/rst/dataset/celeba/img_align_celeba/000012.jpg')
#img2 = torch.rand((3, 218, 178)) * 255
#img2 = read_image('/home/rst/workspace/vae/result/dog.jpg')
#img2 = img2[:, 100 : 100+128, 100 : 100+128]

#img3 = torch.rand((3, 128, 128))
#img3 = torch.rand((3, 218, 178)) * 255
#img3 = torch.ones((3, 218, 178))
img3 = torch.zeros((3, 218, 178))
img = torch.stack((img1/ 255., img2 / 255., img3))

model = BernoulliMLP('cpu')
#model = UNet('cpu')
#model = ResNet34Bernoulli('cpu')
state_dict = torch.load('/home/rst/workspace/vae/result/BernoulliMLP.pt')
#state_dict = torch.load('/home/rst/workspace/vae/result/BernoulliUNet.pt')
#state_dict = torch.load('/home/rst/workspace/vae/result/BernoulliResNet34.pt')
model.load_state_dict(state_dict)

img, _, _ = model(img)

# img1 = img[0].permute(1, 2, 0).detach().numpy()
# img2 = img[1].permute(1, 2, 0).detach().numpy()
# img3 = img[2].permute(1, 2, 0).detach().numpy()

img1 = img[0].view(3, 218, 178).permute(1, 2, 0).detach().numpy()
img2 = img[1].view(3, 218, 178).permute(1, 2, 0).detach().numpy()
img3 = img[2].view(3, 218, 178).permute(1, 2, 0).detach().numpy()

imsave('/home/rst/workspace/vae/result/img1.png', img1)
imsave('/home/rst/workspace/vae/result/img2.png', img2)
imsave('/home/rst/workspace/vae/result/img3.png', img3)