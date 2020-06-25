import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import sys
sys.path.append("home/bix/Christoph/owncloud/transfer_learning")
import torch
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from misc_functions import get_domain_adaptation_images,load_adapted_resnet_features,predict_test_images,preprocess_example_image


def plot_eigenspectrum(x):
    values = np.linalg.svd(x,compute_uv=False)
    plt.bar(range(len(values)), values, align='center')
    plt.ylabel("Eigenvalue")
    # plt.tight_layout()
    plt.xlabel("Index")
    plt.xticks([0, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.savefig("nsdqs_spectra.pdf", transparent=True)
    plt.show()

def plot_spetrum(data):
    plt.figure()
    x = data.detach().cpu().numpy()
    values = np.linalg.svd(x.T @ x ,compute_uv=False)
    values = (values - values.min()) / (values.max()-values.min())
    plt.bar(range(len(values)), values, align='center')
    plt.ylabel("Eigenvalue")
    # plt.tight_layout()
    plt.xlabel("Index")
    # plt.xticks([0, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.savefig("plots/cdan_feature_spectrum.png", transparent=True)
    # plt.show(block=False)

def plot_eigs(data,y,i):
    plt.figure()
    # x = data.detach().cpu().numpy()
    # l,values,r = np.linalg.svd(x,compute_uv=True)
    # d =  l[:,[0,10]] @ np.diag(values[[0,10]])
    # plt.scatter(d[:, 0], d[:, 1], s=10, marker="s", c=y.flatten(),cmap=plt.get_cmap("tab20"))
    # plt.savefig("plots/scatter_"+str(i)+".pdf", transparent=True)
    # # plt.show(block=False)
    # # plt.show(block=False)

    plt.figure()
    values = np.linalg.svd(x.T @ x ,compute_uv=False)
    plt.bar(range(len(values)), values, align='center')
    plt.ylabel("Eigenvalue")
    # plt.tight_layout()
    plt.xlabel("Index")
    # plt.xticks([0, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.savefig("plots/spectra"+str(i)+".pdf", transparent=True)
    # plt.show(block=False)

    plt.figure()
    s = data[:len(y[y==1]),:].detach().cpu().numpy()
    t = data[len(y[y==1]):,:].detach().cpu().numpy()
    sv = np.linalg.svd(s.T @ s ,compute_uv=False)
    tv = np.linalg.svd(t.T @ t,compute_uv=False)
    diff = np.square( (sv - tv)**2 )
    plt.bar(range(len(diff)), diff, align='center')
    plt.ylabel("Difference")
    # plt.tight_layout()
    plt.xlabel("Index")
    # plt.xticks([0, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.savefig("plots/diff"+str(i)+".pdf", transparent=True)
    # plt.show(block=False)


example_list = get_domain_adaptation_images()
pretrained_model = load_adapted_resnet_features("snapshot/san/CDAN_on_amazon_vs_webcam_True.pth.tar")
dic = ""
pretrained_model.eval()

data = torch.empty([1,3,224,224])
for index in example_list:
    (original_image, prep_img, target_class, file_name_to_export) =\
        preprocess_example_image(index)
    data = torch.cat((data,prep_img),0)
data = data[1:]
data = data.cuda()
predictions = pretrained_model._modules['0'].feature_layers(data)
predictions = predictions.reshape(6,2048)
x = predictions.T @ predictions

x = x.detach().cpu().numpy()
values,d= np.linalg.eig(x)
# values = (values - values.min()) / (values.max()-values.min())

bars = values[:10]
plt.figure()
plt.bar(range(len(bars)), bars, align='center')
plt.ylabel("Eigenvalue")
# plt.tight_layout()
plt.xlabel("Index")
# plt.xticks([0, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
plt.savefig("plots/cdan_feature_spectrum.png", transparent=True)
