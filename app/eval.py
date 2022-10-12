# 내장
import os

# 프로젝트
from config import CONFIG
from model import Network

# 서드파티
import loader
import torch
import matplotlib.pyplot as plt
import numpy as np


CH_FIRST_TO_LAST = (1, 2, 0)


def unnormalize(img):
    return img / 2 + 0.5
    

def imshow(img):
    img = unnormalize(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg))
    plt.show()


def eval(test_data_loader):
    network = Network()
    network.load_state_dict(torch.load(os.path.join(CONFIG.model_save_dir, CONFIG.model_name)))
    images, labels = next(iter(test_data_loader))
    outputs = network(images)
    _, predicted = torch.max(outputs, 1)
    print('GT:\t', ' '.join(f'{CONFIG.classes[labels[j]]:5s}' for j in range(CONFIG.batch_size)))
    print('PRED:\t', ' '.join(f'{CONFIG.classes[predicted[j]]:5s}' for j in range(CONFIG.batch_size)))
    # imshow(torchvision.utils.make_grid(images))
    
    
if __name__ == '__main__':
    eval(loader.CIFAR10.test_data_loader)