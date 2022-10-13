# 내장
import os

# 프로젝트
from config import CONFIG
from model import ClassificationNetwork

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
    network = ClassificationNetwork(test_data_loader.dataset.n_class)
    network.load_state_dict(torch.load(os.path.join(CONFIG.model_save_dir, CONFIG.model_name)))
    images, labels = next(iter(test_data_loader))
    outputs = network(images)
    _, gts = torch.max(labels, 1)
    _, predicted = torch.max(outputs, 1)
    for j in range(CONFIG.batch_size):
        print(f'\nGroundTruth:\t {test_data_loader.dataset.categories[gts[j]]["name"]}')
        print(f'Predicted:\t {test_data_loader.dataset.categories[predicted[j]]["name"]}')
    # imshow(torchvision.utils.make_grid(images))
    
    
if __name__ == '__main__':
    eval(loader.ClassificatonCOCODataLoader(CONFIG.coco_test_path).data_loader)