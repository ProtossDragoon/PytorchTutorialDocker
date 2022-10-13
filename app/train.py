# 내장
import os

# 프로젝트
from config import CONFIG
from model import ClassificationNetwork
import loader

# 서드파티
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


def train(train_data_loader):
    network = ClassificationNetwork(train_data_loader.dataset.n_class)
    ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(CONFIG.epochs):
        for i, data in enumerate(pbar:=tqdm.tqdm(train_data_loader)):
            inputs, labels = data # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            optimizer.zero_grad() # 변화도(Gradient) 매개변수를 0으로 만들고
            outputs = network(inputs) # 순전파 + 역전파 + 최적화를 한 후
            loss = ce(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f'Epoch: {epoch + 1}, '
                f'Loss: {loss.item() / (i + 1):.5f}')
    print('Finished Training')
    
    torch.save(
        network.state_dict(),
        os.path.join(CONFIG.model_save_dir, CONFIG.model_name))
    print('Model saved')
    
    
if __name__ == '__main__':
    train(loader.ClassificatonCOCODataLoader(CONFIG.coco_train_path).data_loader)