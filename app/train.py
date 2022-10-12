# 내장
import os

# 프로젝트
from config import CONFIG
from model import Network
import loader

# 서드파티
import torch
import torch.nn as nn
import torch.optim as optim


def train(train_data_loader):
    network = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(CONFIG.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            inputs, labels = data # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            optimizer.zero_grad() # 변화도(Gradient) 매개변수를 0으로 만들고
            outputs = network(inputs) # 순전파 + 역전파 + 최적화를 한 후
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    
    torch.save(
        network.state_dict(),
        os.path.join(CONFIG.model_save_dir, CONFIG.model_name))
    print('Model saved')
    
    
if __name__ == '__main__':
    train(loader.CIFAR10.train_data_loader)