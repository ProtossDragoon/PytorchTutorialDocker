# 프로젝트
from train import train
from eval import eval
import loader


def main():
    train(loader.CIFAR10.train_data_loader)
    eval(loader.CIFAR10.test_data_loader)


if __name__ == '__main__':
    main()