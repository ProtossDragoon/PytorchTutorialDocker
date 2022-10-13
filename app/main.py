# 프로젝트
from config import CONFIG
from train import train
from eval import eval
import loader


def main():
    train(loader.ClassificatonCOCODataLoader(CONFIG.coco_train_path).data_loader)
    eval(loader.ClassificatonCOCODataLoader(CONFIG.coco_test_path).data_loader)


if __name__ == '__main__':
    main()