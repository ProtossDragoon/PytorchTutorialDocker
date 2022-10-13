# Pytorch Image Classification Tutorial with Docker

[pytorch classification tutorial](https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) 을 기반으로 작성한 간단한 소스 코드를 docker 환경에서 실행합니다.

## Command

```
git clone https://github.com/ProtossDragoon/PytorchTutorialDocker.git
cd PytorchTutorialDocker
docker-compose up
```

- 컨테이너 환경에서 간단한 파이토치 분류모델이 Kaggle 의 [children-vs-adult](https://www.kaggle.com/datasets/die9origephit/children-vs-adults-images) 데이터셋을 바탕으로 학습됩니다.
- 모델 가중치는 `PytorchTutorialDocker/result` 디렉터리에 `network.pt` 파일로 저장됩니다.

## Log

```
pytorchtutorialdocker-train-and-eval-1 | loading annotations into memory...
pytorchtutorialdocker-train-and-eval-1 | Done (t=0.00s)
pytorchtutorialdocker-train-and-eval-1 | creating index...
pytorchtutorialdocker-train-and-eval-1 | index created!
pytorchtutorialdocker-train-and-eval-1 | /usr/local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
pytorchtutorialdocker-train-and-eval-1 |   warn(f"Failed to load image Python extension: {e}")
Epoch: 1, Loss: 0.00408: 100%|██████████| 170/170 [00:28<00:00,  5.92it/s]
pytorchtutorialdocker-train-and-eval-1 | Model saved
pytorchtutorialdocker-train-and-eval-1 | loading annotations into memory...
pytorchtutorialdocker-train-and-eval-1 | Done (t=0.00s)
pytorchtutorialdocker-train-and-eval-1 | creating index...
pytorchtutorialdocker-train-and-eval-1 | index created!
pytorchtutorialdocker-train-and-eval-1 | 
pytorchtutorialdocker-train-and-eval-1 | GroundTruth:    child
pytorchtutorialdocker-train-and-eval-1 | Predicted:      child
pytorchtutorialdocker-train-and-eval-1 | 
pytorchtutorialdocker-train-and-eval-1 | GroundTruth:    adult
pytorchtutorialdocker-train-and-eval-1 | Predicted:      child
pytorchtutorialdocker-train-and-eval-1 | 
pytorchtutorialdocker-train-and-eval-1 | GroundTruth:    adult
pytorchtutorialdocker-train-and-eval-1 | Predicted:      child
pytorchtutorialdocker-train-and-eval-1 | 
pytorchtutorialdocker-train-and-eval-1 | GroundTruth:    child
pytorchtutorialdocker-train-and-eval-1 | Predicted:      child
```