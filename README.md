# Pytorch Image Classification Tutorial with Docker

간단한 [pytorch classification tutorial](https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) 을 docker 환경에서 실행합니다.

## Command

```
git clone https://github.com/ProtossDragoon/PytorchTutorialDocker.git
cd PytorchTutorialDocker
docker-compose up
```

- 컨테이너 환경에서 간단한 파이토치 분류모델이 CIFAR-10 데이터셋을 바탕으로 학습됩니다.
- 모델 가중치는 `PytorchTutorialDocker/result` 디렉터리에 `network.pt` 파일로 저장됩니다.

## Log

```
pytorchtutorialdocker-train-and-eval-1 | /usr/local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
pytorchtutorialdocker-train-and-eval-1 |   warn(f"Failed to load image Python extension: {e}")
pytorchtutorialdocker-train-and-eval-1 | Files already downloaded and verified
pytorchtutorialdocker-train-and-eval-1 | Files already downloaded and verified
pytorchtutorialdocker-train-and-eval-1 | [1,  2000] loss: 2.204
pytorchtutorialdocker-train-and-eval-1 | [1,  4000] loss: 1.819
pytorchtutorialdocker-train-and-eval-1 | [1,  6000] loss: 1.649
pytorchtutorialdocker-train-and-eval-1 | [1,  8000] loss: 1.532
pytorchtutorialdocker-train-and-eval-1 | [1, 10000] loss: 1.468
pytorchtutorialdocker-train-and-eval-1 | [1, 12000] loss: 1.437
pytorchtutorialdocker-train-and-eval-1 | Finished Training
pytorchtutorialdocker-train-and-eval-1 | Model saved
pytorchtutorialdocker-train-and-eval-1 | GT: cat   ship  ship  plane
pytorchtutorialdocker-train-and-eval-1 | PRED: cat   car   car   ship 
```
