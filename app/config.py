# 서드파티
import dotmap
import os


CONFIG = dotmap.DotMap({
    'batch_size': 4,
    'epochs': 1,
    'coco_train_path': os.path.join('.', 'data', 'AdultChild', 'train.json'),
    'coco_test_path': os.path.join('.', 'data', 'AdultChild', 'test.json'),
    'model_save_dir': os.path.join('.', 'result'),
    'model_name': 'network.pt'
})