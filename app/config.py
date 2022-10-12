# 서드파티
import dotmap
import os


CONFIG = dotmap.DotMap({
    'batch_size': 4,
    'epochs': 1,
    'classes': ('plane', 
                'car', 
                'bird', 
                'cat', 
                'deer', 
                'dog', 
                'frog', 
                'horse', 
                'ship', 
                'truck'),
    'model_save_dir': os.path.join('.', 'result'),
    'model_name': 'network.pt'
})