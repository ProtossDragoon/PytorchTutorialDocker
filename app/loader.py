# 프로젝트
from config import CONFIG

# 서드파티
from pycocotools.coco import COCO
from PIL import Image
import torch
import torchvision.transforms as transforms


class COCOHelper():
    pass


class ClassificatonCOCO(torch.utils.data.Dataset, COCOHelper):
    def __init__(self, json_path):
        self.json_path = json_path
        self.coco = COCO(self.json_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.n_class = len(self.coco.getCatIds())
        self.categories = self.coco.loadCats(self.coco.getCatIds())

    def __len__(self):
        return len(self.coco.imgs)
    
    def __getitem__(self, index):
        img_id   = list(self.coco.imgs.keys())[index]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img      = Image.open(img_path).convert('RGB')
        img      = self.transform(img)
        ann_ids  = self.coco.getAnnIds(imgIds=img_id)
        ann      = self.coco.loadAnns(ann_ids)[0] # each image has only one annotation
        target   = torch.zeros((self.n_class,))
        target[ann['category_id']] = 1.0
        return img, target
    
    
class ClassificatonCOCODataLoader:
    def __init__(self, json_path):
        self.dataset = ClassificatonCOCO(json_path)
        self.batch_size = CONFIG.batch_size
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4)